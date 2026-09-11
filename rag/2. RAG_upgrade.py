'''
코드 분석은 ipynb 파일에서
'''

import os
import re
import numpy as np
import chromadb

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI


# =========================================================
# 기본 설정
# =========================================================

load_dotenv()

DATA_PATH = "data/school.txt"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# multilingual reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OPENAI_MODEL = os.getenv("OPENAI_MODEL")


# =========================================================
# Embedding Model
# =========================================================

embedder = SentenceTransformer(EMBEDDING_MODEL)


# =========================================================
# Level 3
# Chunk + Overlap
# =========================================================

def load_sections(path):
    """
    [도서관], [학사] 같은 section을 읽는다.
    """

    sections = []

    current_section = "일반"
    buffer = []

    with open(path, "r", encoding="utf-8") as f:

        for raw_line in f:

            line = raw_line.strip()

            if not line:
                continue

            # [도서관]
            match = re.fullmatch(r"\[(.+)]", line)

            if match:

                # 이전 section 저장
                if buffer:
                    sections.append(
                        (
                            current_section,
                            " ".join(buffer)
                        )
                    )

                current_section = match.group(1)
                buffer = []

            else:
                buffer.append(line)

    # 마지막 section
    if buffer:
        sections.append(
            (
                current_section,
                " ".join(buffer)
            )
        )

    return sections


def chunk_with_overlap(
    text,
    chunk_size=35,
    overlap=10
):
    """
    단어 기준 chunking.

    예:
    chunk_size = 35
    overlap = 10

    chunk1:
    [0 ~ 34]

    chunk2:
    [25 ~ 59]

    앞 chunk의 마지막 10단어가
    다음 chunk에 다시 들어간다.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "overlap은 chunk_size보다 작아야 합니다."
        )

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end == len(words):
            break

        # overlap
        start = end - overlap

    return chunks


def build_documents(path):

    sections = load_sections(path)

    documents = []

    global_index = 0

    for section, text in sections:

        chunks = chunk_with_overlap(
            text,
            chunk_size=35,
            overlap=10
        )

        for local_index, chunk in enumerate(chunks):

            documents.append(
                {
                    "id": f"chunk-{global_index:03d}",

                    "text": chunk,

                    "metadata": {
                        "source": os.path.basename(path),
                        "section": section,
                        "chunk_index": local_index
                    }
                }
            )

            global_index += 1

    return documents


documents = build_documents(DATA_PATH)


print("\n===== CHUNKS =====")

for doc in documents:
    print(
        doc["id"],
        doc["metadata"],
        doc["text"]
    )


# =========================================================
# Embedding
# =========================================================

texts = [
    doc["text"]
    for doc in documents
]

document_embeddings = embedder.encode(
    texts,
    normalize_embeddings=True
)


# =========================================================
# Level 1
# 직접 Cosine Similarity
# =========================================================

def cosine_search(
    question,
    top_k=3
):

    query_embedding = embedder.encode(
        question,
        normalize_embeddings=True
    )

    # normalize 되어 있기 때문에
    # dot product == cosine similarity

    scores = (
        document_embeddings
        @ query_embedding
    )

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:

        doc = documents[index]

        results.append(
            {
                **doc,
                "score": float(scores[index])
            }
        )

    return results


# =========================================================
# Level 2
# Chroma Vector DB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="school_rag"
)


collection.upsert(
    ids=[
        doc["id"]
        for doc in documents
    ],

    documents=[
        doc["text"]
        for doc in documents
    ],

    embeddings=document_embeddings.tolist(),

    metadatas=[
        doc["metadata"]
        for doc in documents
    ]
)


def vector_search(
    question,
    top_k=10,
    where=None
):

    query_embedding = embedder.encode(
        question,
        normalize_embeddings=True
    )

    query_args = {
        "query_embeddings": [
            query_embedding.tolist()
        ],
        "n_results": min(
            top_k,
            len(documents)
        ),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    # =====================================================
    # Level 4
    # Metadata filtering
    # =====================================================

    if where is not None:
        query_args["where"] = where

    result = collection.query(
        **query_args
    )

    hits = []

    if not result["ids"]:
        return hits

    for (
        doc_id,
        text,
        metadata,
        distance
    ) in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):

        hits.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "vector_distance": float(distance)
            }
        )

    return hits


# =========================================================
# Level 5
# BM25
# =========================================================

def tokenize(text):
    """
    아주 단순한 tokenizer.

    실제 한국어 서비스라면
    Kiwi / MeCab 등을 붙이는 것이 좋다.
    """

    return re.findall(
        r"[가-힣A-Za-z0-9]+",
        text.lower()
    )


tokenized_corpus = [
    tokenize(doc["text"])
    for doc in documents
]

bm25 = BM25Okapi(
    tokenized_corpus
)


def metadata_matches(
    metadata,
    where
):

    if where is None:
        return True

    # 이번 실습에서는
    # {"section": "도서관"}
    # 같은 equality filter만 지원

    for key, value in where.items():

        if metadata.get(key) != value:
            return False

    return True


def bm25_search(
    question,
    top_k=10,
    where=None
):

    query_tokens = tokenize(question)

    scores = bm25.get_scores(
        query_tokens
    )

    candidates = []

    for index, score in enumerate(scores):

        doc = documents[index]

        if not metadata_matches(
            doc["metadata"],
            where
        ):
            continue

        candidates.append(
            {
                **doc,
                "bm25_score": float(score)
            }
        )

    candidates.sort(
        key=lambda x: x["bm25_score"],
        reverse=True
    )

    return candidates[:top_k]


# =========================================================
# Level 5
# Hybrid Search
#
# Vector Search + BM25
#          ↓
#          RRF
# =========================================================

def hybrid_search(
    question,
    top_k=10,
    where=None
):

    vector_hits = vector_search(
        question,
        top_k=top_k,
        where=where
    )

    bm25_hits = bm25_search(
        question,
        top_k=top_k,
        where=where
    )

    fused = {}

    RRF_K = 60


    # ---------------------------------
    # Vector rank
    # ---------------------------------

    for rank, hit in enumerate(
        vector_hits,
        start=1
    ):

        doc_id = hit["id"]

        if doc_id not in fused:

            fused[doc_id] = {
                **hit,
                "rrf_score": 0
            }

        fused[doc_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )


    # ---------------------------------
    # BM25 rank
    # ---------------------------------

    for rank, hit in enumerate(
        bm25_hits,
        start=1
    ):

        doc_id = hit["id"]

        if doc_id not in fused:

            fused[doc_id] = {
                **hit,
                "rrf_score": 0
            }

        fused[doc_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )


    results = list(
        fused.values()
    )

    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# Level 6
# Reranker
# =========================================================

_reranker = None


def get_reranker():

    global _reranker

    if _reranker is None:

        print(
            "\nReranker 모델 로딩..."
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

    return _reranker


def rerank(
    question,
    candidates,
    top_k=3
):

    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [
        (
            question,
            candidate["text"]
        )
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs
    )

    results = []

    for candidate, score in zip(
        candidates,
        scores
    ):

        result = dict(candidate)

        result["rerank_score"] = float(
            np.asarray(score).squeeze()
        )

        results.append(result)

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# Level 7
# Citation 만들기
# =========================================================

def attach_citations(
    hits
):

    cited = []

    for index, hit in enumerate(
        hits,
        start=1
    ):

        item = dict(hit)

        item["citation_id"] = (
            f"C{index}"
        )

        cited.append(item)

    return cited


def build_context(
    cited_hits
):

    context_parts = []

    for hit in cited_hits:

        citation = hit[
            "citation_id"
        ]

        section = hit[
            "metadata"
        ].get(
            "section",
            "Unknown"
        )

        source = hit[
            "metadata"
        ].get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"""
[{citation}]
source: {source}
section: {section}
text: {hit["text"]}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# LLM
# =========================================================

_llm_client = None


def get_llm_client():

    global _llm_client

    if _llm_client is None:

        if not os.getenv(
            "OPENAI_API_KEY"
        ):
            raise RuntimeError(
                "OPENAI_API_KEY가 없습니다."
            )

        if not OPENAI_MODEL:
            raise RuntimeError(
                "OPENAI_MODEL을 .env에 설정하세요."
            )

        _llm_client = OpenAI()

    return _llm_client


def generate_answer(
    question,
    hits
):

    cited_hits = attach_citations(
        hits
    )

    context = build_context(
        cited_hits
    )

    client = get_llm_client()

    response = client.responses.create(

        model=OPENAI_MODEL,

        instructions="""
너는 RAG 기반 질의응답 시스템이다.

반드시 제공된 참고 문서만 사용해서 답변한다.

규칙:

1. 참고 문서에 없는 사실을 만들지 않는다.
2. 사실을 말할 때 반드시 [C1], [C2] 형태로 근거를 표시한다.
3. 관련 근거가 없다면 "제공된 문서에서 확인할 수 없습니다."라고 답한다.
4. citation 번호는 제공된 번호만 사용한다.
5. 짧고 명확하게 답한다.
""",

        input=f"""
[참고 문서]

{context}


[사용자 질문]

{question}
"""
    )

    return (
        response.output_text.strip(),
        cited_hits
    )


# =========================================================
# Level 8
# Grounding / Hallucination Verification
# =========================================================

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def extract_numbers(text):

    return re.findall(
        r"\d+(?:\.\d+)?",
        text
    )


def grounding_check(
    answer,
    cited_hits,
    similarity_threshold=0.35
):

    evidence_map = {
        hit["citation_id"]: hit["text"]
        for hit in cited_hits
    }

    sentences = split_sentences(
        answer
    )

    results = []

    overall_pass = True


    for sentence in sentences:

        # "정보 없음" 답변은 별도 처리
        if (
            "제공된 문서에서 확인할 수 없습니다"
            in sentence
        ):

            results.append(
                {
                    "sentence": sentence,
                    "supported": True,
                    "reason": "정보 부족 응답"
                }
            )

            continue


        # -------------------------------
        # citation 찾기
        # -------------------------------

        citations = re.findall(
            r"\[(C\d+)]",
            sentence
        )


        if not citations:

            overall_pass = False

            results.append(
                {
                    "sentence": sentence,
                    "supported": False,
                    "reason": "citation 없음"
                }
            )

            continue


        # 잘못된 citation 검사

        valid_citations = [
            citation
            for citation in citations
            if citation in evidence_map
        ]


        if not valid_citations:

            overall_pass = False

            results.append(
                {
                    "sentence": sentence,
                    "supported": False,
                    "reason": "존재하지 않는 citation"
                }
            )

            continue


        # citation 제거한 실제 claim

        claim = re.sub(
            r"\[C\d+]",
            "",
            sentence
        ).strip()


        evidence_text = " ".join(
            evidence_map[citation]
            for citation
            in valid_citations
        )


        # -------------------------------
        # Semantic support
        # -------------------------------

        claim_embedding = embedder.encode(
            claim,
            normalize_embeddings=True
        )

        evidence_embedding = embedder.encode(
            evidence_text,
            normalize_embeddings=True
        )

        similarity = float(
            np.dot(
                claim_embedding,
                evidence_embedding
            )
        )


        # -------------------------------
        # 숫자 hallucination 검사
        # -------------------------------

        claim_numbers = extract_numbers(
            claim
        )

        evidence_numbers = extract_numbers(
            evidence_text
        )


        numbers_ok = all(
            number in evidence_numbers
            for number in claim_numbers
        )


        supported = (
            similarity
            >= similarity_threshold
            and numbers_ok
        )


        if not supported:
            overall_pass = False


        results.append(
            {
                "sentence": sentence,
                "supported": supported,
                "similarity": round(
                    similarity,
                    4
                ),
                "numbers_ok": numbers_ok,
                "citations": valid_citations
            }
        )


    return (
        overall_pass,
        results
    )


# =========================================================
# Level 9
# Query Rewrite
# =========================================================

def rewrite_query(
    original_question,
    previous_query,
    grounding_results
):

    client = get_llm_client()

    failures = [
        result
        for result in grounding_results
        if not result["supported"]
    ]

    failure_text = "\n".join(
        str(failure)
        for failure in failures
    )


    response = client.responses.create(

        model=OPENAI_MODEL,

        instructions="""
너는 RAG 검색 질의 최적화기다.

검색 실패 원인을 보고
문서 검색에 더 적합한 한국어 검색 질의 하나를 작성한다.

규칙:

- 원래 질문의 의미는 유지한다.
- 핵심 명사와 조건을 명확하게 만든다.
- 필요한 경우 동의어를 추가한다.
- 답을 직접 하지 않는다.
- 검색 질의 한 줄만 반환한다.
""",

        input=f"""
원래 사용자 질문:
{original_question}

이전 검색 질의:
{previous_query}

Grounding 실패 정보:
{failure_text}

새로운 검색 질의:
"""
    )

    return response.output_text.strip()


# =========================================================
# Level 9
# Retrieve → Generate → Verify → Retry
# =========================================================

def rag(
    question,
    where=None,
    max_retries=2
):

    retrieval_query = question


    for attempt in range(
        max_retries + 1
    ):

        print(
            f"\n========== ATTEMPT {attempt + 1} =========="
        )

        print(
            "Retrieval Query:",
            retrieval_query
        )


        # ---------------------------------
        # Level 5
        # Hybrid Retrieval
        # ---------------------------------

        candidates = hybrid_search(
            retrieval_query,
            top_k=10,
            where=where
        )


        print(
            "\n[Hybrid Search]"
        )

        for candidate in candidates[:5]:

            print(
                candidate["id"],
                round(
                    candidate["rrf_score"],
                    5
                ),
                candidate["text"]
            )


        # ---------------------------------
        # Level 6
        # Reranker
        # ---------------------------------

        reranked = rerank(
            retrieval_query,
            candidates,
            top_k=3
        )


        print(
            "\n[Reranked]"
        )

        for hit in reranked:

            print(
                hit["id"],
                round(
                    hit["rerank_score"],
                    4
                ),
                hit["text"]
            )


        # ---------------------------------
        # Level 7
        # Generate + Citation
        # ---------------------------------

        answer, cited_hits = (
            generate_answer(
                question,
                reranked
            )
        )


        print(
            "\n[Answer]"
        )

        print(answer)


        # ---------------------------------
        # Level 8
        # Grounding
        # ---------------------------------

        grounded, grounding_results = (
            grounding_check(
                answer,
                cited_hits
            )
        )


        print(
            "\n[Grounding]"
        )

        for result in grounding_results:
            print(result)


        # 검증 성공
        if grounded:

            print(
                "\nGROUNDING PASS"
            )

            return answer


        print(
            "\nGROUNDING FAIL"
        )


        # retry 횟수 소진
        if attempt >= max_retries:
            break


        # ---------------------------------
        # Level 9
        # Query Rewrite
        # ---------------------------------

        retrieval_query = (
            rewrite_query(
                question,
                retrieval_query,
                grounding_results
            )
        )


        print(
            "\n[Rewritten Query]"
        )

        print(
            retrieval_query
        )


    return (
        "충분한 근거를 확보하지 못했습니다."
    )


# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":

    while True:

        question = input(
            "\n질문을 입력하세요 (exit 종료): "
        ).strip()

        if question.lower() == "exit":
            break


        answer = rag(
            question
        )

        print(
            "\n================ FINAL ================"
        )

        print(answer)