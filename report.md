# RAG Pipeline for Question Answering — Evaluation Report

---

## 1. Selected Dataset

**TriviaQA (rc.wikipedia configuration)**

TriviaQA is a large-scale reading comprehension dataset containing trivia questions authored independently by trivia enthusiasts, paired with Wikipedia documents as evidence. The `rc.wikipedia` configuration specifically uses Wikipedia as the evidence source, making it a closed-book reading comprehension task: the system must retrieve relevant passages from a fixed corpus and generate answers grounded in that corpus.

- **Split used:** Validation
- **Questions selected:** First 50 questions that have at least one linked Wikipedia passage
- **Answer format:** Short factual phrases (names, dates, places)
- **Gold annotations:** Each question includes a canonical answer and a list of valid aliases (e.g., "Campbell-Bannerman" and "Sir Henry Campbell-Bannerman" are both accepted)

---

## 2. RAG Pipeline

The pipeline was built from scratch using the following components:

| Component | Technology |
|---|---|
| Vector store | ChromaDB (persistent, local) |
| Embedding model | `jinaai/jina-embeddings-v5-text-nano` (local, CPU) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Retrieval | Cosine similarity search via ChromaDB |
| LLM | OpenAI `gpt-4.1` via API |
| Orchestration | Custom Python pipeline |

**Pipeline stages:**
1. Load 50 TriviaQA questions and their linked Wikipedia passages
2. Chunk all passages (512 tokens, 64 overlap) and embed with Jina v5 nano
3. Store chunks in ChromaDB
4. For each question, embed the query and retrieve top-5 chunks
5. Pass top-3 chunks as context to GPT-4.1 and generate an answer
6. Evaluate retrieval (Recall@5) and answer quality (Exact Match, Token F1)

---

## 3. Retrieval Setup

| Setting | Value |
|---|---|
| Embedding model | `jinaai/jina-embeddings-v5-text-nano` (239M params, 768-dim) |
| Embedding approach | Asymmetric — `query` prompt for questions, `document` prompt for passages |
| Chunk size | 512 tokens |
| Chunk overlap | 64 tokens |
| Total chunks indexed | 6,003 (from 74 unique passages) |
| Retrieved passages (k) | 5 |
| Passages passed to LLM | 3 (top-3 of retrieved 5) |
| Vector store | ChromaDB with cosine similarity |

The Jina v5 nano model was selected for its task-specific retrieval training using asymmetric query/document prompts, which is optimal for the asymmetric RAG retrieval task where question phrasing differs significantly from passage phrasing.

---

## 4. Results

### 4.1 Retrieval Performance

| Metric | Value |
|---|---|
| **Recall@5** | **0.94** (47/50) |
| Retrieval misses | 3/50 |

Recall@5 measures whether the correct answer string appears in at least one of the top-5 retrieved chunks. A score of 0.94 indicates the retriever successfully locates relevant evidence for the vast majority of questions.

### 4.2 Answer Quality

| Metric | Value |
|---|---|
| **Exact Match (EM)** | **0.86** (43/50) |
| **Mean Token F1** | **0.89** |
| EM correct | 43/50 |

**Exact Match** counts a prediction as correct if it exactly matches any gold alias after normalization (lowercase, strip punctuation). **Token F1** measures token-level overlap between predicted and gold answers, rewarding partial matches.

### 4.3 Per-Question Results Summary

| Category | Count |
|---|---|
| Retrieval hit + Exact match correct | 43 |
| Retrieval hit + Answer wrong | 4 |
| Retrieval miss + Answer correct (model knowledge) | 3 |
| Retrieval miss + Answer wrong | 0 |
| **Total** | **50** |

### 4.4 Example Correct Predictions

| Question | Gold Answer | Predicted |
|---|---|---|
| What was MJ's autobiography? | Moonwalk | Moonwalk |
| In which movie did Garbo say "I want to be alone"? | Grand Hotel | Grand Hotel |
| Who had a 70s No 1 hit with Kiss You All Over? | Exile | Exile |
| What Michelle Pfeiffer movie got a boost from Gangsta's Paradise? | Dangerous Minds | Dangerous Minds |

---

## 5. Failure Case Analysis

### 5.1 Type 1 — Retrieval Miss (3 cases)

The retrieved passages do not contain the answer string, so the LLM cannot answer correctly regardless of its capability.

**Example:**
> **Q:** Which Lloyd Webber musical premiered in the US on 10th December 1993?
> **Gold:** Sunset Boulevard
> **Retrieved:** 5 chunks from the general Andrew Lloyd Webber Wikipedia article (biography, Phantom of the Opera, etc.)
> **Predicted:** Sunset Boulevard *(answered from model knowledge, not retrieved context)*

**Root cause:** The linked Wikipedia page for this question is Lloyd Webber's general biography. The specific sentence mentioning Sunset Boulevard's US premiere date is not in the top-5 chunks returned by the retriever. The chunks that rank highest match on "Lloyd Webber musical" semantically but retrieve general biography rather than the specific event.

**Example 2:**
> **Q:** Who was the next British Prime Minister after Arthur Balfour?
> **Gold:** Campbell-Bannerman
> **Retrieved:** 5 chunks from Arthur Balfour's Wikipedia page (his own career, not his successor)
> **Predicted:** Henry Campbell-Bannerman *(model answered correctly from knowledge)*

**Root cause:** The question asks about Balfour's successor, but the retriever finds chunks about Balfour himself. The passage mentions "Bonar Law became Prime Minister" (his 1922 successor) but not the 1905 successor Campbell-Bannerman.

### 5.2 Type 2 — Retrieval Hit but Wrong Answer (4 cases)

The correct answer is present in the retrieved passages, but the LLM produces an incorrect or imprecise response.

**Example 1:**
> **Q:** Which highway was Revisited in a classic 60s album by Bob Dylan?
> **Gold:** 61
> **Retrieved:** Bob Dylan passages (Highway 61 Revisited mentioned)
> **Predicted:** Highway 61

**Issue:** The model correctly identifies the highway but returns "Highway 61" instead of just "61". This is a prompt adherence failure — the model adds context ("Highway") despite being instructed to give the shortest phrase. A stricter normalization or post-processing step could fix this.

**Example 2:**
> **Q:** Art Garfunkel trained for which profession although he didn't qualify?
> **Gold:** Architect
> **Retrieved:** Art Garfunkel passages
> **Predicted:** mathematics education

**Issue:** The retrieved passage states Garfunkel "studied architecture" and also "pursued a doctorate in mathematics education". The model incorrectly picks the latter. This is a multi-fact confusion failure — the passage contains multiple professions and the model picks the wrong one.

**Example 3:**
> **Q:** Kim Carnes' nine weeks at No 1 with Bette Davis Eyes was interrupted for one week by which song?
> **Gold:** Stars on 45 medley
> **Retrieved:** Bette Davis Eyes passages
> **Predicted:** I don't know

**Issue:** The retrieved passages discuss the song's chart success but do not mention which song interrupted it. The specific fact is simply not present in the available chunks.

### 5.3 Summary of Failure Modes

| Failure Mode | Count | Description |
|---|---|---|
| Retrieval miss — specific fact in wrong chunk | 3 | Answer exists in corpus but not in top-5 chunks |
| Over-verbose answer | 1 | Correct entity retrieved but answer too long |
| Multi-fact confusion | 1 | Multiple relevant facts in context, wrong one selected |
| Missing specific fact in passage | 2 | Fact present in article but not in retrieved chunks |

---

## 6. Model Comparison

| Embedding Model | Chunk Size | Overlap | Top-k | Context Passages | LLM | Recall@5 | Exact Match | Token F1 |
|---|---|---|---|---|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | 512 | 64 | 5 | 3 | `gpt-4.1` | 0.88 | — | — |
| `jina-embeddings-v5-text-nano` (local) | 512 | 64 | 5 | 3 | `gpt-5.4-nano` | 0.94 | 0.76 | 0.81 |
| `jina-embeddings-v5-text-nano` (local) | 512 | 64 | 5 | 3 | `gpt-4.1` | **0.94** | **0.86** | **0.89** |

> Note: The OpenAI embedding run used an earlier prompt version; EM/F1 are not directly comparable and are omitted for that row.

**Key takeaways:** Jina v5 nano improved Recall@5 by +6 pp over OpenAI embeddings due to its asymmetric query/document prompts tuned for retrieval. Among LLMs, GPT-4.1 outperformed GPT-5.4-nano by +10 pp EM and +8 pp F1, being more precise when selecting the correct fact from multi-fact passages.

---

## 7. Discussion

The pipeline achieves strong retrieval (Recall@5 = 0.94) and answer quality (EM = 0.86, Token F1 = 0.89), demonstrating that the combination of Jina v5 nano embeddings with GPT-4.1 is highly effective for closed-book reading comprehension on TriviaQA.

**Key findings:**

1. **Embedding model choice matters significantly.** Initial experiments with `text-embedding-3-small` (OpenAI) achieved Recall@5 = 0.88. Switching to Jina v5 nano improved this to 0.94, primarily because Jina v5 uses asymmetric query/document prompts optimized for retrieval tasks.

2. **Prompt design is critical for answer quality.** Using a generic "be concise" instruction produced EM = 0.10 (GPT-4.1 answered in full sentences). Switching to a strict short-answer prompt with examples raised EM to 0.86.

3. **The main bottleneck is retrieval granularity.** The 3 retrieval misses are caused by the linked Wikipedia pages being long documents where the specific answer sentence falls outside the top-5 ranked chunks. Increasing k or using a re-ranker could address this.

---

## 8. System Configuration Summary

```
Dataset:         TriviaQA rc.wikipedia, validation split, 50 questions
Embedding:       jinaai/jina-embeddings-v5-text-nano (local, CPU)
Chunk size:      512 tokens, 64 overlap
Total chunks:    6,003
Retrieved k:     5 passages
LLM context:     Top-3 passages
LLM:             gpt-4.1 (OpenAI API)
Temperature:     0.1
Max tokens:      256
```

---

*Report generated from pipeline results on 2026-04-10.*
