# Retrieval-Augmented Generation for Help.mail.ru

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline tailored to the help.mail.ru knowledge base. It combines:

1. **Retrieval** — fast vector-based search over pre-indexed document chunks
2. **Generation** — large-language-model (LLM) answer synthesis conditioned on retrieved context

Users’ questions are answered by first retrieving the top-k most relevant passages and then generating a concise, context-aware reply.

## Notebooks

The repository includes two Jupyter notebooks:

* `notebook_ru.ipynb`: the full pipeline with comments in Russian.
* `notebook_en.ipynb`: the full pipeline with comments in English.

---

## Theoretical Background

### 1. RAG Formulation

RAG decomposes the answer distribution as

$$
p(a \mid q) \;=\; \sum_{d \in \mathcal{D}_k} p_{\mathrm{gen}}(a \mid q, d)\;p_{\mathrm{ret}}(d \mid q)
$$

where

* $q$ is the user query,
* $\mathcal{D}_k$ are the top-$k$ retrieved documents,
* $p_{\mathrm{ret}}(d\mid q)\propto \exp\big(\mathrm{sim}(E(q),E(d))\big)$ via cosine similarity,
* $p_{\mathrm{gen}}(a\mid q,d)$ is the LLM’s conditional generation model.

### 2. Embedding & Similarity

Each text chunk $t$ is mapped to a vector $E(t)\in\mathbb{R}^d$. We retrieve by ranking on

$$
\mathrm{sim}(u,v) \;=\; \frac{u\cdot v}{\|u\|\;\|v\|}\,
$$

which FAISS indexes for sub-linear nearest-neighbor search.

### 3. Chunking Strategy

Long pages are split into overlapping chunks to fit embedding models’ context windows. Given text length $L$, chunk size $C$, overlap $O$, the number of chunks is

$$
N \;=\; \left\lceil \frac{L - O}{\,C - O\,}\right\rceil.
$$

Overlap $O$ preserves semantic continuity across boundaries.

### 4. Uniformity & Alignment (Wang & Isola, 2020)

We evaluate each embedding model’s geometry via:

* **Uniformity**

  $$
    \mathcal{U}(X) = \log\!\Biggl[\frac{1}{n(n-1)}\sum_{i\neq j} \exp\bigl(-t\,\|x_i - x_j\|^2\bigr)\Biggr],
  $$

  where $X=\{x_i\}$ are normalized embeddings and $t$ a temperature (e.g. $t=2$). Lower (more negative) $\mathcal{U}$ indicates a more even spread on the hypersphere.

* **Alignment**
  We simulate a “positive” view $x_i^+$ via random dropout and compute

  $$
    \mathcal{A}(X) = \frac{1}{n}\sum_{i=1}^n \bigl\|\,\hat x_i - \hat x_i^+ \bigr\|^2,
  $$

  where $\hat x = x/\|x\|$. Smaller $\mathcal{A}$ reflects stability under perturbations.

### 5. t-SNE Visualization

We project high-dimensional embeddings to 2D with t-SNE by minimizing

$$
\mathrm{KL}(P\;\|\;Q) \;=\; \sum_{i\neq j} p_{ij} \,\log\frac{p_{ij}}{q_{ij}},
$$

where $p_{ij}$ and $q_{ij}$ are pairwise affinities in high and low dimensions respectively.

---

## Implementation

1. **Crawling & Extraction**

   * Recursive URL loader up to depth 3
   * BeautifulSoup + regex cleanup of boilerplate

2. **Chunking**

   * `RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)`

3. **Embeddings**

   * Models: `deepvk/USER-bge-m3`, `paraphrase-multilingual-MiniLM-L12-v2`, `LaBSE`
   * Computation via `HuggingFaceEmbeddings`

4. **Indexing**

   * FAISS index saved locally

5. **Metrics**

   * Uniformity & alignment computed in PyTorch
   * Comparison across embedding models

6. **Visualization**

   * t-SNE + Plotly interactive scatter

---

## Results & Visualization


![Visualizing embeddings in space](images/image.png)

*Visualizing embeddings in space*

| Model                                                       | Uniformity | Alignment |
| ----------------------------------------------------------- | ---------: | --------: |
| deepvk/USER-bge-m3                                          |    –1.7844 |    0.1024 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |    –6.7717 |    0.1057 |
| sentence-transformers/LaBSE                                 |    –1.6217 |    0.1029 |

> **Best pick:** MiniLM, for its highly uniform geometry conducive to efficient retrieval.

---

## References

* **Wang & Isola (2020).** *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.*
  [https://arxiv.org/pdf/2005.10242](https://arxiv.org/pdf/2005.10242)
* **Lewis et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
* **FAISS**: Facebook AI Similarity Search — [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* **HuggingFace Transformers** — [https://huggingface.co/transformers](https://huggingface.co/transformers)
