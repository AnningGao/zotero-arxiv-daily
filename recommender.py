import numpy as np
import mlx.core as mx
from mlx_embedding_models.embedding import EmbeddingModel
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='bge-m3') -> list[ArxivPaper]:
    # Use MLX-based embedding model for Apple Silicon
    # bge-m3 is SOTA with 1024 dims and 8192 max context length
    encoder = EmbeddingModel.from_registry(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    # time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1)/2)
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    # Compute cosine similarity using MLX
    corpus_feature = mx.array(corpus_feature)
    candidate_feature = mx.array(candidate_feature)
    # Normalize for cosine similarity
    corpus_norm = corpus_feature / mx.linalg.norm(corpus_feature, axis=1, keepdims=True)
    candidate_norm = candidate_feature / mx.linalg.norm(candidate_feature, axis=1, keepdims=True)
    sim = candidate_norm @ corpus_norm.T  # [n_candidate, n_corpus]
    sim = np.array(sim)  # Convert back to numpy for scoring
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate