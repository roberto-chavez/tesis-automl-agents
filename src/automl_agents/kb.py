"""Base de conocimiento RAG sobre ChromaDB (R1.1).

Los embeddings se calculan fuera de Chroma y se le pasan explícitamente, de modo que
el backend de embeddings sea intercambiable sin depender de la versión de Chroma:
  - "sentence-transformers": modelo multilingüe (recomendado; `pip install .[embeddings]`)
  - "hash": embeddings léxicos deterministas, sin descargas (modo offline / tests)
  - "auto": intenta el primero y cae al segundo.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import chromadb

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_STOP = set(
    "a al algo de del la las el los un una unos unas y o u e en con por para que se su sus es son "
    "como mas menos muy sin sobre entre lo le les si no ni the of and or to in for with is are".split()
)


def _norm(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    toks = re.findall(r"[a-z0-9]+", text)
    return [t[:5] for t in toks if t not in _STOP and len(t) > 1]  # stemming por prefijo


class HashEmbedder:
    name = "hash"
    dim = 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            toks = _norm(t)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if "_" not in f else 0.5
            v = [math.log1p(x) for x in v]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class SentenceTransformerEmbedder:
    name = "st"

    def __init__(self, model="paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model)

    def embed(self, texts):
        return self._m.encode(texts, normalize_embeddings=True).tolist()


def make_embedder(backend: str = "auto"):
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if backend != "auto":
                raise
    return HashEmbedder()


@dataclass
class Doc:
    id: str
    title: str
    text: str
    kind: str
    score: float = 0.0

    def to_dict(self):
        return {"id": self.id, "title": self.title, "kind": self.kind, "score": round(self.score, 3)}


class KnowledgeBase:
    def __init__(self, persist_dir: str | Path | None = None, backend: str = "auto",
                 knowledge_dir: Path = KNOWLEDGE_DIR):
        self.embedder = make_embedder(backend)
        self.knowledge_dir = Path(knowledge_dir)
        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self.client = chromadb.EphemeralClient()
        # El nombre incluye el backend: cada uno tiene su propia dimensionalidad.
        self.collection = self.client.get_or_create_collection(
            f"kb_{self.embedder.name}", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._docs: dict[str, Doc] = {}
        self.build()

    def _load(self) -> list[Doc]:
        docs = []
        for p in sorted(self.knowledge_dir.glob("*.md")):
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            title = lines[0].lstrip("# ").strip()
            docs.append(Doc(p.stem, title, "\n".join(lines[1:]).strip(), p.stem.split("_")[0]))
        return docs

    def build(self) -> int:
        docs = self._load()
        self._docs = {d.id: d for d in docs}
        texts = [f"{d.title}. {d.text}" for d in docs]
        self.collection.upsert(
            ids=[d.id for d in docs], documents=texts,
            embeddings=self.embedder.embed(texts),
            metadatas=[{"title": d.title, "kind": d.kind} for d in docs],
        )
        return len(docs)

    def query(self, text: str, k: int = 3) -> list[Doc]:
        res = self.collection.query(query_embeddings=self.embedder.embed([text]), n_results=k)
        out = []
        for i, dist in zip(res["ids"][0], res["distances"][0]):
            d = self._docs[i]
            out.append(Doc(d.id, d.title, d.text, d.kind, score=1.0 - float(dist)))
        return out

    def retrieve_many(self, queries: list[str], k: int = 3) -> list[Doc]:
        seen: dict[str, Doc] = {}
        for q in queries:
            for d in self.query(q, k):
                if d.id not in seen or d.score > seen[d.id].score:
                    seen[d.id] = d
        return sorted(seen.values(), key=lambda d: -d.score)

    def evaluate(self, reference: list[dict] | None = None, k: int = 3) -> dict:
        """Precisión@k y exhaustividad@k frente a un conjunto de referencia etiquetado a mano."""
        if reference is None:
            reference = json.loads((self.knowledge_dir / "reference_queries.json").read_text("utf-8"))
        precs, recs = [], []
        for item in reference:
            got = {d.id for d in self.query(item["query"], k)}
            rel = set(item["relevant"])
            hit = len(got & rel)
            precs.append(hit / k)
            recs.append(hit / len(rel))
        return {
            "k": k, "n_queries": len(reference), "backend": self.embedder.name,
            "precision_at_k": round(sum(precs) / len(precs), 3),
            "recall_at_k": round(sum(recs) / len(recs), 3),
        }
