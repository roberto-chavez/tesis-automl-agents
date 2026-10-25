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
from dataclasses import dataclass, asdict, replace
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
    source: str | None = None
    date: str | None = None
    scope: str | None = None
    version: str = ""

    def __post_init__(self):
        # Mantener KnowledgeBase utilizable como módulo autónomo en el inspector Colab.
        data = {k: v for k, v in asdict(self).items() if k not in ("score", "version")}
        self.version = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False,
                                                separators=(",", ":")).encode()).hexdigest()

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "text"}


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
        suffix = ("_" + hashlib.sha256(str(self.knowledge_dir.resolve()).encode()).hexdigest()[:12]
                  if self.knowledge_dir.resolve() != KNOWLEDGE_DIR.resolve() else "")
        self.collection = self.client.get_or_create_collection(
            f"kb_{self.embedder.name}{suffix}", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._docs: dict[str, Doc] = {}
        self.build()

    def _load(self) -> list[Doc]:
        docs = []
        for p in sorted(self.knowledge_dir.glob("*.md")):
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            metadata = {}
            if lines and lines[0] == "---" and "---" in lines[1:]:
                end = lines.index("---", 1)
                for line in lines[1:end]:
                    key, sep, value = line.partition(":")
                    if sep and key.strip() in ("source", "date", "scope", "kind"):
                        metadata[key.strip()] = value.strip().strip("\"'")
                lines = lines[end + 1:]
            while lines and not lines[0].strip():
                lines.pop(0)
            if not lines:
                continue
            title = lines[0].lstrip("# ").strip()
            docs.append(Doc(p.stem, title, "\n".join(lines[1:]).strip(),
                            metadata.pop("kind", p.stem.split("_")[0]), **metadata))
        return docs

    def build(self) -> int:
        docs = self._load()
        self._docs = {d.id: d for d in docs}
        texts = [f"{d.title}. {d.text}" for d in docs]
        obsolete = set(self.collection.get()["ids"]) - set(self._docs)
        if obsolete:
            self.collection.delete(ids=list(obsolete))
        if not docs:
            return 0
        self.collection.upsert(
            ids=[d.id for d in docs], documents=texts,
            embeddings=self.embedder.embed(texts),
            metadatas=[{k: v for k, v in d.to_dict().items() if v is not None} for d in docs],
        )
        return len(docs)

    def query(self, text: str, k: int = 3, *, kinds: list[str] | None = None) -> list[Doc]:
        eligible = [d for d in self._docs.values() if not kinds or d.kind in kinds]
        if k < 1 or not eligible:
            return []
        kwargs = {"where": {"kind": {"$in": kinds}}} if kinds else {}
        res = self.collection.query(query_embeddings=self.embedder.embed([text]),
                                    n_results=min(k, len(eligible)), **kwargs)
        out = []
        for i, dist in zip(res["ids"][0], res["distances"][0]):
            d = self._docs[i]
            out.append(replace(d, score=1.0 - float(dist)))
        return out

    def retrieve_many(self, queries: list[str], k: int = 3, *, kinds: list[str] | None = None,
                      max_documents: int | None = None, max_tokens: int | None = None) -> list[Doc]:
        seen: dict[str, Doc] = {}
        for q in queries:
            for d in self.query(q, k, kinds=kinds):
                if d.id not in seen or d.score > seen[d.id].score:
                    seen[d.id] = d
        result, used = [], 0
        for d in sorted(seen.values(), key=lambda d: -d.score):
            tokens = math.ceil(len((d.title + "\n" + d.text).encode("utf-8")) / 3)
            if max_documents is not None and len(result) >= max_documents:
                break
            if max_tokens is not None and used + tokens > max_tokens:
                continue
            result.append(d)
            used += tokens
        return result

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


class EvidenceManager:
    """Límites acumulados de versiones únicas, compartidos por todos los roles.

    El host serializa llamadas con run_lock. Ledger, evidencia y trazabilidad se
    confirman en una transacción; repetir una versión nunca renueva su cargo.
    """
    def __init__(self, store, cfg):
        self.store = store
        values = {}
        if store.get("rag_limits") is None:
            values["rag_limits"] = {"documents": cfg.rag_max_documents, "tokens": cfg.rag_max_tokens}
        if store.get("rag_ledger") is None:
            # Migrar también si la reanudación no realiza otra consulta RAG.
            evidence = {key: asdict(Doc(**data)) for key, data in store.get("tool_evidence", {}).items()}
            values["tool_evidence"] = evidence
            values["rag_ledger"] = {d["id"] + ":" + d["version"]:
                math.ceil(len((d["title"] + "\n" + d["text"]).encode("utf-8")) / 3)
                for d in evidence.values()}
        with store.db:
            for key, value in values.items():
                store.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def retrieve(self, kb, query, k, *, role, action_id, kinds=None):
        from .context import SCOPE_ROLES, token_count
        from .state_store import fingerprint
        if kb is None:
            docs = []
        else:
            import inspect
            parameters = inspect.signature(kb.query).parameters
            filters_supported = "kinds" in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
            docs = kb.query(query, k, kinds=kinds) if kinds and filters_supported else kb.query(query, k)
        evidence = self.store.get("tool_evidence", {})
        ledger = self.store.get("rag_ledger", {})
        # Migración de evidencia de ejecuciones previas sin ledger.
        for data in evidence.values():
            d = Doc(**data)
            ledger.setdefault(d.id + ":" + d.version, token_count(d.title + "\n" + d.text)[0])
        limits = self.store.get("rag_limits")
        tokens = sum(ledger.values())
        refs, seen, omitted = [], set(), 0
        artifacts = {}
        for raw in docs:
            d = Doc(**{name: getattr(raw, name) for name in Doc.__dataclass_fields__ if hasattr(raw, name)})
            if (kinds and d.kind not in kinds) or d.scope not in (None, "general", "train", "validation"):
                continue
            if role == "configuration" and d.scope == "validation":
                continue
            key = d.id + ":" + d.version
            if key in seen:
                continue
            seen.add(key)
            cost = token_count(d.title + "\n" + d.text)[0]
            if key not in ledger:
                if len(ledger) >= limits["documents"] or tokens + cost > limits["tokens"]:
                    omitted += 1
                    continue
                ledger[key] = cost
                tokens += cost
            evidence[d.id] = asdict(d)
            # Una guía con alcance validation tampoco se puede abrir desde configuración.
            scope = "validation" if d.scope == "validation" else "evidence"
            version = fingerprint(d.text)
            artifact_id = fingerprint({"scope": scope, "version": version})
            artifacts["safe_artifact:" + artifact_id] = {"data": d.text, "scope": scope,
                "roles": list(SCOPE_ROLES[scope]),
                "version": version}
            refs.append({**d.to_dict(), "artifact_id": artifact_id})
        retrieval_id = fingerprint({"action_id": action_id, "role": role})
        result = {"documents": refs, "retrieval_id": retrieval_id,
                  "evidence_revision": fingerprint([(d["id"], d["version"]) for d in refs]),
                  "usage": {"documents": len(ledger), "tokens": tokens}, "limits": limits,
                  "omitted": omitted, "token_count_method": "estimated_utf8_bytes"}
        values = {"tool_evidence": evidence, "rag_ledger": ledger,
                  "rag_retrieval:" + retrieval_id: {"query": query, "k": k, "kinds": kinds, "role": role,
                    "action_id": action_id, **result}, **artifacts}
        with self.store.db:
            for key, value in values.items():
                self.store.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))
        return result
