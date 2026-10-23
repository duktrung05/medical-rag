class BGEM3Encoder:
    """Pinned BGE-M3 adapter with distinct document/query input lengths."""

    def __init__(self, model_name: str = "BAAI/bge-m3", *, revision: str,
                 document_max_length: int = 512, query_max_length: int = 512,
                 device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers to enable BGE-M3") from exc
        from src.medrag.core.model_revision import require_cached_revision

        require_cached_revision(model_name, revision)
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self.revision = revision
        self.document_max_length = document_max_length
        self.query_max_length = query_max_length

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.model.max_seq_length = self.document_max_length
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        self.model.max_seq_length = self.query_max_length
        return self.model.encode(texts, normalize_embeddings=True).tolist()

