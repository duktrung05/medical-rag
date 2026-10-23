from src.medrag.core.contracts import Passage, Source


def verify(passage: Passage, source: Source) -> bool:
    return (passage.corpus_version == source.corpus_version
            and passage.official_id == source.official_id
            and passage.url == source.url
            and passage.source_sha256 == source.text_sha256
            and passage.source_start is not None and passage.source_end is not None
            and 0 <= passage.source_start < passage.source_end <= len(source.text)
            and source.text[passage.source_start:passage.source_end] == passage.text)

