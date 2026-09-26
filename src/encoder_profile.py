"""Model-specific pooling and input conventions, shared by configuration and retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderProfile:
    name: str
    pooling: str
    query_prefix: str
    passage_prefix: str


def resolve_encoder_profile(model_name: str | None, profile: str = 'auto',
                            pooling: str | None = None, query_prefix: str | None = None,
                            passage_prefix: str | None = None) -> EncoderProfile:
    name = (model_name or '').casefold().replace('\\', '/').rstrip('/').split('/')[-1]
    if profile == 'auto':
        profile = 'bge_m3' if name == 'bge-m3' else 'e5' if name.startswith(('multilingual-e5-', 'e5-')) else 'generic'
    defaults = {
        'bge_m3': ('cls', '', ''),
        'e5': ('mean', 'query: ', 'passage: '),
        'generic': ('mean', '', ''),
    }
    if profile not in defaults:
        raise ValueError(f'Unknown encoder profile: {profile}')
    default_pooling, default_query, default_passage = defaults[profile]
    resolved = EncoderProfile(profile, pooling or default_pooling,
                              default_query if query_prefix is None else query_prefix,
                              default_passage if passage_prefix is None else passage_prefix)
    if resolved.pooling not in ('cls', 'mean'):
        raise ValueError(f'Unknown pooling: {resolved.pooling}')
    if profile in ('bge_m3', 'e5') and (resolved.pooling, resolved.query_prefix, resolved.passage_prefix) != defaults[profile]:
        raise ValueError(f'{profile} requires pooling={default_pooling!r}, '
                         f'query_prefix={default_query!r}, passage_prefix={default_passage!r}')
    return resolved
