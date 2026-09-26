"""Resume the pinned official Reranker model in ranges and verify its published SHA256."""

import concurrent.futures
import hashlib
import time
from pathlib import Path

from pip._vendor import requests

URL = 'https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e/model.safetensors'
SHA256 = 'd9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286'
SIZE = 2271071852
CHUNK = 4 * 1024 * 1024


def main():
    directory = Path('.cache/huggingface/reranker_parts_4mb')
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 3600
    count = (SIZE + CHUNK - 1) // CHUNK

    def fetch(index):
        start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
        part = directory / f'{index:04d}.part'
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            if time.monotonic() >= deadline:
                raise TimeoutError('Reranker model download exceeded 3600 seconds')
            try:
                with requests.get(URL + f"?range={index}", headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=30) as response:
                    response.raise_for_status()
                    expected = f'bytes {start}-{end}/{SIZE}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError(f'Server did not honor range: {response.headers.get("Content-Range")}')
                    with part.open('wb') as handle:
                        for block in response.iter_content(1024 * 1024):
                            if time.monotonic() >= deadline:
                                raise TimeoutError('Reranker model download exceeded 3600 seconds')
                            handle.write(block)
                if part.stat().st_size != end - start + 1:
                    raise ValueError('Incomplete range')
                return
            except Exception:
                if attempt == 3:
                    raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if completed % 10 == 0 or completed == count:
                print(f'Reranker model ranges: {completed}/{count}', flush=True)
    model_file = Path('.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e/model.safetensors')
    temporary = model_file.with_suffix('.safetensors.partial')
    digest = hashlib.sha256()
    with temporary.open('wb') as handle:
        for i in range(count):
            with (directory / f'{i:04d}.part').open('rb') as part:
                for block in iter(lambda: part.read(1024 * 1024), b''):
                    digest.update(block)
                    handle.write(block)
    if digest.hexdigest() != SHA256 or temporary.stat().st_size != SIZE:
        raise ValueError('Official Reranker model SHA256/size verification failed; refusing publication')
    temporary.replace(model_file)
    print(f'Verified SHA256={SHA256}; model_file={model_file}', flush=True)


if __name__ == '__main__':
    main()
