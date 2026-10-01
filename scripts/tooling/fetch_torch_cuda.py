"""Resume the pinned official CUDA wheel in ranges and verify its published SHA256."""

import concurrent.futures
import hashlib
import time
from pathlib import Path

from pip._vendor import requests

URL = 'https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl'
SHA256 = '2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30'
SIZE = 3273024349
CHUNK = 16 * 1024 * 1024


def main():
    directory = Path('.cache/pip/torch_cuda_parts')
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 1800
    count = (SIZE + CHUNK - 1) // CHUNK

    def fetch(index):
        start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
        part = directory / f'{index:04d}.part'
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            if time.monotonic() >= deadline:
                raise TimeoutError('CUDA wheel download exceeded 1800 seconds')
            try:
                with requests.get(URL, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=60) as response:
                    response.raise_for_status()
                    expected = f'bytes {start}-{end}/{SIZE}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError(f'Server did not honor range: {response.headers.get("Content-Range")}')
                    with part.open('wb') as handle:
                        for block in response.iter_content(1024 * 1024):
                            if time.monotonic() >= deadline:
                                raise TimeoutError('CUDA wheel download exceeded 1800 seconds')
                            handle.write(block)
                if part.stat().st_size != end - start + 1:
                    raise ValueError('Incomplete range')
                return
            except Exception:
                if attempt == 3:
                    raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if completed % 10 == 0 or completed == count:
                print(f'CUDA wheel ranges: {completed}/{count}', flush=True)
    wheel = directory.parent / 'torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
    temporary = wheel.with_suffix('.whl.partial')
    digest = hashlib.sha256()
    with temporary.open('wb') as handle:
        for i in range(count):
            with (directory / f'{i:04d}.part').open('rb') as part:
                for block in iter(lambda: part.read(1024 * 1024), b''):
                    digest.update(block)
                    handle.write(block)
    if digest.hexdigest() != SHA256 or temporary.stat().st_size != SIZE:
        raise ValueError('Official CUDA wheel SHA256/size verification failed; refusing installation')
    temporary.replace(wheel)
    print(f'Verified SHA256={SHA256}; wheel={wheel}', flush=True)


if __name__ == '__main__':
    main()
