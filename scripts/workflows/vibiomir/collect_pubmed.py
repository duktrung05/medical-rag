"""Collect a pinned PubMed ID queue with bounded requests and durable resume."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import email.utils
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq

ENDPOINT = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
TERMINAL = ('usable_abstract', 'metadata_only', 'unavailable')
DOC_SCHEMA = pa.schema([
    ('id', pa.int64()), ('official_url', pa.string()), ('pmid', pa.string()),
    ('gpu_shard', pa.int8()), ('status', pa.string()), ('title', pa.string()),
    ('text', pa.string()), ('language', pa.string()), ('metadata_json', pa.string()),
    ('raw_file', pa.string()), ('raw_sha256', pa.string()),
    ('text_sha256', pa.string()), ('error', pa.string()), ('updated_at', pa.string()),
])


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        with temp.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode())


def node_text(node):
    return ''.join(node.itertext()).strip() if node is not None else ''


def parse_records(body):
    """Match records by PMID and retain abstract labels, inline text and language."""
    if b'<!ENTITY' in body.upper():
        raise ValueError('XML entity declarations are not accepted')
    root = ET.fromstring(body)
    if root.tag != 'PubmedArticleSet':
        raise ValueError('Expected PubmedArticleSet, received ' + root.tag)
    if root.find('.//ERROR') is not None:
        raise ValueError('PubMed returned an XML ERROR')
    records = {}
    for record in root:
        if record.tag not in ('PubmedArticle', 'PubmedBookArticle'):
            raise ValueError('Unexpected PubMed record: ' + record.tag)
        citation = record.find('MedlineCitation')
        if citation is None:
            citation = record.find('BookDocument')
        if citation is None:
            raise ValueError('Missing citation')
        pmid = node_text(citation.find('PMID'))
        if not pmid.isdigit() or pmid in records:
            raise ValueError('Missing or duplicate PMID in response')
        article = citation.find('Article') if record.tag == 'PubmedArticle' else citation
        if article is None:
            raise ValueError('Missing article')
        title = node_text(article.find('ArticleTitle'))
        title_source = 'ArticleTitle'
        if not title and record.tag == 'PubmedBookArticle':
            title = node_text(article.find('Book/BookTitle'))
            title_source = 'BookTitle'
        parts = []
        sections = []
        for section in article.findall('Abstract/AbstractText'):
            text = node_text(section)
            label = section.get('Label', '')
            sections.append({'label': label, 'category': section.get('NlmCategory'), 'text': text})
            if text:
                parts.append(f'{label}: {text}' if label else text)
        abstract = '\n\n'.join(parts)
        language = ','.join(node_text(x) for x in article.findall('Language'))
        if not language and record.tag == 'PubmedBookArticle':
            language = ','.join(node_text(x) for x in article.findall('Book/Language'))
        identifiers = {x.get('IdType', ''): node_text(x) for x in record.findall('.//ArticleIdList/ArticleId')}
        metadata = {'record_type': record.tag, 'sections': sections, 'identifiers': identifiers,
                    'other_abstracts': [{'language': x.get('Language'), 'text': node_text(x)}
                                       for x in citation.findall('OtherAbstract')],
                    'publication_date': node_text(article.find('Journal/JournalIssue/PubDate')) or node_text(article.find('Book/PubDate')),
                    'title_source': title_source, 'extractor_version': 'pubmed-xml-v2'}
        text = f'{title}\n\n{abstract}'.strip()
        records[pmid] = {'pmid': pmid, 'title': title, 'text': text, 'language': language,
                         'status': 'usable_abstract' if abstract else 'metadata_only',
                         'metadata_json': json.dumps(metadata, ensure_ascii=False),
                         'text_sha256': sha(text.encode())}
    return records


class RateLimiter:
    def __init__(self, rps=1):
        self.gap = 1 / rps
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            delay = self.next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_at = time.monotonic() + self.gap

    def cooldown(self, seconds):
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic() + seconds)


class ResourceGate:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_time = 0.0
        self.previous_cpu = None
        self.blocked = False

    def wait(self):
        while True:
            with self.lock:
                if time.monotonic() - self.last_time > (10 if self.blocked else 30):
                    memory = {}
                    for line in Path('/proc/meminfo').read_text().splitlines():
                        key, value = line.split(':', 1)
                        memory[key] = int(value.strip().split()[0]) * 1024
                    values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
                    cpu = (sum(values), values[3] + values[4])
                    utilization = 0.0
                    if self.previous_cpu and cpu[0] > self.previous_cpu[0]:
                        utilization = 100 * (1 - (cpu[1] - self.previous_cpu[1]) / (cpu[0] - self.previous_cpu[0]))
                    self.previous_cpu = cpu
                    self.last_time = time.monotonic()
                    self.blocked = (utilization > 70 or memory['MemAvailable'] < 8 * 1024**3
                                    or memory['MemAvailable'] < 0.15 * memory['MemTotal'])
                    if self.blocked:
                        print('Resource pause: CPU %.1f%%, available RAM %.1f GiB' %
                              (utilization, memory['MemAvailable'] / 1024**3), flush=True)
                blocked = self.blocked
            if not blocked:
                return
            time.sleep(5)


def retry_after(value):
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, AttributeError):
            return 30.0


def fetch_batch(pmids, raw_dir, limiter, gate, timeout=60):
    batch_key = sha(','.join(pmids).encode())
    path = raw_dir / (batch_key + '.xml.gz')
    sidecar = path.with_suffix('.json')
    if path.exists() and sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text())
            if meta['requested_pmids'] == pmids and file_sha(path) == meta['gzip_sha256']:
                records = parse_records(gzip.decompress(path.read_bytes()))
                return {'records': records, 'raw_file': str(path), 'raw_sha256': meta['gzip_sha256'],
                        'attempts': [], 'error': None, 'cached': True}
        except (ValueError, OSError, KeyError, ET.ParseError, EOFError):
            pass
    attempts = []
    params = {'db': 'pubmed', 'retmode': 'xml', 'id': ','.join(pmids), 'tool': 'ViBioMIR-PubMed-Collector'}
    for name, variable in [('email', 'NCBI_EMAIL'), ('api_key', 'NCBI_API_KEY')]:
        if os.environ.get(variable):
            params[name] = os.environ[variable]
    for number in range(4):
        gate.wait()
        limiter.wait()
        start = now()
        error = None
        http_status = None
        delay = [5, 15, 45, 0][number] + random.uniform(0, 1)
        try:
            request = urllib.request.Request(ENDPOINT, data=urllib.parse.urlencode(params).encode(),
                      headers={'User-Agent': 'ViBioMIR-PubMed-Collector/1.0',
                               'Content-Type': 'application/x-www-form-urlencoded'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                http_status = response.status
                body = response.read(16 * 1024**2 + 1)
            if len(body) > 16 * 1024**2:
                raise ValueError('Response exceeds 16 MiB; requires smaller batch')
            records = parse_records(body)
            if set(records) - set(pmids):
                raise ValueError('Response contains unrequested PMIDs')
            compressed = gzip.compress(body, compresslevel=5, mtime=0)
            digest = sha(compressed)
            atomic_bytes(path, compressed)
            atomic_json(sidecar, {'requested_pmids': pmids, 'returned_pmids': list(records),
                        'gzip_sha256': digest, 'body_sha256': sha(body), 'fetched_at': now()})
            attempts.append({'start': start, 'end': now(), 'http_status': http_status, 'error': None})
            return {'records': records, 'raw_file': str(path), 'raw_sha256': digest,
                    'attempts': attempts, 'error': None, 'cached': False}
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            error = 'HTTP ' + str(exc.code)
            if exc.code == 429:
                delay = max(delay, retry_after(exc.headers.get('Retry-After')))
                limiter.cooldown(delay)
            if exc.code in (400, 404, 410):
                attempts.append({'start': start, 'end': now(), 'http_status': http_status, 'error': error})
                break
        except (OSError, ValueError, ET.ParseError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:200]
        attempts.append({'start': start, 'end': now(), 'http_status': http_status, 'error': error})
        if number < 3:
            time.sleep(delay)
    return {'records': {}, 'attempts': attempts, 'error': error or 'fetch_failed', 'cached': False}


def connect(path, revision, queue_hash):
    con = sqlite3.connect(path)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA cache_size=-8192')
    con.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, official_url TEXT NOT NULL, pmid TEXT UNIQUE NOT NULL,
            gpu_shard INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            title TEXT DEFAULT '', text TEXT DEFAULT '', language TEXT DEFAULT '',
            metadata_json TEXT DEFAULT '{}', raw_file TEXT DEFAULT '', raw_sha256 TEXT DEFAULT '',
            text_sha256 TEXT DEFAULT '', error TEXT DEFAULT '', updated_at TEXT DEFAULT '', attempts INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS attempts (
            batch_key TEXT, requested_pmids TEXT, start TEXT, end TEXT, http_status INTEGER, error TEXT);
        CREATE INDEX IF NOT EXISTS document_status ON documents(status);
    ''')
    for key, value in [('revision', revision), ('queue_sha256', queue_hash)]:
        old = con.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        if old and old[0] != value:
            raise ValueError('State input mismatch: ' + key)
        con.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', (key, value))
    con.commit()
    return con


def persist_result(con, entries, result):
    pmids = [x['pmid'] for x in entries]
    for attempt in result['attempts']:
        con.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)',
                    (sha(','.join(pmids).encode()), json.dumps(pmids), attempt['start'], attempt['end'],
                     attempt['http_status'], attempt['error']))
    for entry in entries:
        record = result['records'].get(entry['pmid'])
        if record:
            con.execute('''UPDATE documents SET status=?, title=?, text=?, language=?, metadata_json=?,
                         raw_file=?,raw_sha256=?,text_sha256=?,error='',updated_at=?,attempts=attempts+? WHERE id=?''',
                        (record['status'], record['title'], record['text'], record['language'], record['metadata_json'],
                         result['raw_file'], result['raw_sha256'], record['text_sha256'], now(),
                         len(result['attempts']), entry['id']))
        else:
            con.execute('''UPDATE documents SET status=?,error=?,updated_at=?,attempts=attempts+? WHERE id=?''',
                        ('fetch_failed' if result['error'] else 'missing_record', result['error'] or 'PMID absent from response',
                         now(), len(result['attempts']), entry['id']))
    con.commit()


def write_export(con, output, selected_ids, label):
    directory = output / label
    directory.mkdir(exist_ok=True)
    count = 0
    shard = 0
    rows = []
    columns = ','.join(f.name for f in DOC_SCHEMA)
    # Selected IDs live in a temporary table to avoid SQLite parameter limits.
    con.execute('CREATE TEMP TABLE IF NOT EXISTS selected (id INTEGER PRIMARY KEY)')
    con.execute('DELETE FROM selected')
    con.executemany('INSERT INTO selected VALUES (?)', ((x,) for x in selected_ids))
    for row in con.execute(f'SELECT {columns} FROM documents WHERE id IN (SELECT id FROM selected) ORDER BY id'):
        rows.append(dict(zip(DOC_SCHEMA.names, row)))
        if len(rows) >= 5000:
            destination = directory / f'part_{shard:05d}.parquet'
            temp = destination.with_suffix('.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=DOC_SCHEMA), temp, compression='zstd')
            os.replace(temp, destination)
            count += len(rows)
            rows = []
            shard += 1
    if rows:
        destination = directory / f'part_{shard:05d}.parquet'
        temp = destination.with_suffix('.tmp')
        pq.write_table(pa.Table.from_pylist(rows, schema=DOC_SCHEMA), temp, compression='zstd')
        os.replace(temp, destination)
        count += len(rows)
        shard += 1
    for path in directory.glob('part_*.parquet'):
        if int(path.stem.split('_')[1]) >= shard:
            path.unlink()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--pilot-count', type=int, default=0)
    args = parser.parse_args()
    args.queue = args.queue.resolve()
    args.output = args.output.resolve()
    os.nice(10)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:min(8, max(1, len(available) // 2))])
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'collector.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    table = pq.read_table(args.queue)
    entries = table.to_pylist()
    if len({x['id'] for x in entries}) != len(entries) or len({x['pmid'] for x in entries}) != len(entries):
        raise ValueError('Duplicate official IDs or PMIDs')
    if len(entries) != 240000:
        raise ValueError('Expected 240000 pinned IDs')
    for entry in entries:
        parts = urllib.parse.urlsplit(entry['official_url'])
        if parts.hostname != 'pubmed.ncbi.nlm.nih.gov' or parts.path.strip('/') != entry['pmid']:
            raise ValueError('PMID URL mismatch')
    con = connect(args.output / 'state.sqlite', args.revision, file_sha(args.queue))
    con.executemany('INSERT OR IGNORE INTO documents(id,official_url,pmid,gpu_shard) VALUES (?,?,?,?)',
                    ((x['id'], x['official_url'], x['pmid'], x['gpu_shard']) for x in entries))
    con.commit()
    if args.pilot_count:
        indexes = [i * (len(entries) - 1) // (args.pilot_count - 1) for i in range(args.pilot_count)]
        selected = [entries[i] for i in indexes]
    else:
        selected = entries
    selected_ids = [x['id'] for x in selected]
    validity = {}
    invalid_ids = []
    for ident, path, digest in con.execute("SELECT id,raw_file,raw_sha256 FROM documents WHERE status IN ('usable_abstract','metadata_only')"):
        if path and not Path(path).is_absolute():
            path = str(Path(path).resolve())
            con.execute('UPDATE documents SET raw_file=? WHERE id=?', (path, ident))
        if path not in validity:
            validity[path] = file_sha(path) if Path(path).is_file() else None
        if not digest or validity[path] != digest:
            invalid_ids.append(ident)
    con.executemany("UPDATE documents SET status='pending',error='Raw checksum invalid on resume' WHERE id=?", ((x,) for x in invalid_ids))
    con.commit()
    raw = args.output / 'raw'
    raw.mkdir(exist_ok=True)
    limiter, gate = RateLimiter(), ResourceGate()
    start = time.monotonic()
    print('Starting', 'pilot' if args.pilot_count else 'full', len(selected), 'IDs; 4 fetch workers; 1 request/s total', flush=True)
    failure_streak = 0
    for pass_number, batch_size in enumerate([200, 50, 10, 1]):
        status_by_id = dict(con.execute('SELECT id,status FROM documents'))
        pending = [x for x in selected if status_by_id[x['id']] not in TERMINAL]
        if not pending:
            break
        print('Pass', pass_number + 1, 'pending', len(pending), 'batch size', batch_size, flush=True)
        groups = (pending[i:i+batch_size] for i in range(0, len(pending), batch_size))
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            running = {}
            for _ in range(4):
                group = next(groups, None)
                if group:
                    running[pool.submit(fetch_batch, [x['pmid'] for x in group], raw, limiter, gate)] = group
            completed = 0
            while running:
                done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED, timeout=30)
                if not done:
                    print('Waiting for network/resource gate; in flight', len(running), flush=True)
                for future in done:
                    group = running.pop(future)
                    result = future.result()
                    persist_result(con, group, result)
                    failure_streak = failure_streak + 1 if result['error'] else 0
                    completed += len(group)
                    counts = dict(con.execute('SELECT status,count(*) FROM documents GROUP BY status'))
                    atomic_json(args.output / 'progress.json', {'updated_at': now(), 'phase': 'pilot' if args.pilot_count else 'full',
                                'total_queue': len(entries), 'selected': len(selected), 'pass': pass_number + 1,
                                'counts': counts, 'elapsed_seconds': round(time.monotonic() - start, 1),
                                'fetch_workers': 4, 'requests_per_second_total': 1})
                    print('Processed', completed, 'in pass; states', counts, flush=True)
                    if failure_streak >= 8:
                        raise RuntimeError('Eight consecutive failed batches; state saved, pause and diagnose before resume')
                    group = next(groups, None)
                    if group:
                        running[pool.submit(fetch_batch, [x['pmid'] for x in group], raw, limiter, gate)] = group
    # Reconcile true unavailable records without inventing unavailable from a network failure.
    statuses = {x['id']: con.execute('SELECT status FROM documents WHERE id=?', (x['id'],)).fetchone()[0] for x in selected}
    for ident, status in statuses.items():
        if status == 'missing_record':
            con.execute("UPDATE documents SET status='unavailable',error='PMID absent after batch and individual refetch' WHERE id=?", (ident,))
    con.commit()
    label = 'pilot_documents' if args.pilot_count else 'documents'
    exported = write_export(con, args.output, selected_ids, label)
    selected_counts = {}
    for ident in selected_ids:
        status = con.execute('SELECT status FROM documents WHERE id=?', (ident,)).fetchone()[0]
        selected_counts[status] = selected_counts.get(status, 0) + 1
    unresolved = sum(n for status, n in selected_counts.items() if status not in TERMINAL)
    report = {'finished_at': now(), 'revision': args.revision, 'queue_sha256': file_sha(args.queue),
              'selected': len(selected), 'exported': exported, 'states': selected_counts,
              'unresolved': unresolved, 'elapsed_seconds': round(time.monotonic() - start, 1),
              'complete': unresolved == 0, 'usable_abstracts': selected_counts.get('usable_abstract', 0),
              'note': 'metadata_only/unavailable are separate from usable abstract coverage'}
    atomic_json(args.output / ('pilot_report.json' if args.pilot_count else 'collection_report.json'), report)
    atomic_json(args.output / 'progress.json', {'updated_at': now(), 'phase': 'pilot' if args.pilot_count else 'full',
                'total_queue': len(entries), 'selected': len(selected), 'counts': dict(con.execute('SELECT status,count(*) FROM documents GROUP BY status')),
                'elapsed_seconds': report['elapsed_seconds'], 'complete': report['complete'],
                'fetch_workers': 4, 'requests_per_second_total': 1})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    con.close()
    return 0 if not unresolved else 2


if __name__ == '__main__':
    raise SystemExit(main())
