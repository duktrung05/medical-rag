"""Meaningful offline checks for PMID mapping, resume and bounded API requests."""
import gzip
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts.workflows.vibiomir.collect_pubmed import (
    connect, fetch_batch, parse_records, persist_result, RateLimiter, sha,
)

XML = b'''<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>22</PMID><Article>
<ArticleTitle>Title <i>inline</i></ArticleTitle><Abstract>
<AbstractText Label="RESULTS">Dose <sup>2</sup> improved.</AbstractText>
<AbstractText Label="CONCLUSIONS">Works.</AbstractText></Abstract><Language>eng</Language>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>11</PMID><Article><ArticleTitle>Short</ArticleTitle>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'''


class PubmedTests(unittest.TestCase):
    def test_mapping_order_inline_and_metadata_only(self):
        rows = parse_records(XML)
        self.assertEqual(list(rows), ['22', '11'])
        self.assertEqual(rows['22']['text'], 'Title inline\n\nRESULTS: Dose 2 improved.\n\nCONCLUSIONS: Works.')
        self.assertEqual(rows['22']['language'], 'eng')
        self.assertEqual(rows['11']['status'], 'metadata_only')

    def test_reject_error_duplicate_and_entity(self):
        for body in [b'<eFetchResult><ERROR>bad</ERROR></eFetchResult>',
                     XML.replace(b'<PMID>11</PMID>', b'<PMID>22</PMID>'),
                     b'<!DOCTYPE x [<!ENTITY test "x">]><PubmedArticleSet/>']:
            with self.assertRaises(ValueError):
                parse_records(body)

    def test_book_title_language_and_abstract_fallback(self):
        body=b'''<PubmedArticleSet><PubmedBookArticle><BookDocument>
        <PMID>33</PMID><Book><BookTitle>Drug class review</BookTitle><Language>eng</Language></Book>
        <Abstract><AbstractText>Evidence.</AbstractText></Abstract></BookDocument></PubmedBookArticle></PubmedArticleSet>'''
        row=parse_records(body)['33']
        self.assertEqual(row['title'],'Drug class review')
        self.assertEqual(row['language'],'eng')
        self.assertEqual(row['text'],'Drug class review\n\nEvidence.')

    def test_partial_batch_persist_and_state_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.sqlite'
            con = connect(path, 'revision', 'queuehash')
            entries = [{'id': 100, 'pmid': '22'}, {'id': 200, 'pmid': '33'}]
            con.executemany('INSERT INTO documents(id,official_url,pmid,gpu_shard) VALUES (?,?,?,?)',
                            [(100, 'url22', '22', 0), (200, 'url33', '33', 1)])
            persist_result(con, entries, {'records': parse_records(XML), 'attempts': [],
                                         'raw_file': 'raw', 'raw_sha256': 'hash', 'error': None})
            self.assertEqual(dict(con.execute('SELECT id,status FROM documents')),
                             {100:'usable_abstract',200:'missing_record'})
            con.close()
            with self.assertRaises(ValueError):
                connect(path, 'other_revision', 'queuehash')

    def test_raw_cache_hash_corruption_requires_refetch(self):
        class Gate:
            def wait(self): pass
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, n): return XML
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            with patch('urllib.request.urlopen', return_value=Response()) as request:
                first = fetch_batch(['22','11'], raw, RateLimiter(1000), Gate())
                self.assertEqual(request.call_count, 1)
                second = fetch_batch(['22','11'], raw, RateLimiter(1000), Gate())
                self.assertTrue(second['cached'])
                self.assertEqual(request.call_count, 1)
                Path(first['raw_file']).write_bytes(b'corrupt')
                repaired = fetch_batch(['22','11'], raw, RateLimiter(1000), Gate())
                self.assertEqual(request.call_count, 2)
                self.assertEqual(gzip.decompress(Path(repaired['raw_file']).read_bytes()), XML)

    def test_rate_limiter_shared_threads(self):
        limiter = RateLimiter(50)
        timestamps = []
        def worker():
            limiter.wait()
            timestamps.append(time.monotonic())
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        ordered = sorted(timestamps)
        self.assertTrue(all(b-a >= 0.018 for a,b in zip(ordered,ordered[1:])))


if __name__ == '__main__':
    unittest.main()
