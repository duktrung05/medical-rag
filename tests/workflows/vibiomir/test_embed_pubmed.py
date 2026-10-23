import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from scripts.workflows.vibiomir.embed_pubmed import verify_and_merge, part_fingerprint


class EmbeddingMergeTests(unittest.TestCase):
    def fixture(self, base, duplicate=False):
        state = base/'state.sqlite'
        con = sqlite3.connect(state)
        con.execute('CREATE TABLE documents(id INTEGER,status TEXT)')
        con.executemany('INSERT INTO documents VALUES (?,?)',[(11,'usable_abstract'),(22,'usable_abstract')])
        con.commit(); con.close()
        for shard, ident in [(0,11),(1,11 if duplicate else 22)]:
            directory=base/f'shard_{shard}'; directory.mkdir()
            vectors=np.zeros((1,1024),dtype=np.float16);vectors[0,0]=1
            path=directory/'part_00000.npz'
            np.savez(path,ids=np.array([ident]),vectors=vectors,token_counts=np.array([100]))
            path.with_suffix('.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':1}))
            (directory/'complete.json').write_text(json.dumps({'parts':1,'device':shard}))
        return state

    def test_merge_id_vector_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);state=self.fixture(base)
            verify_and_merge(state,base,'revision')
            self.assertEqual(np.load(base/'pubmed_document_ids.npy').tolist(),[11,22])
            self.assertEqual(np.load(base/'pubmed_document_vectors.npy').shape,(2,1024))
            self.assertTrue(json.loads((base/'embedding_report.json').read_text())['complete'])

    def test_duplicate_shard_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);state=self.fixture(base,duplicate=True)
            with self.assertRaises(ValueError):verify_and_merge(state,base,'revision')

    def test_text_or_config_change_invalidates_cache(self):
        first=part_fingerprint([(11,'text','hash1')],{'sequence':384})
        self.assertNotEqual(first,part_fingerprint([(11,'text','hash2')],{'sequence':384}))
        self.assertNotEqual(first,part_fingerprint([(11,'text','hash1')],{'sequence':256}))


if __name__=='__main__':unittest.main()
