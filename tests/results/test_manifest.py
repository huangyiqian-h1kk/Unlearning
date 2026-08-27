import hashlib, importlib.util, json, pathlib, tarfile, tempfile, unittest

ROOT=pathlib.Path(__file__).parents[2]
P=ROOT/"scripts/results/extract_archived_metrics.py"; S=importlib.util.spec_from_file_location("extract",P); e=importlib.util.module_from_spec(S); S.loader.exec_module(e)

class ManifestTests(unittest.TestCase):
 def test_manifest_shape_and_hashes(self):
  with open(ROOT/'results/paper/manifest.json') as f: m=json.load(f)
  self.assertEqual(m['probe_order'],['ATT','IDeq','ID']); self.assertEqual(len(m['experiments']),25)
  for a in m['archives']: self.assertEqual(e.sha256_file(ROOT/a['path']),a['sha256'])
  ids={x['id'] for x in m['experiments']}; self.assertEqual(len(ids),25)
 def test_member_hash_validation_and_missing(self):
  with tempfile.TemporaryDirectory() as d:
   arc=pathlib.Path(d)/'a.tar.gz'; payload=b'{"x":1}'
   import io
   with tarfile.open(arc,'w:gz') as t:
    i=tarfile.TarInfo('ok.json');i.size=len(payload);t.addfile(i,io.BytesIO(payload))
   ah=e.sha256_file(arc); mh=hashlib.sha256(payload).hexdigest()
   self.assertEqual(e.read_member(arc,ah,'ok.json',mh),payload)
   with self.assertRaises(FileNotFoundError): e.read_member(arc,ah,'missing.json',mh)
   with self.assertRaises(ValueError): e.read_member(arc,'0'*64,'ok.json',mh)
 def test_builder_has_no_tar_dependency(self):
  text=(ROOT/'scripts/results/build_tables.py').read_text(); self.assertNotIn('tarfile',text); self.assertNotIn('clinicia_provenance_bundle',text)

if __name__=='__main__': unittest.main()
