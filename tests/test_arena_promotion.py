import json
from pathlib import Path
import tempfile
import unittest

from scripts.promote_arena_v2 import promote, ROOT


class PromotionTests(unittest.TestCase):
    def paths(self):
        return [ROOT/'outputs/arena_v2/ppo.pt', ROOT/'outputs/arena_v2_cosine/diffusion.pt',
                ROOT/'outputs/arena_v2_cosine/critic.pt', ROOT/'outputs/arena_v2/ppo_acceptance.json',
                ROOT/'outputs/arena_v2_cosine/diffusion_acceptance.json']

    def test_verified_reports_can_build_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'deployment.json'
            manifest = promote(*self.paths(), output)
            self.assertEqual(set(manifest['models']), {'ppo', 'diffusion', 'critic'})
            self.assertEqual(json.loads(output.read_text()), manifest)

    def test_failed_acceptance_does_not_overwrite_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths()
            report = json.loads(paths[3].read_text())
            report['summary']['improved'] = False
            failed = Path(directory)/'failed.json'
            failed.write_text(json.dumps(report))
            paths[3] = failed
            output = Path(directory)/'deployment.json'
            output.write_text('existing accepted deployment')
            with self.assertRaises(ValueError):
                promote(*paths, output)
            self.assertEqual(output.read_text(), 'existing accepted deployment')


if __name__ == '__main__':
    unittest.main()
