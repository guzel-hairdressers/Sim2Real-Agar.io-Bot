import json
from pathlib import Path
import tempfile
import unittest

from scripts.promote_split_combat import promote, ROOT


class SplitPromotionTests(unittest.TestCase):
    directory = ROOT/'outputs/split_combat_v3'

    def paths(self):
        return [self.directory/'combat_acceptance.json', self.directory/'ppo_retention.json',
                self.directory/'diffusion_retention.json']

    def test_verified_combat_and_retention_build_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'deployment.json'
            manifest = promote(self.directory, *self.paths(), output)
            self.assertEqual(set(manifest['models']), {'ppo','diffusion','critic'})
            self.assertEqual(json.loads(output.read_text()),manifest)

    def reject_combat_mutation(self, mutate, expected):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths()
            report = json.loads(paths[0].read_text())
            mutate(report)
            failed = Path(directory)/'failed.json'
            failed.write_text(json.dumps(report)); paths[0] = failed
            output = Path(directory)/'deployment.json'
            output.write_text('previous deployment')
            with self.assertRaisesRegex(ValueError,expected):
                promote(self.directory, *paths, output)
            self.assertEqual(output.read_text(),'previous deployment')

    def test_changed_checkpoint_is_rejected(self):
        self.reject_combat_mutation(lambda r:r['checkpoint_hashes'].update({'ppo.pt':'0'*64}),
                                    'differs from combat checkpoint')

    def test_changed_source_is_rejected(self):
        self.reject_combat_mutation(lambda r:r['source_hashes'].update({'src/arena_policies.py':'0'*64}),
                                    'Evaluated source has changed')

    def test_missing_source_hash_is_rejected(self):
        self.reject_combat_mutation(lambda r:r['source_hashes'].pop('envs/agar_env.py'),
                                    'omits required source hashes')

    def test_duplicate_cases_are_rejected(self):
        self.reject_combat_mutation(lambda r:r['rows'].append(r['rows'][0]),
                                    'distinct fresh cases')

    def test_failed_combat_cannot_pass_by_changing_summary(self):
        def no_captures(report):
            for row in report['rows']:
                row['candidate']['projectile_capture'] = False
        self.reject_combat_mutation(no_captures,'failed physical combat acceptance')

    def test_failed_retention_does_not_overwrite_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths()
            report = json.loads(paths[1].read_text())
            for row in report['rows']:
                row['groups']['new']['final_mass'] = 0
            failed = Path(directory)/'failed.json'
            failed.write_text(json.dumps(report)); paths[1] = failed
            output = Path(directory)/'deployment.json'
            output.write_text('previous deployment')
            with self.assertRaisesRegex(ValueError,'failed v2 retention tolerance'):
                promote(self.directory, *paths, output)
            self.assertEqual(output.read_text(),'previous deployment')


if __name__ == '__main__':
    unittest.main()
