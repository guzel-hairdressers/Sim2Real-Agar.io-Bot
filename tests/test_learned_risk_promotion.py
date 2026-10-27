import json
from pathlib import Path
import tempfile
import unittest

from scripts.promote_learned_risk import ROOT, full_match_protocol, promote


class LearnedRiskPromotionTests(unittest.TestCase):
    def report(self):
        return json.loads((ROOT/'outputs/learned_risk_v7/ppo_development_retention.json').read_text())

    def test_actual_development_matches_have_complete_timelines(self):
        full_match_protocol(self.report())

    def test_shortened_or_incomplete_match_cannot_claim_full_duration(self):
        for mutation in ('timeline','roster','steps'):
            report=self.report()
            if mutation=='timeline':report['rows'][0]['timeline'].pop()
            elif mutation=='roster':report['rows'][0]['players'].pop()
            else:report['protocol']['steps_per_match']=60
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                full_match_protocol(report)

    def test_archived_candidate_cannot_overwrite_deployment(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'deployment.json';output.write_text('previous accepted deployment')
            with self.assertRaisesRegex(ValueError,'Evaluated source changed'):
                promote(ROOT/'outputs/learned_risk_v7',output)
            self.assertEqual(output.read_text(),'previous accepted deployment')


if __name__=='__main__':unittest.main()
