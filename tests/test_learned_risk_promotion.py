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

    def reject_current_mutation(self,name,mutate,expected):
        current=ROOT/'outputs/learned_risk_v8'
        names=['acceptance_protocol.json','ppo.pt','diffusion.pt','outcomes.json',
               'ppo_retention.json','diffusion_retention.json','league.json']
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for item in names:
                if item==name:
                    report=json.loads((current/item).read_text());mutate(report)
                    (folder/item).write_text(json.dumps(report))
                else:(folder/item).symlink_to(current/item)
            output=folder/'deployment.json';output.write_text('previous accepted deployment')
            with self.assertRaisesRegex(ValueError,expected):promote(folder,output)
            self.assertEqual(output.read_text(),'previous accepted deployment')

    def test_current_survival_failure_cannot_build_accepted_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'deployment.json'
            with self.assertRaisesRegex(ValueError,'ppo failed navigation retention'):
                promote(ROOT/'outputs/learned_risk_v8',output)
            self.assertFalse(output.exists())

    def test_changed_evaluated_source_is_rejected(self):
        self.reject_current_mutation('outcomes.json',
            lambda r:r['source_hashes'].update({'envs/agar_env.py':'0'*64}),
            'Evaluated source changed')

    def test_changed_evaluated_checkpoint_is_rejected(self):
        self.reject_current_mutation('outcomes.json',
            lambda r:r['checkpoint_hashes'].update({'ppo':'0'*64}),
            'Outcome checkpoint hash mismatch')

    def test_duplicate_outcome_case_is_rejected(self):
        def duplicate(report):
            row=next(r for r in report['rows'] if r['algorithm']=='ppo' and r['label'] and
                     r['candidate']['deaths']==r['baseline']['deaths']==0 and
                     r['candidate']['utility']>=r['baseline']['utility'])
            report['rows'].append(row)
        self.reject_current_mutation('outcomes.json',duplicate,'Duplicated outcome cases')

    def test_forged_success_summary_cannot_hide_physical_violation(self):
        def violate(report):
            report['rows'][0]['candidate']['physical_violations']+=1
            for value in report['summary'].values():value['passed']=True
        self.reject_current_mutation('outcomes.json',violate,'failed outcome acceptance')

    def test_failed_retention_cannot_be_hidden_by_summary(self):
        def fail(report):
            report['summary']['passed']=True
        self.reject_current_mutation('ppo_retention.json',fail,'failed navigation retention')

    def test_protocol_cannot_change_after_fitting(self):
        self.reject_current_mutation('acceptance_protocol.json',
            lambda r:r.update(final_encounter_seed=r['final_encounter_seed']+1),
            'Declared acceptance protocol changed after fitting')


if __name__=='__main__':unittest.main()
