import copy
import unittest

from src.evaluate_balanced_ppo import paired_seed, summarize, ROOT


class PairedEvaluationTests(unittest.TestCase):
    def test_same_controller_produces_identical_focus_outcomes(self):
        row = paired_seed((6400001, 12, str(ROOT / 'outputs/split_combat_v3/ppo.pt')))
        self.assertEqual(len(row['cases']), 2)
        for case in row['cases']:
            self.assertEqual(case['baseline'], case['candidate'])
            self.assertEqual(row['control_world'], case['replacement_world'])

    def test_two_seats_are_one_bootstrap_cluster_and_deaths_cannot_hide(self):
        player = dict(final_mass=50, full_deaths=0, captures=0, splits=0,
                      projectile_captures=0, wall_contact_frames=0,
                      physical_violations=0, nonphysical_overrides=0,
                      nonphysical_split_changes=0, control_mode='fully_learned')
        rows = [dict(seed=i, cases=[dict(candidate=copy.deepcopy(player),
                                           baseline=copy.deepcopy(player)) for _ in range(2)])
                for i in range(12)]
        good = summarize(rows)
        self.assertTrue(good['passed'])
        self.assertEqual(good['seed_clusters'], 12)
        self.assertEqual(good['paired_focus_cases'], 24)
        self.assertEqual(good['mass_difference_seed_bootstrap_95_ci'], [0, 0])
        rows[0]['cases'][0]['candidate']['full_deaths'] = 3
        self.assertFalse(summarize(rows)['passed'])
        with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
            summarize(rows + [rows[0]])

    def test_unmeasured_movement_is_not_zero_and_split_changes_fail(self):
        p = dict(final_mass=50, full_deaths=0, captures=0, splits=0,
                 projectile_captures=0, wall_contact_frames=0,
                 physical_violations=0, nonphysical_overrides=None,
                 nonphysical_split_changes=0, control_mode='assisted_movement_learned_split')
        rows = [dict(seed=i, cases=[dict(candidate=copy.deepcopy(p), baseline=copy.deepcopy(p))
                                    for _ in range(2)]) for i in range(12)]
        result = summarize(rows)
        self.assertIsNone(result['candidate']['nonphysical_overrides'])
        self.assertTrue(result['passed'])
        rows[0]['cases'][0]['candidate']['nonphysical_split_changes'] = 1
        self.assertFalse(summarize(rows)['passed'])


if __name__ == '__main__':
    unittest.main()
