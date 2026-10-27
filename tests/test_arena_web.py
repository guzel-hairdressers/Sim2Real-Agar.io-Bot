import time
import unittest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from scripts.play_agar_web import app


class ArenaWebTests(unittest.TestCase):
    def test_experimental_preview_identifies_failed_gate_and_physical_mode(self):
        directory=Path(__file__).resolve().parents[1]/'outputs/learned_risk_v8/experimental'
        with patch.dict('os.environ',{'ARENA_POLICY_DIR':str(directory)}), TestClient(app) as client:
            status=client.get('/status').json()
            self.assertEqual(status['split_control'],'learned_risk_physical_only')
            self.assertEqual(status['acceptance_status'],'experimental_failed_ppo_survival_gate')
            self.assertIn('Experimental learned splits',client.get('/').text)

    def test_accepted_models_render_and_controls_reset(self):
        import scripts.play_agar_web as web
        with TestClient(app) as client:
            deadline = time.monotonic()+3
            while not web.arena_mgr.latest_jpeg and time.monotonic() < deadline:
                time.sleep(.02)
            status = client.get('/status').json()
            self.assertTrue(status['ready'])
            self.assertIsNone(status['error'])
            self.assertEqual(status['observation_version'], 2)
            self.assertEqual(status['control_dt'], 1/30)
            self.assertEqual(status['split_control'],web.arena_mgr.model_manifest.get('split_control','legacy_risk_filter'))
            self.assertEqual(set(status['models']), {'ppo', 'diffusion', 'critic'})
            image = cv2.imdecode(np.frombuffer(web.arena_mgr.latest_jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (675, 1200, 3))
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.post('/action', json={'dx': 50, 'dy': 30, 'split': False, 'sprint': False}).status_code, 200)
            self.assertEqual(client.post('/reset').json(), {'status':'resetting'})
        self.assertFalse(web.arena_mgr.is_running)


if __name__ == '__main__':
    unittest.main()
