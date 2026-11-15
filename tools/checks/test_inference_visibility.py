"""Check optional per-hand visibility on the viewer's real NPZ loading path."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.visualization.inference_viewer import ViewerServer, load_predictions


class VisibilityTests(unittest.TestCase):
    def load(self, visibility=None, *, batch=0, batched=False):
        fields: dict[str, Any] = {
            'contact_prob': np.zeros((3, 2, 778), dtype=np.float32),
            'distance_m': np.zeros((3, 2, 778), dtype=np.float32),
            'force_normalized_10hz': np.zeros((3, 2, 778, 3), dtype=np.float32),
        }
        if visibility is not None:
            fields['hand_visibility_prob_10hz'] = np.asarray(visibility, dtype=np.float32)
        if batched:
            fields = {key: np.stack([value, value[::-1]]) for key, value in fields.items()}
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.npz'
            np.savez(path, **fields)
            return load_predictions(path, batch, {}, None)[0]

    def test_per_frame_left_right_order(self):
        values = [[0.9, 0.1], [0.2, 0.8], [0.5, 1]]
        payload = self.load(values)
        np.testing.assert_allclose(payload['handVisibility'], values)
        self.assertEqual(len(payload['handVisibility']), len(payload['times']))

    def test_batch_selection(self):
        values = [[0.9, 0.1], [0.2, 0.8], [0.5, 1]]
        np.testing.assert_allclose(self.load(values, batch=1, batched=True)['handVisibility'],
                                   values[::-1])

    def test_old_prediction_without_visibility(self):
        self.assertIsNone(self.load()['handVisibility'])

    def test_invalid_visibility(self):
        for values in ([[0.2, 0.8]], [[0.2], [0.5], [0.8]],
                       [[0.2, 0.8], [-0.1, 1], [0.5, 0.5]],
                       [[0.2, 0.8], [0.5, 1.1], [0.5, 0.5]],
                       [[0.2, 0.8], [float('nan'), 1], [0.5, 0.5]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.load(values)

    def test_api_only_service_and_health(self):
        with TemporaryDirectory() as directory:
            server = ViewerServer(('127.0.0.1', 0), Path(directory), api_only=True)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base + '/api/health', timeout=5) as response:
                    self.assertEqual(json.load(response),
                                     {'status': 'ok', 'service': 'tactohand-inference'})
                with self.assertRaises(HTTPError) as failure:
                    urlopen(base + '/', timeout=5)
                self.assertEqual(failure.exception.code, 404)
                failure.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    @unittest.skipUnless(shutil.which('node'), 'Node is optional for the UI check')
    def test_ui_updates_with_frame_and_clears_legacy_values(self):
        html = (Path(__file__).resolve().parents[2]
                / 'training/visualization/inference_viewer.html').read_text()
        script = re.search(r'<script>(.*?)</script>', html, re.S)
        assert script is not None
        ui = re.search(r'function drawVisibility\(\) \{.*?\n\}', script[1], re.S)
        assert ui is not None
        harness = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
let source = '';
process.stdin.on('data', chunk => source += chunk);
process.stdin.on('end', () => {
  const {script, ui} = JSON.parse(source);
  new vm.Script(script); // Parse the entire page script, not just the badge function.
  const context = {visibilityReadouts: [{dataset:{}}, {dataset:{}}], frame:0,
    data:{handVisibility:[[.9,.1],[.2,.8],[.5,1]]}};
  vm.createContext(context);
  vm.runInContext(ui, context);
  context.drawVisibility();
  assert.equal(context.visibilityReadouts[0].dataset.state, 'visible');
  assert.match(context.visibilityReadouts[0].textContent, /90\.0%/);
  assert.equal(context.visibilityReadouts[1].dataset.state, 'not-visible');
  context.frame = 1;
  context.drawVisibility();
  assert.equal(context.visibilityReadouts[0].dataset.state, 'not-visible');
  assert.equal(context.visibilityReadouts[1].dataset.state, 'visible');
  assert.match(context.visibilityReadouts[1].textContent, /80\.0%/);
  context.frame = 2;
  context.drawVisibility();
  assert.equal(context.visibilityReadouts[0].dataset.state, 'not-visible');
  context.data = {handVisibility:null};
  context.drawVisibility();
  for (const badge of context.visibilityReadouts) {
    assert.equal(badge.dataset.state, 'unavailable');
    assert.equal(badge.textContent, 'Visibility unavailable');
  }
});
'''
        subprocess.run(['node', '-e', harness],
                       input=json.dumps({'script': script[1], 'ui': ui[0]}),
                       text=True, check=True, capture_output=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
