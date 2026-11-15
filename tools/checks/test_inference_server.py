"""Exercise server-owned prediction settings through the real HTTP API."""
from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.visualization.inference_dashboard import DashboardServer
from training.visualization.inference_viewer import ViewerServer


@contextmanager
def serving(server):
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.checkpoints = self.root / 'checkpoints'
        self.checkpoints.mkdir()
        for name in ('tactohand', 'other'):
            (self.checkpoints / f'{name}.ckpt').write_bytes(b'not loaded by the HTTP API')
            (self.checkpoints / f'{name}.yaml').write_text('{}')
        outputs = self.root / 'outputs'
        outputs.mkdir()
        self.server = ViewerServer(('127.0.0.1', 0), outputs, checkpoints=self.checkpoints,
                                   device='auto', api_only=True)
        port = self.enterContext(serving(self.server))
        self.backend = f'http://127.0.0.1:{port}'
        # These tests exercise the actual dashboard proxy as well as the GPU API.
        dashboard = DashboardServer(('127.0.0.1', 0), self.backend)
        self.port = self.enterContext(serving(dashboard))
        assert self.server.jobs is not None
        self.enqueue = self.enterContext(patch.object(self.server.jobs.executor, 'submit'))

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def configure(self, checkpoint='tactohand.ckpt', device='cpu', enabled=True):
        return self.request('POST', '/api/server', json.dumps(dict(
            checkpoint=checkpoint, device=device, enabled=enabled)), {'Content-Type': 'application/json'})

    def upload(self, suffix=''):
        return self.request('POST', '/api/jobs?filename=test.mp4' + suffix, b'video',
                            {'Content-Type': 'application/octet-stream'})

    def test_settings_and_upload_use_server_checkpoint_and_device(self):
        self.assertEqual(self.request('GET', '/api/server')[1]['checkpoint'], 'tactohand.ckpt')
        self.assertEqual(self.configure('other.ckpt', 'cuda:1')[0], 200)
        status, job = self.upload()
        self.assertEqual(status, 202)
        args = self.enqueue.call_args.args
        self.assertEqual(args[-3:], (self.checkpoints / 'other.ckpt',
                                     self.checkpoints / 'other.yaml', 'cuda:1'))
        self.configure(device='cpu')
        self.assertEqual(args[-1], 'cuda:1')  # Queued jobs retain their original device.
        self.assertEqual(self.request('GET', '/api/jobs/' + job['id'])[1]['status'], 'queued')

    def test_pause_rejects_new_requests_but_keeps_existing_jobs(self):
        _, job = self.upload()
        self.assertEqual(self.configure(enabled=False)[0], 200)
        self.assertEqual(self.upload()[0], 400)
        self.assertEqual(self.request('GET', '/api/jobs/' + job['id'])[0], 200)
        self.configure(enabled=True)
        self.assertEqual(self.upload()[0], 202)

    def test_client_cannot_override_server_checkpoint(self):
        self.assertEqual(self.upload('&checkpoint=other.ckpt')[0], 400)
        self.assertEqual(self.upload('&checkpoint=tactohand.ckpt')[0], 202)

    def test_invalid_settings_leave_current_configuration_unchanged(self):
        for checkpoint, device in [('missing.ckpt', 'cpu'), ('other.ckpt', 'cuda:bad')]:
            self.assertEqual(self.configure(checkpoint, device)[0], 400)
        state = self.request('GET', '/api/server')[1]
        self.assertEqual((state['checkpoint'], state['device']), ('tactohand.ckpt', 'auto'))

    def test_direct_viewer_self_connection_does_not_proxy_recursively(self):
        self.port = self.server.server_port
        status, body = self.request('POST', '/dashboard/connection',
            json.dumps({'backend_url': self.backend}), {'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        self.assertTrue(body['local'])
        self.assertEqual(self.request('GET', '/api/server')[0], 200)


if __name__ == '__main__':
    unittest.main()
