"""Exercise the local dashboard against an independent HTTP API (no ML deps)."""
from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Thread
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.visualization.inference_dashboard import CHUNK_BYTES, DashboardServer


class BackendHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/gpu/api/runs/example/video':
            assert self.headers.get('Range') == 'bytes=2-5'
            self.send_response(206)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', '4')
            self.send_header('Content-Range', 'bytes 2-5/10')
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(b'2345')
            return
        self.send_response(404)
        self.send_header('Content-Type', 'application/json')
        body = b'{"error":"missing run"}'
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_HEAD = do_GET

    def do_POST(self):
        assert self.path == '/gpu/api/jobs?filename=test.mp4&checkpoint=latest.ckpt'
        assert self.headers['Content-Type'] == 'application/octet-stream'
        body = self.rfile.read(int(self.headers['Content-Length']))
        # Echo the bytes to check both upload and response streaming.
        self.send_response(202)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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


class DashboardTests(unittest.TestCase):
    def setUp(self):
        backend = ThreadingHTTPServer(('127.0.0.1', 0), BackendHandler)
        port = self.enterContext(serving(backend))
        dashboard = DashboardServer(('127.0.0.1', 0), f'http://127.0.0.1:{port}/gpu')
        self.port = self.enterContext(serving(dashboard))

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_local_page_and_missing_route(self):
        status, _, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'<title>TactoHand', body)
        self.assertEqual(self.request('GET', '/not-an-api')[0], 404)

    def test_upload_and_response_cross_multiple_chunks(self):
        body = bytes(range(256)) * (CHUNK_BYTES // 256 * 3 + 7)
        status, _, echoed = self.request(
            'POST', '/api/jobs?filename=test.mp4&checkpoint=latest.ckpt', body,
            {'Content-Type': 'application/octet-stream'})
        self.assertEqual(status, 202)
        self.assertEqual(echoed, body)

    def test_video_range_and_head(self):
        for method in ('GET', 'HEAD'):
            status, headers, body = self.request(
                method, '/api/runs/example/video', headers={'Range': 'bytes=2-5'})
            self.assertEqual(status, 206)
            self.assertEqual(headers['Content-Range'], 'bytes 2-5/10')
            self.assertEqual(headers['Content-Length'], '4')
            self.assertEqual(body, b'2345' if method == 'GET' else b'')

    def test_backend_error_preserved(self):
        status, _, body = self.request('GET', '/api/runs/missing')
        self.assertEqual(status, 404)
        self.assertIn(b'missing run', body)

    def test_reject_chunked_upload(self):
        self.assertEqual(self.request('POST', '/api/jobs', body=b'',
                                      headers={'Transfer-Encoding': 'chunked'})[0], 400)

    def test_invalid_backend_url(self):
        for url in ('file:///tmp/api', 'http://host:99999', 'http://user:pass@host',
                    'http://host?query=1', 'http://host#fragment'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                DashboardServer(('127.0.0.1', 0), url)

    def test_backend_unavailable_and_local_page_still_works(self):
        self.request('GET', '/')
        # Reserve a port without listening for a deterministic refused connection.
        import socket
        with socket.socket() as reserved:
            reserved.bind(('127.0.0.1', 0))
            server = DashboardServer(('127.0.0.1', 0),
                                     f'http://127.0.0.1:{reserved.getsockname()[1]}')
            with serving(server) as port:
                self.port = port
                self.assertEqual(self.request('GET', '/api/checkpoints')[0], 502)
                self.assertEqual(self.request('GET', '/')[0], 200)


if __name__ == '__main__':
    unittest.main()
