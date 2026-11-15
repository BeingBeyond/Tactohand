"""Serve the prediction dashboard locally and stream API traffic to a GPU host.

This module uses only the Python standard library. It can also be run directly
beside inference_viewer.html without installing the training package.
"""
from __future__ import annotations

import argparse
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit


CHUNK_BYTES = 256 * 1024
RESPONSE_HEADERS = (
    'Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges',
    'Cache-Control', 'X-Content-Type-Options',
)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend_url: str, timeout: float = 300):
        backend = urlsplit(backend_url)
        if (backend.scheme not in ('http', 'https') or not backend.hostname
                or backend.username is not None or backend.password is not None
                or backend.query or backend.fragment):
            raise ValueError('Use an http(s) backend URL without credentials, query or fragment')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Backend timeout must be positive and finite')
        self.backend = backend
        self.backend_host = backend.hostname
        self.backend_port = backend.port
        self.backend_timeout = timeout
        self.html = Path(__file__).with_name('inference_viewer.html').read_bytes()
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):

    @property
    def dashboard(self) -> DashboardServer:
        return cast(DashboardServer, self.server)

    def log_message(self, format, *args):
        pass

    def reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def error(self, status: int, detail: str) -> None:
        self.reply(status, json.dumps({'error': detail}).encode(), 'application/json')

    def proxy(self) -> None:
        backend = self.dashboard.backend
        connection_type = HTTPSConnection if backend.scheme == 'https' else HTTPConnection
        connection = connection_type(self.dashboard.backend_host, self.dashboard.backend_port,
                                     timeout=self.dashboard.backend_timeout)
        response_started = False
        try:
            headers = {key: self.headers[key] for key in ('Range', 'Content-Type') if key in self.headers}
            length = 0
            if self.command == 'POST':
                if self.headers.get('Transfer-Encoding'):
                    self.error(400, 'Use Content-Length uploads, not Transfer-Encoding')
                    return
                try:
                    length = int(self.headers['Content-Length'])
                    if length < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    self.error(411, 'A nonnegative Content-Length is required')
                    return
                headers['Content-Length'] = str(length)
            target = backend.path.rstrip('/') + self.path
            connection.putrequest(self.command, target)
            for key, value in headers.items():
                connection.putheader(key, value)
            connection.endheaders()
            while length:
                chunk = self.rfile.read(min(length, CHUNK_BYTES))
                if not chunk:
                    raise OSError('Upload ended before the declared Content-Length')
                connection.send(chunk)
                length -= len(chunk)
            response = connection.getresponse()
            self.send_response(response.status)
            for key in RESPONSE_HEADERS:
                value = response.getheader(key)
                if value is not None:
                    self.send_header(key, value)
            self.end_headers()
            response_started = True
            if self.command != 'HEAD':
                while chunk := response.read(CHUNK_BYTES):
                    self.wfile.write(chunk)
        except (OSError, ValueError, HTTPException) as error:
            if not response_started:
                try:
                    self.error(502, f'Inference backend request failed: {error}')
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            connection.close()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ('/', '/index.html'):
            self.reply(200, self.dashboard.html, 'text/html; charset=utf-8')
        elif self.path.startswith('/api/'):
            self.proxy()
        else:
            self.error(404, 'Not found')

    do_HEAD = do_GET

    def do_POST(self) -> None:
        if self.path.startswith('/api/'):
            self.proxy()
        else:
            self.error(404, 'Not found')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url', required=True, help='GPU API URL, e.g. http://127.0.0.1:8766')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--backend-timeout', type=float, default=300, help='backend socket timeout in seconds')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('Use a valid port')
    try:
        server = DashboardServer((args.host, args.port), args.backend_url, args.backend_timeout)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    with server:
        print(f'Inference backend: {args.backend_url}', flush=True)
        print(f'Open http://{args.host}:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
