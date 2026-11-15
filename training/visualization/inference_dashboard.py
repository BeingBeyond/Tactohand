"""Serve the prediction dashboard locally and stream API traffic to a GPU host.

This module uses only the Python standard library. It can also be run directly
beside inference_viewer.html without installing the training package.
"""
from __future__ import annotations

import argparse
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
import json
import math
from pathlib import Path
from typing import cast
from urllib.parse import SplitResult, quote, unquote, urlsplit


CHUNK_BYTES = 256 * 1024
RESPONSE_HEADERS = (
    'Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges',
    'Cache-Control', 'X-Content-Type-Options',
)


def parse_backend_url(value: str) -> SplitResult:
    backend = urlsplit(value)
    if (backend.scheme not in ('http', 'https') or not backend.hostname
            or backend.username is not None or backend.password is not None
            or backend.query or backend.fragment):
        raise ValueError('Use an http(s) server URL without credentials, query or fragment')
    _ = backend.port  # Validate the port before accepting the connection.
    return backend


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend_url: str = '', timeout: float = 300):
        if backend_url:
            parse_backend_url(backend_url)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Backend timeout must be positive and finite')
        self.backend_url = backend_url.rstrip('/')
        self.backend_timeout = timeout
        self.local_backend = False
        self.html = Path(__file__).with_name('inference_viewer.html').read_bytes()
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):

    @property
    def dashboard(self) -> DashboardServer:
        return cast(DashboardServer, self.server)

    def log_message(self, format, *args):
        pass

    def reply(self, status: int, body: bytes, content_type: str,
              headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def error(self, status: int, detail: str) -> None:
        self.reply(status, json.dumps({'error': detail}).encode(), 'application/json')

    @property
    def connection_cookie(self) -> str:
        return f'tactohand_backend_{self.dashboard.server_port}'

    def selected_backend(self) -> str:
        cookies = SimpleCookie(self.headers.get('Cookie', ''))
        value = cookies.get(self.connection_cookie)
        target = unquote(value.value) if value else self.dashboard.backend_url
        return self.normalize_target(target)

    def normalize_target(self, target: str) -> str:
        if target:
            backend = parse_backend_url(target)
            own_host = urlsplit('http://' + self.headers.get('Host', '')).hostname
            if ((backend.port or (443 if backend.scheme == 'https' else 80)) == self.dashboard.server_port
                    and backend.hostname in {own_host, 'localhost', '127.0.0.1', '0.0.0.0', '::1'}):
                if not self.dashboard.local_backend:
                    raise ValueError('This is the client dashboard; connect to an inference server port')
                return ''
        return target

    def read_json(self) -> dict:
        if self.headers.get_content_type() != 'application/json':
            raise ValueError('Send settings as application/json')
        if self.headers.get('Transfer-Encoding'):
            raise ValueError('Use Content-Length for settings')
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 8192:
            raise ValueError('Settings must be between 1 and 8192 bytes')
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError('Settings must be a JSON object')
        return value

    def handle_connection(self) -> bool:
        if urlsplit(self.path).path != '/dashboard/connection':
            return False
        try:
            headers = None
            target = self.selected_backend() if self.command != 'POST' else ''
            if self.command == 'POST':
                requested = self.read_json().get('backend_url')
                if not isinstance(requested, str):
                    raise ValueError('Enter an inference server URL')
                target = self.normalize_target(requested.strip().rstrip('/'))
                headers = {'Set-Cookie': f'{self.connection_cookie}={quote(target, safe="")}; '
                           'Path=/; HttpOnly; SameSite=Strict'}
            body = {'backend_url': target, 'local': self.dashboard.local_backend and not target}
            self.reply(200, json.dumps(body).encode(), 'application/json', headers)
        except (ValueError, OSError) as error:
            self.error(400, str(error))
        return True

    def proxy(self) -> None:
        try:
            target_url = self.selected_backend()
            if not target_url:
                self.error(503, 'Connect an inference server in the Client panel first')
                return
            backend = parse_backend_url(target_url)
        except ValueError as error:
            self.error(400, str(error))
            return
        assert backend.hostname is not None
        connection_type = HTTPSConnection if backend.scheme == 'https' else HTTPConnection
        connection = connection_type(backend.hostname, backend.port,
                                     timeout=min(10, self.dashboard.backend_timeout) if urlsplit(self.path).path == '/api/health'
                                     else self.dashboard.backend_timeout)
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
        if self.handle_connection():
            return
        path = urlsplit(self.path).path
        if path in ('/', '/index.html'):
            self.reply(200, self.dashboard.html, 'text/html; charset=utf-8')
        elif self.path.startswith('/api/'):
            self.proxy()
        else:
            self.error(404, 'Not found')

    do_HEAD = do_GET

    def do_POST(self) -> None:
        if self.handle_connection():
            return
        if self.path.startswith('/api/'):
            self.proxy()
        else:
            self.error(404, 'Not found')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url', default='', help='initial GPU API URL; can also be set in the Client panel')
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
