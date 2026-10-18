"""Reverse proxy 127.0.0.1:18081 -> https://hf-mirror.com that keeps clients on the mirror.

hf-mirror answers paginated listings (/api/.../tree/...) with `Link: <https://huggingface.co/...>; rel="next"`,
and huggingface_hub follows that absolute URL into the blocked domain (e.g. `datasets` resolving data_files in
allenai/c4/en, which has >1000 files). This proxy rewrites huggingface.co / hf-mirror.com in Link and Location
headers to itself and streams everything else through unchanged. Containers use HF_ENDPOINT=http://127.0.0.1:18081.
"""

import http.client
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "hf-mirror.com"
SELF = "http://127.0.0.1:18081"
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade", "host"}
CTX = ssl.create_default_context()


def rewrite(value: str) -> str:
    return value.replace("https://huggingface.co", SELF).replace("https://hf-mirror.com", SELF)


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self):
        body = None
        if (n := int(self.headers.get("Content-Length") or 0)) > 0:
            body = self.rfile.read(n)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Host"] = UPSTREAM
        conn = http.client.HTTPSConnection(UPSTREAM, context=CTX, timeout=300)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            length = resp.getheader("Content-Length")
            for k, v in resp.getheaders():
                if k.lower() in HOP:
                    continue
                if k.lower() in ("link", "location"):
                    v = rewrite(v)
                self.send_header(k, v)
            if length is None and self.command != "HEAD":
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            if self.command != "HEAD":
                while chunk := resp.read(1 << 20):
                    self.wfile.write(chunk)
        except Exception as e:  # upstream failure -> 502 so clients retry
            self.log_message("upstream error %s %s: %s", self.command, self.path, e)
            try:
                self.send_error(502, str(e)[:200])
            except Exception:
                pass
        finally:
            conn.close()

    do_GET = do_HEAD = do_POST = _forward


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18081), Proxy).serve_forever()
