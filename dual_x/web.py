"""Minimal read-only storefront with explicit affiliate disclosure."""
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .storage import connect

def render_products(rows):
    cards = []
    for platform, title, price, url in rows:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            continue
        cards.append('<article><h2>'+escape(title)+'</h2><p>'+escape(platform)+
                     ' · ฿'+escape(price)+'</p><a rel="nofollow sponsored noopener noreferrer" '+
                     'target="_blank" href="'+escape(url, quote=True)+'">ดูสินค้าที่ร้านค้า</a></article>')
    return ('<!doctype html><html lang="th"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DUAL-X สินค้าแนะนำ</title>'
            '<style>body{font:16px system-ui;max-width:960px;margin:auto;padding:24px}'
            'article{border:1px solid #ddd;border-radius:12px;padding:18px;margin:12px 0}'
            'a{color:#075ab5}</style><h1>DUAL-X Commerce</h1>'
            '<p>ลิงก์บางรายการเป็น Affiliate เราอาจได้รับค่าคอมมิชชันเมื่อคุณซื้อสินค้า</p>'
            + ''.join(cards) + '</html>')

def serve(db_path="dual_x.sqlite3", host="127.0.0.1", port=8080):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            with connect(db_path) as db:
                rows = db.execute("SELECT platform,title,price,affiliate_url FROM products ORDER BY updated_at DESC LIMIT 100").fetchall()
            body = render_products(rows).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
if __name__ == "__main__":
    serve()
