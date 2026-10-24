import subprocess, time, json, urllib.request, socket, base64, os, struct

def make_ws_handshake(sock, host, port, path):
    key = base64.b64encode(os.urandom(16)).decode('utf-8')
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
    sock.sendall(req.encode('utf-8'))
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = sock.recv(4096)
        if not chunk: break
        resp += chunk

def send_ws_frame(sock, message):
    data = message.encode('utf-8')
    length = len(data)
    frame = bytearray([0x81])
    mask = os.urandom(4)
    if length <= 125: frame.append(0x80 | length)
    elif length <= 65535: frame.append(0x80 | 126); frame.extend(struct.pack('>H', length))
    else: frame.append(0x80 | 127); frame.extend(struct.pack('>Q', length))
    frame.extend(mask)
    frame.extend(bytearray(b ^ mask[i % 4] for i, b in enumerate(data)))
    sock.sendall(frame)

def recv_ws_frame(sock):
    header = sock.recv(2)
    if not header: return None
    b1, b2 = header[0], header[1]
    is_masked = (b2 & 0x80) != 0
    length = b2 & 0x7F
    if length == 126: length = struct.unpack('>H', sock.recv(2))[0]
    elif length == 127: length = struct.unpack('>Q', sock.recv(8))[0]
    mask = sock.recv(4) if is_masked else None
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        if not chunk: break
        data.extend(chunk)
    if is_masked:
        data = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
    return data.decode('utf-8', errors='ignore')

def main():
    port = 9232
    chrome_proc = subprocess.Popen([
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '--headless=new',
        f'--remote-debugging-port={port}',
        '--remote-allow-origins=*',
        '--disable-gpu',
        '--ignore-certificate-errors',
        '--window-size=390,844',
        '--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        '--no-first-run',
        '--no-default-browser-check',
        'https://morpheme.games/'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{port}/json', headers={'Host': f'127.0.0.1:{port}'})
        with urllib.request.urlopen(req) as r: tabs = json.loads(r.read().decode())
        page_tabs = [t for t in tabs if t.get('type') == 'page']
        ws_url = page_tabs[0]['webSocketDebuggerUrl']
        path = "/" + ws_url.split("/", 3)[3]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', port))
        make_ws_handshake(s, '127.0.0.1', port, path)
        send_ws_frame(s, json.dumps({"id": 1, "method": "Runtime.enable"}))
        send_ws_frame(s, json.dumps({"id": 2, "method": "Page.enable"}))
        time.sleep(2)

        nav_cmd = """(() => {
            if (window.switchPage) window.switchPage('page-tools');
            const navTools = document.getElementById('nav-tools-btn');
            if (navTools) navTools.click();
            setTimeout(() => {
                if (window.showTool) window.showTool('store');
                const btn = document.querySelector('.tool-nav-btn[data-tool="store"]');
                if (btn) btn.click();
            }, 600);
        })()"""
        send_ws_frame(s, json.dumps({"id": 10, "method": "Runtime.evaluate", "params": {"expression": nav_cmd}}))
        time.sleep(3)

        send_ws_frame(s, json.dumps({"id": 20, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
        start = time.time()
        while time.time() - start < 8:
            s.settimeout(0.5)
            try:
                f = recv_ws_frame(s)
                if f:
                    msg = json.loads(f)
                    if msg.get('id') == 20:
                        data = msg.get('result', {}).get('data')
                        if data:
                            out_p = '/Users/jeffbabiak/.gemini/antigravity/brain/9a2d6b3e-9af9-42bc-95f3-2470eb462a17/mobile_store_current.png'
                            with open(out_p, 'wb') as out:
                                out.write(base64.b64decode(data))
                            print(f"Saved screenshot to {out_p}")
                        break
            except socket.timeout: pass
    finally:
        chrome_proc.terminate()

if __name__ == '__main__':
    main()
