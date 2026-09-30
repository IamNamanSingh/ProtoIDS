#!/usr/bin/env python
"""
Execute code inside a kernel of an ALREADY RUNNING Jupyter server - i.e. the
same kernel the Colab frontend is attached to via "Connect to local runtime".

Talking to the running server (instead of a fresh `python` process) means the
notebook sees exactly the state the user is looking at: imported modules, the
working directory, variables defined in earlier cells, and the same
interpreter/GPU the Colab session is using.

    scripts/colab_exec.py --file cell.py
    scripts/colab_exec.py -c "import torch; print(torch.cuda.is_available())"
    cat cell.py | scripts/colab_exec.py -

The token comes from the URL Jupyter printed at startup, e.g.
    http://127.0.0.1:8888/?token=abc123...
    ->  --token abc123     (or set COLAB_JUPYTER_TOKEN)
"""
import argparse
import base64
import json
import os
import sys
import uuid
from urllib.parse import urlencode, urlparse

try:
    import websocket  # websocket-client
except ImportError:
    sys.exit("Need websocket-client: pip install websocket-client")

TIMEOUT = 60 * 60 * 6  # long-running training cells are expected


def api(base: str, token: str, path: str, method: str = "GET", body=None):
    import urllib.request
    url = f"{base}{path}"
    sep = "&" if "?" in url else "?"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url + sep + urlencode({"token": token}), data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> HTTP {exc.code} {exc.read()[:200]!r}")


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def pick_kernel(base: str, token: str, kernel_id: str | None, name: str) -> str:
    kernels = api(base, token, "/api/kernels") or []
    if kernel_id:
        for k in kernels:
            if k["id"] == kernel_id:
                return kernel_id
        raise SystemExit(f"kernel {kernel_id} not found. available: "
                         f"{[k['id'] for k in kernels]}")
    # Prefer a running kernel (that is the one the Colab UI is attached to).
    running = [k for k in kernels if k.get("execution_state") == "idle" and k.get("connections")]
    if running:
        print(f"[colab_exec] attaching to running kernel {running[0]['id']}",
              file=sys.stderr)
        return running[0]["id"]
    if kernels:
        print(f"[colab_exec] attaching to kernel {kernels[0]['id']}", file=sys.stderr)
        return kernels[0]["id"]
    print("[colab_exec] no kernel yet, starting one", file=sys.stderr)
    created = api(base, token, "/api/kernels", "POST",
                  body={"name": name, "path": os.getcwd()})
    return created["id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("COLAB_JUPYTER_URL",
                                                    "http://127.0.0.1:8888"))
    ap.add_argument("--token", default=os.environ.get("COLAB_JUPYTER_TOKEN"))
    ap.add_argument("--kernel", default=None, help="kernel id (default: running one)")
    ap.add_argument("--kernel-name", default="python3")
    ap.add_argument("-c", dest="cmd", default=None, help="code to run")
    ap.add_argument("-f", "--file", default=None, help="file with the code to run")
    args = ap.parse_args()

    # A full notebook URL is a convenient way to pass the token.
    if args.url and "token=" in args.url and not args.token:
        qs = urlparse(args.url).query
        args.token = dict(p.split("=", 1) for p in qs.split("&") if "=" in p).get("token")
    base = args.url.rstrip("/")
    base = base.split("?", 1)[0].rstrip("/")
    if not args.token:
        raise SystemExit("No token. Pass --token <tok> or set COLAB_JUPYTER_TOKEN "
                         "(it is in the URL Jupyter printed at startup).")

    if args.cmd:
        code = args.cmd
    elif args.file:
        code = open(args.file).read()
    elif not sys.stdin.isatty():
        code = sys.stdin.read()
    else:
        raise SystemExit("nothing to run: pass -c, --file, or pipe on stdin")

    kid = pick_kernel(base, args.token, args.kernel, args.kernel_name)
    ws_url = f"ws://{urlparse(base).netloc}/api/kernels/{kid}/channels?" + \
        urlencode({"token": args.token, "session_id": uuid.uuid4().hex})

    ws = websocket.create_connection(ws_url, timeout=TIMEOUT,
                                     max_size=512 * 1024 * 1024)
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "colab_exec", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "content": {
            "code": code, "silent": False, "store_history": False,
            "user_expressions": {}, "allow_stdin": False, "stop_on_error": True,
        },
        "channel": "shell",
    }))

    rc = 0
    while True:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            print("\n[timeout]", file=sys.stderr)
            rc = 124
            break
        if not raw:
            break
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mtype = msg["header"]["msg_type"]
        content = msg["content"]
        if mtype == "stream":
            sys.stdout.write(content["text"])
            sys.stdout.flush()
        elif mtype in ("execute_result", "display_data"):
            sys.stdout.write((content.get("data", {}).get("text/plain")) or "")
            sys.stdout.flush()
        elif mtype == "error":
            rc = 1
            print(f"\n\033[31m{content['ename']}: {content['evalue']}\033[0m",
                  file=sys.stderr)
            for line in content.get("traceback", []):
                # strip the ANSI colour codes tornado adds
                print(line.replace("\x1b[0m", "").replace("\x1b[31m", "")
                      .replace("\x1b[1;31m", ""), file=sys.stderr)
        elif mtype == "execute_reply":
            rc = 1 if content.get("status") == "error" else rc
            break
    ws.close()
    print(f"\n[{'ok' if rc == 0 else 'failed'}]", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
