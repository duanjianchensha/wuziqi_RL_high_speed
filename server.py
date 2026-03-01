"""
server.py — Web 服务入口

用法:
  python server.py               # 默认 http://127.0.0.1:8000
  python server.py --port 8080
  python server.py --host 0.0.0.0 --port 8000  # 局域网访问
"""

import argparse
import multiprocessing as mp
import sys
import os

# 确保项目根目录在 PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="五子棋 AlphaZero Web 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="开发模式（热重载，仅用于调试）")
    return p.parse_args()


if __name__ == "__main__":
    # Windows spawn 保护（FastAPI worker 内可能涉及多进程）
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    import uvicorn
    print(f"启动五子棋对战服务  →  http://{args.host}:{args.port}")
    print("在浏览器中打开上述地址，即可开始人机对战。")
    print("按 Ctrl+C 停止服务。\n")

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
