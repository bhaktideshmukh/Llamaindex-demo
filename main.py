import asyncio
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "components":
        from components import demo
    elif cmd == "tools":
        from tools import demo
    elif cmd == "agents":
        from agents import demo
    elif cmd == "workflow":
        from workflows import demo
    elif cmd == "chat":
        from agents import chat as demo
    else:
        print("Usage: python main.py [components|tools|agents|workflow|chat]")
        return
    asyncio.run(demo())


if __name__ == "__main__":
    main()