"""Trendora AI: Terminal-ის ვერსია. გაშვება: python main.py"""
import uuid

from agent import build_graph, run_turn


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Trendora AI მზადაა. გასასვლელად დაწერე: exit")

    while True:
        text = input("\nშენ: ").strip()
        if text.lower() in ("exit", "quit", "გასვლა"):
            break
        if not text:
            continue

        out = run_turn(graph, config, user_text=text)

        # თუ გრაფი დადასტურებას ითხოვს, ვკითხოთ მომხმარებელს და გავაგრძელოთ
        while out["type"] == "interrupt":
            print("\n⚠️ ", out["payload"]["message"])
            answer = input("დაწერე 'დიახ' ან 'არა': ").strip().lower()
            out = run_turn(graph, config, resume=(answer in ("დიახ", "კი", "yes", "y")))

        print("\nTrendora:", out["text"])


if __name__ == "__main__":
    main()
