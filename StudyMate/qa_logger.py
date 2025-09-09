from datetime import datetime

class QALogger:
    def __init__(self):
        self.items = []  # list of dicts: {time, question, answer, sources}

    def add(self, question: str, answer: str, sources: list[str]):
        self.items.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "sources": sources or []
        })

    def to_text(self) -> str:
        lines = []
        for i, it in enumerate(self.items, 1):
            lines.append(f"""=== Q&A #{i} ===
Time: {it['time']}
Q: {it['question']}

A: {it['answer']}

Sources:
{chr(10).join(f'- {s}' for s in it['sources']) if it['sources'] else '- (none)'}
""")
        return "\n".join(lines)
