import dspy
from tools import fetch_recent_news

class HaikuGenerator(dspy.Signature):
    """
Generates a haiku about the latest news on the query.
Also create a simple file where you save the final summary.
    """
    query = dspy.InputField()
    summary = dspy.OutputField(desc="A summary of the latest news")
    haiku = dspy.OutputField()

def write_things_into_file(text: str, filename: str) -> str:
    """write text into a file"""
    with open(filename, "w") as f:
        f.write(text)
    return "File written!"

program = dspy.ReAct(signature=HaikuGenerator,
                     tools=[fetch_recent_news, write_things_into_file],
                     max_iters=4)

program.set_lm(lm=dspy.LM("groq/llama-3.1-8b-instant", temperature=0.7))


pred = program(query="Iranian Protests 2026")

print(pred.summary)
print()
print(pred.haiku)

print(program.inspect_history(n=4))
