import dspy
import asyncio
from idea_gen import IdeaGenerator
from joke_gen import JokeGenerator

# import mlflow
# mlflow.autolog()
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Tool calling")

dspy.configure(lm=dspy.LM("groq/llama-3.1-8b-instant"), temperature=1)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

idea_generator = IdeaGenerator(num_samples=5)
joke_generator = JokeGenerator(num_reflection_steps=2)

def main(query):
    idea = idea_generator.call(query=query)
    joke = joke_generator(joke_idea=idea)
    return joke


if __name__ == "__main__":
    query = input("Query: \n")
    output = main(query)
    print(output)


