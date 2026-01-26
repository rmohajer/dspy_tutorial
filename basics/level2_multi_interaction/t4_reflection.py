# need to run this in terminal

"""
Docstring for level2_multi_interaction.t4_reflection
Query
  ↓
PromptRefiner (optimize prompt)
  ↓
Optimized Query
  ↓
Multiple JokeIdeas
  ↓
Judge (select best idea)
  ↓
Iterative Joke Refinement
"""

import time
import dspy
import asyncio

# from dspy.teleprompt.mipro_optimizer_v2 import MIPROOptimizerV2, select_best_prompt
from print_utils import print
from typing import List, Optional
from pydantic import BaseModel, Field

# Uncomment this to use mlflow
# import mlflow
# mlflow.autolog()
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Reflection")


dspy.configure(lm=dspy.LM("groq/llama-3.1-8b-instant"), temperature=1)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)


# ---------------------------
# Data models
# ---------------------------
class JokeIdea(BaseModel):
    setup: str
    contradiction: str
    punchline: str


# ---------------------------
# Signatures
# ---------------------------
class PromptRefiner(dspy.Signature):
    """
    You are an expert comedy writer.
    Rewrite the user query into a clearer, funnier, more specific prompt
    that will help another comedian generate a great joke.
    """
    query: str = dspy.InputField()
    refined_query: str = dspy.OutputField(
        description="An optimized prompt for generating a funny joke"
    )


class QueryToIdea(dspy.Signature):
    """
    Generate a structured joke idea.
    """
    query: str = dspy.InputField()
    joke_idea: JokeIdea = dspy.OutputField()


class IdeaToJoke(dspy.Signature):
    """
    Turn a joke idea into a polished joke.
    If a draft exists, improve timing, punchiness, and humor.
    """
    joke_idea: JokeIdea = dspy.InputField()
    joke_draft: Optional[str] = dspy.InputField()
    joke: str = dspy.OutputField()


class JokeJudge(dspy.Signature):
    """
    Rank joke ideas from funniest (1) to worst (N).
    """
    joke_idea: List[JokeIdea] = dspy.InputField()
    joke_ratings: List[int] = dspy.OutputField()



def check_score_goodness(args, pred):
    num_samples = len(args["joke_idea"])
    same_length = len(pred.joke_ratings) == num_samples
    all_ranks_present = all([(i + 1) in pred.joke_ratings for i in range(num_samples)])
    return 1 if (same_length and all_ranks_present) else 0


class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, num_samples=2, num_reflection_steps=2):
        # Prompt optimizer
        self.prompt_refiner = dspy.ChainOfThought(PromptRefiner)

        # Core joke pipeline
        self.query_to_idea = dspy.ChainOfThought(QueryToIdea)
        self.idea_to_joke = dspy.ChainOfThought(IdeaToJoke)
        self.judge = dspy.Refine(
            module=dspy.ChainOfThought(JokeJudge),
            N=3,
            reward_fn=check_score_goodness,
            threshold=1,
        )

        self.num_samples = num_samples
        self.num_reflection_steps = num_reflection_steps
        

    def forward(self, query: str):

        # 1️⃣ Prompt optimization
        refined = self.prompt_refiner(query=query)
        refined_query = refined.refined_query

        print("\n🔧 Optimized Prompt:\n", refined_query)

        # 2️⃣ Generate multiple joke ideas
        joke_ideas = []
        for _ in range(self.num_samples):
            joke_idea = self.query_to_idea(query=refined_query)
            print("\n💡 Generated Joke Idea:\n", joke_idea)
            joke_ideas.append(joke_idea)

        print("\n💡 Generated Joke Ideas:\n", joke_ideas)

        # 3️⃣ Judge ideas
        judge_result = self.judge(joke_idea=joke_ideas)
        ratings = judge_result.joke_ratings
        best_idx = ratings.index(1)

        selected_idea = joke_ideas[best_idx]
        print("\n🏆 Selected Joke Idea:\n", selected_idea)

        # 4️⃣ Iterative joke refinement
        joke = None
        for i in range(self.num_reflection_steps):
            joke_out = self.idea_to_joke(
                joke_idea=selected_idea,
                joke_draft=joke,
            )
            joke = joke_out.joke
            print(f"\n🔁 Refinement step {i+1}:\n{joke}")

        return joke


def main():
    joke_generator = ConditionalJokeGenerator()
    start_time = time.time()
    joke = joke_generator(
        query="Write a joke about AI that has to do with them turning rogue."
    )

    print("---")
    print(joke)
    print(time.time() - start_time)


if __name__ == "__main__":
    main()