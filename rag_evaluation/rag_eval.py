import json
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

# Compatibility shim for ragas with current langchain-community
import langchain_google_vertexai
sys.modules.setdefault("langchain_community.chat_models.vertexai", langchain_google_vertexai)

current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(current_dir, "rag_generation_output2.json")
OUTPUT_CSV = os.path.join(current_dir, "rag_evaluation_report2.csv")

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

answer_relevancy.strictness = 1 
# Evaluator Setup
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
)


def run_evaluation():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing generation output at {INPUT_FILE}. Run generate_answers.py first.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert list of dicts to Hugging Face Dataset format
    dataset_dict = {
        "question": [item["question"] for item in raw_data],
        "contexts": [item["contexts"] for item in raw_data],
        "answer": [item["answer"] for item in raw_data],
        "ground_truth": [item["ground_truth"] for item in raw_data],
    }
    dataset = Dataset.from_dict(dataset_dict)


    print(f"Running RAGAS evaluation on {len(dataset)} samples...")

    run_config = RunConfig(
        timeout=240,
        max_workers=1,
        max_retries=10,
        max_wait=60
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    df = results.to_pandas()

    # Column normalization across ragas schema variations
    df = df.rename(columns={
        "user_input": "question",
        "retrieved_contexts": "contexts",
        "response": "answer",
        "reference": "ground_truth"
    })

    display_cols = [c for c in ["question", "context_precision", "context_recall", "faithfulness", "answer_relevancy"] if c in df.columns]
    print("\nEvaluation Summary:")
    print(df[display_cols])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nReport saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_evaluation()