import json
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

load_dotenv(override=True)

# Compatibility shim for ragas with current langchain-community
# import langchain_google_vertexai
# sys.modules.setdefault("langchain_community.chat_models.vertexai", langchain_google_vertexai)

current_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
INPUT_FILE = os.path.join(current_dir, "rag_generation_output_similarity_decomposed_20261108_201738.json")
OUTPUT_CSV = os.path.join(current_dir, f"rag_evaluation_report_{timestamp}.csv")

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


def sanitize_contexts(ctx):
  """Ensures retrieved_contexts is strictly list[str] to prevent Pydantic list_type errors."""
  if isinstance(ctx, list):
    # Flatten if nested or ensure all items are strings
    return [str(c) for c in ctx if c]
  elif isinstance(ctx, str):
    # If saved as a single giant string, split by double newline or wrap
    chunks = [c.strip() for c in ctx.split("\n\n") if c.strip()]
    return chunks if chunks else [ctx]
  return [str(ctx)]


def run_evaluation():
  if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Missing generation output at {INPUT_FILE}. Run generate_answers.py"
        " first."
    )

  with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  # Standardize keys into Ragas v0.2+ expected schema
  standardized_data = []
  for item in raw_data:
    q = item.get("question") or item.get("user_input", "")
    ans = item.get("answer") or item.get("response", "")
    gt = item.get("ground_truth") or item.get("reference", "")
    raw_ctx = item.get("contexts") or item.get("retrieved_contexts", [])

    standardized_data.append({
        "user_input": q,
        "response": ans,
        "reference": gt,
        "retrieved_contexts": sanitize_contexts(raw_ctx),
        "is_out_of_scope": item.get("is_out_of_scope", False),
    })

  # Separate in-scope items for retrieval metrics (Akshay's suggestion)
  in_scope_items = [
      item for item in standardized_data if not item["is_out_of_scope"]
  ]

  print(
      f"Loaded {len(standardized_data)} total samples ({len(in_scope_items)}"
      " in-scope for retrieval scores)."
  )

  run_config = RunConfig(
      timeout=240, max_workers=1, max_retries=10, max_wait=60
  )

  # ------------------------------------------------------------------
  # Phase 1: Retrieval Metrics (Evaluated strictly on in-scope questions)
  # ------------------------------------------------------------------
  retrieval_dataset = Dataset.from_dict({
      "user_input": [x["user_input"] for x in in_scope_items],
      "response": [x["response"] for x in in_scope_items],
      "reference": [x["reference"] for x in in_scope_items],
      "retrieved_contexts": [x["retrieved_contexts"] for x in in_scope_items],
  })

  print(
      "\nRunning Retrieval Evaluation (context_precision, context_recall) on"
      f" {len(in_scope_items)} queries..."
  )
  retrieval_results = evaluate(
      dataset=retrieval_dataset,
      metrics=[context_precision, context_recall],
      llm=evaluator_llm,
      embeddings=evaluator_embeddings,
      run_config=run_config,
  )
  df_retrieval = retrieval_results.to_pandas()

  # ------------------------------------------------------------------
  # Phase 2: Generation Metrics (Evaluated across all 30 questions)
  # ------------------------------------------------------------------
  full_dataset = Dataset.from_dict({
      "user_input": [x["user_input"] for x in standardized_data],
      "response": [x["response"] for x in standardized_data],
      "reference": [x["reference"] for x in standardized_data],
      "retrieved_contexts": [
          x["retrieved_contexts"] for x in standardized_data
      ],
  })

  print(
      "\nRunning Generation Evaluation (faithfulness, answer_relevancy) on all"
      f" {len(standardized_data)} queries..."
  )
  gen_results = evaluate(
      dataset=full_dataset,
      metrics=[faithfulness, answer_relevancy],
      llm=evaluator_llm,
      embeddings=evaluator_embeddings,
      run_config=run_config,
  )
  df_gen = gen_results.to_pandas()

  # ------------------------------------------------------------------
  # Merge & Export Consolidated Report
  # ------------------------------------------------------------------
  # Merge retrieval metrics back into full dataframe
  df_final = pd.merge(
      df_gen,
      df_retrieval[
          ["user_input", "context_precision", "context_recall"]
      ].drop_duplicates(subset=["user_input"]),
      on="user_input",
      how="left",
  )

  # Rename for display
  df_final = df_final.rename(
      columns={
          "user_input": "question",
          "response": "answer",
          "reference": "ground_truth",
          "retrieved_contexts": "contexts",
      }
  )

  cols_order = [
      "question",
      "context_precision",
      "context_recall",
      "faithfulness",
      "answer_relevancy",
  ]
  display_cols = [c for c in cols_order if c in df_final.columns]

  print("\n" + "=" * 80)
  print("AERORAG BENCHMARK RESULTS")
  print("=" * 80)
  print(df_final[display_cols].to_string(index=False))

  print("\n" + "=" * 40)
  print("AVERAGE SCORES")
  print("=" * 40)
  print(f"Context Precision (In-Scope): {df_final['context_precision'].mean():.4f}")
  print(f"Context Recall (In-Scope):    {df_final['context_recall'].mean():.4f}")
  print(f"Faithfulness (Overall):       {df_final['faithfulness'].mean():.4f}")
  print(f"Answer Relevancy (Overall):   {df_final['answer_relevancy'].mean():.4f}")
  print("=" * 40)

  df_final.to_csv(OUTPUT_CSV, index=False)
  print(f"\nReport saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
  run_evaluation()