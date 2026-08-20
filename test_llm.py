import sys
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel

print("1. Connecting to local Ollama model (llama3.2)...")

try:
    # Native DeepEval Ollama integration
    local_llm = OllamaModel(model="llama3.2", base_url="http://localhost:11434")

    # 1. Define test case inputs
    test_case = LLMTestCase(
        input="What is the policy for returning a product?",
        actual_output="You can return any unused item within 30 days for a full refund."
    )

    # 2. Set up metric using local Ollama evaluator
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=local_llm)

    print("2. Running evaluation... (giving Ollama ~10 seconds to score)")
    relevancy_metric.measure(test_case)

    print("\n--- EVALUATION RESULTS ---")
    print(f"Score:  {relevancy_metric.score}")
    print(f"Passed: {relevancy_metric.is_successful()}")
    print(f"Reason: {relevancy_metric.reason}")

except Exception as e:
    print(f"\nERROR OCCURRED: {e}", file=sys.stderr)