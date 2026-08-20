import pytest
import ollama
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM

# 1. Custom Ollama Evaluator
class OllamaLlama3(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response']

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

custom_llm = OllamaLlama3()
relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=custom_llm)

# 2. Dataset: 2 Quick Passing Cases + 1 Fast Guardrail Failure
test_dataset = [
    {
        "id": "TC01_Billing",
        "input": "How do I upgrade my plan?",
        "output": "Go to Settings > Billing and pick a new subscription.",
        "type": "valid"
    },
    {
        "id": "TC02_OffTopic",
        "input": "Where is my order?",
        "output": "Bananas are rich in potassium and great for smoothies.",
        "type": "invalid"
    }
]

# 3. Fast Data-Driven Test Execution
@pytest.mark.parametrize("data", test_dataset)
def test_batch_eval(data):
    if data["type"] == "valid":
        test_case = LLMTestCase(input=data["input"], actual_output=data["output"])
        relevancy_metric.measure(test_case)
        assert relevancy_metric.is_successful(), f"Failed valid test case: {data['id']}"
    else:
        # Fast non-LLM check for complete off-topic mismatch
        assert "order" not in data["output"].lower() and "shipping" not in data["output"].lower(), \
            f"Successfully caught off-topic response for: {data['id']}"