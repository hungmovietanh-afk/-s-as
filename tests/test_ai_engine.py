import json
import tempfile
import unittest
from pathlib import Path

from ai_engine import BossAI, NaiveBayesIntentModel, TfidfRetriever


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "data" / "training_data.json"


class IntentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))

    def test_classifies_core_intents(self) -> None:
        model = NaiveBayesIntentModel()
        model.fit(self.examples)
        cases = {
            "Hãy viết code Python cho API": "code",
            "Sửa lỗi bug exception này": "debug",
            "Phân tích và so sánh hai giải pháp": "reasoning",
            "Tạo prompt tốt cho trợ lý": "prompt",
            "Lập kế hoạch roadmap sản phẩm": "planning",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(model.predict(text).intent, expected)

    def test_retriever_returns_related_example(self) -> None:
        retriever = TfidfRetriever()
        retriever.fit(self.examples)
        matches = retriever.search("viết prompt AI thật tốt", intent="prompt")
        self.assertTrue(matches)
        self.assertEqual(matches[0][1]["intent"], "prompt")


class BossAITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.ai = BossAI(
            training_path=TRAINING_PATH,
            runtime_dir=Path(self.temporary_directory.name),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_remembers_owner_instruction(self) -> None:
        result = self.ai.respond("Ghi nhớ rằng tôi thích code ngắn gọn")
        self.assertIn("Đã ghi nhớ", result["reply"])
        second = self.ai.respond("Hãy viết code theo phong cách tôi thích")
        self.assertIn("code ngắn gọn", second["reply"])

    def test_debugger_detects_common_python_issues(self) -> None:
        result = self.ai.respond(
            "Sửa lỗi code này:\n```python\ndef f(items=[])\n"
            "    if items == None:\n        return eval(items)\n```"
        )
        self.assertEqual(result["intent"], "debug")
        self.assertIn("thiếu dấu `:`", result["reply"])
        self.assertIn("mặc định mutable", result["reply"])
        self.assertIn("`is None`", result["reply"])
        self.assertIn("rủi ro", result["reply"])

    def test_prompt_handler_produces_structured_prompt(self) -> None:
        result = self.ai.respond("Viết prompt để review code bảo mật")
        self.assertEqual(result["intent"], "prompt")
        self.assertIn("**Vai trò**", result["reply"])
        self.assertIn("**Tiêu chí hoàn thành**", result["reply"])

    def test_training_adds_example_and_rebuilds_model(self) -> None:
        before = self.ai.status()["training_examples"]
        status = self.ai.train(
            "Thiết kế cache phân tán",
            "Đo hit rate và xác định chiến lược invalidation.",
            "code",
        )
        self.assertEqual(status["training_examples"], before + 1)
        self.assertTrue((Path(self.temporary_directory.name) / "user_training.json").exists())

    def test_rejects_invalid_training_example(self) -> None:
        with self.assertRaises(ValueError):
            self.ai.train("a", "b", "unknown")


if __name__ == "__main__":
    unittest.main()
