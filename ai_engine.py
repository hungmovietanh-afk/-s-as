from __future__ import annotations

import json
import math
import random
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


INTENTS = {
    "code": "Viết code",
    "debug": "Sửa lỗi",
    "reasoning": "Suy luận",
    "prompt": "Viết prompt",
    "planning": "Lập kế hoạch",
    "general": "Trợ lý chung",
}

TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ỹ0-9_+#.-]+", re.UNICODE)
CODE_BLOCK_PATTERN = re.compile(r"```(?:[a-zA-Z0-9_+#.-]+)?\s*\n?(.*?)```", re.DOTALL)


def tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("đ", "d")
    tokens = TOKEN_PATTERN.findall(normalized)
    features = list(tokens)
    features.extend(f"{left}::{right}" for left, right in zip(tokens, tokens[1:]))
    return features


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    scores: dict[str, float]


class NaiveBayesIntentModel:
    def __init__(self) -> None:
        self.intent_documents: Counter[str] = Counter()
        self.intent_terms: dict[str, Counter[str]] = defaultdict(Counter)
        self.intent_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.document_count = 0

    def fit(self, examples: Iterable[dict[str, str]]) -> None:
        self.intent_documents.clear()
        self.intent_terms.clear()
        self.intent_totals.clear()
        self.vocabulary.clear()
        self.document_count = 0

        for example in examples:
            intent = example["intent"]
            terms = tokenize(example["instruction"])
            self.intent_documents[intent] += 1
            self.intent_terms[intent].update(terms)
            self.intent_totals[intent] += len(terms)
            self.vocabulary.update(terms)
            self.document_count += 1

    def predict(self, text: str) -> Prediction:
        if not self.document_count:
            return Prediction("general", 1.0, {"general": 1.0})

        terms = tokenize(text)
        vocabulary_size = max(len(self.vocabulary), 1)
        log_scores: dict[str, float] = {}

        for intent, document_count in self.intent_documents.items():
            prior = (document_count + 1) / (
                self.document_count + len(self.intent_documents)
            )
            score = math.log(prior)
            denominator = self.intent_totals[intent] + vocabulary_size
            for term in terms:
                score += math.log((self.intent_terms[intent][term] + 1) / denominator)
            log_scores[intent] = score

        highest = max(log_scores.values())
        probabilities = {
            intent: math.exp(score - highest) for intent, score in log_scores.items()
        }
        total = sum(probabilities.values())
        normalized = {intent: value / total for intent, value in probabilities.items()}
        best_intent = max(normalized, key=normalized.get)
        return Prediction(best_intent, normalized[best_intent], normalized)


class TfidfRetriever:
    def __init__(self) -> None:
        self.examples: list[dict[str, str]] = []
        self.idf: dict[str, float] = {}
        self.vectors: list[dict[str, float]] = []

    def fit(self, examples: list[dict[str, str]]) -> None:
        self.examples = examples
        document_frequency: Counter[str] = Counter()
        documents: list[list[str]] = []
        for example in examples:
            terms = tokenize(example["instruction"])
            documents.append(terms)
            document_frequency.update(set(terms))

        count = max(len(documents), 1)
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._vectorize_terms(terms) for terms in documents]

    def _vectorize_terms(self, terms: list[str]) -> dict[str, float]:
        frequencies = Counter(terms)
        vector = {
            term: (1 + math.log(frequency)) * self.idf.get(term, 0.0)
            for term, frequency in frequencies.items()
        }
        magnitude = math.sqrt(sum(value * value for value in vector.values()))
        if not magnitude:
            return {}
        return {term: value / magnitude for term, value in vector.items()}

    def search(
        self, query: str, intent: str | None = None, limit: int = 3
    ) -> list[tuple[float, dict[str, str]]]:
        query_vector = self._vectorize_terms(tokenize(query))
        matches: list[tuple[float, dict[str, str]]] = []
        for example, vector in zip(self.examples, self.vectors):
            if intent and example["intent"] != intent:
                continue
            similarity = sum(
                query_value * vector.get(term, 0.0)
                for term, query_value in query_vector.items()
            )
            if similarity > 0:
                matches.append((similarity, example))
        return sorted(matches, key=lambda item: item[0], reverse=True)[:limit]


class OwnerMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.items: list[dict[str, str]] = self._load()

    def _load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def add(self, content: str) -> None:
        item = {
            "content": content.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self.items.append(item)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def relevant(self, query: str, limit: int = 3) -> list[str]:
        query_terms = set(tokenize(query))
        scored: list[tuple[int, str]] = []
        for item in self.items:
            content = item["content"]
            score = len(query_terms.intersection(tokenize(content)))
            if score:
                scored.append((score, content))
        return [content for _, content in sorted(scored, reverse=True)[:limit]]


class BossAI:
    def __init__(
        self,
        training_path: Path | None = None,
        runtime_dir: Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent
        self.training_path = training_path or root / "data" / "training_data.json"
        self.runtime_dir = runtime_dir or root / "data" / "runtime"
        self.user_training_path = self.runtime_dir / "user_training.json"
        self.memory = OwnerMemory(self.runtime_dir / "owner_memory.json")
        self.intent_model = NaiveBayesIntentModel()
        self.retriever = TfidfRetriever()
        self.examples: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._reload_models()

    def _read_examples(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            item
            for item in data
            if isinstance(item, dict)
            and item.get("intent") in INTENTS
            and item.get("instruction")
            and item.get("response")
        ]

    def _reload_models(self) -> None:
        examples = self._read_examples(self.training_path)
        examples.extend(self._read_examples(self.user_training_path))
        self.examples = examples
        self.intent_model.fit(examples)
        self.retriever.fit(examples)

    def status(self) -> dict[str, object]:
        return {
            "name": "Sếp AI Core",
            "version": "0.1.0",
            "architecture": "Naive Bayes + TF-IDF + specialist modules",
            "training_examples": len(self.examples),
            "vocabulary_size": len(self.intent_model.vocabulary),
            "memory_items": len(self.memory.items),
            "external_ai": False,
            "intents": INTENTS,
        }

    def train(self, instruction: str, response: str, intent: str) -> dict[str, object]:
        instruction = instruction.strip()
        response = response.strip()
        if len(instruction) < 3 or len(response) < 3:
            raise ValueError("Ví dụ huấn luyện phải có ít nhất 3 ký tự.")
        if intent not in INTENTS:
            raise ValueError("Nhóm năng lực không hợp lệ.")

        with self._lock:
            user_examples = self._read_examples(self.user_training_path)
            user_examples.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "intent": intent,
                }
            )
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            self.user_training_path.write_text(
                json.dumps(user_examples, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._reload_models()
        return self.status()

    def respond(self, message: str, session_id: str = "default") -> dict[str, object]:
        del session_id
        message = message.strip()
        if not message:
            raise ValueError("Mệnh lệnh không được để trống.")
        if len(message) > 20_000:
            raise ValueError("Mệnh lệnh vượt quá 20.000 ký tự.")

        remembered = self._remember_if_requested(message)
        if remembered:
            return self._result(
                remembered,
                Prediction("general", 1.0, {"general": 1.0}),
                ["Nhận diện lệnh ghi nhớ", "Lưu vào bộ nhớ chủ sở hữu"],
            )

        prediction = self.intent_model.predict(message)
        intent = self._apply_intent_rules(message, prediction.intent)
        if intent != prediction.intent:
            prediction = Prediction(intent, max(prediction.confidence, 0.82), prediction.scores)

        handlers = {
            "code": self._handle_code,
            "debug": self._handle_debug,
            "reasoning": self._handle_reasoning,
            "prompt": self._handle_prompt,
            "planning": self._handle_planning,
            "general": self._handle_general,
        }
        reply, trace = handlers[intent](message)
        memory = self.memory.relevant(message)
        if memory:
            reply += "\n\n**Bộ nhớ liên quan**\n" + "\n".join(
                f"- {item}" for item in memory
            )
            trace.append("Đối chiếu bộ nhớ chủ sở hữu")
        return self._result(reply, prediction, trace)

    def _result(
        self, reply: str, prediction: Prediction, trace: list[str]
    ) -> dict[str, object]:
        return {
            "reply": reply,
            "intent": prediction.intent,
            "intent_label": INTENTS[prediction.intent],
            "confidence": round(prediction.confidence, 4),
            "trace": trace,
        }

    def _remember_if_requested(self, message: str) -> str | None:
        patterns = [
            r"^(?:hãy\s+)?ghi nhớ(?:\s+rằng)?\s*[:,-]?\s*(.+)$",
            r"^(?:hãy\s+)?nhớ(?:\s+rằng)?\s*[:,-]?\s*(.+)$",
            r"^từ giờ(?:\s+hãy)?\s*[:,-]?\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, message, re.IGNORECASE | re.DOTALL)
            if match and match.group(1).strip():
                content = match.group(1).strip()
                self.memory.add(content)
                return f"Đã ghi nhớ: **{content}**"
        return None

    def _apply_intent_rules(self, message: str, predicted: str) -> str:
        lowered = message.lower()
        code = self._extract_code(message)
        if code and any(term in lowered for term in ("lỗi", "bug", "sai", "fix", "debug")):
            return "debug"
        keywords = {
            "prompt": ("prompt", "câu lệnh ai", "system instruction"),
            "debug": ("traceback", "exception", "sửa lỗi", "debug", "bug"),
            "planning": ("kế hoạch", "lộ trình", "roadmap", "chia việc"),
            "reasoning": ("phân tích", "suy luận", "so sánh", "tại sao"),
            "code": ("viết code", "lập trình", "hàm ", "class ", "api ", "component"),
        }
        for intent, terms in keywords.items():
            if any(term in lowered for term in terms):
                return intent
        return predicted

    def _extract_code(self, message: str) -> str:
        blocks = CODE_BLOCK_PATTERN.findall(message)
        return "\n\n".join(block.strip() for block in blocks if block.strip())

    def _handle_debug(self, message: str) -> tuple[str, list[str]]:
        code = self._extract_code(message)
        findings = self._analyze_code(code) if code else []
        response = [
            "## Chẩn đoán",
            "Tôi sẽ tách lỗi thành triệu chứng, nguyên nhân có thể và cách xác minh.",
        ]
        if findings:
            response.append("\n### Phát hiện trực tiếp")
            response.extend(f"- {finding}" for finding in findings)
        elif code:
            response.append(
                "\nChưa thấy lỗi cú pháp phổ biến bằng bộ phân tích tĩnh nhỏ. "
                "Cần traceback, đầu vào gây lỗi và kết quả mong muốn để khoanh vùng chính xác."
            )
        else:
            response.append(
                "\nHãy gửi code trong khối ba dấu backtick, thông báo lỗi đầy đủ, "
                "đầu vào và kết quả mong muốn."
            )
        response.extend(
            [
                "\n### Quy trình sửa",
                "1. Tạo ca tái hiện nhỏ nhất.",
                "2. Xác định dòng đầu tiên thuộc code dự án trong stack trace.",
                "3. Kiểm tra giá trị và kiểu dữ liệu ngay trước dòng lỗi.",
                "4. Sửa nguyên nhân gốc, không chỉ bắt và bỏ qua exception.",
                "5. Thêm test cho ca lỗi và một ca biên liên quan.",
            ]
        )
        return "\n".join(response), [
            "Nhận diện yêu cầu sửa lỗi",
            "Quét mẫu lỗi tĩnh",
            "Lập quy trình tái hiện và xác minh",
        ]

    def _analyze_code(self, code: str) -> list[str]:
        findings: list[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack: list[tuple[str, int]] = []
        closing = {value: key for key, value in pairs.items()}
        for line_number, line in enumerate(code.splitlines(), start=1):
            stripped = re.sub(r"(['\"]).*?\1", "", line)
            for character in stripped:
                if character in pairs:
                    stack.append((character, line_number))
                elif character in closing:
                    if not stack or stack[-1][0] != closing[character]:
                        findings.append(
                            f"Dòng {line_number}: dấu `{character}` không có cặp mở phù hợp."
                        )
                    else:
                        stack.pop()
            trimmed = line.strip()
            if re.match(r"^(def|class|if|elif|else|for|while|try|except|with)\b", trimmed):
                if not trimmed.endswith(":") and not trimmed.endswith(("{", ";")):
                    findings.append(f"Dòng {line_number}: có thể thiếu dấu `:`.")
            if re.search(r"\bexcept\s*:", trimmed):
                findings.append(
                    f"Dòng {line_number}: `except:` quá rộng; nên bắt exception cụ thể."
                )
            if re.search(r"\bdef\s+\w+\([^)]*=\s*(\[\]|\{\})", trimmed):
                findings.append(
                    f"Dòng {line_number}: đối số mặc định mutable có thể giữ "
                    "trạng thái ngoài ý muốn."
                )
            if "== None" in line or "!= None" in line:
                findings.append(
                    f"Dòng {line_number}: trong Python nên dùng `is None` hoặc `is not None`."
                )
            execution_match = re.search(r"\b(eval|exec)\s*\(", line)
            if execution_match:
                findings.append(
                    f"Dòng {line_number}: `{execution_match.group(1)}` "
                    "có rủi ro thực thi dữ liệu không tin cậy."
                )
        for opening, line_number in stack:
            findings.append(
                f"Dòng {line_number}: dấu `{opening}` chưa được đóng bằng `{pairs[opening]}`."
            )
        return findings[:10]

    def _handle_prompt(self, message: str) -> tuple[str, list[str]]:
        subject = re.sub(
            r"(?i)^(hãy\s+)?(viết|tạo|cải thiện|sửa)?\s*(một\s+)?prompt\s*(cho|để)?\s*",
            "",
            message,
        ).strip()
        if not subject:
            subject = "[mô tả nhiệm vụ]"
        prompt = f"""## Prompt đã cấu trúc

**Vai trò**
Bạn là chuyên gia phù hợp nhất để thực hiện nhiệm vụ sau.

**Mục tiêu**
{subject}

**Quy trình**
1. Xác định mục tiêu, dữ liệu đầu vào và ràng buộc còn thiếu.
2. Chia nhiệm vụ thành các phần có thể kiểm chứng.
3. Thực hiện lần lượt, ưu tiên độ chính xác hơn suy đoán.
4. Tự kiểm tra kết quả theo tiêu chí hoàn thành.
5. Nếu thiếu dữ liệu quan trọng, hỏi tối đa 3 câu ngắn trước khi làm.

**Yêu cầu đầu ra**
- Trả lời rõ ràng, có cấu trúc.
- Nêu giả định quan trọng.
- Đưa ví dụ cụ thể khi hữu ích.
- Không bịa dữ kiện; đánh dấu phần chưa chắc chắn.

**Tiêu chí hoàn thành**
Kết quả dùng được ngay, đúng ràng buộc và có bước kiểm chứng."""
        return prompt, [
            "Trích xuất mục tiêu prompt",
            "Bổ sung vai trò, quy trình và ràng buộc",
            "Định nghĩa tiêu chí hoàn thành",
        ]

    def _handle_reasoning(self, message: str) -> tuple[str, list[str]]:
        reply = f"""## Phân tích có cấu trúc

**Vấn đề:** {message}

### 1. Mục tiêu cần đạt
- Xác định kết quả cuối cùng có thể đo hoặc kiểm chứng.
- Tách điều bắt buộc khỏi điều mong muốn.

### 2. Dữ kiện và giả định
- Liệt kê dữ kiện đã biết từ yêu cầu.
- Chưa mặc định những thông tin không được cung cấp.
- Kiểm tra các giả định có thể làm thay đổi kết luận.

### 3. Các hướng giải quyết
- **Hướng tối giản:** giải quyết phần cốt lõi với ít phụ thuộc nhất.
- **Hướng cân bằng:** tối ưu chất lượng, thời gian và khả năng bảo trì.
- **Hướng mở rộng:** đầu tư lớn hơn để tăng hiệu năng hoặc quy mô.

### 4. Cách ra quyết định
Chấm từng hướng theo tác động, chi phí, rủi ro, khả năng đảo ngược và thời gian
kiểm chứng. Ưu tiên thử nghiệm nhỏ giúp loại bỏ bất định lớn nhất trước.

### 5. Bước tiếp theo
Chuyển phương án được chọn thành đầu việc, tiêu chí nghiệm thu và phép đo. Gửi
thêm dữ kiện cụ thể nếu bạn muốn tôi đưa ra kết luận thay vì khung phân tích."""
        return reply, [
            "Tách mục tiêu và ràng buộc",
            "So sánh nhiều hướng giải quyết",
            "Xây tiêu chí quyết định và kiểm chứng",
        ]

    def _handle_planning(self, message: str) -> tuple[str, list[str]]:
        reply = f"""## Kế hoạch thực thi

**Mục tiêu:** {message}

1. **Chốt phạm vi** — đầu ra, người dùng, ràng buộc, tiêu chí thành công.
2. **Khảo sát hiện trạng** — tài nguyên sẵn có, phụ thuộc, rủi ro lớn nhất.
3. **Thiết kế bản nhỏ nhất** — một luồng hoàn chỉnh có thể chạy và đo được.
4. **Triển khai theo lát dọc** — mỗi phần gồm code, kiểm tra và tài liệu.
5. **Xác minh** — test chức năng, lỗi biên, hiệu năng và bảo mật phù hợp.
6. **Phát hành có kiểm soát** — log, sao lưu, rollback và theo dõi chỉ số.
7. **Lặp lại** — ưu tiên cải tiến dựa trên dữ liệu sử dụng thực tế.

**Định nghĩa hoàn thành:** sản phẩm chạy được, có phép kiểm tra lặp lại, giới hạn
được ghi rõ và có phương án phục hồi khi lỗi."""
        return reply, [
            "Chuyển mục tiêu thành các giai đoạn",
            "Gắn kiểm chứng vào từng giai đoạn",
            "Định nghĩa điều kiện hoàn thành",
        ]

    def _handle_code(self, message: str) -> tuple[str, list[str]]:
        language = self._detect_language(message)
        matches = self.retriever.search(message, intent="code", limit=1)
        learned = ""
        if matches and matches[0][0] >= 0.42:
            learned = f"\n\n### Mẫu kiến thức gần nhất\n{matches[0][1]['response']}"
        reply = f"""## Thiết kế triển khai ({language})

**Yêu cầu:** {message}

### Trước khi viết code
1. Chốt chữ ký đầu vào/đầu ra và hành vi khi dữ liệu sai.
2. Liệt kê ca thường, ca biên và lỗi cần trả về.
3. Chọn giải pháp đơn giản nhất đáp ứng độ phức tạp mong muốn.

### Cấu trúc đề xuất
- Tách logic thuần khỏi I/O để dễ kiểm thử.
- Validate tại biên hệ thống.
- Trả lỗi có ngữ cảnh, không nuốt exception.
- Viết test cho luồng chính, giá trị rỗng, giới hạn và dữ liệu sai.

Để tôi viết bản code hoàn chỉnh thay vì khung, hãy cung cấp: ngôn ngữ/phiên bản,
chữ ký mong muốn, ví dụ input-output và môi trường đang dùng.{learned}"""
        return reply, [
            "Nhận diện ngôn ngữ và loại tác vụ",
            "Truy xuất ví dụ code đã học",
            "Thiết kế contract, ca biên và kiểm thử",
        ]

    def _detect_language(self, message: str) -> str:
        lowered = message.lower()
        mapping = {
            "TypeScript": ("typescript", ".ts", "react", "next.js"),
            "JavaScript": ("javascript", "node.js", ".js"),
            "Python": ("python", "django", "flask", "fastapi"),
            "Go": ("golang", " go ", ".go"),
            "Rust": ("rust", "cargo"),
            "Java": ("java", "spring"),
            "C#": ("c#", ".net"),
        }
        for language, terms in mapping.items():
            if any(term in lowered for term in terms):
                return language
        return "ngôn ngữ phù hợp với dự án"

    def _handle_general(self, message: str) -> tuple[str, list[str]]:
        matches = self.retriever.search(message, limit=2)
        if matches and matches[0][0] >= 0.5:
            reply = matches[0][1]["response"]
            source = "Truy xuất ví dụ huấn luyện gần nhất"
        else:
            options = [
                "Tôi đã nhận mệnh lệnh. Hãy bổ sung đầu ra mong muốn và ràng buộc "
                "quan trọng để tôi thực hiện chính xác.",
                "Tôi có thể xử lý yêu cầu này theo hướng code, sửa lỗi, phân tích, "
                "lập kế hoạch hoặc viết prompt. Hãy chỉ rõ kết quả bạn muốn nhận.",
            ]
            reply = random.choice(options)
            source = "Tổng hợp phản hồi nền"
        return reply, [
            "Phân loại ý định",
            source,
            "Kiểm tra bộ nhớ liên quan",
        ]
