# Sếp AI

Web chatbot “AI nhân viên” chạy bằng engine tự xây dựng, không gọi API AI và không tải model bên thứ ba.

## MVP hiện có

- Mô hình phân loại ý định Naive Bayes được huấn luyện từ dữ liệu riêng trong repo.
- Bộ truy xuất kiến thức TF-IDF tự viết.
- Các mô-đun chuyên biệt cho lập trình, phân tích lỗi, suy luận có cấu trúc và viết prompt.
- Bộ nhớ dài hạn dành cho chủ sở hữu.
- Giao diện dạy thêm ví dụ để model tái huấn luyện ngay.
- Web UI responsive, lịch sử chat lưu trên trình duyệt.
- Engine AI không có dependency model/API bên ngoài; lớp web dùng FastAPI và Uvicorn.

Đây là model nhỏ được huấn luyện từ số 0, không phải LLM và chưa thể đạt chất lượng của các model hàng đầu. Kiến trúc được thiết kế để dữ liệu, thuật toán, bộ nhớ và quá trình nâng cấp đều nằm dưới quyền kiểm soát của chủ sở hữu.

## Chạy

Yêu cầu Python 3.10 trở lên.

```bash
python3 -m pip install .
python3 server.py
```

Mở `http://127.0.0.1:8000`.

Để bảo vệ bản triển khai mạng:

```bash
BOSS_AI_OWNER_TOKEN='thay-bang-token-dai-ngau-nhien' \
BOSS_AI_HOST='0.0.0.0' \
python3 server.py
```

Nhập cùng token trong mục **Khóa chủ sở hữu** trên giao diện.

Khi chạy trên Fly.io, app dùng `/data/runtime` để lưu dữ liệu trên volume và có
thể đọc SHA-256 của khóa từ `data/owner_token.sha256`. Có thể thay thế bằng
`BOSS_AI_OWNER_TOKEN` hoặc `BOSS_AI_OWNER_TOKEN_SHA256`.

## Kiểm tra

```bash
make check
```

## Dạy AI

Mở **Dạy AI** trên giao diện, nhập:

1. Mệnh lệnh hoặc câu hỏi mẫu.
2. Câu trả lời mong muốn.
3. Nhóm năng lực phù hợp.

Ví dụ mới được lưu ở `data/runtime/user_training.json`, sau đó model phân loại và chỉ mục truy xuất được huấn luyện lại. Bộ nhớ chủ sở hữu nằm ở `data/runtime/owner_memory.json`. Hai tệp runtime không được commit.

## Cấu trúc

```text
ai_engine.py             model, retrieval, memory và bộ xử lý chuyên môn
main.py                  FastAPI app, bảo mật, API và ASGI entrypoint
server.py                local startup wrapper
data/training_data.json  dữ liệu huấn luyện gốc thuộc dự án
static/                  ứng dụng web
tests/                   unit test
```

## API nội bộ

- `GET /api/status`
- `POST /api/chat` với `{ "message": "...", "session_id": "..." }`
- `POST /api/train` với `{ "instruction": "...", "response": "...", "intent": "..." }`
- `GET /health`

Nếu `BOSS_AI_OWNER_TOKEN` được đặt, mọi endpoint `/api/*` yêu cầu header `X-Owner-Key`.

## Giới hạn an toàn

Engine ưu tiên mệnh lệnh của chủ sở hữu trong phạm vi năng lực được cấp, nhưng không tự chạy lệnh hệ thống hay sửa file. Các công cụ có tác động tới máy chủ nên được bổ sung qua sandbox, nhật ký và bước xác nhận riêng thay vì cho chatbot quyền thực thi trực tiếp.