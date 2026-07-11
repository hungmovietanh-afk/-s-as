const chat = document.querySelector("#chat");
const hero = document.querySelector("#hero");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const trainingDrawer = document.querySelector("#trainingDrawer");
const settingsDrawer = document.querySelector("#settingsDrawer");
const drawerBackdrop = document.querySelector("#drawerBackdrop");
const trainingForm = document.querySelector("#trainingForm");
const settingsForm = document.querySelector("#settingsForm");
const ownerKeyInput = document.querySelector("#ownerKey");
const historyKey = "boss-ai-chat-v1";

let history = loadHistory();
let busy = false;

function authHeaders() {
  const key = sessionStorage.getItem("boss-ai-owner-key");
  return key ? { "X-Owner-Key": key } : {};
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(options.headers || {}),
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Yêu cầu thất bại.");
  }
  return payload;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(value) {
  const codeBlocks = [];
  let escaped = escapeHtml(value).replace(/```(?:[\w+#.-]+)?\n?([\s\S]*?)```/g, (_, code) => {
    const token = `@@CODE_${codeBlocks.length}@@`;
    codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return token;
  });

  escaped = escaped
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^(\d+)\. (.+)$/gm, "<li data-ordered>$2</li>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li data-ordered>.*?<\/li>\n?)+/gs, (block) => `<ol>${block.replaceAll(" data-ordered", "")}</ol>`)
    .replace(/(<li>.*?<\/li>\n?)+/gs, "<ul>$&</ul>")
    .split(/\n{2,}/)
    .map((block) => {
      if (/^<(h2|h3|ul|ol|pre)/.test(block)) return block;
      return `<p>${block.replaceAll("\n", "<br>")}</p>`;
    })
    .join("");

  codeBlocks.forEach((block, index) => {
    escaped = escaped.replace(`<p>@@CODE_${index}@@</p>`, block).replace(`@@CODE_${index}@@`, block);
  });
  return escaped;
}

function loadHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(historyKey) || "[]");
    return Array.isArray(saved) ? saved.slice(-30) : [];
  } catch {
    return [];
  }
}

function saveHistory() {
  localStorage.setItem(historyKey, JSON.stringify(history.slice(-30)));
}

function renderMessage(item, persist = false) {
  hero.classList.add("hidden");
  const element = document.createElement("article");
  element.className = `message ${item.role}`;
  const name = item.role === "user" ? "Sếp" : "Sếp AI";
  const avatar = item.role === "user" ? "VA" : "S";
  const badge =
    item.role === "assistant" && item.intent
      ? `<span class="intent-badge">${escapeHtml(item.intent)}</span>`
      : "";
  const trace =
    item.trace?.length
      ? `<div class="trace">Xử lý: ${item.trace.map(escapeHtml).join(" · ")}</div>`
      : "";
  element.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-body">
      <div class="message-meta"><strong>${name}</strong>${badge}</div>
      <div class="message-content">${renderMarkdown(item.content)}</div>
      ${trace}
    </div>`;
  messages.append(element);
  if (persist) {
    history.push(item);
    saveHistory();
  }
  chat.scrollIntoView({ block: "end", behavior: "smooth" });
  return element;
}

function renderTyping() {
  hero.classList.add("hidden");
  const element = document.createElement("article");
  element.className = "message assistant";
  element.innerHTML = `
    <div class="message-avatar">S</div>
    <div class="message-body">
      <div class="message-meta"><strong>Sếp AI</strong></div>
      <div class="typing"><i></i><i></i><i></i></div>
    </div>`;
  messages.append(element);
  return element;
}

async function sendMessage(message) {
  if (!message.trim() || busy) return;
  busy = true;
  sendButton.disabled = true;
  renderMessage({ role: "user", content: message.trim() }, true);
  messageInput.value = "";
  resizeInput();
  const typing = renderTyping();
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message: message.trim(),
        session_id: crypto.randomUUID?.() || "browser",
      }),
    });
    typing.remove();
    renderMessage(
      {
        role: "assistant",
        content: result.reply,
        intent: `${result.intent_label} · ${Math.round(result.confidence * 100)}%`,
        trace: result.trace,
      },
      true,
    );
    await refreshStatus();
  } catch (error) {
    typing.remove();
    renderMessage(
      {
        role: "assistant",
        content: `Không thể xử lý yêu cầu: **${error.message}**`,
        intent: "Lỗi kết nối",
      },
      false,
    );
  } finally {
    busy = false;
    sendButton.disabled = false;
    messageInput.focus();
  }
}

function resizeInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
}

function openDrawer(drawer) {
  closeDrawers();
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  drawerBackdrop.classList.add("open");
}

function closeDrawers() {
  document.querySelectorAll(".drawer").forEach((drawer) => {
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
  });
  drawerBackdrop.classList.remove("open");
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    document.querySelector("#modelName").textContent = status.name;
    document.querySelector("#modelMeta").textContent =
      `${status.training_examples} ví dụ · ${status.vocabulary_size} đặc trưng`;
    const values = [
      status.architecture,
      `${status.training_examples} ví dụ`,
      `${status.vocabulary_size} đặc trưng`,
      `${status.memory_items} mục`,
    ];
    document.querySelectorAll("#modelStats dd").forEach((element, index) => {
      element.textContent = values[index];
    });
  } catch {
    document.querySelector("#modelName").textContent = "Cần khóa truy cập";
    document.querySelector("#modelMeta").textContent = "Mở Cấu hình để xác thực";
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener("input", resizeInput);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

document.querySelector("#newChatButton").addEventListener("click", () => {
  history = [];
  saveHistory();
  messages.replaceChildren();
  hero.classList.remove("hidden");
  messageInput.focus();
});

document.querySelectorAll("#openTrainingButton, #headerTrainingButton").forEach((button) => {
  button.addEventListener("click", () => openDrawer(trainingDrawer));
});

document.querySelector("#openSettingsButton").addEventListener("click", () => {
  ownerKeyInput.value = sessionStorage.getItem("boss-ai-owner-key") || "";
  openDrawer(settingsDrawer);
});

document.querySelectorAll("[data-close-drawer]").forEach((button) => {
  button.addEventListener("click", closeDrawers);
});

drawerBackdrop.addEventListener("click", closeDrawers);

trainingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = document.querySelector("#trainingStatus");
  status.className = "form-status";
  status.textContent = "Đang tái huấn luyện...";
  try {
    const result = await api("/api/train", {
      method: "POST",
      body: JSON.stringify({
        instruction: document.querySelector("#trainingInstruction").value,
        response: document.querySelector("#trainingResponse").value,
        intent: document.querySelector("#trainingIntent").value,
      }),
    });
    status.textContent = `${result.message} Tổng: ${result.status.training_examples} ví dụ.`;
    trainingForm.reset();
    await refreshStatus();
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message;
  }
});

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  sessionStorage.setItem("boss-ai-owner-key", ownerKeyInput.value);
  const status = document.querySelector("#settingsStatus");
  status.className = "form-status";
  status.textContent = "Đã lưu khóa trong phiên này.";
  await refreshStatus();
});

document.querySelector("#menuButton").addEventListener("click", () => {
  document.querySelector(".sidebar").classList.toggle("open");
});

history.forEach((item) => renderMessage(item));
refreshStatus();
messageInput.focus();
