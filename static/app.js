const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadStatus = document.getElementById("uploadStatus");
const docList = document.getElementById("docList");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

function setStatus(text, type) {
  uploadStatus.textContent = text;
  uploadStatus.className = `status ${type}`;
  uploadStatus.classList.remove("hidden");
}

async function loadDocuments() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    renderDocList(data.documents || []);
  } catch (err) {
    docList.innerHTML = `<li class="empty">Failed to load documents</li>`;
  }
}

function renderDocList(docs) {
  if (!docs.length) {
    docList.innerHTML = `<li class="empty">No documents indexed yet</li>`;
    return;
  }

  const uniqueDocsMap = new Map();
  for (const d of docs) {
    if (!uniqueDocsMap.has(d.filename)) {
      uniqueDocsMap.set(d.filename, d);
    }
  }
  const uniqueDocs = Array.from(uniqueDocsMap.values());

  docList.innerHTML = uniqueDocs
    .map(
      (d) => `
    <li class="doc-item">
      <div class="meta">
        <div class="name">${escapeHtml(d.filename)}</div>
        <div class="info">${d.doc_type} · ${d.chunk_count} chunks</div>
      </div>
      <button class="btn danger" data-id="${d.doc_id}">Delete</button>
    </li>
  `
    )
    .join("");

  docList.querySelectorAll(".btn.danger").forEach((btn) => {
    btn.addEventListener("click", () => deleteDocument(btn.dataset.id));
  });
}

async function deleteDocument(docId) {
  if (!confirm("Delete this document from the knowledge base?")) return;
  try {
    await fetch(`/api/documents/${docId}`, { method: "DELETE" });
    await loadDocuments();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

async function uploadFiles(files) {
  if (!files.length) return;

  const formData = new FormData();
  for (const f of files) formData.append("files", f);

  setStatus("Indexing... OCR may take a minute for images.", "loading");
  browseBtn.disabled = true;

  try {
    const res = await fetch("/api/ingest", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Upload failed");
    }

    const ok = data.results?.length || 0;
    const errCount = data.errors?.length || 0;
    let msg = `Indexed ${ok} file(s).`;
    if (errCount) msg += ` ${errCount} error(s): ${data.errors.join("; ")}`;
    setStatus(msg, errCount && !ok ? "error" : "success");
    await loadDocuments();
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    browseBtn.disabled = false;
    fileInput.value = "";
  }
}

function addMessage(role, text, sources) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  // Escape HTML first to prevent XSS
  let safeText = escapeHtml(text);

  // Strip numerical citations like [1] or heavy brackets 【...】
  safeText = safeText.replace(/【[^】]+】|\[\d+\]/g, '');

  // Convert inline citations like [filename.ext, page X] into neat UI badges
  safeText = safeText.replace(/\[([^\]]+\.[a-zA-Z0-9]{2,5}(?:,\s*page\s*\d+)?)]/g, (match, p1) => {
    return `<span class="citation-badge">${p1}</span>`;
  });

  let sourcesHtml = "";
  if (sources?.length) {
    sourcesHtml = `
      <details class="sources">
        <summary>${sources.length} source(s)</summary>
        ${sources
          .map(
            (s) => `
          <div class="source-item">
            <strong>${escapeHtml(s.file)}</strong>
            ${s.page ? ` (p. ${escapeHtml(s.page)})` : ""}
            <br>
            <div class="source-snippet" onclick="this.classList.toggle('expanded')">
              ${escapeHtml(s.snippet)}
            </div>
            <div class="expand-hint">Click snippet to expand/collapse</div>
          </div>
        `
          )
          .join("")}
      </details>
    `;
  }

  div.innerHTML = `
    <div class="bubble">${safeText}</div>
    ${sourcesHtml}
  `;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

async function sendChat(message) {
  sendBtn.disabled = true;
  addMessage("user", message);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Chat failed");
    addMessage("assistant", data.answer, data.sources);
  } catch (err) {
    addMessage("assistant", "Error: " + err.message);
  } finally {
    sendBtn.disabled = false;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

dropzone.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener("change", () => uploadFiles([...fileInput.files]));

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  uploadFiles([...e.dataTransfer.files]);
});

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg) return;
  chatInput.value = "";
  sendChat(msg);
});

loadDocuments();
