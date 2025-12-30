import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/dist/transformers.min.js";

let embedder;

/* Load model */
async function loadModel() {
  embedder = await pipeline(
    "feature-extraction",
    "Xenova/paraphrase-MiniLM-L3-v2"
  );
  console.log("Model loaded");
}

loadModel();

/* Mean pooling */
function meanPooling(tensor) {
  const { data, dims } = tensor; // dims = [1, tokens, hidden]
  const tokens = dims[1];
  const hidden = dims[2];

  const pooled = new Float32Array(hidden).fill(0);

  for (let t = 0; t < tokens; t++) {
    for (let h = 0; h < hidden; h++) {
      pooled[h] += data[t * hidden + h];
    }
  }

  for (let h = 0; h < hidden; h++) {
    pooled[h] /= tokens;
  }

  return pooled;
}

/* Cosine similarity */
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/* Button click */
document.getElementById("compareBtn").addEventListener("click", async () => {
  const n1 = document.getElementById("name1").value.trim();
  const n2 = document.getElementById("name2").value.trim();

  if (!n1 || !n2) {
    alert("Enter both names");
    return;
  }

  const e1 = await embedder(n1);
  const e2 = await embedder(n2);

  const vec1 = meanPooling(e1);
  const vec2 = meanPooling(e2);

  const score = cosineSimilarity(vec1, vec2);
  const percent = (score * 100).toFixed(2);

  document.getElementById("result").innerText =
    `Similarity Score: ${percent}% → ${score >= 0.70 ? "MATCH ✅" : "NO MATCH ❌"}`;
});
