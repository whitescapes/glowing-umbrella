// server.js
import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Use Railway-provided port or default 3000
const PORT = process.env.PORT || 3000;

// Endpoint for OpenAI-style chat requests
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const apiKey = process.env.NIM_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: "NIM_API_KEY not set" });
    }

    const response = await fetch("https://api.nvidia.com/nim/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.json(data);

  } catch (err) {
    console.error("Error calling NIM API:", err);
    res.status(500).json({ error: "Failed to contact NIM API" });
  }
});

// Health check endpoint
app.get("/", (req, res) => {
  res.send("NIM API Proxy is running!");
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
