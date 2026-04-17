// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Toggles
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// ---------------------------
// MODEL MAPPING (SAFE)
// ---------------------------
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4o': 'meta/llama-3.1-70b-instruct',
  'claude-3-opus': 'meta/llama-3.1-70b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-8b-instruct',
  'gemini-pro': 'meta/llama-3.1-70b-instruct'
};

// ---------------------------
// SAFETY SYSTEM
// ---------------------------
const FALLBACK_MODEL = 'meta/llama-3.1-70b-instruct';

const SAFE_MODELS = new Set([
  'meta/llama-3.1-8b-instruct',
  'meta/llama-3.1-70b-instruct',
  'nvidia/llama-3.1-nemotron-ultra-253b-v1'
]);

const DEEPSEEK_ALLOWED = new Set([
  'deepseek-ai/deepseek-v3'
]);

function resolveModel(model) {
  let nimModel = MODEL_MAPPING[model] || FALLBACK_MODEL;

  // Block unsafe models
  if (!SAFE_MODELS.has(nimModel)) {
    if (nimModel.includes('deepseek') && DEEPSEEK_ALLOWED.has(nimModel)) {
      return nimModel;
    }
    return FALLBACK_MODEL;
  }

  return nimModel;
}

// ---------------------------
// HEALTH CHECK
// ---------------------------
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// ---------------------------
// LIST MODELS (OpenAI style)
// ---------------------------
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// ---------------------------
// CHAT COMPLETIONS
// ---------------------------
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = resolveModel(model);

    console.log("Requested:", model);
    console.log("Resolved:", nimModel);

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false,
      extra_body: ENABLE_THINKING_MODE
        ? { chat_template_kwargs: { thinking: true } }
        : undefined
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json'
      }
    );

    // ---------------------------
    // STREAMING
    // ---------------------------
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      response.data.on('data', (chunk) => {
        res.write(chunk);
      });

      response.data.on('end', () => res.end());
      return;
    }

    // ---------------------------
    // NORMAL RESPONSE
    // ---------------------------
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices,
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);

  } catch (error) {
    console.error('Proxy error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message
    });

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message,
        type: 'nim_proxy_error',
        code: error.response?.status || 500
      }
    });
  }
});

// ---------------------------
// 404 HANDLER
// ---------------------------
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ---------------------------
// START SERVER
// ---------------------------
app.listen(PORT, () => {
  console.log(`NIM Proxy running on port ${PORT}`);
  console.log(`Base: ${NIM_API_BASE}`);
  console.log(`Reasoning: ${SHOW_REASONING}`);
  console.log(`Thinking: ${ENABLE_THINKING_MODE}`);
});
