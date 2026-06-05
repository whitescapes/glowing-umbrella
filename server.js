// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Startup Validation ───────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const PROXY_API_KEY = process.env.PROXY_API_KEY;

// 🔴 FIX #1: Fail fast if required env vars are missing
if (!NIM_API_KEY) {
  console.error('FATAL: NIM_API_KEY environment variable is not set');
  process.exit(1);
}
if (!PROXY_API_KEY) {
  console.error('FATAL: PROXY_API_KEY environment variable is not set');
  process.exit(1);
}

// ─── Feature Toggles ─────────────────────────────────────────────────────────

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// ─── Model Mapping ────────────────────────────────────────────────────────────

// Maps OpenAI-style model names to NVIDIA NIM model IDs.
// Aliases reflect approximate capability tiers, not exact equivalence.
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// ─── Middleware ───────────────────────────────────────────────────────────────

app.use(cors());

// 🔴 FIX #6: Limit request body size to prevent memory exhaustion
app.use(express.json({ limit: '1mb' }));

// 🔴 FIX #2: Authenticate all incoming requests against PROXY_API_KEY.
// Health check is intentionally excluded so load balancers can probe it.
app.use((req, res, next) => {
  if (req.path === '/health') return next();

  const authHeader = req.headers['authorization'] || '';
  const key = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : authHeader;

  if (!key || key !== PROXY_API_KEY) {
    return res.status(401).json({
      error: { message: 'Unauthorized', type: 'auth_error', code: 401 }
    });
  }
  next();
});

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Resolve the NIM model to use for a given requested model name.
 * 🔴 FIX #3: Removed the live probe request that burned API quota on every
 * unknown model name. Instead, fall through to deterministic heuristics.
 */
function resolveNimModel(requestedModel) {
  if (MODEL_MAPPING[requestedModel]) {
    return MODEL_MAPPING[requestedModel];
  }

  // Heuristic fallback — no live API probe
  const m = requestedModel.toLowerCase();
  if (m.includes('gpt-4') || m.includes('claude-opus') || m.includes('405b')) {
    return 'meta/llama-3.1-405b-instruct';
  }
  if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
}

/**
 * Build a streaming SSE data chunk that closes the <think> tag.
 */
function makeCloseThinkChunk() {
  return {
    choices: [{ delta: { content: '</think>\n\n' }, index: 0 }]
  };
}

// ─── Routes ───────────────────────────────────────────────────────────────────

// Health check — unauthenticated so load balancers / uptime monitors can use it
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models (OpenAI-compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000), // 🟡 FIX #9: Unix seconds throughout
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

// Chat completions — main proxy endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = resolveNimModel(model);

    // 🟡 FIX #10: Use ?? so that explicit falsy values (0, false) are respected
    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 8000,
      stream: stream ?? false,
      // 🟡 FIX #7: chat_template_kwargs goes at the top level, not in extra_body
      ...(ENABLE_THINKING_MODE && { chat_template_kwargs: { thinking: true } })
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    // ── Streaming path ──────────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');

        // 🟠 FIX #5: Keep the last (potentially incomplete) line in the buffer
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                delta.content = combinedContent;
              } else {
                // Strip reasoning from the output entirely
                delta.content = content ?? '';
              }

              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            // Passthrough unparseable lines (e.g. comments)
            res.write(line + '\n\n');
          }
        }
      });

      response.data.on('end', () => {
        // 🟠 FIX #4: Close any open <think> tag if the stream ended mid-reasoning
        if (reasoningStarted) {
          res.write(`data: ${JSON.stringify(makeCloseThinkChunk())}\n\n`);
        }

        // 🟠 FIX #5: Flush any remaining buffered data
        if (buffer.trim()) {
          res.write(buffer + '\n\n');
        }

        res.write('data: [DONE]\n\n');
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

    // ── Non-streaming path ──────────────────────────────────────────────────
    } else {
      const nowSeconds = Math.floor(Date.now() / 1000);

      const openaiResponse = {
        id: `chatcmpl-${nowSeconds}`, // 🟡 FIX #9: Consistent seconds-based ID
        object: 'chat.completion',
        created: nowSeconds,
        model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content ?? '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
          }

          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage ?? {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    // 🟡 FIX #8: Don't leak internal error details to clients in production
    const clientMessage =
      process.env.NODE_ENV === 'production'
        ? 'An internal error occurred'
        : error.message;

    res.status(error.response?.status || 500).json({
      error: {
        message: clientMessage,
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check:    http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode:     ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
