# Splash

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash serves a small set of models to coding agents and to any OpenAI or
Anthropic compatible client, on one Mac. On a 48 GB M5 Pro it decodes
Qwen3.8-27B at 2× the speed of the next-fastest engine and, with a 32K context
cached, returns the first token in 282 ms. Its kernels, draft model, and memory
plan are specialized for each model it serves. That is why it is fast, and why
there is nothing to configure.

## Q8 fork: run the 8-bit model without compiling

This branch adds 8-bit weight support. The release below is **prebuilt**:
no Xcode, no Metal toolchain, no `make`. You need an Apple Silicon Mac on
macOS 26.4 or later, and 48 GB of unified memory (64 GB recommended; the
weights alone are 27 GB).

```bash
curl -fsSL https://raw.githubusercontent.com/npanj/splash/q8/install-q8.sh | sh
splash-q8 serve --model nitinpanj/Qwen3.8-27B-Splash-HQ
```

- **What the installer does:** downloads the release archive from
  [GitHub Releases](https://github.com/npanj/splash/releases/tag/1.0-q8),
  checks its SHA-256, and installs a `splash-q8` command. It bundles its own
  Python, so there's nothing else to install.
- **It doesn't replace Homebrew's `splash`.** Both can live side by side and
  share the downloaded model files.
- **The first run downloads about 27 GB** from Hugging Face and verifies it.
  Later runs start from disk.
- **Downloaded the archive in a browser instead?** Unpack it and run
  `xattr -dr com.apple.quarantine splash-1.0-q8-arm64-macos26`. Then start
  `python/bin/python3 install/launcher.py serve --model nitinpanj/Qwen3.8-27B-Splash-HQ`
  from inside that folder.
- **Official 4-bit models still work:** `splash-q8 serve --model incoai/Qwen3.8-27B-Splash`.

**Building from source instead?** `make -j4` needs full Xcode, not just the
Command Line Tools. With Xcode 26 you also need the separate Metal toolchain:
`xcodebuild -downloadComponent MetalToolchain`. If you see
`cannot execute tool 'metal'` or `missing Metal Toolchain`, that is the fix.

## Quick start

Apple M3 or newer, macOS 26.4 or later, [Homebrew](https://brew.sh), and 36 GB
of unified memory (48 GB or more recommended).

```bash
brew install incoai/tap/splash
splash serve --model incoai/Qwen3.8-27B-Splash
```

The first run downloads and verifies the model package, checks available
memory, and starts serving on `127.0.0.1:8000`.

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes
```

Press Ctrl+C in the server terminal to stop Splash.

## Use the API

Splash speaks OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses
(`/v1/responses`), and Anthropic Messages (`/v1/messages`), all with streaming,
tool calls, JSON Schema output, images, and inline PDFs. `/tokenize` and
`/apply-template` return token IDs and the rendered prompt without running the
model.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "incoai/Qwen3.8-27B-Splash",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

`model` is optional. If you set it, it must match the package you served.
Reasoning is on by default. `"reasoning_effort": "none"` turns it off, and
Qwen3.8-27B also takes `low`, `medium`, and `xhigh`.

## Models

| Package (`--model`) | Contents | Download |
| --- | --- | ---: |
| [`incoai/Qwen3.8-27B-Splash`](https://huggingface.co/incoai/Qwen3.8-27B-Splash) | Qwen3.8-27B, 4-bit, with its DFlash 2 draft | 17.4 GB |
| [`incoai/Qwen3.6-35B-A3B-Splash`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-Splash) | Qwen3.6-35B-A3B, 4-bit, with its DFlash 2 draft | 20.9 GB |

`--model` takes any `owner/repo` that holds a Splash package, a format
[DEVELOPMENT.md](DEVELOPMENT.md#model-packages) describes. Plain MLX or
Transformers checkpoints do not work. Private repositories need `HF_TOKEN`.
Packages download into the Hugging Face cache, and `brew upgrade splash` keeps
them, along with model links and agent sessions.

## Settings

There is no config file. The server binds `127.0.0.1:8000`, one server at a
time. Context supports up to the model’s native 256K window; usable capacity
depends on available memory.

`splash serve` accepts these optional flags:

- `--max-memory`: ceiling on Metal allocations, e.g. `28G`. Default: auto.
- `--max-context`: context limit, up to `256K`, e.g. `100K`. Default: auto.
- `--max-image-pixels`: maximum resized pixels per image. Default: 4,194,304.
- `--allowed-host`: extra HTTP `Host` name to accept, for a proxy. Repeatable.
- `--api-key`: require this key on API requests, as a bearer token or
  `x-api-key`. Defaults to `SPLASH_API_KEY`.
- `--no-webui`: turn off the chat page.

If the model does not fit in the memory available, startup prints a memory
budget breakdown and stops.

Authentication is off by default. Set `SPLASH_API_KEY` in the shell that runs
`splash serve` and in the shell that runs an agent, and both sides use it.
Health and readiness probes stay public.

- Experimental cache offloading: [PR #3](https://github.com/incoai/splash/pull/3)
  adds SSD offloading for KV cache and GDN states. Build from that branch and
  set `--max-cache-disk 8G` to enable it. This helps preserve reusable prefixes
  when RAM is limited, reducing repeated prefill.

## Performance

Measured on an M5 Pro (16-core GPU, 48 GB): selected SPEED-Bench coding prompts
over HTTP, a 1,024-token output limit, reasoning on (medium for the 27B). The
ratio in each cell is against the next-fastest engine we measured.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

Splash led on every measure at every prompt length we tested, and the lead
grows with load: 3.8× at four concurrent 32K requests on the 35B. The
[launch post](https://inco.ai/blog/splash/) has the method and the full
comparison against oMLX, Lily, uzu, and Ollama.

## Design

The runtime, scheduler, cache, and API are shared. Everything else is rebuilt
per model:

- **A draft trained for the model.** Speculative decoding is the decode path in
  Splash, not an option. Every model ships with its own [DFlash
  2](https://inco.ai/blog/dflash2/) draft, and one pass of the target verifies a
  block of tokens in parallel.
- **Kernels compiled for exact shapes.** Fused Metal kernels, written and tuned
  by our in-house kernel agents for the model's dimensions, read weights packed
  for them and mapped zero-copy from disk. They ship precompiled: no Xcode, no
  compiler toolchain, nothing tuned on your machine.
- **A memory plan computed for this machine.** Context, KV capacity, and batch
  limits are worked out at startup from the memory Metal recommends, less the
  weights, the draft, and each request's state.

The [launch post](https://inco.ai/blog/splash/) covers the design in depth.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): building from source, tests, model packages,
  and release packaging.
- Apache-2.0, see [LICENSE](LICENSE). Model weights keep their own licenses.
