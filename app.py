"""
Interactive Transformer Architecture Visualizer

A Streamlit dashboard for exploring nanoGPT and nanochat model architectures.
Displays model components, parameter counts, and architecture diagrams.

Sources:
- FLOPs calculation: https://github.com/karpathy/nanochat/discussions/420#discussion-9319591
- PaLM paper (Appendix B): https://arxiv.org/abs/2204.02311
- nanoGPT: https://github.com/karpathy/nanoGPT
- nanochat: https://github.com/karpathy/nanochat
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import graphviz

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="Transformer Architecture Visualizer",
    page_icon="🔬",
    layout="wide",
)


def format_params(n: int) -> str:
    """Format parameter count with appropriate suffix."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def render_svg_with_tooltips(dot: graphviz.Digraph, height: int = 800) -> None:
    """Render a graphviz diagram as interactive SVG with working tooltips using viz.js (no system graphviz needed)."""
    import json

    # Get the DOT source from the graphviz object
    dot_source = dot.source

    # Escape the DOT source for JavaScript
    dot_source_escaped = json.dumps(dot_source)

    # HTML with viz.js and svg-pan-zoom loaded from CDN
    html_content = f"""
    <style>
        #graph-container {{
            width: 100%;
            height: {height}px;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        #graph-container svg {{
            width: 100%;
            height: 100%;
        }}
        .node:hover {{
            cursor: pointer;
        }}
        .node:hover ellipse,
        .node:hover polygon,
        .node:hover path {{
            stroke-width: 2px;
            stroke: #333;
        }}
        #loading {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: #666;
        }}
        #error {{
            color: #dc3545;
            padding: 20px;
        }}
        /* Custom tooltip styling */
        .viz-tooltip {{
            position: fixed;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
            max-width: 500px;
            white-space: pre-wrap;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            pointer-events: none;
            line-height: 1.4;
        }}
        .viz-tooltip .cmt {{
            color: #6a9955;
        }}
        .viz-tooltip .kw {{
            color: #569cd6;
        }}
    </style>

    <div id="loading">Loading diagram...</div>
    <div id="error" style="display: none;"></div>
    <div id="graph-container" style="display: none;"></div>
    <div class="viz-tooltip" id="tooltip" style="display: none;"></div>

    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <script type="module">
        import {{ Graphviz }} from 'https://cdn.jsdelivr.net/npm/@hpcc-js/wasm@2.16.1/dist/graphviz.js';

        const dotSource = {dot_source_escaped};

        try {{
            const graphviz = await Graphviz.load();
            const svg = graphviz.dot(dotSource, 'svg');

            document.getElementById('loading').style.display = 'none';
            const container = document.getElementById('graph-container');
            container.style.display = 'block';
            container.innerHTML = svg;

            // Initialize svg-pan-zoom
            const svgElement = container.querySelector('svg');
            svgElement.setAttribute('width', '100%');
            svgElement.setAttribute('height', '100%');

            const panZoom = svgPanZoom(svgElement, {{
                zoomEnabled: true,
                controlIconsEnabled: true,
                fit: true,
                center: true,
                minZoom: 0.25,
                maxZoom: 10,
                zoomScaleSensitivity: 0.3
            }});

            // Enhanced tooltip handling
            const tooltip = document.getElementById('tooltip');
            const nodes = container.querySelectorAll('.node, .cluster');

            nodes.forEach(node => {{
                // Graphviz creates two tooltip sources:
                // 1. <title> element with node ID (default)
                // 2. <a xlink:title="..."> wrapper with actual tooltip (from tooltip attr)

                // Find the actual tooltip from xlink:title attribute on <a> element
                const anchor = node.querySelector('a');
                const xlinkTitle = anchor?.getAttribute('xlink:title') || anchor?.getAttributeNS('http://www.w3.org/1999/xlink', 'title');

                // Remove all <title> elements (these contain node IDs, not our tooltips)
                node.querySelectorAll('title').forEach(t => t.remove());

                // Remove xlink:title to prevent native browser tooltip
                if (anchor) {{
                    anchor.removeAttribute('xlink:title');
                    anchor.removeAttributeNS('http://www.w3.org/1999/xlink', 'title');
                }}

                // Only add custom tooltip if we found actual tooltip content
                if (xlinkTitle) {{
                    node.addEventListener('mouseenter', (e) => {{
                        // Order matters: keywords first, then comments, so we don't match inside our own tags
                        let html = xlinkTitle
                            .replace(/&/g, '&amp;')
                            .replace(/</g, '&lt;')
                            .replace(/>/g, '&gt;')
                            .replace(/\\b(self|def|class|return|if|else|for|in|import|from|None|True|False)\\b/g, '⟦KW⟧$1⟦/KW⟧')
                            .replace(/(#[^\\n]*)/g, '⟦CMT⟧$1⟦/CMT⟧')
                            .replace(/⟦KW⟧/g, '<span class="kw">').replace(/⟦\\/KW⟧/g, '</span>')
                            .replace(/⟦CMT⟧/g, '<span class="cmt">').replace(/⟦\\/CMT⟧/g, '</span>');
                        tooltip.innerHTML = html;
                        tooltip.style.display = 'block';
                    }});

                    node.addEventListener('mousemove', (e) => {{
                        tooltip.style.left = (e.clientX + 15) + 'px';
                        tooltip.style.top = (e.clientY + 15) + 'px';
                    }});

                    node.addEventListener('mouseleave', () => {{
                        tooltip.style.display = 'none';
                    }});
                }}
            }});

        }} catch (err) {{
            document.getElementById('loading').style.display = 'none';
            document.getElementById('error').style.display = 'block';
            document.getElementById('error').textContent = 'Error rendering diagram: ' + err.message;
            console.error('Graphviz error:', err);
        }}
    </script>
    """
    components.html(html_content, height=height + 20, scrolling=True)


# =============================================================================
# Code Snippets for Tooltips
# =============================================================================

# nanoGPT code snippets (from nanoGPT/model.py)
# Notation: d=n_embd, h=n_head, d_h=d/h (head_dim), V=vocab_size, T=block_size, L=n_layer
NANOGPT_CODE = {
    "wte": """# Token Embedding (model.py:127)
nn.Embedding(config.vocab_size, config.n_embd)

# Weight shape: (V, d) - vocab_size × n_embd
# Weight-tied with lm_head (model.py:138)
self.transformer.wte.weight = self.lm_head.weight""",

    "wpe": """# Position Embedding (model.py:128)
nn.Embedding(config.block_size, config.n_embd)

# Weight shape: (T, d) - block_size × n_embd
# Each position gets a learned d-dimensional vector""",

    "ln": """# LayerNorm with optional bias (model.py:18-27)
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

# Weight shape: (d,) - n_embd
# Bias shape: (d,) if bias=True, else None
# Normalizes over the last dimension (embedding dim)""",

    "c_attn": """# Combined QKV projection (model.py:35)
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

# Weight shape: (d, 3d) - n_embd × 3*n_embd
# Bias shape: (3d,) if bias=True
# Projects input to concatenated [Q, K, V]
# Split into Q, K, V (model.py:56)
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# Each of Q, K, V has shape (B, T, d) then reshaped to (B, h, T, d_h)""",

    "attn_op": """# Scaled dot-product attention (model.py:62-71)
# Input Q, K, V shapes: (B, h, T, d_h)
# Attention scores: Q @ K^T = (B, h, T, d_h) @ (B, h, d_h, T) = (B, h, T, T)
# Scale by 1/sqrt(d_h) for stable gradients
if self.flash:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, h, T, T) @ (B, h, T, d_h) = (B, h, T, d_h)
# Output shape: (B, h, T, d_h) -> reshaped to (B, T, d)""",

    "attn_proj": """# Output projection (model.py:37)
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

# Weight shape: (d, d) - n_embd × n_embd
# Bias shape: (d,) if bias=True
# Projects concatenated heads back to residual stream
# Input: (B, T, d), Output: (B, T, d)""",

    "mlp_fc": """# Up projection / first FC layer (model.py:82)
self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

# Weight shape: (d, 4d) - n_embd × 4*n_embd
# Bias shape: (4d,) if bias=True
# Expands from d to 4d (standard GPT-2 ratio)
# Input: (B, T, d), Output: (B, T, 4d)""",

    "gelu": """# GELU activation (model.py:83, 88-89)
self.gelu = nn.GELU()
x = self.gelu(x)

# No learnable parameters
# Applied element-wise: (B, T, 4d) -> (B, T, 4d)
# GELU(x) ≈ x * sigmoid(1.702 * x)""",

    "mlp_proj": """# Down projection / second FC layer (model.py:84)
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

# Weight shape: (4d, d) - 4*n_embd × n_embd
# Bias shape: (d,) if bias=True
# Compresses from 4d back to d
# Input: (B, T, 4d), Output: (B, T, d)""",

    "ln_f": """# Final LayerNorm (model.py:131)
ln_f = LayerNorm(config.n_embd, bias=config.bias)

# Weight shape: (d,) - n_embd
# Bias shape: (d,) if bias=True
# Pre-norm architecture requires this final norm
# Input: (B, T, d), Output: (B, T, d)""",

    "lm_head": """# Language model head (model.py:133)
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

# Weight shape: (d, V) - n_embd × vocab_size
# No bias (bias=False)
# Weight-tied with token embedding (model.py:138)
self.transformer.wte.weight = self.lm_head.weight
# Input: (B, T, d), Output: (B, T, V)""",

    "block": """# Transformer Block (model.py:94-106)
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # pre-norm attention
        x = x + self.mlp(self.ln_2(x))   # pre-norm MLP
        return x

# Input/Output shape: (B, T, d) - preserved through residuals""",
}

# nanochat code snippets (from nanochat/gpt.py)
# Notation: d=n_embd, h=n_head, k=n_kv_head, d_h=d/h (head_dim), V=vocab_size, V'=padded_vocab, T=seq_len, L=n_layer
NANOCHAT_CODE = {
    "wte": """# Token Embedding with vocab padding (gpt.py:155, 159)
padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
nn.Embedding(padded_vocab_size, config.n_embd)

# Weight shape: (V', d) - padded_vocab_size × n_embd
# Padding to multiple of 64 for tensor core efficiency""",

    "norm": """# RMSNorm - purely functional, no learnable params (gpt.py:48-50)
def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# No learnable parameters! (unlike LayerNorm)
# Normalizes by RMS: x / sqrt(mean(x²) + eps)
# Input/Output shape unchanged: (B, T, d) -> (B, T, d)""",

    "scale_in": """# Per-layer residual scaling (gpt.py:167-168, 359-360)
self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

# Shape: (L,) - one scalar per layer for each
# Applied before each block (gpt.py:359-360)
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
# Blends residual stream with initial embedding x0""",

    "c_q": """# Query projection (gpt.py:71)
self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)

# Weight shape: (d, h×d_h) = (d, d) - n_embd × n_embd
# No bias (all nanochat linears are bias=False)
# Input: (B, T, d), Output: (B, T, d) -> reshaped to (B, T, h, d_h)""",

    "c_k": """# Key projection - fewer heads for GQA (gpt.py:72)
self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

# Weight shape: (d, k×d_h) - n_embd × (n_kv_head × head_dim)
# With GQA: k < h, so this is smaller than Q projection
# Example: h=12, k=4, d_h=64 -> (768, 256) instead of (768, 768)
# Input: (B, T, d), Output: (B, T, k×d_h) -> reshaped to (B, T, k, d_h)""",

    "c_v": """# Value projection - fewer heads for GQA (gpt.py:73)
self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

# Weight shape: (d, k×d_h) - n_embd × (n_kv_head × head_dim)
# Same size as K projection (GQA shares KV head count)
# Input: (B, T, d), Output: (B, T, k×d_h) -> reshaped to (B, T, k, d_h)""",

    "rotary": """# Rotary Position Embeddings + QK Normalization (gpt.py:53-59, 86-88)

# RoPE: Applied to Q and K only (not V)
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

# QK-Norm: Applied to Q and K only (not V)
q, k = norm(q), norm(k)  # RMSNorm before attention

# Why Q and K only, not V?
# - Attention scores = Q @ K^T (V not involved)
# - RoPE: rotating Q and K encodes RELATIVE position (i-j) in their dot product
# - QK-Norm: bounds Q and K magnitudes to prevent attention logit explosion
# - V carries content that flows through - no position or stability concerns
# No learnable parameters""",

    "flash_attn": """# Flash Attention 3 (gpt.py:93-95)
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

# FA3 handles GQA automatically: broadcasts k KV heads to h Q heads
# Input Q: (B, T, h, d_h), K: (B, T, k, d_h), V: (B, T, k, d_h)
# Output: (B, T, h, d_h) - same shape as Q
# window_size enables sliding window attention (memory efficient)
# No learnable parameters - just efficient attention computation""",

    "attn_proj": """# Output projection (gpt.py:74)
self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

# Weight shape: (d, d) - n_embd × n_embd
# Projects concatenated attention heads back to residual stream
# Input: (B, T, d), Output: (B, T, d)""",

    "mlp_fc": """# Up projection (gpt.py:119)
self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

# Weight shape: (d, 4d) - n_embd × 4*n_embd
# Expands from d to 4d (standard transformer ratio)
# Input: (B, T, d), Output: (B, T, 4d)""",

    "relu_sq": """# Squared ReLU activation (gpt.py:124)
x = F.relu(x).square()

# No learnable parameters
# Applied element-wise: (B, T, 4d) -> (B, T, 4d)
# ReLU²(x) = max(0, x)² - sparser than GELU, helps with training""",

    "mlp_proj": """# Down projection (gpt.py:120)
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

# Weight shape: (4d, d) - 4*n_embd × n_embd
# Compresses from 4d back to d
# Input: (B, T, 4d), Output: (B, T, d)""",

    "lm_head": """# Language model head - NOT weight-tied (gpt.py:162)
self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

# Weight shape: (d, V') - n_embd × padded_vocab_size
# Unlike nanoGPT, this is SEPARATE from embedding (not tied)
# This adds V'×d extra parameters but allows different representations
# Input: (B, T, d), Output: (B, T, V')""",

    "softcap": """# Logit soft-capping (gpt.py:364-368)
softcap = 15
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size]  # remove padding: (B,T,V') -> (B,T,V)
logits = logits.float()  # fp32 for numerical stability
logits = softcap * torch.tanh(logits / softcap)  # squash to [-15, 15]

# Prevents extreme logits that can cause training instability
# tanh smoothly caps values: large |x| -> ±softcap""",

    "block": """# Transformer Block (gpt.py:129-138)
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)  # pre-norm
        x = x + self.mlp(norm(x))  # pre-norm
        return x

# Input/Output shape: (B, T, d) - preserved through residuals
# Note: No LayerNorm modules - uses functional RMSNorm (no params)""",
}


# =============================================================================
# Parameter Calculation Functions (matching actual PyTorch implementations)
# =============================================================================

def calc_nanogpt_params(n_layer: int, n_head: int, n_embd: int,
                        vocab_size: int, block_size: int, bias: bool) -> dict:
    """
    Calculate nanoGPT parameter counts.

    Architecture:
    - Token embedding: vocab_size × n_embd (weight-tied with lm_head)
    - Position embedding: block_size × n_embd
    - Per block:
        - LayerNorm 1: n_embd (weight) + n_embd (bias if enabled)
        - Attention c_attn: n_embd × 3*n_embd + 3*n_embd (bias if enabled)
        - Attention c_proj: n_embd × n_embd + n_embd (bias if enabled)
        - LayerNorm 2: n_embd (weight) + n_embd (bias if enabled)
        - MLP c_fc: n_embd × 4*n_embd + 4*n_embd (bias if enabled)
        - MLP c_proj: 4*n_embd × n_embd + n_embd (bias if enabled)
    - Final LayerNorm: n_embd (weight) + n_embd (bias if enabled)
    - lm_head: weight-tied with token embedding, no separate params
    """
    head_dim = n_embd // n_head

    # Embeddings
    wte_params = vocab_size * n_embd  # token embedding
    wpe_params = block_size * n_embd  # position embedding

    # Per-block calculations
    ln_params = n_embd + (n_embd if bias else 0)  # LayerNorm: weight + optional bias

    # Attention: c_attn (combined QKV) and c_proj
    c_attn_weight = n_embd * (3 * n_embd)
    c_attn_bias = (3 * n_embd) if bias else 0
    c_proj_weight = n_embd * n_embd
    c_proj_bias = n_embd if bias else 0
    attn_params = c_attn_weight + c_attn_bias + c_proj_weight + c_proj_bias

    # MLP: c_fc and c_proj
    mlp_fc_weight = n_embd * (4 * n_embd)
    mlp_fc_bias = (4 * n_embd) if bias else 0
    mlp_proj_weight = (4 * n_embd) * n_embd
    mlp_proj_bias = n_embd if bias else 0
    mlp_params = mlp_fc_weight + mlp_fc_bias + mlp_proj_weight + mlp_proj_bias

    # Per block total
    block_params = (2 * ln_params) + attn_params + mlp_params

    # Final layer norm
    ln_f_params = ln_params

    # lm_head is weight-tied with wte, so no additional params
    lm_head_params = 0

    # Totals
    total_params = wte_params + wpe_params + (n_layer * block_params) + ln_f_params
    non_embedding_params = total_params - wpe_params  # exclude position embeddings

    return {
        "total": total_params,
        "non_embedding": non_embedding_params,
        "wte": wte_params,
        "wpe": wpe_params,
        "block": block_params,
        "ln": ln_params,
        "attn": attn_params,
        "attn_qkv": c_attn_weight + c_attn_bias,
        "attn_proj": c_proj_weight + c_proj_bias,
        "mlp": mlp_params,
        "mlp_fc": mlp_fc_weight + mlp_fc_bias,
        "mlp_proj": mlp_proj_weight + mlp_proj_bias,
        "ln_f": ln_f_params,
        "lm_head": lm_head_params,
        "head_dim": head_dim,
    }


def calc_nanochat_params(n_layer: int, n_head: int, n_kv_head: int, n_embd: int,
                         vocab_size: int) -> dict:
    """
    Calculate nanochat parameter counts.

    Architecture differences from nanoGPT:
    - No position embeddings (uses rotary embeddings - no params)
    - RMSNorm has no learnable params
    - No bias in any linear layers
    - Separate Q, K, V projections (supports GQA)
    - lm_head is NOT weight-tied (separate params)
    - Per-layer resid_lambdas and x0_lambdas scalars
    """
    # Vocab size gets padded to multiple of 64
    padded_vocab_size = ((vocab_size + 63) // 64) * 64
    head_dim = n_embd // n_head

    # Token embedding (padded vocab)
    wte_params = padded_vocab_size * n_embd

    # Per-block calculations (no LayerNorm params in nanochat - RMSNorm is functional)
    # Attention: separate Q, K, V projections
    c_q_params = n_embd * (n_head * head_dim)  # Query projection
    c_k_params = n_embd * (n_kv_head * head_dim)  # Key projection (GQA)
    c_v_params = n_embd * (n_kv_head * head_dim)  # Value projection (GQA)
    c_proj_params = n_embd * n_embd  # Output projection
    attn_params = c_q_params + c_k_params + c_v_params + c_proj_params

    # MLP: same 4x expansion, no bias
    mlp_fc_params = n_embd * (4 * n_embd)
    mlp_proj_params = (4 * n_embd) * n_embd
    mlp_params = mlp_fc_params + mlp_proj_params

    # Per block total
    block_params = attn_params + mlp_params

    # lm_head (NOT weight-tied, uses padded vocab)
    lm_head_params = n_embd * padded_vocab_size

    # Per-layer scalars
    resid_lambdas = n_layer
    x0_lambdas = n_layer

    # Totals
    total_params = wte_params + (n_layer * block_params) + lm_head_params + resid_lambdas + x0_lambdas

    return {
        "total": total_params,
        "non_embedding": total_params,  # nanochat counts all params
        "wte": wte_params,
        "block": block_params,
        "attn": attn_params,
        "attn_q": c_q_params,
        "attn_k": c_k_params,
        "attn_v": c_v_params,
        "attn_proj": c_proj_params,
        "mlp": mlp_params,
        "mlp_fc": mlp_fc_params,
        "mlp_proj": mlp_proj_params,
        "lm_head": lm_head_params,
        "resid_lambdas": resid_lambdas,
        "x0_lambdas": x0_lambdas,
        "head_dim": head_dim,
        "padded_vocab_size": padded_vocab_size,
        "n_kv_head": n_kv_head,
    }


# =============================================================================
# Memory and Compute Estimation Functions
# =============================================================================

DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


def format_bytes(n_bytes: int) -> str:
    """Format byte count with appropriate suffix (binary units, 1024-based)."""
    KiB = 1024
    MiB = 1024 ** 2
    GiB = 1024 ** 3
    TiB = 1024 ** 4

    if n_bytes >= TiB:
        return f"{n_bytes/TiB:.2f} TB"
    elif n_bytes >= GiB:
        return f"{n_bytes/GiB:.2f} GB"
    elif n_bytes >= MiB:
        return f"{n_bytes/MiB:.2f} MB"
    elif n_bytes >= KiB:
        return f"{n_bytes/KiB:.1f} KB"
    return f"{n_bytes} B"


def format_flops(flops: int) -> str:
    """Format FLOP count with appropriate suffix."""
    if flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops/1e3:.1f} KFLOPs"
    return f"{flops} FLOPs"


def calc_memory_params(total_params: int, dtype: str = "bf16") -> int:
    """Calculate memory required for model parameters."""
    return total_params * DTYPE_BYTES[dtype]


def calc_memory_kv_cache(
    n_layer: int,
    n_kv_head: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    dtype: str = "bf16"
) -> int:
    """
    Calculate KV cache memory for inference.

    KV cache stores K and V tensors for all layers to avoid recomputation
    during autoregressive generation.

    Per layer: 2 (K and V) × batch × seq_len × n_kv_head × head_dim × bytes
    """
    bytes_per_elem = DTYPE_BYTES[dtype]
    per_layer = 2 * batch_size * seq_len * n_kv_head * head_dim * bytes_per_elem
    return n_layer * per_layer


def calc_memory_activations(
    n_layer: int,
    n_head: int,
    n_embd: int,
    batch_size: int,
    seq_len: int,
    dtype: str = "bf16",
    flash_attention: bool = True
) -> int:
    """
    Estimate activation memory for training (forward pass).

    Main activations per layer:
    - Input to attention/MLP: (B, T, d)
    - Q, K, V: 3 × (B, T, d)
    - Attention output: (B, T, d)
    - MLP intermediate: (B, T, 4d)
    - Attention scores: (B, h, T, T) - only if not using Flash Attention!

    This is a rough estimate - actual memory depends on framework, checkpointing, etc.
    """
    bytes_per_elem = DTYPE_BYTES[dtype]

    # Per-layer activation memory (rough estimate)
    # Residual stream saved for backward: (B, T, d)
    residual = batch_size * seq_len * n_embd * bytes_per_elem

    # QKV activations: 3 × (B, T, d)
    qkv = 3 * batch_size * seq_len * n_embd * bytes_per_elem

    # Attention output before projection: (B, T, d)
    attn_out = batch_size * seq_len * n_embd * bytes_per_elem

    # MLP intermediate (after up-projection): (B, T, 4d)
    mlp_intermediate = batch_size * seq_len * (4 * n_embd) * bytes_per_elem

    # Attention scores: (B, h, T, T) - this is the big one!
    # Flash Attention avoids materializing this
    if flash_attention:
        attn_scores = 0
    else:
        attn_scores = batch_size * n_head * seq_len * seq_len * bytes_per_elem

    per_layer = residual + qkv + attn_out + mlp_intermediate + attn_scores

    # Add ~20% overhead for gradients, optimizer states held during backward, etc.
    per_layer = int(per_layer * 1.2)

    return n_layer * per_layer


def calc_flops_per_token(
    n_layer: int,
    n_head: int,
    n_kv_head: int,
    n_embd: int,
    vocab_size: int,
    seq_len: int,
    forward_only: bool = True
) -> int:
    """
    Estimate FLOPs per token.

    Based on nanochat's estimate_flops() and the PaLM paper (Appendix B):
    https://github.com/karpathy/nanochat/discussions/420#discussion-9319591
    https://arxiv.org/abs/2204.02311

    Training formula (from nanochat):
        flops_per_token = 6 * (nparams - embedding_params) + 12 * l * h * q * t

    Where:
    - 6× multiplier: 2 FLOPs/param forward (multiply + accumulate) + 4 FLOPs/param backward
    - 12 * l * h * q * t: attention context term (Q @ K^T and attn @ V matmuls)
    - l = n_layer, h = n_head, q = head_dim, t = seq_len
    - Note: h * q = n_embd, so attention term = 12 * n_layer * n_embd * seq_len

    Forward-only uses 2× and 4× instead of 6× and 12×.

    We exclude embedding params (just a lookup, not a matmul) following nanochat/Chinchilla.
    """
    head_dim = n_embd // n_head

    # Count matmul parameters per layer (excluding embeddings)
    # Attention: Q, K, V projections + output projection
    # With GQA, K and V projections are smaller (n_kv_head instead of n_head)
    attn_q = n_embd * n_embd  # Q projection: (d, d)
    attn_k = n_embd * (n_kv_head * head_dim)  # K projection: (d, k×d_h)
    attn_v = n_embd * (n_kv_head * head_dim)  # V projection: (d, k×d_h)
    attn_proj = n_embd * n_embd  # Output projection: (d, d)

    # MLP: up and down projections
    mlp_up = n_embd * (4 * n_embd)  # (d, 4d)
    mlp_down = (4 * n_embd) * n_embd  # (4d, d)

    params_per_layer = attn_q + attn_k + attn_v + attn_proj + mlp_up + mlp_down
    total_matmul_params = n_layer * params_per_layer

    # LM head (unembedding): (d, V)
    lm_head_params = n_embd * vocab_size
    total_matmul_params += lm_head_params

    # Base FLOPs: 2 × params for forward (multiply + accumulate)
    # Training: 6 × params (2 forward + 4 backward)
    multiplier = 2 if forward_only else 6
    base_flops = multiplier * total_matmul_params

    # Attention context-dependent FLOPs per layer (not included in 2N approximation):
    # Q @ K^T: 2 × T × d_attn per token (d_attn = n_head × head_dim = n_embd)
    # attn @ V: 2 × T × d_attn per token
    # Total: 4 × T × d_attn per token per layer
    # Note: GQA doesn't reduce this - K,V are broadcast to all query heads
    attn_context_flops_per_layer = 4 * seq_len * n_embd
    total_attn_context_flops = n_layer * attn_context_flops_per_layer
    if not forward_only:
        total_attn_context_flops *= 3  # backward adds ~2× more

    return base_flops + total_attn_context_flops


def calc_generation_flops(
    n_layer: int,
    n_head: int,
    n_kv_head: int,
    n_embd: int,
    vocab_size: int,
    prompt_len: int,
    gen_tokens: int,
    use_kv_cache: bool = True
) -> int:
    """
    Calculate total FLOPs to generate tokens autoregressively.

    Without KV cache: Must recompute K, V for ALL previous tokens at each step.
    With KV cache: Only compute K, V for the new token, retrieve cached values.

    The key insight:
    - Without KV cache: generating token t requires a forward pass over all t tokens
      Total = sum(t for t in 1..N) = N(N+1)/2 token-passes
    - With KV cache: generating token t requires forward pass for 1 new token
      Total = N token-passes (but attention still looks at full context)
    """
    head_dim = n_embd // n_head

    # FLOPs for projections (QKV + output + MLP + LM head) per token
    # These scale with number of tokens processed
    qkv_proj = n_embd * (n_embd + 2 * n_kv_head * head_dim)  # Q + K + V projections
    attn_proj = n_embd * n_embd
    mlp = 2 * n_embd * (4 * n_embd)  # up + down
    lm_head = n_embd * vocab_size
    proj_flops_per_token = 2 * n_layer * (qkv_proj + attn_proj + mlp) + 2 * lm_head

    # Attention FLOPs per layer: scales with context length
    # Q @ K^T + attn @ V = 4 * d * context_len per token
    def attn_flops_for_context(context_len: int) -> int:
        return 4 * n_embd * context_len * n_layer

    total_flops = 0

    if use_kv_cache:
        # With KV cache:
        # - Prompt: process all prompt tokens at once (like training)
        # - Generation: each new token only computes its own projections
        #   but attention still looks at full context

        # Prompt processing (prefill)
        total_flops += prompt_len * proj_flops_per_token
        total_flops += prompt_len * attn_flops_for_context(prompt_len // 2)  # avg context

        # Generation: 1 new token at a time
        for i in range(gen_tokens):
            context_len = prompt_len + i + 1
            total_flops += proj_flops_per_token  # only 1 token's projections
            total_flops += attn_flops_for_context(context_len)  # attention over full context
    else:
        # Without KV cache:
        # Must recompute everything from scratch at each step
        for i in range(gen_tokens):
            context_len = prompt_len + i + 1
            # Process ALL tokens (prompt + generated so far)
            total_flops += context_len * proj_flops_per_token
            total_flops += context_len * attn_flops_for_context(context_len // 2)

    return total_flops


# =============================================================================
# Graphviz Diagram Generation
# =============================================================================

def create_nanogpt_diagram(params: dict, config: dict) -> graphviz.Digraph:
    """Create a graphviz diagram for nanoGPT architecture."""
    dot = graphviz.Digraph(comment='nanoGPT Architecture')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.6')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')

    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    code = NANOGPT_CODE  # Code snippets for tooltips

    # Input
    dot.node('input', 'Input Tokens\n(batch, seq_len)', fillcolor='#e8f4f8',
             tooltip='Input token IDs of shape (batch_size, sequence_length)')

    # Token Embedding
    dot.node('wte', f'Token Embedding\nwte: {format_params(params["wte"])}\n({config["vocab_size"]} × {n_embd})', fillcolor='#d4edda',
             tooltip=code["wte"])

    # Position Embedding
    dot.node('wpe', f'Position Embedding\nwpe: {format_params(params["wpe"])}\n({config["block_size"]} × {n_embd})', fillcolor='#d4edda',
             tooltip=code["wpe"])

    # Embedding sum
    dot.node('emb_sum', 'Add + Dropout', fillcolor='#fff3cd', shape='ellipse',
             tooltip='Add token and position embeddings, then apply dropout')

    # Create a subgraph for transformer blocks
    with dot.subgraph(name='cluster_blocks') as blocks:
        blocks.attr(label=f'Transformer Blocks (×{n_layer})', style='dashed', color='gray',
                   tooltip=code["block"])

        # Show one representative block with details
        with blocks.subgraph(name='cluster_block') as block:
            block.attr(label=f'Block (×{n_layer})\nTotal per block: {format_params(params["block"])}',
                      style='rounded', color='#6c757d', bgcolor='#f8f9fa',
                      tooltip=code["block"])

            # LayerNorm 1
            block.node('ln1', f'LayerNorm\n{format_params(params["ln"])}', fillcolor='#e2e3e5',
                      tooltip=code["ln"])

            # Attention subgraph
            with block.subgraph(name='cluster_attn') as attn:
                attn.attr(label=f'Multi-Head Attention: {format_params(params["attn"])}', style='rounded', color='#007bff', bgcolor='#cce5ff')
                attn.node('qkv', f'QKV Projection\nc_attn: {format_params(params["attn_qkv"])}\n({n_embd} → {3*n_embd})', fillcolor='#b8daff',
                         tooltip=code["c_attn"])
                attn.node('attn_op', f'Scaled Dot-Product\nAttention\n({n_head} heads)', fillcolor='#b8daff', shape='ellipse',
                         tooltip=code["attn_op"])
                attn.node('attn_proj', f'Output Projection\nc_proj: {format_params(params["attn_proj"])}\n({n_embd} → {n_embd})', fillcolor='#b8daff',
                         tooltip=code["attn_proj"])

            # Residual 1
            block.node('res1', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse',
                      tooltip='x = x + attn(ln_1(x))\nResidual connection around attention')

            # LayerNorm 2
            block.node('ln2', f'LayerNorm\n{format_params(params["ln"])}', fillcolor='#e2e3e5',
                      tooltip=code["ln"])

            # MLP subgraph
            with block.subgraph(name='cluster_mlp') as mlp:
                mlp.attr(label=f'Feed-Forward Network (MLP): {format_params(params["mlp"])}', style='rounded', color='#28a745', bgcolor='#d4edda')
                mlp.node('mlp_fc', f'Up Projection\nc_fc: {format_params(params["mlp_fc"])}\n({n_embd} → {4*n_embd})', fillcolor='#c3e6cb',
                        tooltip=code["mlp_fc"])
                mlp.node('gelu', 'GELU\nActivation', fillcolor='#c3e6cb', shape='ellipse',
                        tooltip=code["gelu"])
                mlp.node('mlp_proj', f'Down Projection\nc_proj: {format_params(params["mlp_proj"])}\n({4*n_embd} → {n_embd})', fillcolor='#c3e6cb',
                        tooltip=code["mlp_proj"])

            # Residual 2
            block.node('res2', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse',
                      tooltip='x = x + mlp(ln_2(x))\nResidual connection around MLP')

    # Final LayerNorm
    dot.node('ln_f', f'Final LayerNorm\nln_f: {format_params(params["ln_f"])}', fillcolor='#e2e3e5',
             tooltip=code["ln_f"])

    # LM Head
    dot.node('lm_head', f'Language Model Head\nProjects to vocabulary\n(weight-tied with wte)\n({n_embd} → {config["vocab_size"]})', fillcolor='#f5c6cb',
             tooltip=code["lm_head"])

    # Output
    dot.node('output', f'Output Logits\n(batch, seq_len, {config["vocab_size"]})', fillcolor='#e8f4f8',
             tooltip='Output logits for each token position\nApply softmax to get probabilities')

    # Edges
    dot.edge('input', 'wte')
    dot.edge('input', 'wpe')
    dot.edge('wte', 'emb_sum')
    dot.edge('wpe', 'emb_sum')
    dot.edge('emb_sum', 'ln1')
    dot.edge('ln1', 'qkv')
    dot.edge('qkv', 'attn_op')
    dot.edge('attn_op', 'attn_proj')
    dot.edge('attn_proj', 'res1')
    dot.edge('emb_sum', 'res1', style='dashed', color='gray')
    dot.edge('res1', 'ln2')
    dot.edge('ln2', 'mlp_fc')
    dot.edge('mlp_fc', 'gelu')
    dot.edge('gelu', 'mlp_proj')
    dot.edge('mlp_proj', 'res2')
    dot.edge('res1', 'res2', style='dashed', color='gray')
    dot.edge('res2', 'ln_f')
    dot.edge('ln_f', 'lm_head')
    dot.edge('lm_head', 'output')

    return dot


def create_nanochat_diagram(params: dict, config: dict) -> graphviz.Digraph:
    """Create a graphviz diagram for nanochat architecture."""
    dot = graphviz.Digraph(comment='nanochat Architecture')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.6')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')

    n_layer = config['n_layer']
    n_head = config['n_head']
    n_kv_head = config['n_kv_head']
    n_embd = config['n_embd']
    head_dim = params['head_dim']
    code = NANOCHAT_CODE  # Code snippets for tooltips

    gqa_ratio = n_head // n_kv_head
    gqa_label = f"GQA {n_head}:{n_kv_head}" if gqa_ratio > 1 else "MHA"

    # Input
    dot.node('input', 'Input Tokens\n(batch, seq_len)', fillcolor='#e8f4f8',
             tooltip='Input token IDs of shape (batch_size, sequence_length)')

    # Token Embedding
    dot.node('wte', f'Token Embedding\nwte: {format_params(params["wte"])}\n({params["padded_vocab_size"]} × {n_embd})', fillcolor='#d4edda',
             tooltip=code["wte"])

    # RMSNorm (no params)
    dot.node('norm_emb', 'RMSNorm\n(no params)', fillcolor='#e2e3e5',
             tooltip=code["norm"])

    # Save x0 for skip connection
    dot.node('save_x0', 'Save Initial Embedding\n(x₀ for residual blending)', fillcolor='#fff3cd', shape='ellipse',
             tooltip='x0 = norm(wte(idx))\nSaved for blending back into residual stream at each layer')

    # Create a subgraph for transformer blocks
    with dot.subgraph(name='cluster_blocks') as blocks:
        blocks.attr(label=f'Transformer Blocks (×{n_layer})', style='dashed', color='gray',
                   tooltip=code["block"])

        # Show one representative block with details
        with blocks.subgraph(name='cluster_block') as block:
            block.attr(label=f'Block (×{n_layer})\nTotal per block: {format_params(params["block"])}',
                      style='rounded', color='#6c757d', bgcolor='#f8f9fa',
                      tooltip=code["block"])

            # Per-layer scaling
            block.node('scale_in',
                      f'Residual Scaling\nBlends current hidden state with\ninitial embedding (per-layer learnable)\nresid_lambdas: {n_layer}, x0_lambdas: {n_layer}',
                      fillcolor='#ffeeba', shape='box',
                      tooltip=code["scale_in"])

            # RMSNorm (no params)
            block.node('norm1', 'RMSNorm\n(no params)', fillcolor='#e2e3e5',
                      tooltip=code["norm"])

            # Attention subgraph
            with block.subgraph(name='cluster_attn') as attn:
                attn.attr(label=f'Grouped-Query Attention ({gqa_label}): {format_params(params["attn"])}',
                         style='rounded', color='#007bff', bgcolor='#cce5ff')

                # Separate Q, K, V projections
                attn.node('q_proj', f'Query Projection\nc_q: {format_params(params["attn_q"])}\n({n_embd} → {n_head}×{head_dim})', fillcolor='#b8daff',
                         tooltip=code["c_q"])
                attn.node('k_proj', f'Key Projection\nc_k: {format_params(params["attn_k"])}\n({n_embd} → {n_kv_head}×{head_dim})', fillcolor='#b8daff',
                         tooltip=code["c_k"])
                attn.node('v_proj', f'Value Projection\nc_v: {format_params(params["attn_v"])}\n({n_embd} → {n_kv_head}×{head_dim})', fillcolor='#b8daff',
                         tooltip=code["c_v"])

                attn.node('rotary', 'Rotary Position Encoding\n+ QK Normalization\n(no learnable params)', fillcolor='#e7f1ff', shape='ellipse',
                         tooltip=code["rotary"])
                attn.node('flash_attn', f'Scaled Dot-Product Attention\n(Flash Attention 3)\n{n_head} Q heads, {n_kv_head} KV heads', fillcolor='#b8daff', shape='ellipse',
                         tooltip=code["flash_attn"])
                attn.node('attn_proj', f'Output Projection\nc_proj: {format_params(params["attn_proj"])}\n({n_embd} → {n_embd})', fillcolor='#b8daff',
                         tooltip=code["attn_proj"])

            # Residual 1
            block.node('res1', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse',
                      tooltip='x = x + self.attn(norm(x), ...)\nResidual connection around attention')

            # RMSNorm 2 (no params)
            block.node('norm2', 'RMSNorm\n(no params)', fillcolor='#e2e3e5',
                      tooltip=code["norm"])

            # MLP subgraph
            with block.subgraph(name='cluster_mlp') as mlp:
                mlp.attr(label=f'Feed-Forward Network (MLP): {format_params(params["mlp"])}', style='rounded', color='#28a745', bgcolor='#d4edda')
                mlp.node('mlp_fc', f'Up Projection\nc_fc: {format_params(params["mlp_fc"])}\n({n_embd} → {4*n_embd})', fillcolor='#c3e6cb',
                        tooltip=code["mlp_fc"])
                mlp.node('relu_sq', 'ReLU² Activation\n(squared ReLU)', fillcolor='#c3e6cb', shape='ellipse',
                        tooltip=code["relu_sq"])
                mlp.node('mlp_proj', f'Down Projection\nc_proj: {format_params(params["mlp_proj"])}\n({4*n_embd} → {n_embd})', fillcolor='#c3e6cb',
                        tooltip=code["mlp_proj"])

            # Residual 2
            block.node('res2', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse',
                      tooltip='x = x + self.mlp(norm(x))\nResidual connection around MLP')

    # Final RMSNorm (no params)
    dot.node('norm_f', 'RMSNorm\n(no params)', fillcolor='#e2e3e5',
             tooltip=code["norm"])

    # LM Head (NOT weight-tied)
    dot.node('lm_head', f'Language Model Head\nProjects to vocabulary\nlm_head: {format_params(params["lm_head"])}\n({n_embd} → {params["padded_vocab_size"]})\n(separate weights, not tied)', fillcolor='#f5c6cb',
             tooltip=code["lm_head"])

    # Softcap
    dot.node('softcap', 'Logit Soft-Capping\nSmooths extreme values\n15 · tanh(x/15)', fillcolor='#e8f4f8', shape='ellipse',
             tooltip=code["softcap"])

    # Output
    dot.node('output', f'Output Logits\n(batch, seq_len, {config["vocab_size"]})', fillcolor='#e8f4f8',
             tooltip='Output logits for each token position\nApply softmax to get probabilities')

    # Edges
    dot.edge('input', 'wte')
    dot.edge('wte', 'norm_emb')
    dot.edge('norm_emb', 'save_x0')
    dot.edge('save_x0', 'scale_in')
    dot.edge('scale_in', 'norm1')
    dot.edge('norm1', 'q_proj')
    dot.edge('norm1', 'k_proj')
    dot.edge('norm1', 'v_proj')
    dot.edge('q_proj', 'rotary')
    dot.edge('k_proj', 'rotary')
    dot.edge('rotary', 'flash_attn')
    dot.edge('v_proj', 'flash_attn')
    dot.edge('flash_attn', 'attn_proj')
    dot.edge('attn_proj', 'res1')
    dot.edge('scale_in', 'res1', style='dashed', color='gray')
    dot.edge('res1', 'norm2')
    dot.edge('norm2', 'mlp_fc')
    dot.edge('mlp_fc', 'relu_sq')
    dot.edge('relu_sq', 'mlp_proj')
    dot.edge('mlp_proj', 'res2')
    dot.edge('res1', 'res2', style='dashed', color='gray')
    dot.edge('res2', 'norm_f')
    dot.edge('norm_f', 'lm_head')
    dot.edge('lm_head', 'softcap')
    dot.edge('softcap', 'output')

    return dot


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.title("Transformer Architecture Visualizer")
    st.markdown("Interactive exploration of nanoGPT and nanochat model architectures · [GitHub](https://github.com/MarkDunne/transformer-viz)")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Model Configuration")

        # Model type selector
        model_type = st.radio(
            "Architecture",
            ["nanochat", "nanoGPT"],
            help="nanochat: Modern with GQA, RoPE, etc. nanoGPT: Classic GPT-2 style."
        )

        # GitHub links
        st.caption("[nanoGPT](https://github.com/karpathy/nanoGPT) · [nanochat](https://github.com/znation/nanochat)")

        st.divider()

        # Preset configurations
        st.subheader("Presets")
        if model_type == "nanoGPT":
            preset = st.selectbox(
                "Load preset",
                ["nanogpt-tiny", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
            )
            presets = {
                "nanogpt-tiny": {"n_layer": 3, "n_head": 3, "n_embd": 48, "block_size": 11, "vocab_size": 3},
                "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768, "block_size": 1024, "vocab_size": 50304},
                "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024, "block_size": 1024, "vocab_size": 50304},
                "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280, "block_size": 1024, "vocab_size": 50304},
                "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600, "block_size": 1024, "vocab_size": 50304},
            }
        else:
            preset = st.selectbox(
                "Load preset",
                ["nanochat-small", "nanochat-medium", "nanochat-large"],
            )
            presets = {
                "nanochat-small": {"n_layer": 12, "n_head": 6, "n_kv_head": 6, "n_embd": 768, "sequence_len": 1024, "vocab_size": 50304},
                "nanochat-medium": {"n_layer": 24, "n_head": 16, "n_kv_head": 4, "n_embd": 1024, "sequence_len": 2048, "vocab_size": 50304},
                "nanochat-large": {"n_layer": 36, "n_head": 16, "n_kv_head": 4, "n_embd": 1536, "sequence_len": 4096, "vocab_size": 50304},
            }

        defaults = presets[preset]

        st.divider()
        st.subheader("Parameters")

        # Common parameters
        n_layer = st.slider("n_layer (depth)", 1, 128, defaults.get("n_layer", 12),
                           help="Number of transformer blocks")

        # n_embd first, then constrain n_head to valid divisors
        n_embd = st.select_slider("n_embd (width)",
                                  options=[48, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1600, 2048, 4096],
                                  value=defaults.get("n_embd", 768),
                                  help="Embedding dimension / hidden size")

        # Compute valid n_head values (divisors of n_embd, capped at 128)
        valid_heads = [h for h in range(1, min(n_embd, 128) + 1) if n_embd % h == 0]
        default_head = defaults.get("n_head", 12)
        if default_head not in valid_heads:
            # Snap to nearest valid divisor
            default_head = min(valid_heads, key=lambda x: abs(x - default_head))
        n_head = st.select_slider("n_head (attention heads)",
                                  options=valid_heads,
                                  value=default_head,
                                  help="Number of attention heads (must divide n_embd evenly)")

        vocab_size = st.number_input("vocab_size", 1, 200000, defaults.get("vocab_size", 50304),
                                     help="Vocabulary size")

        # Model-specific parameters
        if model_type == "nanoGPT":
            block_size = st.slider("block_size (context)", 1, 8192, defaults.get("block_size", 1024),
                                  help="Maximum sequence length / context window")
            bias = st.checkbox("bias", value=True, help="Include bias in linear layers and LayerNorm")

            config = {
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "vocab_size": vocab_size,
                "block_size": block_size,
                "bias": bias,
            }
        else:
            # GQA configuration
            st.markdown("**Group-Query Attention (GQA)**")
            # n_kv_head must divide n_head evenly
            valid_kv_heads = [h for h in range(1, n_head + 1) if n_head % h == 0]
            default_kv = defaults.get("n_kv_head", n_head)
            if default_kv not in valid_kv_heads:
                default_kv = valid_kv_heads[-1]  # default to MHA
            n_kv_head = st.select_slider("n_kv_head (KV heads)",
                                        options=valid_kv_heads,
                                        value=default_kv,
                                        help="Number of key/value heads. Less than n_head enables GQA.")

            gqa_ratio = n_head // n_kv_head
            if gqa_ratio == 1:
                st.info("MHA: Each query head has its own KV pair")
            elif n_kv_head == 1:
                st.info(f"MQA: All {n_head} query heads share 1 KV pair")
            else:
                st.info(f"GQA: {gqa_ratio} query heads share each KV pair")

            # Note: nanochat uses RoPE, so sequence length is not an architectural
            # constraint (unlike nanoGPT's block_size). It only affects inference
            # memory/compute, configured in Estimation Settings below.

            config = {
                "n_layer": n_layer,
                "n_head": n_head,
                "n_kv_head": n_kv_head,
                "n_embd": n_embd,
                "vocab_size": vocab_size,
            }

        # Shape notation legend with actual values
        st.divider()
        st.subheader("Shape Notation")

        # Compute derived values
        head_dim = n_embd // n_head
        # For nanoGPT, block_size is architectural; for nanochat, use preset default for estimation
        seq_len = config.get("block_size", defaults.get("sequence_len", 1024))
        kv_heads = config.get("n_kv_head", n_head)

        st.markdown(f"""
| Symbol | Meaning | Value |
|--------|---------|-------|
| `d` | n_embd | **{n_embd}** |
| `h` | n_head | **{n_head}** |
| `k` | n_kv_head | **{kv_heads}** |
| `d_h` | head dim (d/h) | **{head_dim}** |
| `V` | vocab_size | **{vocab_size:,}** |
| `T` | seq length | **{seq_len:,}** |
| `L` | n_layer | **{n_layer}** |
| `B` | batch size | *(see below)* |
""")
        st.caption("Hover over diagram nodes for detailed shape info")

        # Memory/compute estimation settings
        st.divider()
        st.subheader("Estimation Settings")
        st.caption("Configure scenario for memory & compute estimates")
        est_batch_size = st.number_input("Concurrent users", 1, 100000, 1, help="Number of parallel requests/users (each needs their own KV cache)")
        est_prompt_len = st.number_input("Prompt length", 1, 1000000, seq_len, help="Number of tokens in the input prompt")
        est_gen_tokens = st.number_input("Tokens to generate", 1, 100000, 100, help="Number of new tokens to generate after the prompt")
        est_dtype = st.selectbox("Dtype", ["bf16", "fp16", "fp32", "int8"], help="Data type for memory calculations")

        # Total context = prompt + generated
        est_total_context = est_prompt_len + est_gen_tokens

    # Calculate parameters
    if model_type == "nanoGPT":
        params = calc_nanogpt_params(
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_embd=config["n_embd"],
            vocab_size=config["vocab_size"],
            block_size=config["block_size"],
            bias=config["bias"],
        )
    else:
        params = calc_nanochat_params(
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_kv_head=config["n_kv_head"],
            n_embd=config["n_embd"],
            vocab_size=config["vocab_size"],
        )

    # Main content area - two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Architecture Diagram")
        st.caption("💡 Hover over nodes to see the corresponding code")

        # Generate and display diagram
        if model_type == "nanoGPT":
            diagram = create_nanogpt_diagram(params, config)
        else:
            diagram = create_nanochat_diagram(params, config)

        render_svg_with_tooltips(diagram, height=1200)

    with col2:
        st.header("Parameter Summary")

        # Key metrics
        st.metric("Total Parameters", format_params(params["total"]))
        st.metric("Non-Embedding Parameters", format_params(params["non_embedding"]))

        st.divider()

        # Detailed breakdown
        st.subheader("Component Breakdown")

        breakdown_data = {
            "Component": [],
            "Parameters": [],
            "% of Total": [],
        }

        if model_type == "nanoGPT":
            components = [
                ("Token Embedding (wte)", params["wte"]),
                ("Position Embedding (wpe)", params["wpe"]),
                (f"Attention (×{n_layer})", params["attn"] * n_layer),
                (f"MLP (×{n_layer})", params["mlp"] * n_layer),
                (f"LayerNorm (×{2*n_layer + 1})", params["ln"] * (2 * n_layer + 1)),
            ]
        else:
            components = [
                ("Token Embedding (wte)", params["wte"]),
                (f"Attention (×{n_layer})", params["attn"] * n_layer),
                (f"MLP (×{n_layer})", params["mlp"] * n_layer),
                ("LM Head", params["lm_head"]),
                ("Per-layer scalars", params["resid_lambdas"] + params["x0_lambdas"]),
            ]

        for name, count in components:
            breakdown_data["Component"].append(name)
            breakdown_data["Parameters"].append(format_params(count))
            breakdown_data["% of Total"].append(f"{100 * count / params['total']:.1f}%")

        st.table(breakdown_data)

        st.divider()

        # Architecture details
        st.subheader("Architecture Details")
        st.markdown(f"**Head dimension:** {params['head_dim']}")
        st.markdown(f"**MLP hidden size:** {4 * n_embd}")

        if model_type == "nanochat":
            st.markdown(f"**Padded vocab size:** {params['padded_vocab_size']}")
            gqa_ratio = n_head // n_kv_head
            st.markdown(f"**GQA ratio:** {gqa_ratio}:1 ({n_head} Q heads : {n_kv_head} KV heads)")
            kv_cache_reduction = (n_head - n_kv_head) / n_head * 100
            if kv_cache_reduction > 0:
                st.markdown(f"**KV cache reduction:** {kv_cache_reduction:.0f}%")

        st.divider()

        # Memory and Compute Estimates
        st.subheader("Memory & Compute")

        # Get n_kv_head (same as n_head for nanoGPT)
        n_kv_head_for_calc = config.get("n_kv_head", n_head)

        # Calculate estimates
        param_memory = calc_memory_params(params["total"], est_dtype)
        kv_cache_memory = calc_memory_kv_cache(
            n_layer=n_layer,
            n_kv_head=n_kv_head_for_calc,
            head_dim=params["head_dim"],
            batch_size=est_batch_size,
            seq_len=est_total_context,  # KV cache holds prompt + generated
            dtype=est_dtype
        )
        activation_memory = calc_memory_activations(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            batch_size=est_batch_size,
            seq_len=est_total_context,
            dtype=est_dtype,
            flash_attention=True
        )
        flops_fwd = calc_flops_per_token(
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head_for_calc,
            n_embd=n_embd,
            vocab_size=vocab_size,
            seq_len=est_total_context,
            forward_only=True
        )
        flops_train = calc_flops_per_token(
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head_for_calc,
            n_embd=n_embd,
            vocab_size=vocab_size,
            seq_len=est_total_context,
            forward_only=False
        )

        # Display memory estimates
        st.markdown(f"**Parameter Memory ({est_dtype}):** {format_bytes(param_memory)}")
        st.caption(f"`N × bytes = {format_params(params['total'])} × {DTYPE_BYTES[est_dtype]}`")

        st.markdown(f"**KV Cache ({est_batch_size} users × {est_total_context:,} ctx):** {format_bytes(kv_cache_memory)}")
        st.caption(f"`2 × L × T × k × d_h × bytes = 2 × {n_layer} × {est_total_context:,} × {n_kv_head_for_calc} × {params['head_dim']} × {DTYPE_BYTES[est_dtype]}`")

        st.markdown(f"**Activations (training, est.):** {format_bytes(activation_memory)}")
        st.caption("Rough estimate: residuals + QKV + MLP intermediate + 20% overhead")

        st.divider()

        # Display compute estimates
        st.markdown(f"**FLOPs/token (forward):** {format_flops(flops_fwd)}")
        st.caption(f"`2N + 4×L×d×T = 2×{format_params(params['total'])} + 4×{n_layer}×{n_embd}×{est_total_context:,}`")

        st.markdown(f"**FLOPs/token (training):** {format_flops(flops_train)}")
        st.caption(f"`6N + 12×L×d×T` (forward + backward)")

        st.caption("[FLOPs methodology](https://github.com/karpathy/nanochat/discussions/420#discussion-9319591)")

        # GQA comparison for nanochat
        if model_type == "nanochat" and n_kv_head_for_calc < n_head:
            # Calculate what KV cache would be without GQA
            kv_cache_mha = calc_memory_kv_cache(
                n_layer=n_layer,
                n_kv_head=n_head,  # Full MHA
                head_dim=params["head_dim"],
                batch_size=est_batch_size,
                seq_len=est_total_context,
                dtype=est_dtype
            )
            savings = (kv_cache_mha - kv_cache_memory) / kv_cache_mha * 100
            st.divider()
            st.markdown("**GQA Savings:**")
            st.markdown(f"- KV cache with MHA: {format_bytes(kv_cache_mha)}")
            st.markdown(f"- KV cache with GQA: {format_bytes(kv_cache_memory)}")
            st.markdown(f"- **Reduction: {savings:.0f}%**")

        # KV Cache value for generation
        st.divider()
        st.markdown(f"**Generation Cost ({est_prompt_len:,} prompt → {est_gen_tokens:,} tokens):**")

        # Calculate FLOPs with and without KV cache
        prompt_len = est_prompt_len
        flops_with_cache = calc_generation_flops(
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head_for_calc,
            n_embd=n_embd,
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            gen_tokens=est_gen_tokens,
            use_kv_cache=True
        )
        flops_without_cache = calc_generation_flops(
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head_for_calc,
            n_embd=n_embd,
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            gen_tokens=est_gen_tokens,
            use_kv_cache=False
        )
        speedup = flops_without_cache / flops_with_cache

        st.markdown(f"- With KV cache: {format_flops(flops_with_cache)}")
        st.markdown(f"- Without KV cache: {format_flops(flops_without_cache)}")
        st.markdown(f"- **Speedup: {speedup:.1f}×**")
        st.caption(f"With cache: `2N×G + 4Ld×(PG + G²/2)` — only new token through linear layers")
        st.caption(f"Without: `2N×Σt + 4Ld×Σt²` — all tokens recomputed each step")

        st.divider()

        # Config summary
        st.subheader("Configuration")
        st.json(config)


if __name__ == "__main__":
    main()
