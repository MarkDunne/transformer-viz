"""
Interactive Transformer Architecture Visualizer

A Streamlit dashboard for exploring nanoGPT and nanochat model architectures.
Displays model components, parameter counts, and architecture diagrams.
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
NANOGPT_CODE = {
    "wte": """# Token Embedding (model.py:127)
nn.Embedding(config.vocab_size, config.n_embd)

# Weight-tied with lm_head (model.py:138)
self.transformer.wte.weight = self.lm_head.weight""",

    "wpe": """# Position Embedding (model.py:128)
nn.Embedding(config.block_size, config.n_embd)""",

    "ln": """# LayerNorm with optional bias (model.py:18-27)
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None""",

    "c_attn": """# Combined QKV projection (model.py:35)
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

# Split into Q, K, V (model.py:56)
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)""",

    "attn_op": """# Scaled dot-product attention (model.py:62-71)
if self.flash:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v""",

    "attn_proj": """# Output projection (model.py:37)
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

# Applied after attention (model.py:75)
y = self.resid_dropout(self.c_proj(y))""",

    "mlp_fc": """# Up projection / first FC layer (model.py:82)
self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)""",

    "gelu": """# GELU activation (model.py:83, 88-89)
self.gelu = nn.GELU()
x = self.gelu(x)""",

    "mlp_proj": """# Down projection / second FC layer (model.py:84)
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)""",

    "ln_f": """# Final LayerNorm (model.py:131)
ln_f = LayerNorm(config.n_embd, bias=config.bias)""",

    "lm_head": """# Language model head (model.py:133)
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

# Weight-tied with token embedding (model.py:138)
self.transformer.wte.weight = self.lm_head.weight""",

    "block": """# Transformer Block (model.py:94-106)
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x""",
}

# nanochat code snippets (from nanochat/gpt.py)
NANOCHAT_CODE = {
    "wte": """# Token Embedding with vocab padding (gpt.py:155, 159)
padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
nn.Embedding(padded_vocab_size, config.n_embd)""",

    "norm": """# RMSNorm - purely functional, no learnable params (gpt.py:48-50)
def norm(x):
    return F.rms_norm(x, (x.size(-1),))""",

    "scale_in": """# Per-layer residual scaling (gpt.py:167-168, 359-360)
self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

# Applied before each block (gpt.py:359-360)
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
x = block(x, cos_sin, self.window_sizes[i], kv_cache)""",

    "c_q": """# Query projection (gpt.py:71)
self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)""",

    "c_k": """# Key projection - fewer heads for GQA (gpt.py:72)
self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)""",

    "c_v": """# Value projection - fewer heads for GQA (gpt.py:73)
self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)""",

    "rotary": """# Rotary embeddings (gpt.py:53-59, 86-88)
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

# QK normalization (gpt.py:88)
q, k = norm(q), norm(k)""",

    "flash_attn": """# Flash Attention 3 (gpt.py:93-95)
# FA3 handles GQA automatically when n_kv_heads < n_heads
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)""",

    "attn_proj": """# Output projection (gpt.py:74)
self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)""",

    "mlp_fc": """# Up projection (gpt.py:119)
self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)""",

    "relu_sq": """# Squared ReLU activation (gpt.py:124)
x = F.relu(x).square()""",

    "mlp_proj": """# Down projection (gpt.py:120)
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)""",

    "lm_head": """# Language model head - NOT weight-tied (gpt.py:162)
self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)""",

    "softcap": """# Logit soft-capping (gpt.py:364-368)
softcap = 15
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size]  # remove padding
logits = softcap * torch.tanh(logits / softcap)  # smooth extreme values""",

    "block": """# Transformer Block (gpt.py:129-138)
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x""",
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
                         vocab_size: int, sequence_len: int) -> dict:
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
    st.markdown("Interactive exploration of nanoGPT and nanochat model architectures")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Model Configuration")

        # Model type selector
        model_type = st.radio(
            "Architecture",
            ["nanoGPT", "nanochat"],
            help="nanoGPT: Classic GPT-2 style. nanochat: Modern with GQA, RoPE, etc."
        )

        st.divider()

        # Preset configurations
        st.subheader("Presets")
        if model_type == "nanoGPT":
            preset = st.selectbox(
                "Load preset",
                ["Custom", "nanogpt-tiny", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
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
                ["Custom", "nanochat-small", "nanochat-medium", "nanochat-large"],
            )
            presets = {
                "nanochat-small": {"n_layer": 12, "n_head": 6, "n_kv_head": 6, "n_embd": 768, "sequence_len": 1024, "vocab_size": 50304},
                "nanochat-medium": {"n_layer": 24, "n_head": 12, "n_kv_head": 4, "n_embd": 1024, "sequence_len": 2048, "vocab_size": 50304},
                "nanochat-large": {"n_layer": 36, "n_head": 16, "n_kv_head": 4, "n_embd": 1536, "sequence_len": 4096, "vocab_size": 50304},
            }

        # Get defaults from preset or use sensible defaults
        if preset != "Custom" and preset in presets:
            defaults = presets[preset]
        else:
            if model_type == "nanoGPT":
                defaults = {"n_layer": 12, "n_head": 12, "n_embd": 768, "block_size": 1024, "vocab_size": 50304}
            else:
                defaults = {"n_layer": 12, "n_head": 6, "n_kv_head": 6, "n_embd": 768, "sequence_len": 1024, "vocab_size": 50304}

        st.divider()
        st.subheader("Parameters")

        # Common parameters
        n_layer = st.slider("n_layer (depth)", 1, 64, defaults.get("n_layer", 12),
                           help="Number of transformer blocks")
        n_head = st.slider("n_head (attention heads)", 1, 32, defaults.get("n_head", 12),
                          help="Number of attention heads (query heads for GQA)")
        n_embd = st.select_slider("n_embd (width)",
                                  options=[48, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1600, 2048, 4096],
                                  value=defaults.get("n_embd", 768),
                                  help="Embedding dimension / hidden size")
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
            sequence_len = st.slider("sequence_len (context)", 1, 8192, defaults.get("sequence_len", 1024),
                                    help="Maximum sequence length / context window")

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

            config = {
                "n_layer": n_layer,
                "n_head": n_head,
                "n_kv_head": n_kv_head,
                "n_embd": n_embd,
                "vocab_size": vocab_size,
                "sequence_len": sequence_len,
            }

        # Validation
        if n_embd % n_head != 0:
            st.error(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
            st.stop()

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
            sequence_len=config["sequence_len"],
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

        # Config summary
        st.subheader("Configuration")
        st.json(config)


if __name__ == "__main__":
    main()
