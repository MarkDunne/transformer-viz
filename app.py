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

    # Input
    dot.node('input', 'Input Tokens\n(batch, seq_len)', fillcolor='#e8f4f8')

    # Token Embedding
    dot.node('wte', f'Token Embedding\nwte: {format_params(params["wte"])}\n({config["vocab_size"]} × {n_embd})', fillcolor='#d4edda')

    # Position Embedding
    dot.node('wpe', f'Position Embedding\nwpe: {format_params(params["wpe"])}\n({config["block_size"]} × {n_embd})', fillcolor='#d4edda')

    # Embedding sum
    dot.node('emb_sum', 'Add + Dropout', fillcolor='#fff3cd', shape='ellipse')

    # Create a subgraph for transformer blocks
    with dot.subgraph(name='cluster_blocks') as blocks:
        blocks.attr(label=f'Transformer Blocks (×{n_layer})', style='dashed', color='gray')

        # Show one representative block with details
        with blocks.subgraph(name='cluster_block') as block:
            block.attr(label=f'Block (×{n_layer})\nTotal per block: {format_params(params["block"])}',
                      style='rounded', color='#6c757d', bgcolor='#f8f9fa')

            # LayerNorm 1
            block.node('ln1', f'LayerNorm\n{format_params(params["ln"])}', fillcolor='#e2e3e5')

            # Attention subgraph
            with block.subgraph(name='cluster_attn') as attn:
                attn.attr(label=f'Attention: {format_params(params["attn"])}', style='rounded', color='#007bff', bgcolor='#cce5ff')
                attn.node('qkv', f'c_attn (QKV)\n{format_params(params["attn_qkv"])}\n({n_embd} → {3*n_embd})', fillcolor='#b8daff')
                attn.node('attn_op', f'Scaled Dot-Product\nAttention\n({n_head} heads)', fillcolor='#b8daff', shape='ellipse')
                attn.node('attn_proj', f'c_proj\n{format_params(params["attn_proj"])}\n({n_embd} → {n_embd})', fillcolor='#b8daff')

            # Residual 1
            block.node('res1', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse')

            # LayerNorm 2
            block.node('ln2', f'LayerNorm\n{format_params(params["ln"])}', fillcolor='#e2e3e5')

            # MLP subgraph
            with block.subgraph(name='cluster_mlp') as mlp:
                mlp.attr(label=f'MLP: {format_params(params["mlp"])}', style='rounded', color='#28a745', bgcolor='#d4edda')
                mlp.node('mlp_fc', f'c_fc\n{format_params(params["mlp_fc"])}\n({n_embd} → {4*n_embd})', fillcolor='#c3e6cb')
                mlp.node('gelu', 'GELU', fillcolor='#c3e6cb', shape='ellipse')
                mlp.node('mlp_proj', f'c_proj\n{format_params(params["mlp_proj"])}\n({4*n_embd} → {n_embd})', fillcolor='#c3e6cb')

            # Residual 2
            block.node('res2', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse')

    # Final LayerNorm
    dot.node('ln_f', f'Final LayerNorm\nln_f: {format_params(params["ln_f"])}', fillcolor='#e2e3e5')

    # LM Head
    dot.node('lm_head', f'LM Head\n(weight-tied with wte)\n({n_embd} → {config["vocab_size"]})', fillcolor='#f5c6cb')

    # Output
    dot.node('output', f'Output Logits\n(batch, seq_len, {config["vocab_size"]})', fillcolor='#e8f4f8')

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

    gqa_ratio = n_head // n_kv_head
    gqa_label = f"GQA {n_head}:{n_kv_head}" if gqa_ratio > 1 else "MHA"

    # Input
    dot.node('input', 'Input Tokens\n(batch, seq_len)', fillcolor='#e8f4f8')

    # Token Embedding
    dot.node('wte', f'Token Embedding\nwte: {format_params(params["wte"])}\n({params["padded_vocab_size"]} × {n_embd})', fillcolor='#d4edda')

    # RMSNorm (no params)
    dot.node('norm_emb', 'RMSNorm\n(no params)', fillcolor='#e2e3e5')

    # Save x0 for skip connection
    dot.node('save_x0', 'Save x₀\n(for skip connection)', fillcolor='#fff3cd', shape='ellipse')

    # Create a subgraph for transformer blocks
    with dot.subgraph(name='cluster_blocks') as blocks:
        blocks.attr(label=f'Transformer Blocks (×{n_layer})', style='dashed', color='gray')

        # Show one representative block with details
        with blocks.subgraph(name='cluster_block') as block:
            block.attr(label=f'Block (×{n_layer})\nTotal per block: {format_params(params["block"])}',
                      style='rounded', color='#6c757d', bgcolor='#f8f9fa')

            # Per-layer scaling
            block.node('scale_in', f'λ_resid[i] · x + λ_x0[i] · x₀\nresid_lambdas: {n_layer}\nx0_lambdas: {n_layer}',
                      fillcolor='#ffeeba', shape='box')

            # RMSNorm (no params)
            block.node('norm1', 'RMSNorm\n(no params)', fillcolor='#e2e3e5')

            # Attention subgraph
            with block.subgraph(name='cluster_attn') as attn:
                attn.attr(label=f'Attention ({gqa_label}): {format_params(params["attn"])}',
                         style='rounded', color='#007bff', bgcolor='#cce5ff')

                # Separate Q, K, V projections
                attn.node('q_proj', f'c_q (Query)\n{format_params(params["attn_q"])}\n({n_embd} → {n_head}×{head_dim})', fillcolor='#b8daff')
                attn.node('k_proj', f'c_k (Key)\n{format_params(params["attn_k"])}\n({n_embd} → {n_kv_head}×{head_dim})', fillcolor='#b8daff')
                attn.node('v_proj', f'c_v (Value)\n{format_params(params["attn_v"])}\n({n_embd} → {n_kv_head}×{head_dim})', fillcolor='#b8daff')

                attn.node('rotary', 'Rotary Embeddings\n+ QK Norm\n(no params)', fillcolor='#e7f1ff', shape='ellipse')
                attn.node('flash_attn', f'Flash Attention 3\n({n_head} Q heads, {n_kv_head} KV heads)', fillcolor='#b8daff', shape='ellipse')
                attn.node('attn_proj', f'c_proj\n{format_params(params["attn_proj"])}\n({n_embd} → {n_embd})', fillcolor='#b8daff')

            # Residual 1
            block.node('res1', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse')

            # RMSNorm 2 (no params)
            block.node('norm2', 'RMSNorm\n(no params)', fillcolor='#e2e3e5')

            # MLP subgraph
            with block.subgraph(name='cluster_mlp') as mlp:
                mlp.attr(label=f'MLP: {format_params(params["mlp"])}', style='rounded', color='#28a745', bgcolor='#d4edda')
                mlp.node('mlp_fc', f'c_fc\n{format_params(params["mlp_fc"])}\n({n_embd} → {4*n_embd})', fillcolor='#c3e6cb')
                mlp.node('relu_sq', 'ReLU²', fillcolor='#c3e6cb', shape='ellipse')
                mlp.node('mlp_proj', f'c_proj\n{format_params(params["mlp_proj"])}\n({4*n_embd} → {n_embd})', fillcolor='#c3e6cb')

            # Residual 2
            block.node('res2', 'Add (residual)', fillcolor='#fff3cd', shape='ellipse')

    # Final RMSNorm (no params)
    dot.node('norm_f', 'RMSNorm\n(no params)', fillcolor='#e2e3e5')

    # LM Head (NOT weight-tied)
    dot.node('lm_head', f'LM Head\n{format_params(params["lm_head"])}\n({n_embd} → {params["padded_vocab_size"]})\n(NOT weight-tied)', fillcolor='#f5c6cb')

    # Softcap
    dot.node('softcap', 'Logit Softcap\n15 · tanh(x/15)', fillcolor='#e8f4f8', shape='ellipse')

    # Output
    dot.node('output', f'Output Logits\n(batch, seq_len, {config["vocab_size"]})', fillcolor='#e8f4f8')

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

        # Generate and display diagram
        if model_type == "nanoGPT":
            diagram = create_nanogpt_diagram(params, config)
        else:
            diagram = create_nanochat_diagram(params, config)

        st.graphviz_chart(diagram, width='stretch')

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
