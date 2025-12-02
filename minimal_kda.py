"""
Minimal Self-Contained KDA (Kimi Delta Attention) Implementation for Learning

This is a simplified, educational implementation of KDA that demonstrates the core concepts
without the complexity of the full production version. It includes:
1. The core KDA attention mechanism
2. A simple language model using KDA
3. Training loop with synthetic data

Key Concepts:
- Linear Attention with Delta Rule
- Learnable decay (A_log) and time modulation (dt_bias)
- Gating mechanism for enhanced expressiveness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# CORE KDA ATTENTION (Simplified Fused Recurrent Implementation)
# ============================================================================

def simple_kda_attention(q, k, v, g, beta, A_log, dt_bias):
    """
    Simplified KDA attention mechanism using recurrent computation.
    
    Args:
        q: queries [batch, seq_len, num_heads, head_dim]
        k: keys [batch, seq_len, num_heads, head_dim]
        v: values [batch, seq_len, num_heads, head_dim]
        g: forget gates [batch, seq_len, num_heads, head_dim]
        beta: mixing coefficient [batch, seq_len, num_heads]
        A_log: log decay parameter [num_heads]
        dt_bias: time modulation bias [num_heads * head_dim]
        
    Returns:
        o: output [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # L2 normalize queries and keys (important for stability)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    
    # Compute gate values with decay and time modulation
    # This is the "delta" part - it modulates how much we forget/remember
    dt_bias = rearrange(dt_bias, '(h d) -> h d', h=num_heads)
    A = torch.exp(-torch.exp(A_log))  # Decay factor per head
    
    # Apply gating: g controls what to remember/forget
    g = g - dt_bias.unsqueeze(0).unsqueeze(0)  # Apply time modulation
    g = torch.sigmoid(g)  # Gate values in [0, 1]
    
    # Initialize the memory state (key-value matrix)
    # This is the "linear" part - we maintain KV^T matrix
    state = torch.zeros(batch_size, num_heads, head_dim, head_dim, 
                       device=q.device, dtype=q.dtype)
    
    outputs = []
    
    # Recurrent computation over sequence
    for t in range(seq_len):
        q_t = q[:, t]  # [batch, num_heads, head_dim]
        k_t = k[:, t]  # [batch, num_heads, head_dim]
        v_t = v[:, t]  # [batch, num_heads, head_dim]
        g_t = g[:, t]  # [batch, num_heads, head_dim]
        beta_t = beta[:, t]  # [batch, num_heads]
        
        # Read from memory: o_t = q_t @ state
        o_t = torch.einsum('bhd,bhde->bhe', q_t, state)
        
        # Apply gating to output
        o_t = o_t * g_t
        
        # Update memory with delta rule:
        # state_new = decay * state_old + beta * k_t @ v_t^T
        # Decay the old state
        state = A.view(1, num_heads, 1, 1) * state
        
        # Add new key-value association with beta weighting
        kv_outer = torch.einsum('bhd,bhe->bhde', k_t, v_t)
        state = state + beta_t.view(batch_size, num_heads, 1, 1) * kv_outer
        
        outputs.append(o_t)
    
    # Stack outputs
    o = torch.stack(outputs, dim=1)  # [batch, seq_len, num_heads, head_dim]
    
    return o


# ============================================================================
# KDA ATTENTION LAYER
# ============================================================================

class MinimalKDAAttention(nn.Module):
    """Minimal KDA attention layer with all essential components."""
    
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = num_heads * head_dim
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        
        # Gate projection (for forget gate g)
        self.f_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        
        # Beta projection (mixing coefficient)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        
        # Learnable decay and time modulation parameters
        self.A_log = nn.Parameter(torch.log(torch.ones(num_heads) * 8.0))
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim))
        
        # Output projection with gating
        self.g_proj = nn.Linear(hidden_size, self.key_dim, bias=True)
        self.o_proj = nn.Linear(self.key_dim, hidden_size, bias=False)
        
        # RMSNorm for output
        self.o_norm = nn.RMSNorm(head_dim, eps=1e-5)
        
    def forward(self, x):
        """
        Args:
            x: input [batch, seq_len, hidden_size]
        Returns:
            output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = F.silu(self.q_proj(x))  # Activation for stability
        k = F.silu(self.k_proj(x))
        v = F.silu(self.v_proj(x))
        
        # Get gate and beta
        g = self.f_proj(x)
        beta = torch.sigmoid(self.b_proj(x))
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        g = rearrange(g, 'b s (h d) -> b s h d', h=self.num_heads)
        
        # Core KDA attention
        o = simple_kda_attention(q, k, v, g, beta, self.A_log, self.dt_bias)
        
        # Output gating and normalization
        gate = torch.sigmoid(self.g_proj(x))
        gate = rearrange(gate, 'b s (h d) -> b s h d', h=self.num_heads)
        
        # Apply gating and norm per head
        o = self.o_norm(o) * gate
        
        # Flatten and project
        o = rearrange(o, 'b s h d -> b s (h d)')
        o = self.o_proj(o)
        
        return o


# ============================================================================
# SIMPLE MLP
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple feed-forward network with SwiGLU activation."""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================================
# KDA TRANSFORMER BLOCK
# ============================================================================

class KDABlock(nn.Module):
    """Single transformer block with KDA attention."""
    
    def __init__(self, hidden_size, num_heads, head_dim, intermediate_size):
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.attn = MinimalKDAAttention(hidden_size, num_heads, head_dim)
        self.mlp_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.mlp = SimpleMLP(hidden_size, intermediate_size)
        
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.attn_norm(x))
        # MLP with residual connection
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ============================================================================
# MINIMAL KDA LANGUAGE MODEL
# ============================================================================

class MinimalKDALanguageModel(nn.Module):
    """Minimal language model using KDA attention."""
    
    def __init__(
        self,
        vocab_size=256,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        head_dim=64,
        intermediate_size=512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Token embedding
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # KDA blocks
        self.layers = nn.ModuleList([
            KDABlock(hidden_size, num_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        
        # Final norm and output
        self.norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embeddings.weight
        
    def forward(self, input_ids, labels=None):
        """
        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] (optional, for training)
        Returns:
            logits or (loss, logits)
        """
        # Embed tokens
        x = self.embeddings(input_ids)
        
        # Pass through KDA blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            return loss, logits
        
        return logits


# ============================================================================
# TRAINING EXAMPLE
# ============================================================================

def create_synthetic_data(batch_size, seq_len, vocab_size):
    """Create simple synthetic data for demonstration."""
    # Create sequences with simple patterns (e.g., repeating subsequences)
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    return data


def train_minimal_kda():
    """Simple training loop to demonstrate the model works."""
    
    print("=" * 60)
    print("Minimal KDA Language Model - Training Demo")
    print("=" * 60)
    
    # Hyperparameters
    vocab_size = 128
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    head_dim = 32
    intermediate_size = 256
    batch_size = 4
    seq_len = 64
    num_steps = 100
    learning_rate = 1e-3
    
    # Create model
    model = MinimalKDALanguageModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {num_params:,}")
    print(f"Configuration:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Number of heads: {num_heads}")
    print(f"  - Head dimension: {head_dim}")
    print(f"  - Intermediate size: {intermediate_size}")
    print("\n" + "=" * 60)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for step in range(num_steps):
        # Generate batch of synthetic data
        input_ids = create_synthetic_data(batch_size, seq_len, vocab_size)
        
        # Forward pass
        loss, logits = model(input_ids, labels=input_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{num_steps} | Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Test inference
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, vocab_size, (1, 10))
        print(f"\nTest inference:")
        print(f"Input shape: {test_input.shape}")
        logits = model(test_input)
        print(f"Output shape: {logits.shape}")
        print(f"Next token prediction (first 5): {logits[0, -1].topk(5).indices.tolist()}")
    
    return model


if __name__ == "__main__":
    # Run the training demo
    model = train_minimal_kda()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. **Linear Attention**: KDA uses a recurrent state (key-value matrix) instead of 
   full attention matrix, making it O(d²) per step instead of O(n²).

2. **Delta Rule**: The memory state is updated with a delta rule:
   state_new = decay * state_old + beta * k @ v^T
   This allows controlled forgetting and learning.

3. **Learnable Dynamics**: 
   - A_log: Controls decay rate per head
   - dt_bias: Time-dependent modulation
   - beta: Per-token mixing coefficient

4. **Gating**: Multiple gating mechanisms (f_proj, g_proj) control information flow
   and enhance expressiveness.

5. **Efficiency**: The recurrent formulation makes KDA much faster than standard
   attention for long sequences while maintaining strong performance.
    """)
