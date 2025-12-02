# Kimi Deltanet Step 2: The Matrix Memory (Outer Product)

In Step 1, we learned how to forget a single number or a simple vector.
Now, we need to understand the **Shape of Memory**.

In standard RNNs (like LSTM/GRU), the memory is a vector.
In **Kimi Deltanet (and Linear Attention)**, the memory is a **Matrix**.

Why? And how do we build it?

---

## 1. The Problem with Vector Memory

Imagine you want to remember: *"The cat sat."*
*   **Key (Address)**: "Subject"
*   **Value (Content)**: "Cat"

If your memory is just a vector, you have to squash "Subject" and "Cat" into the same list of numbers. It gets messy.

We want a **Grid (Matrix)**:
*   **Rows**: Represent the Keys (Addresses).
*   **Columns**: Represent the Values (Content).

This way, we can say: *"In the 'Subject' row, write 'Cat' into the columns."*

---

## 2. Creating the Grid: The Outer Product

How do we turn two vectors (Key and Value) into a Matrix?
We use the **Outer Product**.

$$ \text{Update Matrix} = K^T \otimes V $$

Let's see it in action with simple numbers.

*   **Key ($K$)**: `[0, 1, 0]` (Points to the 2nd Row)
*   **Value ($V$)**: `[5, 5, 5]` (The data we want to write)

$$
\begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} \times \begin{pmatrix} 5 & 5 & 5 \end{pmatrix} = 
\begin{pmatrix} 
0 & 0 & 0 \\
\mathbf{5} & \mathbf{5} & \mathbf{5} \\
0 & 0 & 0 
\end{pmatrix}
$$

See what happened? The data `[5, 5, 5]` was written **only** into the 2nd row!

### Code Example

```python
import torch

# Define Key and Value vectors
k = torch.tensor([0.0, 1.0, 0.0]) # The "Address"
v = torch.tensor([5.0, 5.0, 5.0]) # The "Content"

# Calculate Outer Product
# Shape: [3] x [3] -> [3, 3]
update_matrix = torch.outer(k, v)

print("Update Matrix:")
print(update_matrix)
# Expected:
# [[0., 0., 0.],
#  [5., 5., 5.],
#  [0., 0., 0.]]
```

---

## 3. The Matrix Update Loop

Now we can upgrade our recurrent loop.
Instead of `memory = memory * decay + input`, it becomes:

$$ S_t = S_{t-1} \cdot \text{decay} + (K_t \otimes V_t) $$

*   $S_t$: The Memory Matrix at time $t$ (Size $D \times D$).
*   $K_t \otimes V_t$: The new information grid we just calculated.

### The Code

```python
def run_matrix_recurrence(keys, values, decay):
    """
    keys: [Seq_Len, Dim]
    values: [Seq_Len, Dim]
    decay: float (0.0 to 1.0)
    """
    seq_len, dim = keys.shape
    
    # Initialize Memory Matrix S (Dim x Dim)
    S = torch.zeros(dim, dim)
    
    outputs = []
    
    for t in range(seq_len):
        k_t = keys[t]
        v_t = values[t]
        
        # 1. Forget (Decay the old matrix)
        S = S * decay
        
        # 2. Remember (Add the new outer product)
        update = torch.outer(k_t, v_t)
        S = S + update
        
        # 3. Store the "strength" of memory for visualization
        outputs.append(S.norm().item())
        
    return outputs

# Test Data
seq_len = 5
dim = 4
keys = torch.randn(seq_len, dim)
values = torch.randn(seq_len, dim)
decay = 0.9

# Run it
norms = run_matrix_recurrence(keys, values, decay)
print("Memory Norms over time:", norms)
```

---

## Summary

You have just implemented the **Linear Attention State Update**!

1.  **Vector Memory** is too simple for complex relationships.
2.  **Matrix Memory** allows us to map specific Keys to specific Values.
3.  **Outer Product ($K \otimes V$)** is the tool that creates this matrix from input vectors.

### Next Step
Now that we have a matrix memory, we can make it powerful. Real models use **Multiple Heads** (many matrices) and a **Learned Decay** (not just a fixed 0.9). That will be Step 3!
