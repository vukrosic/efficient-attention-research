# The Kimi Linear Equation: A Step-by-Step Guide

This guide explains the single most important mathematical equation behind **Kimi Linear Attention (KDA)**. We will assume zero prior knowledge of linear attention math and break it down to its simplest components.

## 1. The Inputs: From Words to Vectors

Before we touch the memory, we have the word itself.
**Word**: "cats"

The model converts this word into two main vectors:
1.  **Key ($\mathbf{k}$)**: The "Address" (Where to write).
    *   *Meaning*: "This word is a **Subject**."
    *   *Vector*: `[1, 0, 0, 0]` (Points to Row 1)
2.  **Value ($\mathbf{v}$)**: The "Content" (What to write).
    *   *Meaning*: "It is **Plural** and **Animate**."
    *   *Vector*: `[1, 1, 0, 0]` (1=Plural, 1=Animate, 0=..., 0=...)

---

## 2. The Memory Matrix: Rows AND Columns

Now, let's look at the **Memory Matrix ($\mathbf{S}$)**. It's a grid where:
*   **ROWS** are defined by the **Key** (The "Trackers").
*   **COLUMNS** are defined by the **Value** (The "Attributes").

$$
\mathbf{S} = \begin{pmatrix}
  & \text{Col 1} & \text{Col 2} & \text{Col 3} & \text{Col 4} \\
  & \text{(Count)} & \text{(Type)} & \text{(Tense)} & \text{(Mood)} \\
\text{Row 1 (Subject)} \rightarrow & \mathbf{0.9} & \mathbf{0.9} & 0.0 & 0.0 \\
\text{Row 2 (Action)} \rightarrow & 0.0 & 0.0 & 0.0 & 0.0 \\
\text{Row 3 (Object)} \rightarrow & 0.0 & 0.0 & 0.0 & 0.0 \\
\text{Row 4 (Context)} \rightarrow & 0.0 & 0.0 & 0.0 & 0.0
\end{pmatrix}
$$

*   **Cell (1,1)**: "Subject" $\times$ "Count". Value `0.9` means "The Subject is Plural".
*   **Cell (1,2)**: "Subject" $\times$ "Type". Value `0.9` means "The Subject is Animate".

---

## 3. The Interaction: How Writing Happens

How do we get those numbers into the matrix? We use the **Outer Product** ($\mathbf{k} \mathbf{v}^\top$).

$$
\mathbf{k} \times \mathbf{v}^\top = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix} \times \begin{pmatrix} 1 & 1 & 0 & 0 \end{pmatrix} = \begin{pmatrix}
\mathbf{1} & \mathbf{1} & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

We then **ADD** this update matrix to the current Memory Matrix $\mathbf{S}$.
*   Since $\mathbf{k}$ was `[1, 0, 0, 0]`, it only "activated" the **First Row**.
*   Since $\mathbf{v}$ was `[1, 1, 0, 0]`, it wrote data into the **First and Second Columns** of that row.

### Reality Check: Does it only update ONE row?
**In this simple example? Yes.**
**In the real world? NO.**

Real AI vectors are **dense**, meaning they have numbers everywhere, not just zeros and ones.
*   **Real Key**: `[0.2, 0.9, -0.1, 0.4]`
*   **What happens**: The outer product $\mathbf{k}\mathbf{v}^\top$ creates a full grid of numbers.
*   **Result**: A single word updates **EVERY row** simultaneously, but with different strengths.
    *   It might update the "Subject" row strongly (0.9).
    *   It might update the "Sentiment" row weakly (0.2).
    *   It allows one word to influence many different concepts at once.

---

## 4. The Clean-Up Crew: Decay vs. Eraser

Writing is easy. **Forgetting is the hard part.**
Let's stick with our $4 \times 4$ matrix. Suppose Row 1 has "Subject: Cats" and Row 2 has "Action: Run" (from a previous sentence).

Now we see a new word: **"Sat"** (New Action).

### Part A: The Decay (The Gentle Fade)
We want to keep the Subject ("Cats") because it's still relevant. But maybe "Run" is getting old.
*   **The Alpha Vector ($\boldsymbol{\alpha}$)**: `[1.0, 0.5, 1.0, 1.0]`
    *   Row 1 (Subject): Multiplied by 1.0 $\rightarrow$ **Stays perfect.**
    *   Row 2 (Action): Multiplied by 0.5 $\rightarrow$ **Fades to 50%.**
    *   Other Rows: Stay perfect.

This is a "soft" forget. It lowers the volume of old memories but doesn't silence them.

### Part B: The Eraser (The Hard Delete)
Now we need to write "Sat" into Row 2. Even though "Run" is faded, it's still there! If we just write "Sat" on top, we get "Run + Sat", which is a mess.
We need to **wipe Row 2 clean** before writing.

*   **The Key ($\mathbf{k}$)**: Points to Row 2 (`[0, 1, 0, 0]`).
*   **The Eraser Matrix ($\mathbf{I} - \mathbf{k}\mathbf{k}^\top$)**:
    $$
    \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} - \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
    $$
*   **The Result**: When we multiply this by our Memory Matrix, **Row 2 becomes all zeros**. Everything else is untouched.

**Crucial Difference**:
*   **Decay**: Happens to *every* row, based on the learned $\alpha$. It's for long-term memory management.
*   **Eraser**: Happens *only* in the direction of the new Key. It's for immediate conflict resolution (clearing the slot you are about to use).

---

## The Equation

This is the formula that updates the model's "memory" at every single step (token) of a sequence:

$$
\mathbf{S}_t = \underbrace{(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)}_{\text{The Eraser}} \underbrace{\operatorname{Diag}(\boldsymbol{\alpha}_t)}_{\text{The Decay}} \mathbf{S}_{t-1} + \underbrace{\beta_t \mathbf{k}_t \mathbf{v}_t^\top}_{\text{The New Info}}
$$

---

## The Cast of Characters (Variables)

Before we explain *how* it works, let's define *what* everything is.

## The Cast of Characters (Variables)

Before we explain *how* it works, let's define *what* everything is with a concrete example.
**Scenario**: We are processing the sentence **"The cat sat"**. We are currently at the word **"sat"**.
**Assumption**: For simplicity, let's say our vectors have size 2 ($d=2$).

*   **$t$ (Time Step)**: The position in the sequence.
    *   *Concept*: The 3rd word, "sat".
    *   *Value*: `3`

*   **$\mathbf{k}_t$ (Key)**: The "category" or "address" of the current word.
    *   *Concept*: A vector representing **"Verb"**.
    *   *Value*: `[0.0, 1.0]` (0 for Noun channel, 1 for Verb channel)
    *   *Deep Dive*:
        *   **Where do they come from?**: These "channels" are **learned** by the neural network during training. The model isn't explicitly taught "Noun" vs "Verb", but it discovers that dedicating specific dimensions to specific grammatical or semantic roles helps it predict the next word.
        *   **How are they used?**: They act as **slots** in the memory matrix. A value of `1.0` in the second position acts like a switch saying: "Open the 2nd drawer of the memory cabinet to put this new info in."
        *   **For what?**: To organize the chaotic stream of words into structured knowledge. By sorting "sat" into the "Verb" slot, the model can later answer "What was the action?" by looking specifically at that slot.

*   **$\mathbf{v}_t$ (Value)**: The "content" or "meaning" of the current word.
    *   *Concept*: A vector representing **"Sitting"**.
    *   *Value*: `[0.5, -0.5]` (Abstract numerical representation of "sit")

*   **$\mathbf{S}_{t-1}$ (Old Memory)**: The memory state *before* seeing "sat".
    *   *Concept*: A matrix containing **"Subject = Cat"**.
    *   *Value*: `[[0.8, 0.2], [0.0, 0.0]]` (Top row stores "Cat" info in Noun channel)

*   **$\boldsymbol{\alpha}_t$ (Alpha - The Decay)**: A list of numbers (0 to 1) deciding what to keep.
    *   *Concept*: "Keep Noun channel (1.0), forget Verb channel (0.1)".
    *   *Value*: `[1.0, 0.1]`

*   **$\beta_t$ (Beta - The Strength)**: A number deciding how strongly to write the new info.
    *   *Concept*: "Write boldly".
    *   *Value*: `0.9`

*   **$\mathbf{I}$ (Identity Matrix)**: The "Do Nothing" matrix.
    *   *Value*: `[[1.0, 0.0], [0.0, 1.0]]`

---

## Crucial Concept: The Shape of Memory (Rows vs. Tokens)

**Q: Does each row in the memory matrix $\mathbf{S}$ correspond to a specific word (token) in the sentence?**

**A: NO!** This is the most important difference between Standard Attention and Linear Attention.

*   **Standard Attention (The Scroll)**:
    *   Keeps a growing list of every word ever seen.
    *   Row 1 = "The", Row 2 = "cat", Row 3 = "sat".
    *   *Problem*: If the book is 1 million words long, the list is 1 million rows long. It gets too big and slow.

*   **Kimi Linear / RNN (The Whiteboard)**:
    *   Compresses everything into a **fixed-size** grid (e.g., $128 \times 128$).
    *   **Rows correspond to FEATURES, not words.**
    *   *Example*:
        *   **Row 1**: Reserved for "Subject" information.
        *   **Row 2**: Reserved for "Action" information.
    *   When we see "The cat" (Subject), we write into **Row 1**.
    *   When we later see "The dog" (new Subject), we must **overwrite** or update **Row 1**.
    *   This is why the **Eraser** (Step 2) is so critical—we have to clear the "Subject" slot to make room for the new subject, because we don't have infinite rows to just add a new one.

---

## The Story of the Equation

Imagine the model's memory ($\mathbf{S}$) is a **whiteboard**. At every step, the model wants to add new information from the current word. But the whiteboard is finite, so it must manage space carefully.

The equation happens in three distinct steps, moving from right to left:

### Step 1: The Decay (Fading Away)
$$ \operatorname{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} $$

*   **What it does**: Before doing anything else, the model looks at its existing memory ($\mathbf{S}_{t-1}$) and decides what to keep and what to let fade.
*   **The Magic of Kimi**: In standard models, $\alpha$ is just a single number (e.g., multiply everything by 0.9). In **Kimi Linear**, $\alpha$ is a **vector** (a list of numbers).
*   **Why it matters**: This means the model can choose to forget *specific* topics (channels) while keeping others perfectly preserved. It's like having a whiteboard where you can wipe the left side but keep the right side untouched.

### Step 2: The Eraser (The Delta Rule)
$$ (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \dots $$

*   **The Problem**: If you just keep writing on a whiteboard without erasing, it becomes an unreadable mess.
*   **The Solution**: This term acts as a targeted eraser.
    *   $\mathbf{k}_t \mathbf{k}_t^\top$ creates a matrix that represents the "direction" of the current topic.
    *   Subtracting this from $\mathbf{I}$ (1) means: "Remove anything in the memory that conflicts with what I am about to write."
*   **Simple English**: "Make space for the new topic ($\mathbf{k}_t$) by clearing out any old, conflicting information about this specific topic."

### Step 3: The Writer (Adding New Info)
$$ + \beta_t \mathbf{k}_t \mathbf{v}_t^\top $$

*   **What it does**: Finally, we write the new information.
*   **$\mathbf{k}_t \mathbf{v}_t^\top$**: This creates a link between the current topic ($\mathbf{k}$) and its meaning ($\mathbf{v}$).
*   **$\beta_t$**: This controls how "bold" the writing is.
*   **Result**: The new association is stamped onto the whiteboard.

---

## Summary: The Full Loop

Putting it all together, here is what the equation says in plain English:

> "Take the **Old Memory** ($\mathbf{S}_{t-1}$).
> First, **fade out** ($\alpha$) the parts we don't need anymore.
> Second, **erase** ($\mathbf{I} - \dots$) any old facts that contradict what we are seeing right now.
> Finally, **write** ($+$) the new information ($\mathbf{k}\mathbf{v}^\top$) onto the clean space."

This process repeats for every single word, allowing the model to remember millions of tokens of context without the memory exploding in size.

---

## Reality Check: It's Not Just "Nouns" and "Verbs"

You asked: *"Is it just Noun or Verb? What determines this?"*

**The "Noun/Verb" labels were just a metaphor.** In a real AI model (like Kimi Linear), it is much more complex and powerful.

### 1. What are the possibilities?
In our example, we used a vector size of $d=2$, giving us 2 "slots".
*   **Real Models**: Use vector sizes like $d=2048$ or $d=4096$.
*   **Possibilities**: This means there are **thousands of different "channels"** or "slots" available to store information.
*   **Examples of what a channel might actually represent**:
    *   Channel 142: "We are currently inside a Python function definition."
    *   Channel 599: "The subject of the sentence was plural."
    *   Channel 1024: "We just saw an opening parenthesis `(` and are waiting for a closing one `)`."
    *   Channel 2000: "The tone of the text is angry."

### 2. What determines these categories?
**Nothing is hard-coded.** No human programmer sits down and types `if word == "cat": use_channel_1`.

*   **The Training Process**: The model starts with random garbage in these channels. It reads trillions of words.
*   **Gradient Descent**: Every time the model makes a mistake (e.g., predicting "is" instead of "are"), the math calculates *how* it should have used its memory better.
*   **Self-Organization**: Over time, the model *automatically discovers* that dedicating Channel 599 to "plurality" helps it make better predictions. It "learns" to route plural nouns to that specific slot.

### 3. Mixed States
Also, a word is rarely just "one thing".
*   **Input Key**: A real key vector $\mathbf{k}$ won't be `[0, 1]`.
*   **Real Key**: It might be `[0.1, 0.7, -0.2, 0.9, ...]`.
*   **Meaning**: "This word is mostly a Verb (0.7), slightly related to coding (0.1), and strongly implies a question (0.9)."
*   It updates **multiple rows** of the memory matrix at once, blending information across thousands of abstract concepts.

### 4. Summary: The Channel Model
**Yes, exactly!** You have the right mental model now.
Think of the memory matrix $\mathbf{S}$ as a bank of **independent trackers**.
*   **Row 1**: Tracks "Concept A" (e.g., Is the subject plural?).
*   **Row 2**: Tracks "Concept B" (e.g., Are we in a quote?).
*   **Row N**: Tracks "Concept N".

At every single step, the model looks at the word and asks: *"Which of these trackers need to be updated, and which should be left alone?"*

---

## Numerical Walkthrough: The Math in Action

Let's trace the numbers step-by-step to see exactly how the matrix changes.

### 0. The Setup
We have a $2 \times 2$ memory matrix ($d=2$).

*   **Initial Memory ($\mathbf{S}_{t-1}$)**:
    $$ \begin{pmatrix} 10 & 10 \\ 2 & 2 \end{pmatrix} $$
    *(Row 1 has strong "Noun" info, Row 2 has weak "Verb" info)*

*   **Input Key ($\mathbf{k}_t$)**: `[0, 1]`
    *(We are seeing a "Verb". We want to update Row 2.)*

*   **Input Value ($\mathbf{v}_t$)**: `[5, 5]`
    *(The new meaning is "5".)*

*   **Decay ($\boldsymbol{\alpha}_t$)**: `[1.0, 0.5]`
    *(Keep Row 1 perfectly, decay Row 2 by half.)*

*   **Strength ($\beta_t$)**: `1.0`
    *(Full overwrite strength.)*

---

### Step 1: The Decay
First, we apply the decay vector $\boldsymbol{\alpha}$ to the rows of the memory.
$$ \operatorname{Diag}(\boldsymbol{\alpha}) \mathbf{S}_{t-1} = \begin{pmatrix} 1.0 & 0 \\ 0 & 0.5 \end{pmatrix} \begin{pmatrix} 10 & 10 \\ 2 & 2 \end{pmatrix} $$

**Result ($\mathbf{S}_{decayed}$)**:
$$ \begin{pmatrix} 10 & 10 \\ 1 & 1 \end{pmatrix} $$
*Observation: Row 1 stayed the same. Row 2 shrank from 2 to 1.*

---

### Step 2: The Eraser Construction
Now we build the eraser matrix using the Key $\mathbf{k}=[0, 1]$.
$$ \mathbf{k}\mathbf{k}^\top = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \begin{pmatrix} 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} $$

Then subtract from Identity:
$$ \mathbf{I} - \beta \mathbf{k}\mathbf{k}^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} - 1.0 \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} $$
*This is our Eraser Matrix. Notice the bottom-right is 0. It will wipe Row 2.*

---

### Step 3: Applying the Eraser
Multiply the Eraser Matrix by our Decayed Memory.
$$ \underbrace{\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}}_{\text{Eraser}} \times \underbrace{\begin{pmatrix} 10 & 10 \\ 1 & 1 \end{pmatrix}}_{\text{Decayed Memory}} $$

**Result ($\mathbf{S}_{erased}$)**:
$$ \begin{pmatrix} 10 & 10 \\ 0 & 0 \end{pmatrix} $$
*Observation: Row 1 is safe. Row 2 has been completely wiped to 0 to make room.*

---

### Step 4: The Writer (New Info)
Calculate the new information term using Key and Value.
$$ \beta \mathbf{k}\mathbf{v}^\top = 1.0 \begin{pmatrix} 0 \\ 1 \end{pmatrix} \begin{pmatrix} 5 & 5 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 5 & 5 \end{pmatrix} $$

---

### Step 5: Final Addition
Add the New Info to the Erased Memory.
$$ \mathbf{S}_t = \underbrace{\begin{pmatrix} 10 & 10 \\ 0 & 0 \end{pmatrix}}_{\text{Erased}} + \underbrace{\begin{pmatrix} 0 & 0 \\ 5 & 5 \end{pmatrix}}_{\text{New}} $$

**Final Result ($\mathbf{S}_t$)**:
$$ \begin{pmatrix} 10 & 10 \\ 5 & 5 \end{pmatrix} $$

### Conclusion
*   **Row 1 (Noun)**: Started at `[10, 10]`. Ended at `[10, 10]`. **Preserved.**
*   **Row 2 (Verb)**: Started at `[2, 2]`. Decayed to `[1, 1]`. Wiped to `[0, 0]`. Replaced by `[5, 5]`. **Updated.**

This is how Kimi Linear manages its finite memory!
