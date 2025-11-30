Let's say we have some text here.

That text is processed by the linear attention and saved into memory.

Memory matrix:
$$
\mathbf{S} = \begin{pmatrix}
  & \text{Col 1} & \text{Col 2} & \text{Col 3} \\
  & \text{(Plural?)} & \text{(Past?)} & \text{(Alive?)} \\
\text{Row 1 (Subject)} \rightarrow & \mathbf{1.0} & 0.0 & \mathbf{1.0} \\
\text{Row 2 (Action)} \rightarrow & 0.0 & \mathbf{1.0} & 0.0
\end{pmatrix}
$$

Each of these rows represents a different concept we are tracking (The "Keys").
Each of the columns represents a specific attribute or detail about that concept (The "Values").

For example:
*   **Row 1 (Subject)**: Tracks information about the subject of the sentence.
    *   `[1.0, 0.0, 1.0]` -> "The subject is Plural (Col 1) and Alive (Col 3)." -> "Cats"
*   **Row 2 (Action)**: Tracks information about the verb/action.
    *   `[0.0, 1.0, 0.0]` -> "The action is Past Tense (Col 2)." -> "Sat"

Next part is "Cat sat and dog ran."


