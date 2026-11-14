
import numpy as np

# Calculate scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):

    # Calculate attention scores
    scores = np.matmul(Q, K.T)

    # Scale scores using the key dimension
    d = K.shape[-1]
    scaled_scores = scores / np.sqrt(d)

    # Apply softmax to get attention weights
    exp_scores = np.exp(
        scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True)
    )
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    # Multiply attention weights with values
    output = np.matmul(attention_weights, V)

    return attention_weights, output


# Input matrices provided in the assignment
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Calculate attention
weights, output = scaled_dot_product_attention(Q, K, V)

# Display results rounded to three decimal places
print("Attention Weights:")
print(np.round(weights, 3))

print("\nFinal Output:")
print(np.round(output, 3))
