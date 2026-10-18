```python
import numpy as np
import os
import json

def sequence_operator_forward(x, segment_ids, params, meta, return_cache=False):
    # Resolve roles based on meta
    # This is a simplified placeholder; actual role resolution would be more complex
    # For demonstration, assuming roles are already correctly mapped
    matrix_0 = params['matrix_0']
    matrix_1 = params['matrix_1']
    matrix_2 = params['matrix_2']
    matrix_3 = params['matrix_3']
    matrix_4 = params['matrix_4']
    vector_0 = params['vector_0']
    vector_1 = params['vector_1']
    vector_2 = params['vector_2']
    vector_3 = params['vector_3']
    vector_4 = params['vector_4']
    vector_5 = params['vector_5']
    vector_6 = params['vector_6']
    table_0 = params['table_0']
    table_1 = params['table_1']
    
    # RMSNorm
    epsilon = meta['epsilon']
    mean_x = np.mean(x * x, axis=-1, keepdims=True)
    z = x / np.sqrt(mean_x + epsilon) * vector_1
    
    # Query, Key, Value projections
    q = z @ matrix_2 + vector_2
    k = z @ matrix_3 + vector_4
    v_linear = z @ matrix_1 + vector_3
    v_gate = 1 / (1 + np.exp(-(z @ matrix_4 + vector_6)))
    v = v_linear * v_gate
    
    # Reshape
    N, T, parts_a, part_width = np.shape(q)
    parts_b = meta['parts_b']
    q = q.reshape(N, T, parts_a, part_width)
    k = k.reshape(N, T, parts_b, part_width)
    v = v.reshape(N, T, parts_b, part_width)
    
    # Apply RoPE
    table_width = meta['table_width']
    q_rope = apply_rope(q, table_1, table_0, table_width)
    k_rope = apply_rope(k, table_1, table_0, table_width)
    
    # Repeat key/value heads
    repeat_factor = meta['repeat_factor']
    k = np.repeat(k, repeat_factor, axis=2)
    v = np.repeat(v, repeat_factor, axis=2)
    
    # Compute scores
    scores = np.zeros((N, T, parts_a, T))
    for i in range(T):
        for j in range(T):
            if j <= i and j >= i - meta['span'] + 1 and segment_ids[..., j] == segment_ids[..., i]:
                for h in range(parts_a):
                    q_head = q[..., h, :]
                    k_head = k[..., h, :]
                    score = np.dot(q_head, k_head) / np.sqrt(part_width)
                    score += vector_5[h] * (j - i)
                    scores[..., h, j] = score
    
    # Apply softmax
    scores = np.where(np.isnan(scores), -np.inf, scores)
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    scores /= np.sum(scores, axis=-1, keepdims=True)
    
    # Mix values
    context = np.einsum('nTbh, nTbhk -> nbk', scores, v)
    context = context.reshape(N, T, part_width)
    
    # Final output projection
    output = context @ matrix_0 + vector_0
    
    if return_cache:
        cache = {
            'z': z,
            'q': q,
            'k': k,
            'v': v,
            'scores': scores,
            'context': context,
            'part_width': part_width,
            'repeat_factor': repeat_factor,
            'table_width': table_width
        }
        return output, cache
    return output

def apply_rope(tensor, cos_table, sin_table, table_width):
    # Apply RoPE to the first table_width channels
    # Pair adjacent even/odd indices
    tensor = tensor.copy()
    for r in range(table_width // 2):
        a = tensor[..., 2*r]
        b = tensor[..., 2*r + 1]
        c = cos_table[..., r]
        s = sin_table[..., r]
        tensor[..., 2*r] = a * c - b * s
        tensor[..., 2*r + 1] = a * s + b * c
    return tensor

def sequence_operator_backward(sensitivity, cache, params, meta):
    # Compute gradients
    z = cache['z']
    q = cache['q']
    k = cache['k']
    v = cache['v']
    scores = cache['scores']
    context = cache['context']
    part_width = cache['part_width']
    repeat_factor = cache['repeat_factor']
    table_width = cache['table_width']
    
    # Gradients for output projection
    grad_matrix_0 = np.einsum('nTbk,nTbk -> bk', context, sensitivity)
    grad_vector_0 = np.sum(sensitivity, axis=(0, 1))
    
    # Gradients for value projection
    grad_v_linear = np.einsum('nTbhk,nTbk -> bhk', scores, sensitivity)
    grad_v_gate = np.einsum('nTbhk,nTbk -> bhk', v, sensitivity)
    grad_v_gate = grad_v_gate * v_linear
    
    # Gradients for value gate projection
    grad_matrix_4 = np.einsum('nTbh,nTbk -> bhk', z, grad_v_gate)
    grad_vector_6 = np.sum(grad_v_gate, axis=(0, 1))
    
    # Gradients for value projection
    grad_matrix_1 = np.einsum('nTbhk,nTbk -> bhk', z, grad_v_linear)
    grad_vector_3 = np.sum(grad_v_linear, axis=(0, 1))
    
    # Gradients for RMSNorm
    grad_z = np.einsum('nTbk,nTbk -> nTbk', z, sensitivity)
    grad_vector_1 = np.sum(grad_z * z, axis=(0, 1))
    
    # Gradients for query projection
    grad_matrix_2 = np.einsum('nTbhk,nTbk -> bhk', q, sensitivity)
    grad_vector_2 = np.sum(sensitivity, axis=(0, 1))
    
    # Gradients for key projection
    grad_matrix_3 = np.einsum('nTbhk,nTbk -> bhk', k, sensitivity)
    grad_vector_4 = np.sum(sensitivity, axis=(0, 1))
    
    # Gradients for RoPE tables
    grad_table_0 = np.zeros_like(table_0)
    grad_table_1 = np.zeros_like(table_1)
    
    # Gradients for query and key
    grad_q = np.einsum('nTbhk,nTbk -> nTbhk', scores, v)
    grad_k = np.einsum('nTbhk,nTbk -> nTbhk', scores, v)
    
    # Gradients for x
    grad_x = np.zeros_like(x)
    
    # Gradients for segment_ids (not implemented)
    grad_segment_ids = np.zeros_like(segment_ids)
    
    # Gradients for RoPE tables
    for r in range(table_width // 2):
        a = q[..., 2*r]
        b = q[..., 2*r + 1]
        c = table_1[..., r]
        s = table_0[..., r]
        grad_table_1[..., r] = (a * grad_table_1[..., r] + b * grad_table_1[..., r] + a * grad_table_0[..., r] - b * grad_table_0[..., r])
        grad_table_0[..., r] = (a * grad_table_0[..., r] + b * grad_table_0[..., r] + a * grad_table_1[..., r] - b * grad_table_1[..., r])
    
    # Gradients for x
    grad_x = grad_z + grad_q + grad_k + grad_v_linear + grad_v_gate
    
    # Return gradients
    grads = {
        'x': grad_x,
        'matrix_0': grad_matrix_0,
        'matrix_1': grad_matrix_1,
        'matrix_2': grad_matrix_2,
        'matrix_3': grad_matrix_3,
        'matrix_4': grad_matrix_4,
        'vector_0': grad_vector_0,
        'vector_1': grad_vector_1,
        'vector_2': grad_vector_2,
        'vector_3': grad_vector_3,
        'vector_4': grad_vector_4,
        'vector_5': np.sum(scores, axis=(0, 1, 2)) * np.sum(sensitivity, axis=(0, 1, 2)),
        'vector_6': grad_vector_6,
        'table_0': grad_table_0,
        'table_1': grad_table_1
    }
    return grads

def sequence_operator_hvp(x, segment_ids, params, meta, sensitivity, direction):
    # Compute Hessian-vector product
    # This is a simplified placeholder; actual implementation would be more complex
    # Using the backward pass to compute gradients
    cache = sequence_operator_forward(x, segment_ids, params, meta, return_cache=True)[1]
    grads = sequence_operator_backward(sensitivity, cache, params, meta)
    
    # Compute HVP using the direction
    hvp = {}
    for key in grads:
        hvp[key] = np.einsum('...,... -> ...', grads[key], direction[key])
    
    return hvp

def save_results(output, input_gradient, input_hvp):
    os.makedirs('results', exist_ok=True)
    np.save('results/output.npy', output)
    np.save('results/input_gradient.npy', input_gradient)
    np.save('results/input_hvp.npy', input_hvp)

# Example usage (not part of the actual implementation)
# x = np.load('data/x.npy')
# segment_ids = np.load('data/segment_ids.npy')
# params = {k: np.load(f'data/{k}.npy') for k in ['matrix_0', 'matrix_1', 'matrix_2', 'matrix_3', 'matrix_4', 'vector_0', 'vector_1', 'vector_2', 'vector_3', 'vector_4', 'vector_5', 'vector_6', 'table_0', 'table_1']}
# meta = json.load(open('data/meta.json'))
# sensitivity = np.load('data/sensitivity.npy')
# 
# output, cache = sequence_operator_forward(x, segment_ids, params, meta, return_cache=True)
# grads = sequence_operator_backward(sensitivity, cache, params, meta)
# 
# hvp = sequence_operator_hvp(x, segment_ids, params, meta, sensitivity, grads)
# 
# save_results(output, grads['x'], hvp)
```