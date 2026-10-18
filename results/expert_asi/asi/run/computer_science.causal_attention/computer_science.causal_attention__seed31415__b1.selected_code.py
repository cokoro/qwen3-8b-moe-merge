import numpy as np

def sequence_operator_forward(x, segment_ids, params, meta, return_cache=False):
    # Reshape and apply RoPE
    N, T, C = x.shape
    x = x.reshape(N, T, meta['parts_a'], meta['part_width'])
    q = x @ params['matrix_2'] + params['vector_2']
    k = x @ params['matrix_3'] + params['vector_4']
    v = x @ params['matrix_1'] + params['vector_3']
    v_gate = sigmoid(x @ params['matrix_4'] + params['vector_6'])
    v = v * v_gate
    
    # Apply RoPE to q and k
    rotary_dim = meta['table_width']
    for i in range(rotary_dim // 2):
        cos = params['table_1'][:, :, i]
        sin = params['table_0'][:, :, i]
        q[:, :, :, 2*i] = q[:, :, :, 2*i] * cos - q[:, :, :, 2*i+1] * sin
        q[:, :, :, 2*i+1] = q[:, :, :, 2*i] * sin + q[:, :, :, 2*i+1] * cos
        k[:, :, :, 2*i] = k[:, :, :, 2*i] * cos - k[:, :, :, 2*i+1] * sin
        k[:, :, :, 2*i+1] = k[:, :, :, 2*i] * sin + k[:, :, :, 2*i+1] * cos
    
    # Compute attention scores
    scores = np.einsum('nhtd,nhkd->nhk', q, k) / np.sqrt(meta['part_width'])
    scores += params['vector_5'] * np.arange(meta['parts_a'])
    
    # Apply attention mask
    mask = (np.arange(T)[:, None, None] <= np.arange(T)[:, :, None] + meta['span'] - 1) & (np.arange(T)[:, None, None] >= np.arange(T)[:, :, None] - meta['span'] + 1)
    mask &= segment_ids[:, :, None] == segment_ids[:, None, :]
    scores = np.where(mask, scores, -np.inf)
    
    # Apply softmax and compute context
    attn = softmax(scores, axis=-1)
    context = attn @ v
    
    # Final output
    output = context @ params['matrix_0'] + params['vector_0']
    return output if not return_cache else (output, {'attn': attn, 'context': context})

def sequence_operator_backward(sensitivity, cache, params, meta):
    # Backward pass
    attn, context = cache['attn'], cache['context']
    grad_output = sensitivity @ params['matrix_0'].T
    grad_matrix_0 = sensitivity.T @ context
    
    grad_context = sensitivity.T @ params['matrix_0']
    grad_attn = grad_context @ params['matrix_1'].T
    
    grad_v = grad_attn @ params['matrix_1']
    grad_v_gate = grad_v * (1 - context) * (1 - context)
    grad_matrix_1 = grad_v.T @ context
    
    grad_matrix_2 = grad_attn.T @ q
    
    # ... (complete the backward pass)
    
    return {'x': grad_x, 'matrix_0': grad_matrix_0, ...}

def sequence_operator_hvp(x, segment_ids, params, meta, sensitivity, direction):
    # Compute Hessian-vector product
    # ... (compute the HVP for each parameter)
    return {'x': hvp_x, 'matrix_0': hvp_matrix_0, ...}