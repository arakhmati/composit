import composit as cnp
import composit.nn


def softmax(input_tensor, *, axis):
    exp_input_tensor = cnp.exp(input_tensor - cnp.max(input_tensor, axis=axis, keepdims=True))
    return exp_input_tensor / cnp.sum(exp_input_tensor, axis=axis, keepdims=True)


def layer_norm(input_tensor, weight, bias, *, epsilon=1e-5):
    """
    mean = cnp.mean(input_tensor, axis=-1, keepdims=True)
    var = cnp.sqrt(cnp.var(input_tensor, axis=-1, keepdims=True) + epsilon)
    output = (input_tensor - mean) / var
    return output * weight + bias
    """
    mean = cnp.mean(input_tensor, axis=-1, keepdims=True)
    input_tensor_minus_mean = input_tensor - mean
    var = cnp.mean(cnp.square(input_tensor_minus_mean), axis=-1, keepdims=True)
    output = input_tensor_minus_mean / cnp.sqrt(var + epsilon)
    output *= weight
    output += bias
    return output


def batch_norm(input_tensor, mean, var, weight, bias, *, epsilon=1e-5, data_format):
    if data_format == "NCHW":
        raise NotImplementedError
    output = input_tensor - mean
    output = output / cnp.sqrt(var + epsilon)
    output *= weight
    output += bias
    return output


def multi_head_attention(
    hidden_states,
    attention_mask,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    *,
    head_size,
):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    num_heads = hidden_size // head_size

    query = hidden_states @ query_weight
    query = query + query_bias
    query = cnp.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = cnp.transpose(query, (0, 2, 1, 3))

    key = hidden_states @ key_weight
    key = key + key_bias
    key = cnp.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = cnp.transpose(key, (0, 2, 3, 1))

    value = hidden_states @ value_weight
    value = value + value_bias
    value = cnp.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = cnp.transpose(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = softmax(attention_scores, axis=-1)

    context_layer = attention_probs @ value

    context_layer = cnp.transpose(context_layer, (0, 2, 1, 3))
    context_layer = cnp.reshape(context_layer, (batch_size, sequence_size, hidden_size))

    self_output = context_layer @ output_weight
    self_output = self_output + output_bias

    return self_output


def feedforward(hidden_states, intermediate_weight, intermediate_bias, output_weight, output_bias):
    hidden_states = hidden_states @ intermediate_weight
    hidden_states = hidden_states + intermediate_bias
    hidden_states = cnp.nn.gelu(hidden_states)
    hidden_states = hidden_states @ output_weight
    hidden_states = hidden_states + output_bias
    return hidden_states


def resnet_module(
    input_tensor,
    right_branch_conv_0_weight,
    right_branch_bn_0_running_mean,
    right_branch_bn_0_running_var,
    right_branch_bn_0_weight,
    right_branch_bn_0_bias,
    right_branch_conv_1_weight,
    right_branch_bn_1_running_mean,
    right_branch_bn_1_running_var,
    right_branch_bn_1_weight,
    right_branch_bn_1_bias,
    right_branch_conv_2_weight,
    right_branch_bn_2_running_mean,
    right_branch_bn_2_running_var,
    right_branch_bn_2_weight,
    right_branch_bn_2_bias,
    left_branch_conv_weight,
    left_branch_bn_running_mean,
    left_branch_bn_running_var,
    left_branch_bn_weight,
    left_branch_bn_bias,
    data_format,
    module_strides=(1, 1),
):
    left_branch = input_tensor
    right_branch = input_tensor

    if left_branch_conv_weight is not None:
        left_branch = cnp.nn.convolution(
            left_branch, left_branch_conv_weight, strides=module_strides, data_format=data_format
        )
        left_branch = batch_norm(
            left_branch,
            left_branch_bn_running_mean,
            left_branch_bn_running_var,
            left_branch_bn_weight,
            left_branch_bn_bias,
            data_format=data_format,
        )

    right_branch = cnp.nn.convolution(right_branch, right_branch_conv_0_weight, strides=(1, 1), data_format=data_format)
    right_branch = batch_norm(
        right_branch,
        right_branch_bn_0_running_mean,
        right_branch_bn_0_running_var,
        right_branch_bn_0_weight,
        right_branch_bn_0_bias,
        data_format=data_format,
    )
    right_branch = cnp.nn.relu(right_branch)

    right_branch = cnp.nn.convolution(
        right_branch, right_branch_conv_1_weight, strides=module_strides, padding=(1, 1), data_format=data_format
    )
    right_branch = batch_norm(
        right_branch,
        right_branch_bn_1_running_mean,
        right_branch_bn_1_running_var,
        right_branch_bn_1_weight,
        right_branch_bn_1_bias,
        data_format=data_format,
    )
    right_branch = cnp.nn.relu(right_branch)

    right_branch = cnp.nn.convolution(right_branch, right_branch_conv_2_weight, strides=(1, 1), data_format=data_format)
    right_branch = batch_norm(
        right_branch,
        right_branch_bn_2_running_mean,
        right_branch_bn_2_running_var,
        right_branch_bn_2_weight,
        right_branch_bn_2_bias,
        data_format=data_format,
    )

    output = cnp.nn.relu(left_branch + right_branch)
    return output


__all__ = [
    "softmax",
    "layer_norm",
    "batch_norm",
    # NLP
    "multi_head_attention",
    "feedforward",
    # CNN
    "resnet_module",
]
