from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
# import re  \p{L} et al. styles not supported in re
import regex as re


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    from cs336_basics.model.modules import Linear
    linear = Linear(in_features=d_in, out_features=d_out, device=in_features.device, dtype=in_features.dtype)
    if weights is not None:
        weights_state = {'weight':weights.to(device=in_features.device, dtype=in_features.dtype)}
        linear._load_from_state_dict(weights_state)
    linear.eval()
    with torch.no_grad():
        out_features = linear(in_features)
    
    return out_features

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    from cs336_basics.model.modules import Embedding
    embedding_layer = Embedding(vocab_size,d_model)
    if weights is not None:
        weight_state = {'weight':weights.to(device=token_ids.device)}
        embedding_layer.load_state_dict(weight_state)
    embedding_layer.eval()
    with torch.no_grad():
        embedings = self.embedding_layer(token_ids)
    return embeddings


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    from cs336_basics.model.modules import SwiGLU
    swiglu_layer = SwiGLU(d_model,d_ff,device=in_features.device, dtype=in_features.dtype)
    state_dict = {
        'w1.weight': w1_weight.to(device=in_features.device),
        'w2.weight': w2_weight.to(device=in_features.device),
        'w3.weight': w3_weight.to(device=in_features.device),
    }
    swiglu_layer.load_state_dict(state_dict)
    swiglu_layer.eval()
    with torch.no_grad():
        swiglu = swiglu_layer(in_features)
    return swiglu        
    


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    from cs336_basics.model.modules import RMSNorm
    rmsnorm_layer = RMSNorm(d_model,eps,device=in_features.device,dtype=in_features.dtype)
    if weights is not None:
        weight_state = {'g_weight':weights.to(device=in_features.device,dtype=in_features.dtype)}
        rmsnorm_layer.load_state_dict(weight_state)
    rmsnorm_out = rmsnorm_layer(in_features)
    return rmsnorm_out


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab,merges=merges,special_tokens=special_tokens)
    # raise NotImplementedError

from collections import defaultdict, Counter

# pre-tokenize GPT-2, GPT-3 used.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# '(?:[sdmt]|ll|ve|re) English abbreviation handling:is, had, am, not, will, have, are
#  ?\p{L}+: ' ?' optional space; \p{L} - letter property (matches letters from any language); + one or more
# [^\s\p{L}\p{N}] - Negated character class (matches characters that are neither whitespace, letters, nor numbers)
# \s+(?!\S) - Matches trailing whitespace (spaces at the end of text), (?!\S) ensures no non-whitespace characters follow
# \s+ - Matches all other whitespace

def to_bytes_tuple(word: str) -> tuple[bytes]:
    # print(word.encode("utf-8"))    # b'hello'
    l = list(word.encode("utf-8")) # [104, 101, 108, 108, 111]
    l = [bytes([x]) for x in l]  # b'h',b'e',b'l',b'l',b'o'
    return tuple(l)
# tuple: once created, no change

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.byte_to_token_id = {v:k for k,v in vocab.items()}
        self.merges = merges

        self.bpe_rank = dict(zip(merges,range(len(merges)))) # merge and corresponding ids; smaller id-->merge earlyer when training = merge earlyer when tokenizing

        # Special Tokens
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        
        # Ensure special tokens are in vocab
        for special_byte in self.special_tokens_bytes:
            if special_byte not in self.byte_to_token_id:
                special_id = len(self.vocab)
                self.byte_to_token_id[special_byte] = special_id
                self.vocab[special_id] = special_byte


    def encode(
        self,
        text:str
    ) -> list[int]:
        """
        Encode the text into a sequence of token ids.
        
        Args:
            text: Input to encode.

        Returns:
            A sequnce of token ids.
        """
        tokens = []

        # special_tokens = self.special_tokens  !! sort by len, avoiding partial matches
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape,sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})",text)
        else:
            parts = [text]
        
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])  # string to UTF-8 encoded bytes
            else:
                tokens.extend(self._tokenize(part))

        return tokens
    

    def decode(
        self,
        ids: list[int]
    ) -> str:
        """
        Decode a sequence of integer token ids into a string.
        Args:
            ids: A list of integer token ids.
        
        Returns:
            decoded string.
        """
        bytes = b''.join(self.vocab[token_id] for token_id in ids)
        return bytes.decode("utf-8",errors="replace")


    def _tokenize(
           self,
           text:str 
    ) -> list[int]:
        """
        tokenize a string sequence without special tokens into token ids.

        Args:
        text: Input to tokenize.

        Return:
            A sequnce of token ids.
        """
        pre_tokens = []
        for m in re.finditer(PAT,text):
            word = m.group(0)
            pre_tokens.append(word)

        token_ids = []
        for token in pre_tokens:
            # convert token to bytes
            byte_tuple = to_bytes_tuple(token)
            
            # BPE merges: bytes pair merges
            merged = self._merges(byte_tuple)

            # token IDs
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        
        return token_ids
    
    def _merges(
            self,
            byte_tuple: tuple[bytes]
    ) -> list[bytes]:
        """
        Apply BPE merges to byte_tuple

        Args:
            byte_tuple: tuple of single-byte token

        Returns:
            List of merged byte tokens after BPE merging.
        """
        word = list(byte_tuple)

        def get_pairs(word:list[bytes]): #byte pair
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char,char))
                prev_char = char
            return pairs
        
        pairs = get_pairs(word)

        if not pairs:
            return word
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_rank.get(pair,float('inf'))) 
            # Get the merge priority, return the pair itself with the smallest bpe_ranks value, not the minimum value.
            if bigram not in self.bpe_ranks:    # inf
                break

            first, second = bigram              # get tuple (b1,b2)

            new_word = []

            i = 0
            while i < len(word):
                try:
                    j = word.index(first,i)     # get i-th b1 index
                except ValueError:
                    new_word.extend(word[i:])   # can't find, add left to new_word
                    break
                else:
                    new_word.extend(word[i:j])  # i~j tokens
                    i = j
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)          # merged bytes
            word = new_word
            if len(word) == 1:
                break                           # stop and return
            else:
                pairs = get_pairs(word)
            
        return word


from tqdm import tqdm
from datetime import datetime
import time
import heapq

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """


    # Initialize Vocabulary
    # print("Initialize Vocabulary.")
    # print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # add special tokens
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    for special_bytes in special_tokens_bytes:
        if special_bytes not in vocab.values():
            vocab[next_id] = special_bytes
            next_id += 1

    # Pre-tokenization which can use parallelism.
    
    # print("Pre-tokenization.")
    # print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pattern = "|".join(map(re.escape,special_tokens))



    #--------------------------------Parallelism------------------------------- 
    # for vocab_size:500 -corpus.en (fewer content), Parallelism cost 

    # pre_token_cnt_list = []
    # with open(input_path, "rb") as f:    
    #     pre_tokens_cnt = defaultdict(int)
    #     num_processes = 4
    #     boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)

    #     # The following is a serial implementation, but you can parallelize this
    #     # by sending each start/end pair to a set of processes.
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         text = f.read(end - start).decode("utf-8", errors="ignore")
    #         chunks = re.split(pattern, text)                #special token not included

    #         for chunk in chunks:
    #             for m in re.finditer(PAT,chunk):            #split word and num
    #                 word = m.group(0)
    #                 pre_tokens_cnt[to_bytes_tuple(word)] += 1  # list byte
    #         pre_token_cnt_list.append(pre_tokens_cnt)
    
    #--------------------------------Parallelism-------------------------------


    #------------------------------Non Parallelism----------------------------- 
    pre_tokens_cnt = defaultdict(int)
    with open(input_path,"r",encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split(pattern,text)             #special token not included

    for chunk in chunks:
        for m in re.finditer(PAT,chunk):        #split word and num
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1  # list byte

    #------------------------------Non Parallelism----------------------------- 

    # print("pre_token_time:",time.time())
    # 0.23 s for Non Parallelism pre_tokenize

    # BPE merge
    merges = []
    
    initial_vocab_size = len(vocab)
    remaining_merges = vocab_size - initial_vocab_size

    pbar = tqdm(total=remaining_merges, desc="BPE Training")

    
    pair_counts = defaultdict(int)
    # count all adjacent byte pairs 
    # pre_tokens_cnt: (b'', b'') -> cnt
    for token, cnt in pre_tokens_cnt.items():
        for i in range(len(token)-1):
            pair = (token[i],token[i+1])
            pair_counts[pair] += cnt
    
    # print("Start BPE training!")
    # print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    while len(vocab) < vocab_size: 

        if not pair_counts:
            break

        # most frequent pair
        max_count = max(pair_counts.values())
        candidates = [k for k,v in pair_counts.items() if v==max_count]
        best_pair = max(candidates)   # python original comparison, by bytes order

        # best_pair = candidates[0]
        a, b = best_pair

        # create a new token
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        # apply merges to all pre-tokens, token from byte list to merged byte list
        changes = []
        updates = defaultdict(int)
        updates_pairs = defaultdict(int)
        for token, cnt in pre_tokens_cnt.items():
            indices = [i for i in range(len(token)-1) if token[i:i+2] == best_pair]
            if indices:
                new_pre_token = []
                pending_right_updates = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)   #new_token: bytes object; not token[i:i+2]: tuple object

                        # update left pair
                        if len(new_pre_token) > 1:  # 确保左边有元素
                            left_pair = (new_pre_token[-2], new_token)
                            updates_pairs[left_pair] += cnt

                            left_old_pair = (token[i-1], token[i])
                            updates_pairs[left_old_pair] -= cnt

                        if i + 2 < len(token):
                            pending_right_updates.append((len(new_pre_token) - 1, i + 2))

                        i += 2
                    else:
                        new_pre_token.append(token[i])

                        pending_to_remove = []
                        for pending_idx, right_pos in pending_right_updates:
                            if right_pos == i:
                                right_pair = (new_pre_token[pending_idx], token[i])
                                updates_pairs[right_pair] += cnt

                                updates_pairs[(token[i-1],token[i])] -= cnt

                                pending_to_remove.append((pending_idx, right_pos))

                        for item in pending_to_remove:
                            pending_right_updates.remove(item)
                        i += 1
                new_pre_token = tuple(new_pre_token)
                updates[new_pre_token] += cnt
                changes.append(token)


        # time saved 1s for same key in updates
        for new_token, cnt in updates.items():
            pre_tokens_cnt[new_token] = pre_tokens_cnt.get(new_token, 0) + cnt

        for old_token in changes:
            del pre_tokens_cnt[old_token]


        # time saved 4s for update pairs
        for pair, cnt in updates_pairs.items():
            pair_counts[pair] += cnt
            
            
        del pair_counts[best_pair]

        merges.append((a,b))

        pbar.update(1)
        pbar.set_postfix({'vocab now': len(vocab)})

    pbar.close()

    return vocab, merges

import os
from typing import BinaryIO

# parallelism method: from ..cs336_basics/pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, list), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            for special_token in split_special_token:
                found_at = mini_chunk.find(special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def process_chunk_mp(args):
    start, end, file_path, pattern, PAT = args
    pre_tokens_cnt = defaultdict(int)
    
    with open(file_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
        chunks = re.split(pattern, text)
        
        for chunk in chunks:
            for m in re.finditer(PAT, chunk):
                word = m.group(0)
                pre_tokens_cnt[to_bytes_tuple(word)] += 1
    
    return pre_tokens_cnt
    