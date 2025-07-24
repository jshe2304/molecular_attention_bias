from .graph_attention_transformer import GraphAttentionTransformer
from .biased_attention_transformer import BiasedAttentionTransformer
from .fixed_attention_transformer import FixedAttentionTransformer
from .graph_pe_transformer import GraphPETransformer

name_to_model = {
    "GraphAttentionTransformer": GraphAttentionTransformer,
    "BiasedAttentionTransformer": BiasedAttentionTransformer,
    "FixedAttentionTransformer": FixedAttentionTransformer,
    "GraphPETransformer": GraphPETransformer,
}