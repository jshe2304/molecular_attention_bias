from .graph_attention_transformer import GraphAttentionTransformer
from .biased_attention_transformer import BiasedAttentionTransformer
from .fixed_attention_transformer import FixedAttentionTransformer
from .graph_pe_transformer import GraphPETransformer

def get_model(model_type: str, *args, **kwargs):
    if model_type == "GraphAttentionTransformer": 
        return GraphAttentionTransformer(*args, **kwargs)
    if model_type == "BiasedAttentionTransformer": 
        return BiasedAttentionTransformer(*args, **kwargs)
    if model_type == "FixedAttentionTransformer": 
        return FixedAttentionTransformer(*args, **kwargs)
    if model_type == "GraphPETransformer": 
        return GraphPETransformer(*args, **kwargs)