#################### Error 1 ####################
building qwen2 model in TE...
[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/shenxiao/toolkits/model_checkpoints_convertor/qwen/hf2mcore_qwen2_dense_and_moe_gqa.py", line 950, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/shenxiao/toolkits/model_checkpoints_convertor/qwen/hf2mcore_qwen2_dense_and_moe_gqa.py", line 943, in main
[rank0]:     mg_model = model_provider()
[rank0]:   File "/workspace/shenxiao/examples/qwen2/pretrain_qwen.py", line 68, in model_provider
[rank0]:     transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
[rank0]:   File "/workspace/shenxiao/megatron_patch/model/qwen2/layer_specs.py", line 50, in get_gpt_layer_with_transformer_engine_spec
[rank0]:     mlp = _get_mlp_module_spec(
[rank0]:   File "/workspace/shenxiao/megatron_patch/model/qwen2/layer_specs.py", line 118, in _get_mlp_module_spec
[rank0]:     linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
[rank0]: NameError: name 'TELayerNormColumnParallelLinear' is not defined

#################### Fix 1 ####################
megatron_patch/model/qwen2/layer_specs.py
```python
    from megatron.core.transformer.custom_layers.transformer_engine import (
        # TEColumnParallelGroupedLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        # TERowParallelGroupedLinear,
        TERowParallelLinear,
    )
```

