# Layer 2 Pretrained LoRA Adapter

LoRA adapter weights for the Layer 2 meta-classifier, fine-tuned on Mistral-7B-v0.1.

## Adapter Details

- **Base model:** `mistralai/Mistral-7B-v0.1`
- **Method:** LoRA (PEFT 0.10.0)
- **Rank (r):** 16
- **Alpha:** 32
- **Dropout:** 0.05
- **Target modules:** q_proj, k_proj, v_proj, o_proj
- **Task type:** CAUSAL_LM

## Files

| File | Committed | Description |
|------|-----------|-------------|
| `adapter_config.json` | Yes | LoRA hyperparameters and configuration |
| `tokenizer.json` | Yes | Tokenizer vocabulary |
| `tokenizer_config.json` | Yes | Tokenizer settings |
| `special_tokens_map.json` | Yes | Special token definitions |
| `adapter_model.safetensors` | No | Model weights (~55MB, git-ignored) |

## Reproducing the Weights

To regenerate `adapter_model.safetensors`, run the Layer 2 training pipeline:

```bash
python -m ml-service.layer2.training.train_lora
```

See `ml-service/layer2/training/` and `ml-service/layer2/config/` for training configuration.
