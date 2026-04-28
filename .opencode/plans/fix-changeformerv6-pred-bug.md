# Fix: ChangeFormerV6 Evaluation IndexError - pred[0] Resolution Mismatch

## Root Cause

`DecoderTransformer_v3` (used by ChangeFormerV5/V6) returns a list of 5 multi-scale predictions:
```
outputs = [p_c4, p_c3, p_c2, p_c1, cp]
```

`main.py` uses `pred[0]` which gets `p_c4` (coarsest, 8x8 resolution) instead of `cp` (final, full resolution).

Error:
```
IndexError: The shape of the mask [16, 256, 256] at index 1 does not match
the shape of the indexed tensor [16, 8, 8] at index 1
```

## Additional Issue

Model return types are inconsistent:
- UNet, SiamUnet, BIT, DTCDSCN: return `[tensor]` (1-element list) - `pred[0]` OK
- V4, V5, V6: return multi-element list - `pred[0]` WRONG (gets coarsest scale)
- V1, V2, V3: return single tensor - `pred[0]` WRONG (gets first batch sample)

## Fix Plan

### Step 1: Fix `main.py` - Change `pred[0]` to `pred[-1]`

Three locations:

**Line 167** (training loss):
```python
# Before:
loss = self.loss_fn(pred[0].contiguous(), Label)
# After:
loss = self.loss_fn(pred[-1].contiguous(), Label)
```

**Line 250** (evaluation loss):
```python
# Before:
loss = self.loss_fn(pred[0].contiguous(), Label)
# After:
loss = self.loss_fn(pred[-1].contiguous(), Label)
```

**Line 255** (evaluation prediction):
```python
# Before:
pred_labels = pred[0].argmax(dim=1)
# After:
pred_labels = pred[-1].argmax(dim=1)
```

### Step 2: Fix `backbone/ChangeFormer.py` - V1/V2/V3 return format

**ChangeFormerV1 forward** (line ~899):
```python
# Before:
return cp
# After:
return [cp]
```

**ChangeFormerV2 forward** (line ~1286):
```python
# Before:
return cp
# After:
return [cp]
```

**ChangeFormerV3 forward** (line ~1326):
```python
# Before:
return cp
# After:
return [cp]
```

## Verification

After applying fixes, all models will return lists. `pred[-1]` correctly gets:
- 1-element list models: the only element (same as before)
- V4/V5/V6: the final full-resolution prediction `cp`
- V1/V2/V3: the only element `cp` (now wrapped in list)
