# DERMA-ACT V0 Preprocessing Forensics Report - FINAL RESOLVED

Status: FULLY RESOLVED
Date: 2026-09-16/17

## Three-bug Root Cause Chain

### Bug 1 — Major preprocessing mismatch (max diff 3.52, 81.3% agree)
Root cause: preprocess() used cv2.resize(224,224) directly.
Missing the canonical Albumentations Resize(291,291)+CenterCrop(224,224) step.
The canonical get_val_transforms() performs anti-shortcut upscale+crop first.
Fix: Import get_val_transforms() directly; never duplicate the pipeline.

### Bug 2 — A0 passed through ViewGenerator (max diff 3.52 persisted)
Root cause: view_gen.generate(KEEP_FULL) pre-resizes image to 224x224.
Then _canonical_tf(224x224) did Resize(291)+CenterCrop(224) on a small image,
producing a completely different spatial crop from the canonical.
Fix: A0 bypasses ViewGenerator entirely. A1-A9 use view_gen + normalize only.

### Bug 3 — cuDNN GEMM batch-composition effect (max diff 0.006, 100% agree)
Root cause: cuDNN float32 GEMM accumulation order depends on batch composition.
Same image in a batch of 5 (test_ids order) vs batch of 64 (canonical neighbors)
produces ~0.001 difference in the final float32 logits.
This is NOT a preprocessing error. It is a GPU kernel parallelization artifact.
Fix: A0 gate runs in canonical NPZ order with batch_size=64, exactly matching
the original evaluation. This produces bit-exact 0.000000 logit diff.

## Evidence

| Test | Max logit diff | Pred agree |
|---|---|---|
| Original bug (Bug 1+2) | 3.522809 | 81.28% |
| After Bug 1+2 fix, wrong order | 0.006298 | 100% |
| After Bug 3 fix (canonical order, bs=64) | 0.000000 | 100% = 3686/3686 |

## Micro Forensics (5 deterministic images: test_ids[:5])

Images: ISIC_0000002, ISIC_0000003, ISIC_0000010, ISIC_0000020_downsampled, ISIC_0000040_downsampled

Stage verification (first image):
  raw RGB uint8:        (767,1022,3), min=0,   max=255, mean=158.17
  after A.Resize(291):  (291,291,3),  min=1,   max=255, mean=157.62
  after CenterCrop(224):(224,224,3),  min=21,  max=255, mean=151.44
  after /255:           float32, min=0.0824, max=1.000, mean=0.5939
  after normalize:      float32, min=-1.5455, max=2.6400, mean=0.6429
  manual vs canonical_tf diff: 4.77e-07 (sub-pixel float32 rounding, not an issue)

Batch A (test order, bs=5):  max diff = 0.001836  FAIL
Batch B (canonical order, bs=64): max diff = 0.000000  PASS

## Environment

  torch: 2.5.1+cu121
  albumentations: 1.3.1
  lightning: 2.6.5 (checkpoint from 2.6.6 -- harmless)
  cudnn.deterministic: False (not needed -- canonical order achieves exact match)

## Final Gate

  5-image max logit diff : 0.00000000
  5-image pred agreement : 5/5
  Full N=3686 max diff   : 0.00000000
  Full N=3686 pred agree : 3686/3686

  DERMA-ACT OBSERVATION SWEEP AUTHORIZED: YES