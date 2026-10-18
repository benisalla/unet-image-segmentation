# Architecture: the model family and why each switch exists

Everything in this repo is one question asked six ways: *what does each design choice
of U-Net actually buy?* The coursework this repo grew out of answered it by flipping
switches on one model class. Here every switch has a plain name and every combination
is a preset you can train with one flag.

| `--model` | skip connections | down-sampling | conv per block | residual | code |
|---|---|---|---|---|---|
| `unet` | concat | 2x2 max-pool | 2 | no | `models.EncoderDecoder` |
| `unet-strided` | concat | 3x3 conv, stride 2 | 2 | no | `models.Down(down="strided")` |
| `unet-single` | concat | 2x2 max-pool | 1 | no | `models.ConvBlock(double=False)` |
| `unet-residual` | concat | 2x2 max-pool | 2 | yes | `models.ConvBlock(residual=True)` |
| `segnet` | **none** | 2x2 max-pool | 2 | no | `models.Up(skip=False)` |
| `fcn` | - | - (no convs) | - | - | `models.FullyConnectedSeg` |

The switches can also be set directly and override the preset:
`unetseg-train --model unet --no-skip --down strided --residual --no-double`.

## The task in one line

Binary segmentation: an image `x ∈ R^{3×H×W}` goes in, one logit per pixel
`z ∈ R^{1×H×W}` comes out; `p = σ(z)` is the probability that the pixel is object, and
`mask = p > t` with a threshold `t` (0.5 by default, see [thresholds](#thresholds)).

## U-Net (Ronneberger et al., 2015)

```
x 3×96×96
│ stem      ConvBlock(3→16)                    16×96×96 ───────────────────────┐ skip
│ down1     maxpool, ConvBlock(16→32)          32×48×48 ─────────────────┐     │
│ down2     maxpool, ConvBlock(32→64)          64×24×24 ───────────┐     │     │
│ down3     maxpool, ConvBlock(64→128)        128×12×12 ─────┐     │     │     │
│ down4     maxpool, ConvBlock(128→256)       256×6×6        │     │     │     │
│ up4       convT 256→128, cat ─────────────────────────────┘ → ConvBlock(256→128) 128×12×12
│ up3       convT 128→64,  cat ───────────────────────────────────┘ → ConvBlock(128→64) 64×24×24
│ up2       convT 64→32,   cat ─────────────────────────────────────────┘ → ConvBlock(64→32) 32×48×48
│ up1       convT 32→16,   cat ───────────────────────────────────────────────┘ → ConvBlock(32→16) 16×96×96
│ head      1×1 conv 16→1                       1×96×96 logits
```

(`--base 16 --depth 4`, the defaults: 1.94 M parameters. The coursework used base 64,
31 M parameters; `--base 64` reproduces that width.)

**Why an encoder.** A 3×3 conv sees 3×3 pixels. To decide whether a pixel belongs to a
cat, the network needs to see the cat. Each 2× down-sampling doubles how far every later
conv "sees" (the receptive field), so after four of them a 3×3 conv at 6×6 covers most of
a 96×96 image. Channels double at each step because there is more to describe about a
large region than about a 3×3 patch, while the cost per layer stays roughly constant
(H·W halves twice as fast as C doubles: H/2·W/2·2C = H·W·C/2).

**Why a decoder.** The answer must be given per pixel, at full resolution. A transposed
conv with kernel 2 and stride 2 (`ConvTranspose2d(k=2, s=2)`) learns how to turn each
low-resolution feature into a 2×2 patch: it is the learned inverse of the pooling.

**Why skip connections.** The bottleneck knows *what* is in the image but, at 6×6, it no
longer knows *where* the edges are to the pixel. The skip connection concatenates the
encoder feature map of the same resolution to the decoder's input:

```
u = ConvT(d_{l+1})                      # what, up-sampled (blurry)
d_l = ConvBlock( concat[ e_l , u ] )    # + where, straight from the encoder (sharp)
```

The decoder can then copy the edges instead of reconstructing them. Concatenation keeps
both sources and lets the next conv learn how to mix them; adding them (`e_l + u`) would
force both into the same feature space, and a max would throw one away. This is the
first thing the comparison measures: `unet` vs `segnet`.

**Odd sizes.** If H is not divisible by 2^depth, pooling floors the size and the up-sampled
map is one pixel short; `Up` pads it before concatenating (tested with a 50×50 input).

## The convolution block: single, double, residual

```
single:    y = ReLU( BN( conv3x3(x) ) )
double:    y = ReLU( BN( conv3x3( ReLU(BN(conv3x3(x))) ) ) )
residual:  y = ReLU( F(x) + S(x) )      F = the single or double body (before its last ReLU)
                                        S = identity, or 1×1 conv + BN when channels change
```

**Why two convs.** Two stacked 3×3 convs cover a 5×5 window with 2·9·C² weights instead of
25·C² for one 5×5 conv, and they put a ReLU in the middle, so the block can represent a
more complex function at the same resolution. `unet-single` halves the parameters
(0.96 M vs 1.94 M) and removes that extra non-linearity at each scale.

**Why residual.** With `y = F(x) + x` the block only has to learn a correction to its
input, and the gradient reaches earlier layers through the identity path without being
multiplied by every layer's Jacobian. It matters most for deep stacks; a 4-level U-Net
with BatchNorm is shallow enough to train without it, so the expected gain here is small
(that is the honest prediction to check in the results). Note that the coursework's flag
named `is_resnet` was the skip-connection switch, not this.

**Why BatchNorm and no conv bias.** BN normalises each channel over the batch, which keeps
activations in a range where ReLU and the learning rate behave, and its own shift makes a
conv bias redundant (`bias=False`).

## Down-sampling: max-pool vs strided convolution

```
maxpool:  y[i,j] = max_{a,b ∈ {0,1}} x[2i+a, 2j+b]         0 weights
strided:  y = ReLU(BN(conv3x3(x, stride=2)))                9·C² weights per level
```

**Max-pool** keeps the strongest response in each 2×2 window. It has no weights, and a
feature that moves by one pixel inside the window gives the same output (local
translation invariance). For "is there an edge / a texture here", the maximum is exactly
the right summary.

**Strided conv** learns its own summary of each window. It can in principle do better
(it can learn max-like or average-like behaviour, or something else), but it costs
weights (2.14 M vs 1.94 M here), and at initialisation it is a random projection that
can throw away the strong activations max-pool keeps for free. In the coursework it
collapsed completely (recall 0), partly because the strided conv had no BN/ReLU and the
learning rate was 4e-3; here it has both and the run uses warm-up.

## SegNet-like encoder-decoder (`segnet`)

The same encoder and decoder with the skip connections removed: the decoder sees only the
bottleneck. Real SegNet (Badrinarayanan et al., 2017) passes the max-pool *indices* to
the decoder for unpooling, which is a lightweight form of "where" information; this
variant drops even that, to isolate the effect of the skips. Expect blurry, rounded
masks: the edges have to be re-invented from a 6×6 map.

## The fully connected baseline (`fcn`)

```
x (3·H·W) → Linear → 2h → h → h/2 → h → 2h → Linear → H·W → reshape 1×H×W
```

(each hidden Linear followed by BatchNorm1d and ReLU; `h = --fcn-hidden`, 256.)

Every output pixel is connected to every input pixel with its own weight. The model has
no idea that neighbouring pixels are related, must learn "edge" separately at every
location, and its first layer alone has 3·96·96·512 = 14 M weights (19.2 M total at
96 px; it grows with H·W). The coursework called it FCN; the name usually means the
opposite, the *Fully Convolutional Network* of Long et al. (2015), which has no dense
layers at all. What this model *can* learn is a prior over where objects usually are
(a blurry average mask), so it scores above zero IoU but cannot follow an individual
shape's outline.

The coursework's version ended with Linear → BN → ReLU, so its logits could not be
negative: `σ(z) ≥ 0.5` everywhere, a pixel was called background only when the ReLU
output was exactly 0, and the BCE of a background pixel could never fall below ln 2.
Here the last layer is a plain Linear.

## Losses

```
BCE      = -mean_i [ y_i log p_i + (1-y_i) log(1-p_i) ]
Dice     = 1 - (2 Σ p_i y_i + ε) / (Σ p_i + Σ y_i + ε)        (per image, then averaged; ε = 1)
bce+dice = BCE + Dice                                          (default)
```

BCE averages over pixels, so when the object is 12% of the image (the shapes) the 88% of
easy background dominates and "predict background" is a strong local minimum early on.
Dice is a ratio of overlap to total size and is blind to the amount of background, so it
pulls directly towards IoU-like overlap, but its gradient is noisy when p is ~0.5
everywhere. The sum gets BCE's smooth per-pixel signal from the first step and Dice's
focus on the object. The coursework suggested exactly this ("Using Dice loss or a
combination of BCE and Dice loss can directly optimize the overlap") as the fix for its
precision/recall problems.

## Thresholds

`mask = σ(z) > t`. With a calibrated model `t = 0.5` minimises pixel errors, but IoU is not
pixel error: if the model is unsure on the border, a lower `t` grows the mask (recall up,
precision down) and a higher `t` shrinks it. `unetseg-eval` sweeps `t ∈ {0.10, 0.15, ...,
0.90}` on the **validation** split, picks the best IoU, and reports the test split at both
0.5 and that `t` (choosing `t` on the test split would tune on the reported numbers).

## What the coursework concluded

The coursework's written discussion, kept in the author's words (numbers are from the
private competition data, where the object was salt in seismic images, and from the
buggy pipeline described in [legacy/README.md](../legacy/README.md)):

**Skip connections.**
> Skip connections are essential in U-Net architectures, and my experiments highlight
> their impact on segmentation performance.
> - **Preserving Fine Details:** Skip connections bridge the encoder and decoder, enabling
>   high-resolution features from early layers to be directly combined with up-sampled
>   features. This is crucial for recovering spatial details lost during down-sampling.
> - **Better Gradient Flow:** By creating direct pathways, skip connections help mitigate
>   vanishing gradient issues, making it easier to train deeper networks.
>
> With skip connections the network achieved higher accuracy (~90.17%) and a better F1
> score (~0.7670). Without skip connections performance dropped slightly (accuracy ~89.00%,
> F1 score ~0.7351), indicating that even a modest removal of skip connections leads to a
> loss of detail and weaker segmentation performance.
>
> **Alternatives to Concatenation:** I focused on the standard approach of concatenating
> feature maps, as it preserves all information from both the encoder and decoder.
> However, other strategies exist: **Element-wise Addition** might blend features too
> aggressively, potentially losing distinctive details; **Max/Min Operations** could
> oversimplify the fusion process, limiting the ability to capture complementary
> information effectively.
>
> **Conclusion:** Skip connections are critical for maintaining spatial details and
> boosting overall performance.

**Max pooling.**
> With standard max pooling, the network achieved an accuracy of about 90.17% and an F1
> score of 0.7670. This approach preserves the strongest activations within each window,
> which helps retain the most salient features for segmentation. When max pooling was
> replaced by a convolution with a stride of 2, the performance degraded substantially
> (accuracy dropped to around 75.99%, and the recall essentially dropped to 0, yielding an
> F1 score of 0). This suggests that the learnable down-sampling in the strided
> convolution did not effectively capture the necessary features, possibly because it
> failed to robustly highlight the dominant responses that are critical in segmentation
> tasks. Another issue with Pooling using a Conv Layer is that it increases the number of
> parameters.
>
> **Conclusion:** While both methods reduce spatial dimensions, our experiments clearly
> show that standard max pooling is far superior in this context. It appears that the
> fixed, non-learnable nature of max pooling better preserves key features compared to
> strided convolutions.

**Threshold for inference.**
> **Precision** measures the ratio of true positive salt pixel predictions to all pixels
> predicted as salt. **Recall** represents the ratio of true positive salt pixel
> predictions to all actual salt pixels.
>
> At threshold 0.5 the model achieved high precision (~0.8871) and reasonable recall
> (~0.6755). Increasing the threshold to 0.75 led to a significant drop in both precision
> (~0.2192) and recall (~0.2070). This suggests that a higher threshold makes the model
> overly conservative, missing many salt pixels while still misclassifying non-salt
> pixels when it does make a prediction.
>
> **Strategies for Improvement:** *Threshold Optimization:* rather than fixing the
> threshold at 0.5, analyzing the precision–recall curve can help determine an optimal
> threshold. *Modified Loss Functions:* using Dice loss or a combination of BCE and Dice
> loss can directly optimize the overlap between predicted and true masks. *Post-Processing
> Techniques:* methods like conditional random fields (CRFs) or morphological operations can
> refine the segmentation masks further.
>
> **Conclusion:** The threshold choice plays a critical role in balancing precision and
> recall. A more systematic approach to threshold selection, along with tailored loss
> functions, can help better align the model with the specific needs of salt segmentation.

**FCN versus encoder-decoder.**
> Variants using the encoder–decoder design performed exceptionally well, achieving
> accuracy above 90% and a strong F1 score (~0.7670). This structure effectively
> captures both global context (via down-sampling) and fine spatial details (via
> up-sampling and skip connections). In contrast, the FCN variant performed significantly
> worse, with accuracy dropping to ~60.18% and an F1 score of just ~0.2565. Without a
> hierarchical structure, it struggled to combine contextual information with
> fine-grained spatial details.
>
> **Conclusion:** The auto-encoder (encoder–decoder) structure is essential for effective
> semantic segmentation. It strikes a balance between capturing high-level context and
> preserving detailed localization.

**Overall opinion.**
> To accurately answer these questions, we must implement and test each approach; there's
> no better way to evaluate their impact. Max pooling is more effective for down-sampling
> in U-Net. Skip connections are essential for merging multi-scale information, and
> concatenation proves to be the best fusion method. The encoder–decoder architecture
> significantly outperforms an FCN. A fixed threshold (e.g., 0.5) provides a reasonable
> balance between precision and recall, but non-optimal thresholds can degrade
> performance.

### Reading those numbers with the bug fixes in mind

* The threshold 0.75 numbers came from a model *trained* with 0.75 used in validation and
  early stopping, and the later sweep used thresholds 40...80 (all background). The
  sweep here (`unetseg-eval`, [evaluation.md](evaluation.md)) is the systematic
  threshold choice the coursework asked for.
* The FCN was handicapped twice: by its design (no locality, no hierarchy) and by its
  final ReLU (no negative logits). The conclusion still holds after the fix; the README
  results show by how much.
* Validation and test were augmented in the coursework, and the model tested was the last
  epoch, not the saved best; all the relative comparisons were therefore noisier than
  they looked. The opinions above are re-tested on the same footing in the README
  results.
