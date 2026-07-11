# Early stage results and findings with ViT-s-16 and resolution 256

## Condition

| Item | Description |
| --- | --- |
| Dataset | one variant of ImageNet‑like from Kaggle with resolution 256×256 |
| ViT | small Vision Transformer with patch size 16 |

## Results

Even after only 1 epoch, the model learns some basic local features, and the details are not lost if we scale up the test image.

But the global features are still bad.

**Epoch 0 at res 256**
![epoch0](./misc/dino_s16_256_plots/epoch_000_256.png)
**Epoch 0 at res 512**
![epoch0](./misc/dino_s16_256_plots/epoch_000_512.png)
**Epoch 0 at res 1024**
![epoch0](./misc/dino_s16_256_plots/epoch_000_1024.png)

When the model is trained after 1, 6, 11 epochs, it already starts to learn interesting things which reflect the image globally.

**Epoch 0 at res 512**
![epoch0](./misc/dino_s16_256_plots/epoch_000_512.png)
**Epoch 5 at res 512**
![epoch5](./misc/dino_s16_256_plots/epoch_005_512.png)
**Epoch 10 at res 512**
![epoch5](./misc/dino_s16_256_plots/epoch_010_512.png)

When we compare the details after 1 and 11 epochs with high resolution, it is clear that the model is learning both global and local features.

**Epoch 0 at res 1024**
![epoch0](./misc/dino_s16_256_plots/epoch_000_1024.png)
**Epoch 10 at res 1024**
![epoch0](./misc/dino_s16_256_plots/epoch_010_1024.png)
