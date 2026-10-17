# MediScan AI — CNN Architecture & Layers Explained
**A Plain-Language, Layer-by-Layer Guide to the Custom Dual-Path Neural Network**

---

## 1. The Big Picture: What Makes Our Model Unique?

Most medical imaging projects take an existing off-the-shelf model like ResNet-50 or VGG-16 that was originally designed to classify pictures of cats, dogs, and cars on the internet. 

In this project, **we built a 100% custom Convolutional Neural Network (CNN) from scratch called `DualPathNet`**, tailored specifically for radiological scans (Brain MRI, Chest X-Rays, and Endoscopy).

### The "Two-Doctor" Analogy
Imagine two specialist doctors examining the exact same medical scan side-by-side:
- **Doctor A (Path A — $3 \times 3$ Filters)** holds a **magnifying glass**. Doctor A focuses on tiny, microscopic details: sharp tumor edges, fine blood vessels, and pinpoint mucosal textures.
- **Doctor B (Path B — $5 \times 5$ Filters)** steps back and looks at the **entire organ**. Doctor B focuses on broad anatomical context: lung symmetry, whole-organ shapes, and diffuse shadows.
- **The Board Meeting (`join_conv` & `CBAM Attention`)**: Both doctors sit together. They merge their findings, highlight the most suspicious region with an attention highlighter, and make a joint diagnosis.

```
                  [ Input Image: 224 x 224 x 3 ]
                                |
               +----------------+----------------+
               |                                 |
       [ Path A: 3x3 Kernels ]           [ Path B: 5x5 Kernels ]
       (Fine details & edges)            (Broad anatomical context)
               |                                 |
          256 Channels                      256 Channels
               |                                 |
               +----------------+----------------+
                                |
                   [ Concatenation: 512 Channels ]
                                |
                     [ join_conv: 1x1 Conv ]
                       (Compress to 256)
                                |
                    [ CBAM Attention Module ]
                   - Channel Attention ("WHAT")
                   - Spatial Attention ("WHERE")
                                |
                     [ Refine Block: 3x3 ]
                                |
                    [ Dual Pooling Head ]
               AvgPool (256)  +  MaxPool (256)
                                |
                     [ 512-D Feature Vector ]
                                |
                   [ Classifier MLP + Dropout ]
                                |
                   [ Final Prediction / Class ]
```

---

## 2. Layer-by-Layer Breakdown (In Simple Terms)

Let's examine every building block in the model code and understand what each layer actually does:

---

### Layer Block 1: `ResidualConvBlock` (The Core Engine)

Every stage of our model is built using `ResidualConvBlock`. Inside each block are four essential components:

#### 1. `Conv2d` (Convolutional Layer)
- **What it does**: A convolutional filter is like a small sliding mathematical magnifying glass (e.g., $3 \times 3$ or $5 \times 5$ pixels) that slides across the image.
- **Simple explanation**: At the start of the network, these filters detect simple features like horizontal edges, curved lines, and contrast differences. Deeper in the network, they detect complex medical patterns like lung fluid infiltrates or tumor boundaries.

#### 2. `BatchNorm2d` (Batch Normalization)
- **What it does**: After convolution, pixel numbers can become wildly large or tiny. BatchNorm rescales the activations so their average is 0 and spread is 1.
- **Simple explanation**: Think of it as **leveling the playing field**. It prevents internal numbers from exploding or vanishing, allowing the model to train 5x faster and stay stable.

#### 3. `SiLU` (Swish Activation Function)
- **Mathematical formula**: $\text{SiLU}(x) = x \cdot \sigma(x)$
- **Why not standard ReLU?**: Standard ReLU harshly zeros out all negative numbers ($\max(0, x)$), which can cause "dead neurons" where parts of the network permanently shut down. SiLU is smooth, continuously differentiable, and allows tiny negative values to trickle through.
- **Simple explanation**: A smooth volume slider rather than a harsh on/off light switch. It lets subtle medical gradients flow smoothly backwards during learning.

#### 4. The Residual Shortcut (Skip Connection)
- **What it does**: Takes the original input $x$ that entered the block and adds it directly to the output:
  $$\text{Output} = \text{SiLU}(\text{Conv}_2(\text{SiLU}(\text{Conv}_1(x)))) + x$$
- **Simple explanation**: Think of an **express highway lane**. In very deep networks, gradient signals get weaker and weaker as they travel backwards (the vanishing gradient problem). The skip connection provides a direct bypass express route so the original information is never lost.

---

### Layer Block 2: The Two Parallel Paths (`PathA` and `PathB`)

Instead of a single straight pipe of convolutions, our network splits into two parallel branches:

| Feature | Path A (Fine Receptive Field) | Path B (Broad Receptive Field) |
| :--- | :--- | :--- |
| **Kernel Size** | $3 \times 3$ pixels | $5 \times 5$ pixels |
| **Field of View** | Tight, focused, localized | Wide, encompassing, contextual |
| **Clinical Focus** | Microvascular tumor boundaries, mucosal pinpoints | Global lung shape, diffuse ground-glass fluid |
| **Stage Channels** | $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ | $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ |

---

### Layer Block 3: `join_conv` (The Synthesis Layer)

- **What it does**: Path A gives 256 feature maps. Path B gives 256 feature maps. We stack them together along the channel dimension (`torch.cat`), resulting in 512 channels.
- **The $1 \times 1$ Convolution**: A $1 \times 1$ convolutional layer acts as a **smart blender**. It blends the 512 fine and broad channels back down to 256 channels without changing the height or width of the image.
- **Simple explanation**: It takes Doctor A's detailed notes and Doctor B's big-picture notes and condenses them into a single, unified medical summary.

---

### Layer Block 4: `CBAM` (Convolutional Block Attention Module)

This is one of the most powerful features of our custom architecture. Human radiologists don't look at every pixel equally; their eyes immediately dart to the lesion. **CBAM teaches the neural network how to pay attention.**

CBAM does this in two sequential steps:

#### Step A: Channel Attention — *"WHAT to look for"*
- Not all 256 feature maps are equally important. Some might detect background bone, others detect scanner artifacts, and others detect tumor tissue.
- **How it works**:
  1. Computes the average and maximum energy of each channel across the whole image.
  2. Passes those values through a small internal brain (a 2-layer MLP).
  3. Produces a weight score between 0 and 1 for each of the 256 channels.
  4. Multiplies the channels by their score: **amplifying tumor-detecting channels and muting background noise channels**.

#### Step B: Spatial Attention — *"WHERE to look"*
- Once the network knows *what* patterns matter, it must find *where* in the $224 \times 224$ image they are located.
- **How it works**:
  1. Compresses all channels into a 2D map showing the average and peak intensities across the image.
  2. Runs a large $7 \times 7$ convolution across this map to identify spatial hotspots.
  3. Produces a 2D heatmap mask (values from 0 to 1).
  4. Multiplies the image features by this heatmap.
- **Simple explanation**: It acts like a **fluorescent highlighter pen**, highlighting the exact tumor or ulcer region while dimming normal healthy background tissue.

---

### Layer Block 5: Dual Pooling (`AdaptiveAvgPool2d` + `AdaptiveMaxPool2d`)

Traditional CNNs only use Average Pooling (which washes out sharp spikes) or Max Pooling (which only grabs the single highest pixel and ignores context).

We use **Dual Pooling**:
1. `AdaptiveAvgPool2d((1, 1))`: Captures the **overall average presence** of pathological signals (yields a 256-number vector).
2. `AdaptiveMaxPool2d((1, 1))`: Captures the **most intense, sharpest hotspot** of the pathology (yields another 256-number vector).
3. **Concatenation**: We glue them together into a rich **512-dimensional diagnostic fingerprint vector**.

- **Simple explanation**: If you want to know how a class did on an exam, you look at both the **average score** AND the **top score**. Dual pooling gives the classifier both perspectives.

---

### Layer Block 6: The Classifier Head (Decision-Making MLP)

The final stage transforms the 512-dimensional fingerprint into final diagnostic probabilities:

1. **`nn.Linear(512, 128)`**: Compresses the 512 medical features into 128 high-level clinical conclusions.
2. **`nn.BatchNorm1d(128)`**: Normalizes these conclusions for numerical stability.
3. **`nn.SiLU()`**: Smooth non-linear activation.
4. **`nn.Dropout(p=0.4)`**: During training, randomly turns off 40% of the connections on every batch.
   - **Simple explanation**: Prevents any single neuron from becoming a "know-it-all". Forces all neurons to learn independently, preventing overfitting.
5. **`nn.Linear(128, num_classes)`**: The final projection layer that outputs raw class logits:
   - For **TumorNet**: 4 outputs (`glioma`, `meningioma`, `notumor`, `pituitary`)
   - For **PneumoNet**: 2 outputs (`NORMAL`, `PNEUMONIA`)
   - For **GastroNet**: 2 outputs (`Normal`, `Ulcer`)

---

## 3. Summary Architecture Table

| Component | Layer Type | Input Shape | Output Shape | Primary Function |
| :--- | :--- | :---: | :---: | :--- |
| **Input** | Raw Scan | - | $3 \times 224 \times 224$ | Medical image input (RGB) |
| **Path A** | $3 \times 3$ ResBlocks | $3 \times 224 \times 224$ | $256 \times 7 \times 7$ | Fine-grained texture & edge extraction |
| **Path B** | $5 \times 5$ ResBlocks | $3 \times 224 \times 224$ | $256 \times 7 \times 7$ | Broad anatomical context extraction |
| **Merge** | `torch.cat` | $2 \times (256 \times 7 \times 7)$ | $512 \times 7 \times 7$ | Feature concatenation |
| **`join_conv`** | $1 \times 1$ Conv + BN + SiLU | $512 \times 7 \times 7$ | $256 \times 7 \times 7$ | Channel reduction & feature synthesis |
| **CBAM (Channel)** | MLP + Sigmoid | $256 \times 7 \times 7$ | $256 \times 7 \times 7$ | Channel reweighting ("What" matters) |
| **CBAM (Spatial)** | $7 \times 7$ Conv + Sigmoid | $256 \times 7 \times 7$ | $256 \times 7 \times 7$ | Pixel heatmap ("Where" to look) |
| **Refine** | $3 \times 3$ ResBlock | $256 \times 7 \times 7$ | $256 \times 7 \times 7$ | Final convolutional smoothing |
| **Dual Pool** | AvgPool + MaxPool | $256 \times 7 \times 7$ | $512 \text{ (1D)}$ | Context + Peak intensity extraction |
| **MLP Dense** | Linear + BN + SiLU | $512$ | $128$ | High-level clinical abstraction |
| **Dropout** | Dropout ($p = 0.4$) | $128$ | $128$ | Regularization against co-adaptation |
| **Classifier** | Linear | $128$ | $K \text{ classes}$ | Final diagnostic prediction logits |
