# [Deep Learning Project Template] Title of Your Term Paper/Project
## Project Metadata
### Authors
- **Team:** ABC
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Deepfake and image-tampering detectors look great on clean test sets, but real-world photos don’t stay clean. People can re-save images to shift JPEG blocks, apply tiny warps to hide resampling traces, use AI fills to smooth seams, or run the file through social apps that strip useful camera noise. In low-light or surveillance footage (often compressed or infrared), these small edits can quietly break today’s detectors.

AN INTRODUCTORY IMAGE GOES HERE

UnFooled tackles this by training the model to expect attacks. We:
1. Practice against a red team of common counter-forensics during training.
2. Combine content cues (what’s in the picture) with physics-like cues (noise patterns, resampling artifacts).
3. Use randomized checks at test time (slight crops/resizes/recompressions) and vote on the result.
The goal: a detector that stays accurate, well-calibrated, and explainable—even when the forger fights back.

## Problem Statement
We treat the task as (a) real vs. fake and (b) where is it fake (a heatmap), even after the image has been tweaked to fool us. Attackers may know our model (white-box), know our general tricks (gray-box), or only see outputs (black-box). They must keep changes hard to notice while keeping the edited content.

We assume messy “chain of custody” (e.g., WhatsApp/Telegram recompression) and surveillance quirks (rolling shutter, LED flicker, NIR).
Our questions:

Q1: Which counter-forensics hurt most, and by how much?

Q2: Does training with a mix of attacks improve worst-case robustness, not just average scores?

Q3: Do small random test-time jitters reduce attack transfer without being slow—and can we abstain when uncertain?

We will report drop in AUC under attack (ΔAUC), worst-case accuracy across attack types, and confidence calibration suitable for legal use.

## Application Area and Project Domain
Targets include law enforcement and media forensics. Users need: a clear real/fake score, a heatmap showing where the tamper likely is, and a confidence readout (with the option to abstain when unsure).

Our pipeline can also work with provenance standards (e.g., C2PA): if signed claims exist, we check them; if not, we rely on physics-style cues. This makes reports useful for internal reviews and courtroom exhibits.

## What is the paper trying to do, and what are you planning to do?
We propose UnFooled, an attack-aware detector that pairs red-team training with randomized test-time defense and two-stream features (content + residuals). During training, each batch is hit with the most damaging of several edits: JPEG re-align + recompress, tiny resampling warps, denoise→regrain (PRNU/noiseprint spoof), seam smoothing, small color/gamma shifts, and social-app transcodes. The model learns both to decide real/fake and to mark tampered pixels.

At test time, we run a few small random transforms (resize/crop phase, gamma tweak, JPEG quality/phase), get multiple predictions, and vote. Under the hood, we use a pretrained backbone (e.g., ResNet-50) plus a forensic residual adapter and a light FPN-style mask head—fast to fine-tune, sensitive to subtle traces. We will report clean vs. attacked metrics side-by-side (ΔAUC, worst-case accuracy, IoU for localization, and calibration/ECE) on standard deepfake/tamper datasets and a surveillance-style split (low-light, heavy compression). Success = small ΔAUC, strong worst-case, and clear, judge-friendly explanations—because a detector that only works when nobody’s trying to fool it isn’t very forensic.


### Project Documents
- **Presentation PDF:** [Project Presentation](/presentation.PDF)
- **Presentation PPTX:** [Project Presentation](/presentation.pptx)
- **Term Paper PDF:** [Term Paper](/report.pdf)
- **Term Paper Latex Files:** [Term Paper Latex files](/report.zip)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/BRAIN-Lab-AI/Deep-Learning-Project)
- 
### Reference GitHub
- [This is a reference Github](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Project UI

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
