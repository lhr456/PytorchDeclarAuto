# PytorchDeclarAuto
声明式的卷积神经网络
# DeclarAuto

DeclarAuto — A declarative and automatic shape-driven neural network construction toolkit (PyTorch).

**Tagline:** Declarative + Automatic → Build CNNs by describing tensor shapes, not kernel math.

## Features
- Declare input & target tensor shapes, auto-compute conv/pool params.
- Safe parameter calculator (avoid negative kernel sizes).
- Apply operations (conv / maxpool / avgpool / upsample) directly on tensors.
- Simple API for teaching, rapid prototyping and building large networks.
- Optional module wrappers for trainable layers.

## Installation
```bash
git clone https://github.com/<your-username>/DeclarAuto.git
cd DeclarAuto
pip install -r requirements.txt
# or
pip install -e .
