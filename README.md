# 🎉 SampleCNN – Easy Object Detection AI 🖼️🤖

![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Issues](https://img.shields.io/github/issues/Behdad-kanaani/SampleCNN-Image-Classification)

This is **Sample CNN Classification AI** for **object detection** using **PyTorch**.

> This is a really simple **sample project**, but super helpful if you want to **start learning object detection**.
> Very easy AI type to test and play with. 🚀✨

---

## 🌟 Project Vibes

```

Behdad-kanaani/SampleCNN-Image-Classification//
│
├── dataset/
│   ├── train/   # Put your training images here! 📸
│   └── val/     # Put your validation images here! ✔️
│
├── sample_cnn.py  # The main magic script ✨
└── README.md      # You’re already here 😉

````

* Organize images in **class subfolders**
* Works for **binary classification**, but you can expand it

---

## ⚡ Getting Started (Quick & Easy)

1. Clone this repo:

```bash
git clone git@github.com:Behdad-kanaani/SampleCNN-Image-Classification.git
cd SampleCNN-Image-Classification
````

2. Make a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install PyTorch and Torchvision:

```bash
pip install torch torchvision
```

---

## 🤖 How the Model Works

* **3 Convolutional Layers** + ReLU + MaxPool
* **Fully Connected Layers**: Flatten → 128 → 2 classes
* **Input**: 224×224×3 images
* **Output**: Scores for 2 classes

Super simple, easy to tweak, easy to test. 💡

---

## 🚀 Training – Let’s Go!

* Batch size: 32
* Epochs: 10 (or more if you want)
* Loss: `CrossEntropyLoss`
* Optimizer: `Adam` with lr=0.0001

```bash
python sample_cnn.py
```

Sample output:

```
Epoch 1/10, Loss: 0.6934
Epoch 2/10, Loss: 0.5123
…
```

---

## 🏆 Validation – Custom Dataset Results

✨ **How well did our SampleCNN perform?**  

| Metric                | Result         |
|-----------------------|----------------|
| 🎯 Accuracy           | **89.79%**     |

```

89.79% Accuracy

```

> These results are calculated on **My Personal DataSet**.  
> Keep experimenting with your data and hyperparameters to boost performance! 🚀  

💡 **Tip:** Generally, the more images you include in your dataset, the better your model can learn and distinguish patterns. Training for more epochs can also help improve accuracy, but make sure not to overfit!

⚠️ **Note:** The dataset is not included in this repository. Please provide your own images in the `dataset/train` and `dataset/val` folders to run the model. 🖼️

---

## 🔧 Customize & Play

* Set Your datasets 🖼️
* Adjust hyperparameters ⚡
* Make it multi-class 🌈
* Experiment and have fun! 😎

---

## 📜 License

This project is licensed under **[GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html)**.

* ✅ You can **use, share, and modify** the code freely
* ❌ If you deploy this project or modify it, you **must make your changes public**

---

## 🌈 Shoutouts & Thanks

* [PyTorch](https://pytorch.org/) – The power behind the AI 💪
* [Torchvision](https://pytorch.org/vision/stable/index.html) – Dataset helpers 🪄

## Donation

Help Me To improve it In Issue Section


If you want, I can also **add cool status badges for model training, downloads, or coverage** to make it look like a professional AI project!  
