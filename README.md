# 🚀 Distributed RLHF Training Pipeline Optimization

This project demonstrates a small-scale, locally-run Reinforcement Learning with Human Feedback (RLHF) training pipeline using open-source tools. It includes profiling and optimization to highlight performance improvements.

## 📌 Project Goals

- Build a minimal RLHF training loop using Hugging Face TRL and GPT-2.
- Profile the training loop to identify CPU/GPU bottlenecks.
- Apply targeted optimizations and compare before/after performance.
- Visualize and report the improvements clearly.

## 🧱 Tech Stack

- Python 3.11
- PyTorch
- Hugging Face Transformers & TRL
- Datasets (Hugging Face)
- TensorBoard for visualization
- cProfile + Snakeviz for profiling

## 📂 Project Structure

    RLHF-project/
    ├── data/             # Contains small subset of OpenAssistant dataset
    ├── scripts/          # Scripts for training, data loading
    ├── profiling/        # .prof files from cProfile
    ├── logs/             # TensorBoard logs
    ├── visualizations/   # Screenshots and graphs showing performance
    ├── .gitignore
    ├── README.md
    └── requirements.txt  # Optional - dependencies


## 🧪 Dataset

- **Name**: OpenAssistant Conversations Dataset (OASST1)
- **Subset**: 200 dialogue samples (~<100MB)
- **Source**: [Hugging Face](https://huggingface.co/datasets/OpenAssistant/oasst1)

## 📈 Optimization Steps

1. Profile baseline training loop with cProfile and TensorBoard.
2. Optimize batch size, data loading, and memory usage.
3. Re-profile and visualize improvements.

## 📊 Visual Results

Before and after optimization screenshots and trace summaries are available in `/visualizations`.

## 🔧 Setup Instructions

1. Clone the repo:
    ```bash
    git clone https://github.com/<your-username>/RLHF-project.git
    cd RLHF-project
    ```

2. Set up Python environment:
    ```bash
    conda create -n rlhf-project python=3.11 -y
    conda activate rlhf-project
    pip install -r requirements.txt  # or install manually if not present
    ```

3. Run training:
    ```bash
    python scripts/train_rlhf.py
    ```

4. Profile:
    ```bash
    python -m cProfile -o profiling/baseline.prof scripts/train_rlhf.py
    snakeviz profiling/baseline.prof
    ```

---

## 📜 License

MIT License

---

## ✨ Acknowledgements

- [OpenAssistant Project](https://open-assistant.io/)
- [Hugging Face TRL](https://huggingface.co/docs/trl)
- [Ray Distributed](https://docs.ray.io/)
