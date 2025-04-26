from transformers import AutoTokenizer, pipeline, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_from_disk
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import copy

# Load GPT-2 and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model.generation_config = GenerationConfig.from_pretrained("gpt2")

ref_model = copy.deepcopy(model)
ref_model.generation_config = GenerationConfig.from_pretrained("gpt2")
for param in ref_model.parameters():
    param.requires_grad = False

# Load sentiment analysis reward model
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define a custom reward model using sentiment analysis
class SentimentRewardModel(nn.Module):
    def forward(self, samples, responses, **kwargs):
        texts = [sample + resp for sample, resp in zip(samples, responses)]
        scores = sentiment_pipe(texts)
        # Use positive sentiment score as reward
        rewards = [s['score'] if s['label'] == 'POSITIVE' else 1 - s['score'] for s in scores]
        return rewards

reward_model = SentimentRewardModel()

# PPO config
ppo_config = PPOConfig(
    batch_size=4,
    learning_rate=1e-5,
    mini_batch_size=1,  # smaller batches for CPU
)

# Load dataset (200 samples already downloaded)
dataset = load_from_disk("./data/oasst1_small")
dataset = dataset.select(range(200))  # subset for quick iteration

# Format dataset into PPO format: {"query": "<text>"}
ppo_dataset = [{"query": sample["text"][:256]} for sample in dataset]


# Set up tensorboard logger
writer = SummaryWriter("logs/basic_rlhf")

# Create PPO trainer (trl >= 0.8)
ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=model,
    ref_model=ref_model,
    processing_class=tokenizer,
    reward_model=reward_model,
    train_dataset=ppo_dataset
)

# Run training
ppo_trainer.train()

writer.close()
