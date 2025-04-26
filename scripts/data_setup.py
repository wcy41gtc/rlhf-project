from datasets import load_dataset

dataset = load_dataset("OpenAssistant/oasst1", split='train[:10000]')  # first 1000 lines as a small subset
dataset.save_to_disk("./data/oasst1_small")
