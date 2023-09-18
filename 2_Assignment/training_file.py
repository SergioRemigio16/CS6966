import os
import torch
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor
from datasets import load_dataset
import random

# Set the random seed for same model output
torch.cuda.manual_seed_all(1217882)
# Set the random seed for the examples extracted
random.seed(1217882)

device = "cuda"

# Get the training dataset
dataset = load_dataset("jmhessel/newyorker_caption_contest", "explanation_from_pixels")
train_data = dataset['train']

# Randomly sample 5 training examples from the train_data
train_samples = random.sample(list(train_data), 2)
# Randomly sample 2 test examples from the train_data
test_samples = random.sample(list(train_data), 5)


checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

# Get the current working directory
current_directory = os.getcwd()

# We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
prompts = [
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[0]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[0]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[0]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[1]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[1]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[1]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        test_samples[0]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        test_samples[0]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: "
    ],
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[0]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[0]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[0]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[1]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[1]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[1]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        test_samples[1]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        test_samples[1]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: "
    ],
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[0]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[0]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[0]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[1]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[1]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[1]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        test_samples[2]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        test_samples[2]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: "
    ],
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[0]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[0]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[0]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[1]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[1]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[1]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        test_samples[3]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        test_samples[3]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: "
    ],
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[0]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[0]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[0]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        train_samples[1]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        train_samples[1]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: ",
        train_samples[1]['label'],
        "<end_of_utterance>",

        "\nUser: Look at this image.",
        test_samples[4]['image'],
        "<end_of_utterance>",
        "\nUser: Now read this caption about the image.",
        test_samples[4]['caption_choices'],
        "<end_of_utterance>",
        "\nUser: I don't understand why this caption is funny. Can you help me understand the joke?",
        "<end_of_utterance>",
        "\nAssistant: "
    ],

]

"""
# --batched mode
inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
"""
# --single sample mode
inputs = processor(prompts[0], return_tensors="pt").to(device)

# Generation args
exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
for i, t in enumerate(generated_text):
    print(f"{i}:\n{t}\n")
