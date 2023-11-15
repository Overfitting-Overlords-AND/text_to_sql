# from datasets import load_dataset
import sentencepiece as spm
import constants
import json
# dataset = load_dataset("roneneldan/TinyStories")

# Read the JSON file
with open('sql_create_context_v4.json', 'r') as json_file:
    data = json.load(json_file)

# Write to a text file
with open('sql_create_context_v4.txt', 'w') as text_file:
    for obj in data:
        for key, value in obj.items():
            text_file.write(f"{value}\n")
        text_file.write("\n")  # Add a newline to separate objects

# Train the SentencePiece model
spm.SentencePieceTrainer.train(
    input="./sql_create_context_v4.txt",
    model_prefix="text_to_sql",
    vocab_size=constants.VOCAB_SIZE,
    pad_id=3,
    input_sentence_size=constants.INPUT_SENTENCE_SIZE,
    shuffle_input_sentence=True
)
