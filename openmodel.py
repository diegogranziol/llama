from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

# Define model parameters
model_args = ModelArgs(
    dim=4096,  # Example dimension
    n_layers=32,  # Number of transformer layers
    n_heads=32,  # Number of attention heads
    vocab_size=50257,  # Adjust based on your tokenizer
    max_seq_len=2048,  # Maximum sequence length
    max_batch_size=1,  # Maximum batch size for inference
    # Add other parameters as needed
)
print('model loaded')
# Initialize the model with random weights
model = Transformer(model_args)

# Initialize the tokenizer (adjust the path as needed)
tokenizer_path = "/jmain02/home/J2AD008/wga41/dxg49-wga41/llama/CodeLlama-13b/tokenizer.model"
tokenizer = Tokenizer(model_path=tokenizer_path)

print('tokenizer loaded')