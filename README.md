# Diego LLM - GPT-2 Chatbot

A GPT-2 chatbot implementation from scratch with a Chainlit web interface.

## Features

- GPT-2 model implementation from scratch
- Interactive web UI using Chainlit
- Automatic model loading on startup
- Support for multiple GPT-2 model sizes (124M, 355M, 774M, 1558M)
- GPU support (CUDA) when available

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Chatbot

### Using the provided scripts:

**Windows:**
```bash
run_chatbot.bat
```

**Linux/Mac:**
```bash
chmod +x run_chatbot.sh
./run_chatbot.sh
```

### Manual start:

```bash
chainlit run app.py --port 8001
```

The chatbot will be available at `http://localhost:8001`

## Model Configuration

By default, the chatbot uses GPT-2 Small (124M parameters) for faster loading. You can change the model size by setting the `GPT_MODEL` environment variable:

```bash
# Windows
set GPT_MODEL=gpt2-medium (355M)
chainlit run app.py --port 8001

# Linux/Mac
export GPT_MODEL=gpt2-medium (355M)
chainlit run app.py --port 8001
```

Available models:
- `gpt2-small (124M)` - Default, fastest loading
- `gpt2-medium (355M)`
- `gpt2-large (774M)`
- `gpt2-xl (1558M)`

## How It Works

1. **Model Loading**: When you start a chat session, the model is automatically downloaded (if not already cached) and loaded into memory. You'll see loading messages until the model is ready.

2. **Chat Interface**: Once loaded, you can start chatting with the GPT-2 model. The model generates responses based on your input.

3. **Generation Parameters**: The chatbot uses:
   - `max_new_tokens`: 100
   - `top_k`: 50
   - `temperature`: 1.0
   - `context_length`: 1024

## Project Structure

- `app.py` - Chainlit application with chat interface
- `main.py` - Main training and model code
- `gptmodel.py` - GPT model architecture
- `gptdownload.py` - Model weight downloading utilities
- `.chainlit/config.toml` - Chainlit configuration

## Notes

- The first run will download the model weights, which may take some time depending on your internet connection.
- Model weights are cached in the `gpt2/` directory after the first download.
- GPU support is automatically detected and used if available.

