import chainlit as cl
import torch
import tiktoken
from gptmodel import GPTModel
from gptdownload import download_and_load_gpt2
from main import load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text
import os

# Global variables to store the loaded model and tokenizer
model = None
tokenizer = None
device = None
model_config = None

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the model when a chat session starts.
    This ensures the model is loaded before the user can chat.
    """
    global model, tokenizer, device, model_config
    
    # Check if model is already loaded
    if model is not None and tokenizer is not None:
        await cl.Message(
            content="✅ Model is ready! You can start chatting.",
            author="System"
        ).send()
        return
    
    # Show loading message
    await cl.Message(
        content="🔄 Loading the GPT-2 model. This may take a moment...",
        author="System"
    ).send()
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration - using GPT-2 small for faster loading
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.0,
            "qkv_bias": True
        }
        
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }
        
        # Choose model size - can be changed via environment variable
        CHOOSE_MODEL = os.getenv("GPT_MODEL", "gpt2-small (124M)")
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
        model_config = BASE_CONFIG
        
        # Extract model size for downloading
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        
        # Download and load GPT-2 weights
        await cl.Message(
            content=f"📥 Downloading {CHOOSE_MODEL} weights (if not already cached)...",
            author="System"
        ).send()
        
        settings, params = download_and_load_gpt2(
            model_size=model_size,
            models_dir="gpt2"
        )
        
        # Initialize model
        await cl.Message(
            content="🔧 Initializing model architecture...",
            author="System"
        ).send()
        
        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.to(device)
        model.eval()
        
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Set random seed for reproducibility
        torch.manual_seed(123)
        
        # Success message
        await cl.Message(
            content=f"✅ Model loaded successfully! Using {CHOOSE_MODEL} on {device}. You can now start chatting!",
            author="System"
        ).send()
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        await cl.Message(
            content=error_msg,
            author="System"
        ).send()
        raise


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and generate responses.
    """
    global model, tokenizer, device, model_config
    
    if model is None or tokenizer is None:
        await cl.Message(
            content="❌ Model is not loaded yet. Please wait...",
            author="System"
        ).send()
        return
    
    # Get user input
    user_input = message.content
    
    # Show that we're generating
    msg = cl.Message(content="", author="Assistant")
    await msg.send()
    
    try:
        # Generate response
        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(user_input, tokenizer).to(device),
                max_new_tokens=100,  # Adjust as needed
                context_size=model_config["context_length"],
                top_k=50,
                temperature=1.0,
                eos_id=50256  # <|endoftext|> token
            )
        
        # Decode the response
        generated_text = token_ids_to_text(token_ids, tokenizer)
        
        # Extract only the generated part (remove the input prompt)
        response_text = generated_text[len(user_input):].strip()
        
        # Clean up the response (remove any remaining prompt artifacts)
        if "### Response:" in response_text:
            response_text = response_text.split("### Response:")[-1].strip()
        
        # Update the message with the response
        msg.content = response_text
        await msg.update()
        
    except Exception as e:
        error_msg = f"❌ Error generating response: {str(e)}"
        msg.content = error_msg
        await msg.update()

