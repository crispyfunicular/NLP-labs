from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch


def load_model():
    """Load the Catalan RoBERTa model for POS tagging, return POS pipeline."""
    # Using the actual Catalan POS tagging model
    model_id = "projecte-aina/roberta-base-ca-v2-cased-pos"

    # Force CPU usage due to CUDA compatibility issues with RTX 5070 Ti
    device = -1  # Force CPU

    # Use float32 for CPU
    dtype = torch.float32

    # Load the tokenizer for preprocessing text
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the pre-trained model for token classification (POS tagging)
    model = AutoModelForTokenClassification.from_pretrained(model_id, dtype=dtype)

    # Create a pipeline that combines tokenizer and model for POS tagging
    pos_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        aggregation_strategy="simple"  # Groups subword tokens together
    )

    return pos_pipeline


def pos_tag(text, pos_pipeline):
    """Perform POS tagging on the given text and return predictions."""
    preds = pos_pipeline(text)
    return preds


def format_pos_output(preds):
    """Format POS tagging predictions for display."""
    for p in preds:
        word = p['word'].replace('▁', '').replace('##', '')  # Clean subword markers
        print(f"{word:15} → {p['entity_group']} (score={p['score']:.2f})")


def main(text=None, interactive=False):
    """Main function for POS tagging experiment."""
    print("Loading model... (this may take a moment)")
    pos_pipeline = load_model()

    if interactive:
        print("Interactive POS tagging mode")
        print("Model loaded! Enter Catalan text for POS tagging (or 'quit' to exit):")

        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                print("Error: Please enter some text")
                continue

            try:
                # Reuse the pre-loaded pipeline
                preds = pos_tag(user_input, pos_pipeline)
                print(f"\nPOS tags for: {user_input}")
                format_pos_output(preds)
                print()
            except Exception as e:
                print(f"Error: {e}")
                print()
    else:
        # Default mode with provided text or default example
        if text is None:
            text = "El gat negre dorm tranquil·lament."

        print(f"Model loaded! POS tagging for: {text}")
        preds = pos_tag(text, pos_pipeline)
        format_pos_output(preds)
if __name__ == "__main__":
    main()