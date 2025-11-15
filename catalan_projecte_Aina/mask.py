from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch


def load_model():
    """Load the Catalan RoBERTa model and tokenizer, return fill_mask pipeline."""
    # Using a different Catalan model that should be available
    # model_id = "BSC-LT/RoBERTa-ca" => requires Protobuf
    model_id = "PlanTL-GOB-ES/roberta-base-ca"

    # Force CPU (Use FP32 for CPU / FP16 for GPU)
    device = -1
    dtype = torch.float32

    # Load the tokenizer for preprocessing text
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the pre-trained model for masked language modeling
    model = AutoModelForMaskedLM.from_pretrained(model_id, dtype=dtype)

    # Create a pipeline that combines tokenizer and model for easy mask filling
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return fill_mask


def mask_fill(text, fill_mask_pipeline, top_k=5):
    """Fill masked tokens in the given text and return predictions."""

    if '<mask>' not in text:
        raise ValueError("Error: Please include <mask> token in your text")

    preds = fill_mask_pipeline(text, top_k=top_k)
    return preds


def format_mask_output(preds):
    """Format mask filling predictions for display."""
    for p in preds:
        print(f"  {p['sequence']}  (score={p['score']:.4f})")


def main(text=None, interactive=False):
    """Main function for mask filling experiment."""
    print("Loading model... (this may take a moment)")
    fill_mask_pipeline = load_model()

    if interactive:
        print("Interactive mask filling mode")
        print("Model loaded! Enter Catalan text with <mask> token (or 'quit' to exit):")

        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            try:
                # Reuse the pre-loaded pipeline
                preds = mask_fill(user_input, fill_mask_pipeline)
                print(f"\nPredictions for: {user_input}")
                format_mask_output(preds)
                print()
            except Exception as e:
                print(f"Error: {e}")
                print()
    else:
        # Default mode with provided text or default example
        if not text:
            # IMPORTANT: RoBERTa uses the token <mask>
            text = "La llengua catalana és <mask> bonica."

        print(f"Model loaded! Filling mask for: {text}")
        preds = mask_fill(text, fill_mask_pipeline)
        format_mask_output(preds)


if __name__ == "__main__":
    main()
