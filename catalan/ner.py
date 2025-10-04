from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch


def load_model():
    """Load the Catalan RoBERTa model for NER, return NER pipeline."""
    # Using the Catalan NER model
    model_id = "projecte-aina/roberta-base-ca-v2-cased-ner"

    # Force CPU usage due to CUDA compatibility issues with RTX 5070 Ti
    device = -1  # Force CPU

    # Use float32 for CPU
    dtype = torch.float32

    # Load the tokenizer for preprocessing text
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the pre-trained model for token classification (NER)
    model = AutoModelForTokenClassification.from_pretrained(model_id, dtype=dtype)

    # Create a pipeline that combines tokenizer and model for NER
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        aggregation_strategy="simple"  # Groups subword tokens together
    )

    return ner_pipeline


def ner_tag(text, ner_pipeline):
    """Perform NER on the given text and return predictions."""
    preds = ner_pipeline(text)
    return preds


def format_ner_output(preds):
    """Format NER predictions for display."""
    if not preds:
        print("No named entities found.")
        return

    for p in preds:
        word = p['word'].replace('▁', '').replace('##', '')  # Clean subword markers
        entity_type = p['entity_group']
        score = p['score']
        start = p.get('start', 'N/A')
        end = p.get('end', 'N/A')
        print(f"{word:20} → {entity_type:10} (score={score:.2f}) [{start}-{end}]")


def main(text=None, interactive=False):
    """Main function for NER experiment."""
    print("Loading model... (this may take a moment)")
    ner_pipeline = load_model()

    if interactive:
        print("Interactive NER mode")
        print("Model loaded! Enter Catalan text for NER (or 'quit' to exit):")

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
                preds = ner_tag(user_input, ner_pipeline)
                print(f"\nNamed entities in: {user_input}")
                format_ner_output(preds)
                print()
            except Exception as e:
                print(f"Error: {e}")
                print()
    else:
        # Default mode with provided text or default example
        if text is None:
            text = "Joan Miró va néixer a Barcelona el 1893 i va estudiar a l'Escola de Belles Arts."

        print(f"Model loaded! NER analysis for: {text}")
        preds = ner_tag(text, ner_pipeline)
        format_ner_output(preds)
if __name__ == "__main__":
    main()