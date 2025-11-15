import ctranslate2
import pyonmttok
from huggingface_hub import snapshot_download


def load_model(combination):

    """Load Aina MT model."""
    model_dir = snapshot_download(repo_id=f"projecte-aina/aina-translator-{combination}", revision="main")
    tokenizer = pyonmttok.Tokenizer(mode="none", sp_model_path = model_dir + "/spm.model")
    translator = ctranslate2.Translator(model_dir)

    return tokenizer, translator


def translate(user_input, tokenizer, translator):
    tokenized = tokenizer.tokenize(user_input)
    translated = translator.translate_batch([tokenized[0]])
    return tokenizer.detokenize(translated[0][0]['tokens'])


def main(text=None, interactive=False, combination="fr-ca"):
    print("Loading model... (this may take a moment)")
    tokenizer, translator = load_model(combination)

    if interactive:
        print("Enter text to be translated (or 'quit' to exit):")

        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Adéu!")
                break
            if not user_input:
                print("Error: Please enter some text")
                continue

            try:
                print(f"Translation for: {user_input}")
                translation = translate(user_input, tokenizer, translator)
                print(translation)
            except Exception as e:
                print(f"Error: {e}")
                print()
    
    else:
        # Default mode with provided text or default example
        if text is None:
            text = "Bienvenue en Catalogne !"

        print(f"Translation for: {text}")
        translation = translate(text, tokenizer, translator)
        print(translation)


if __name__ == "__main__":
    main()