import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# will be overwritten
MODEL_CHECKPOINT = "facebook/m2m100_418M"
SOURCE_LANG = "en"
TARGET_LANG = "de"

# 2. Load the model using 8-bit quantization
# NOTE: bitsandbytes only supports CUDA devices.
if not torch.cuda.is_available():
    # print("Warning: CUDA not available. Falling back to CPU (8-bit loading will be ignored).")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    device = "cpu"
else:
    # This is the key change for memory efficiency:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_CHECKPOINT,
        load_in_8bit=True,  # <--- THIS IS THE ADAPTATION
        device_map="auto",  # Let accelerate handle device placement (essential for bnb)
    )
    # Note: If device_map="auto" is used, the model is already on the GPU(s).
    device = model.device  # This will correctly report the CUDA device.


def translate_8bit(text: str):
    # 1. Load the tokenizer normally
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, src_lang=SOURCE_LANG)

    # print(f"--- 2. Tokenizing and Preparing Input (Device: {device}) ---")

    # 3. Tokenize the input text and ensure tensors are on the correct device
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        device
    )  # Move tensors to the model's device

    print(encoded_input)

    # print(f"--- 3. Generating Translation ---")

    # 4. Generate the translation
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.get_lang_id(TARGET_LANG),
        max_length=512,
    )

    # 5. Decode the generated tokens
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translated_text


# --- Main execution ---
if __name__ == "__main__":
    try:
        MODEL_CHECKPOINT = "facebook/m2m100_418M"
        SOURCE_LANG = "en"
        TARGET_LANG = "de"

        input_text = """
M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many multilingual translation.
It was introduced in this paper and first released in this repository.

The model that can directly translate between the 9,900 directions of 100 languages.
To translate into a target language, the target language id is forced as the first generated token.
To force the target language id as the first generated token,
pass the forced_bos_token_id parameter to the generate method."""

        input_text = input_text.replace("\n", "")
        translation = translate_8bit(input_text)
        print("\n---        Input           ---", input_text)
        print("\n--- ✅ Translation Result ---", translation)

    except Exception as e:
        print("\n--- ❌ An error occurred during translation ---")
        print(f"Error: {e}")
        print("\nPossible solutions: Ensure CUDA drivers are up to date and `bitsandbytes` is installed correctly.")
