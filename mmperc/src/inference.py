import sys

from decoder.bbox import ModelInferenceWrapper

if __name__ == "__main__":
    print("Testing ModelInferenceWrapper instantiation...")
    try:
        model_inference_wrapper = ModelInferenceWrapper()
        print("ModelInferenceWrapper instantiated successfully.")

        results = model_inference_wrapper.infer_a2d2_dataset("/workspace/mmperc/data/a2d2")
        print(results)

    except Exception as e:
        print(f"Error instantiating ModelInferenceWrapper: {e}")
        sys.exit(1)
