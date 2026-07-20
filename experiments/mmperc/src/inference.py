import sys

from decode_a2d2.bbox import ModelInferenceWrapper

import common.params as params

if __name__ == "__main__":
    print("Testing ModelInferenceWrapper instantiation...")
    try:
        model_inference_wrapper = ModelInferenceWrapper()
        print("ModelInferenceWrapper instantiated successfully.")

        results = model_inference_wrapper.infer_a2d2_dataset(
            params.PATH_VALID, "/workspace/mmperc/data/a2d2_output.npz"
        )
        print(results)

    except Exception as e:
        print(f"Error instantiating ModelInferenceWrapper: {e}")
        sys.exit(1)
