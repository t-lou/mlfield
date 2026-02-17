from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLO26n model
    model = YOLO("yolo26n.pt")

    # Perform object detection on an image
    path = Path("home-img-top-2000x1333-new.jpg")

    if path.is_file():
        results = model("home-img-top-2000x1333-new.jpg")  # Predict on an image
        # results[0].show()  # Display results

        for idx, result in enumerate(results):
            print(idx, type(result), dir(result))  # Print the type and available attributes of the result object
            for k, v in result.__dict__.items():
                print(f"  {k}: {type(v)}")  # Print each attribute and its type

        results[0].save("output.jpg")  # Save results to a file

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model
