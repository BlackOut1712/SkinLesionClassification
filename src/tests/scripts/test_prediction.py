import numpy as np
from pathlib import Path

from src.model_loader import (
    load_model,
    load_class_names,
    validate_model
)
from src.validation import validate_image
from src.preprocessing import preprocess_image
from src.inference import predict


def main():

    model = load_model()
    class_names = load_class_names()
    validate_model(model, class_names)

    image_path = Path("dataset/HAM10000_images/ISIC_0024306.jpg")
    image_bytes = image_path.read_bytes()
    image = validate_image(image_bytes)

    image_tensor = preprocess_image(image)

    result = predict(
        model,
        image_tensor,
        class_names
    )

    prediction_idx = int(np.argmax(result))

    print(f"Predicted class: {class_names[prediction_idx]}")
    print(f"Softmax score: {result[class_names[prediction_idx]]:.2%}")

    print("\nClass probabilities:")

    for name, probability in result.items():
        print(f"{name}: {probability:.2%}")


if __name__ == "__main__":
    main()