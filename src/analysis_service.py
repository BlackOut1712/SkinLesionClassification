from src.validation import validate_image
from src.preprocessing import preprocess_image
from src.inference import predict


def analyze_image(
    image_bytes,
    model,
    class_names
) -> dict[str: float]:

    image = validate_image(image_bytes)

    image_tensor = preprocess_image(image)

    return predict(
        model,
        image_tensor,
        class_names
    )