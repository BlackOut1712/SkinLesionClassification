import cv2
import matplotlib.pyplot as plt

from pathlib import Path

from src.gradcam import build_gradcam
from src.services.analysis_service import analyze_image, explain_prediction, prepare_gradcam_visualization
from src.model_loader import load_model, load_class_names, validate_model

def main():

    model = load_model()
    class_names = load_class_names()
    validate_model(model, class_names)

    image_path = Path("dataset/HAM10000_images/ISIC_0024306.jpg")
    image_bytes = image_path.read_bytes()

    result = analyze_image(
        image_bytes=image_bytes,
        model=model,
        class_names=class_names
    )

    gradcam = build_gradcam(model)

    heatmap = explain_prediction(
        result=result,
        gradcam=gradcam,
        target_class=result.predicted_class,
        class_names=class_names
    )

    colored_heatmap, overlay = (
        prepare_gradcam_visualization(
            image=result.image,
            heatmap=heatmap,
            alpha=0.3,
            colormap=cv2.COLORMAP_JET
        )
    )

    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()