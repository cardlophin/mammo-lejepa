from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.pixels.processing import apply_windowing

DICOM_PATH = Path(
    "/Users/more/Desktop/projects/mammo-lejepa/"
    "data/vindr-mammo/images/"
    "0028fb2c7f0b3a5cb9a80cb0e1cdbb91/"
    "7fc1f1bb8bb1a7efaf7104e49c4d8b86.dicom"
)

OUTPUT_DIR = Path("data/vindr-mammo/breast_bounding_boxes")

# Margen extra alrededor de la bounding box de mama.
CROP_MARGIN = 25


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen DICOM de 12/14/16 bits a uint8 usando
    percentiles para reducir el efecto de valores extremos.
    """
    image = image.astype(np.float32)

    low, high = np.percentile(image, [0.5, 99.5])

    if high <= low:
        low = float(image.min())
        high = float(image.max())

    normalized = (image - low) / (high - low + 1e-8)
    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255).astype(np.uint8)


def load_dicom_image(path: Path) -> tuple[pydicom.Dataset, np.ndarray]:
    """
    Lee el DICOM y prepara una imagen uint8 para OpenCV.

    La ventana DICOM sirve para mejorar la presentación visual.
    MONOCHROME1 se invierte para tener la misma convención de
    intensidad que MONOCHROME2.
    """
    ds = pydicom.dcmread(path)

    image = ds.pixel_array.astype(np.float32)

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        image = apply_windowing(image, ds, index=0)

    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = image.max() - image

    return ds, normalize_uint8(image)


def largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Conserva el componente conectado con mayor área.

    Normalmente corresponde a la mama. Elimina automáticamente texto
    DICOM, letras de orientación, pequeños marcadores y ruido.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    if num_labels <= 1:
        raise RuntimeError("No se detectó ningún componente conectado de mama.")

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    largest_mask = np.zeros_like(binary_mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask


def get_breast_bounding_box(
    image_uint8: np.ndarray,
    margin: int = CROP_MARGIN,
) -> tuple[tuple[int, int, int, int], float]:
    """
    Obtiene la bounding box del campo mamario.

    Devuelve:
        - (x_min, y_min, x_max, y_max)
        - umbral Otsu seleccionado

    No devuelve una segmentación final: la máscara binaria se usa sólo
    internamente para estimar la bounding box.
    """
    height, width = image_uint8.shape

    blurred = cv2.GaussianBlur(
        image_uint8,
        ksize=(5, 5),
        sigmaX=0,
    )

    otsu_threshold, binary_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Operación suave para conectar pequeñas discontinuidades del borde
    # sin intentar una segmentación anatómica precisa.
    kernel_size = max(9, int(min(height, width) * 0.006))
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    cleaned_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    breast_component = largest_connected_component(cleaned_mask)

    breast_points = cv2.findNonZero(breast_component)

    if breast_points is None:
        raise RuntimeError("No se encontraron píxeles para calcular la bounding box.")

    x, y, box_width, box_height = cv2.boundingRect(breast_points)

    x_min = max(0, x - margin)
    y_min = max(0, y - margin)

    x_max = min(width, x + box_width + margin)
    y_max = min(height, y + box_height + margin)

    return (x_min, y_min, x_max, y_max), otsu_threshold


def crop_from_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Recorta una imagen usando el formato:

    (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = box

    return image[y_min:y_max, x_min:x_max]


def draw_bounding_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Devuelve una imagen RGB con la bounding box roja dibujada.
    """
    x_min, y_min, x_max, y_max = box

    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    thickness = max(3, image.shape[0] // 600)

    cv2.rectangle(
        output,
        pt1=(x_min, y_min),
        pt2=(x_max, y_max),
        color=(255, 0, 0),
        thickness=thickness,
    )

    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds, image = load_dicom_image(DICOM_PATH)

    breast_box, otsu_threshold = get_breast_bounding_box(
        image,
        margin=CROP_MARGIN,
    )

    cropped_breast = crop_from_box(image, breast_box)
    boxed_image = draw_bounding_box(image, breast_box)

    x_min, y_min, x_max, y_max = breast_box

    box_width = x_max - x_min
    box_height = y_max - y_min

    output_path = OUTPUT_DIR / (f"{DICOM_PATH.stem}_breast_crop.png")

    overlay_path = OUTPUT_DIR / (f"{DICOM_PATH.stem}_breast_bbox.png")

    cv2.imwrite(output_path, cropped_breast)

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(boxed_image, cv2.COLOR_RGB2BGR),
    )

    laterality = ds.get(
        "ImageLaterality",
        ds.get("Laterality", "?"),
    )

    view_position = ds.get("ViewPosition", "?")

    print(f"DICOM: {DICOM_PATH.name}")
    print(f"Vista: {laterality}-{view_position}")
    print(f"Forma original: {image.shape}")
    print(f"Umbral Otsu: {otsu_threshold:.2f}")
    print()
    print(f"Bounding box de mama (x_min, y_min, x_max, y_max): {breast_box}")
    print(f"Anchura de bounding box: {box_width} px")
    print(f"Altura de bounding box: {box_height} px")
    print(f"Forma del crop: {cropped_breast.shape}")
    print()
    print(f"Imagen con bbox: {overlay_path}")
    print(f"Crop de mama: {output_path}")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 10),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[0].set_title(f"DICOM original\n{laterality}-{view_position}")

    axes[1].imshow(boxed_image)
    axes[1].set_title("Bounding box del campo mamario")

    axes[2].imshow(
        cropped_breast,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[2].set_title(f"Crop de mama\n{box_width} × {box_height}")

    for axis in axes:
        axis.axis("off")

    plt.suptitle(
        f"Bounding box de mama — {DICOM_PATH.name}",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
