import cv2
import os
import glob

def images_to_mp4(image_folder, output_path, fps=30):
    image_paths = sorted(
        p for p in glob.glob(os.path.join(image_folder, "*"))
        if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )[::3]

    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")

    first_image = cv2.imread(image_paths[0])

    if first_image is None:
        raise ValueError(f"Cannot read {image_paths[0]}")

    height, width = first_image.shape[:2]

    # Ensure the output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # Try different codecs
    video = None

    for codec in ["mp4v", "avc1"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height)
        )

        if candidate.isOpened():
            video = candidate
            print(f"Using codec: {codec}")
            break

        candidate.release()

        # Remove potentially incomplete output before trying again
        if os.path.exists(output_path):
            os.remove(output_path)

    if video is None:
        raise RuntimeError(
            "Could not initialize VideoWriter. "
            "Try the FFmpeg solution below."
        )

    try:
        for path in image_paths:
            image = cv2.imread(path)

            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue

            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))

            video.write(image)
    finally:
        video.release()

    print(f"Saved {len(image_paths)} frames to {output_path}")


# Example usage
images_to_mp4(
    image_folder="/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/renderseq10",
    output_path="/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/renderseq10/rendervid.mp4",
    fps=10
)