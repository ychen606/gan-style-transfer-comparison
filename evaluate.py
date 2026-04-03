import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# PATHS
TEST_DIR = "data/testA"
CYCLEGAN_DIR = "cyclegan_outputs/photo_to_painting"
MUNIT_DIR = "munit_outputs/photo_to_painting"
SCORE_FILE = "scores.txt"


def load_scores():
    if not os.path.exists(SCORE_FILE):
        return {"count": 0, "cyclegan_total": 0, "munit_total": 0}

    scores = {}
    with open(SCORE_FILE, "r") as f:
        for line in f:
            key, val = line.strip().split(":")
            scores[key.strip()] = float(val.strip())
    return scores


def save_scores(scores):
    with open(SCORE_FILE, "w") as f:
        for k, v in scores.items():
            f.write(f"{k}: {v}\n")


def display_images(test_img, cyc_img, munit_imgs):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(test_img)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(cyc_img)
    plt.title("CycleGAN")
    plt.axis("off")

    for i, img in enumerate(munit_imgs):
        plt.subplot(2, 3, 4 + i)
        plt.imshow(img)
        plt.title(f"MUNIT {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def get_random_image():
    image_names = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg"))]
    return random.choice(image_names)


def evaluate():
    scores = load_scores()

    name = get_random_image()
    base = name.replace(".jpg", "")

    test_path = os.path.join(TEST_DIR, name)
    cyc_path = os.path.join(CYCLEGAN_DIR, f"{base}_output.jpg")
    munit_paths = [
        os.path.join(MUNIT_DIR, f"{base}_style0.jpg"),
        os.path.join(MUNIT_DIR, f"{base}_style1.jpg"),
        os.path.join(MUNIT_DIR, f"{base}_style2.jpg")
    ]

    # check existence
    if not os.path.exists(cyc_path) or not all(os.path.exists(p) for p in munit_paths):
        print("Missing corresponding outputs, try running again.")
        return

    test_img = Image.open(test_path)
    cyc_img = Image.open(cyc_path)
    munit_imgs = [Image.open(p) for p in munit_paths]

    display_images(test_img, cyc_img, munit_imgs)

    try:
        cyc_score = float(input("Score for CycleGAN (0-10): "))
        munit_scores = []
        for i in range(3):
            s = float(input(f"Score for MUNIT style{i} (0-10): "))
            munit_scores.append(s)

        munit_avg = sum(munit_scores) / 3.0

    except:
        print("Invalid input, skipping...")
        return

    scores["count"] += 1
    scores["cyclegan_total"] += cyc_score
    scores["munit_total"] += munit_avg

    save_scores(scores)

    print(f"Saved. Total images scored: {scores['count']}")


if __name__ == "__main__":
    evaluate()