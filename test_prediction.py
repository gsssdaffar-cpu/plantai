from predict import predict_image
import sys


if len(sys.argv) < 2:
    print("Usage:")
    print("python test_prediction.py image_path")
    sys.exit(1)


image_path = sys.argv[1]

prediction, confidence, top_predictions, confidence_status = (
    predict_image(image_path)
)


print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Image      :", image_path)
print("Prediction :", prediction)
print(f"Confidence : {confidence:.2f}%")
print("Status     :", confidence_status)

print()
print("Top 5 predictions:")

for i, item in enumerate(top_predictions, 1):

    if isinstance(item, dict):

        name = item.get(
            "class",
            item.get(
                "name",
                item.get(
                    "prediction",
                    "Unknown"
                )
            )
        )

        probability = item.get(
            "confidence",
            item.get(
                "probability",
                0
            )
        )

    else:

        name = item[0]
        probability = item[1]

    print(
        f"{i}. {name} "
        f"({probability:.2f}%)"
    )


print("=" * 60)