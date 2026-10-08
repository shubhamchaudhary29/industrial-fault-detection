#!/usr/bin/env python3
"""
Industrial Fault Detection - CLI Prediction & Anomaly Highlighting
==================================================================
Usage:
    python predict.py --image path/to/image.png --model cnn [--output out.png]
    python predict.py --image path/to/image.png --model svm
"""

import os
import sys
import argparse
import cv2

from src.infer import FaultInferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="Run automated industrial surface defect inspection on a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input surface image file (PNG, JPG, BMP, etc.).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "svm"],
        help="Classifier model architecture to use ('cnn' or 'svm'). Default is 'cnn'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path for the annotated visualization image.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Target image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join("outputs", f"{base_name}_{args.model}_prediction.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize Engine
    engine = FaultInferenceEngine(model_type=args.model)
    result = engine.predict(args.image)

    # Save visualization
    cv2.imwrite(output_path, result["annotated_bgr"])

    # Terminal report
    print("\n" + "=" * 60)
    print("        INDUSTRIAL FAULT INSPECTION RESULT")
    print("=" * 60)
    print(f"  Input Image     : {args.image}")
    print(f"  Model Engine    : {args.model.upper()}")
    print(f"  Classification  : {result['label'].upper()}")
    print(f"  Confidence      : {result['confidence'] * 100:.2f}%")
    print(f"  Defect Score    : {result['prob_defective'] * 100:.2f}%")
    print(f"  Flaws Detected  : {result['num_defects_found']} region(s)")
    if result["num_defects_found"] > 0:
        for idx, (bx, by, bw, bh) in enumerate(result["bounding_boxes"]):
            print(f"    - Flaw #{idx + 1:<2}   : x={bx}, y={by}, w={bw}, h={bh}")
    print(f"  Annotated Output: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
