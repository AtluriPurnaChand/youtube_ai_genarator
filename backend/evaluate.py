import os
import glob
import logging
import base64
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import json

from analyzer import analyze_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loaders (Stubs for future integration)
# ──────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_samples(self) -> List[Tuple[List[str], str]]:
        """
        Base pattern.
        Returns: A list of tuples, where each tuple is:
        ([base64_frame_1, base64_frame_2, ...], "true_label")
        """
        raise NotImplementedError("Implement in subclass")


class FaceForensicsLoader(DatasetLoader):
    """
    Loader for FaceForensics++
    Contains original (real) and manipulated (fake) videos.
    """
    def load_samples(self):
        logger.info(f"Loading FaceForensics++ from {self.dataset_path}")
        # Build paths to Original and Manipulated folders
        # Read frames -> base64
        # map Original -> 'real_video', Manipulated -> 'deepfake_detected'
        return []


class DFDCLoader(DatasetLoader):
    """
    Loader for the Deepfake Detection Challenge Dataset.
    """
    def load_samples(self):
        logger.info(f"Loading DFDC from {self.dataset_path}")
        return []


class CelebDFLoader(DatasetLoader):
    """
    Loader for Celeb-DF dataset.
    """
    def load_samples(self):
        logger.info(f"Loading Celeb-DF from {self.dataset_path}")
        return []


class DeeperForensicsLoader(DatasetLoader):
    """
    Loader for DeeperForensics-1.0.
    """
    def load_samples(self):
        logger.info(f"Loading DeeperForensics from {self.dataset_path}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Engine
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, loader: DatasetLoader):
        self.loader = loader
        self.y_true = []
        self.y_pred = []
        self.classes = ["real_video", "ai_generated", "cartoon_animation", "video_game", "deepfake_detected"]

    def run_evaluation(self):
        """Runs the pipeline on the dataset and collects predictions."""
        samples = self.loader.load_samples()
        if not samples:
            logger.warning("Dataset contains no samples! (Or loader is not fully implemented)")
            return

        for frames, true_label in samples:
            self.y_true.append(true_label)
            try:
                # We analyze a batch of frames to leverage the backend's temporal aggregation
                result = analyze_batch(frames, metadata={})
                predicted_label = result.get("type", "error")
                self.y_pred.append(predicted_label)
            except Exception as e:
                logger.error(f"Error analyzing batch: {e}")
                self.y_pred.append("error")

    def print_metrics(self):
        """Calculates and prints Precision, Recall, F1, and Confusion Matrix."""
        if not self.y_true:
            logger.info("No data to evaluate.")
            return

        logger.info("--- Evaluation Results ---")

        # Basic Accuracy
        acc = accuracy_score(self.y_true, self.y_pred)
        logger.info(f"Accuracy: {acc:.4f}")

        # Filter out 'error' predictions logically if desired, mostly we treat them as misclassifications
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted', labels=self.classes, zero_division=0
        )
        logger.info(f"Weighted Precision: {precision:.4f}")
        logger.info(f"Weighted Recall:    {recall:.4f}")
        logger.info(f"Weighted F1 Score:  {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)
        logger.info("\nConfusion Matrix:")
        border = "-" * 85
        logger.info(border)
        header = f"{'True \\ Pred':>18} | " + " | ".join([f"{c[:10]:>10}" for c in self.classes])
        logger.info(header)
        logger.info(border)
        for i, row in enumerate(cm):
            row_str = " | ".join([f"{val:10d}" for val in row])
            logger.info(f"{self.classes[i]:>18} | {row_str}")
        logger.info(border)

def run_mock_test():
    """Generates mock evaluation metrics without loading actual images."""
    logger.info("Running Mock Test (Testing metric computation logic)")
    evaluator = Evaluator(None)
    
    # Mock some data indicating a highly accurate system with a few missed deepfakes
    evaluator.y_true = ["real_video"]*50 + ["deepfake_detected"]*50
    evaluator.y_pred = ["real_video"]*48 + ["ai_generated"]*2 + ["deepfake_detected"]*45 + ["real_video"]*5
    
    evaluator.print_metrics()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate AI Media Detector")
    parser.add_argument("--dataset", type=str, choices=["faceforensics", "dfdc", "celebdf", "deeperforensics"])
    parser.add_argument("--path", type=str, help="Path to the dataset directory")
    parser.add_argument("--test-mock", action="store_true", help="Run a mock test of the evaluation metrics")

    args = parser.parse_args()

    if args.test_mock:
        run_mock_test()
    elif args.dataset and args.path:
        if args.dataset == "faceforensics":
            loader = FaceForensicsLoader(args.path)
        elif args.dataset == "dfdc":
            loader = DFDCLoader(args.path)
        elif args.dataset == "celebdf":
            loader = CelebDFLoader(args.path)
        elif args.dataset == "deeperforensics":
            loader = DeeperForensicsLoader(args.path)
        
        evaluator = Evaluator(loader)
        evaluator.run_evaluation()
        evaluator.print_metrics()
    else:
        logger.info("Please specify --dataset and --path, or --test-mock. See --help for more.")
