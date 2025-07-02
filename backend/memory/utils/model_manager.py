"""Model Manager
=============

Handles embedding model versioning, updates, and validation.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages embedding model lifecycle and versioning"""

    def __init__(self, config_path: str = "config/models.yaml"):
        """Initialize model manager with configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.models_dir = Path(self.config["embedding"]["versioning"]["versions_path"])
        self.current_symlink = Path(self.config["embedding"]["versioning"]["model_path"])
        self.golden_records = self._load_golden_records()

    def _load_config(self) -> dict:
        """Load model configuration"""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _load_golden_records(self) -> dict[str, list[float]]:
        """Load golden test records for validation"""
        golden_path = self.models_dir / "golden_records.json"
        if golden_path.exists():
            with open(golden_path) as f:
                return json.load(f)
        return {
            "test_identity": "My name is John Doe",
            "test_knowledge": "The capital of France is Paris",
            "test_temporal": "Tomorrow I have a meeting at 3pm",
        }

    def get_current_model(self) -> SentenceTransformer:
        """Load the current active model"""
        model_name = self.config["embedding"]["primary"]["name"]
        device = self.config["embedding"]["primary"]["device"]

        logger.info(f"Loading embedding model: {model_name}")

        # Check if we should use a local version via symlink
        if self.current_symlink.exists() and self.current_symlink.is_symlink():
            model_path = str(self.current_symlink.resolve())
            logger.info(f"Using local model via symlink: {model_path}")
        else:
            model_path = model_name

        model = SentenceTransformer(model_path, device=device)

        # Apply optimizations
        if self.config["optimization"]["use_fp16"] and device == "cuda":
            model = model.half()

        if self.config["optimization"]["compile_model"] and hasattr(torch, "compile"):
            model = torch.compile(model)

        return model

    def validate_model(self, model: SentenceTransformer) -> tuple[bool, dict[str, float]]:
        """Validate model against golden records"""
        results = {}
        all_valid = True

        for test_name, test_text in self.golden_records.items():
            # Generate embedding
            embedding = model.encode(test_text, convert_to_tensor=False).tolist()

            # If we have previous embeddings, check similarity
            golden_key = f"{test_name}_embedding"
            if golden_key in self.golden_records:
                golden_embedding = self.golden_records[golden_key]
                similarity = self._cosine_similarity(embedding, golden_embedding)
                results[test_name] = similarity

                # Flag if similarity is too low (potential breaking change)
                if similarity < 0.9:
                    logger.warning(f"Low similarity for {test_name}: {similarity}")
                    all_valid = False
            else:
                # First run, save as golden
                results[test_name] = 1.0

        return all_valid, results

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def install_new_model(self, model_name: str, version: str) -> bool:
        """Install a new model version"""
        try:
            # Create version directory
            version_dir = self.models_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)

            # Download and save model
            logger.info(f"Downloading model {model_name}...")
            model = SentenceTransformer(model_name)
            model.save(str(version_dir))

            # Validate before activation
            valid, results = self.validate_model(model)

            # Save validation results
            with open(version_dir / "validation_results.json", "w") as f:
                json.dump(
                    {"timestamp": datetime.now().isoformat(), "valid": valid, "results": results},
                    f,
                    indent=2,
                )

            if not valid:
                logger.error(f"Model validation failed for {model_name}")
                return False

            # Update symlink to new version
            if self.current_symlink.exists():
                self.current_symlink.unlink()
            self.current_symlink.symlink_to(version_dir)

            # Update config
            self.config["embedding"]["versioning"]["current_version"] = version
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f)

            logger.info(f"Successfully installed model version {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to install model: {e}")
            return False

    def rollback_model(self, version: str) -> bool:
        """Rollback to a previous model version"""
        version_dir = self.models_dir / version

        if not version_dir.exists():
            logger.error(f"Version {version} not found")
            return False

        try:
            # Update symlink
            if self.current_symlink.exists():
                self.current_symlink.unlink()
            self.current_symlink.symlink_to(version_dir)

            # Update config
            self.config["embedding"]["versioning"]["current_version"] = version
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f)

            logger.info(f"Rolled back to model version {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback model: {e}")
            return False

    def list_versions(self) -> list[dict[str, any]]:
        """List all available model versions"""
        versions = []

        for version_dir in self.models_dir.glob("*/"):
            if version_dir.is_dir():
                info = {"version": version_dir.name, "path": str(version_dir), "current": False}

                # Check if this is current version
                if self.current_symlink.exists():
                    current_target = self.current_symlink.resolve()
                    if current_target == version_dir.resolve():
                        info["current"] = True

                # Load validation results if available
                validation_file = version_dir / "validation_results.json"
                if validation_file.exists():
                    with open(validation_file) as f:
                        info["validation"] = json.load(f)

                versions.append(info)

        return sorted(versions, key=lambda x: x["version"], reverse=True)

    def compute_embedding_version_hash(self, embedding: list[float]) -> str:
        """Compute a hash for embedding version tracking"""
        # Use first 10 dimensions for version hash
        version_dims = embedding[:10] if len(embedding) >= 10 else embedding
        version_str = ",".join(f"{x:.6f}" for x in version_dims)
        return hashlib.md5(version_str.encode()).hexdigest()[:8]


# Singleton instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get singleton model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
