import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Callable, Optional

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_generator.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FeatureGroup:
    """Configuration for feature groups used in data generation."""
    name: str
    features: List[str]
    asd_probability_range: Tuple[float, float]
    non_asd_probability_range: Tuple[float, float]
    key_features: Dict[str, float] = field(default_factory=dict)
    missingness_probability: float = 0.01
    feature_dependencies: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interaction_effects: Dict[str, Callable[[Dict[str, int]], float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate feature group settings."""
        if not (0 <= self.asd_probability_range[0] <= self.asd_probability_range[1] <= 1):
            raise ValueError(f"Invalid ASD probability range for {self.name}")
        if not (0 <= self.non_asd_probability_range[0] <= self.non_asd_probability_range[1] <= 1):
            raise ValueError(f"Invalid non-ASD probability range for {self.name}")
        if not (0 <= self.missingness_probability <= 1):
            raise ValueError(f"Invalid missingness probability for {self.name}")

@dataclass
class DatasetConfig:
    """Configuration for the dataset generation process."""
    num_samples: int
    asd_prevalence: float
    random_seed: int
    feature_groups: List[FeatureGroup]
    demographics: Dict[str, Any]
    output_dir: str
    output_formats: List[str]
    validation_metrics: Dict[str, float]
    advanced_configs: Optional[Dict[str, Any]] = None

    def validate(self):
        """Validate the configuration settings."""
        if not (100 <= self.num_samples <= 1_000_000):
            raise ValueError("Sample size must be between 100 and 1,000,000.")
        if not (0.01 <= self.asd_prevalence <= 0.5):
            raise ValueError("ASD prevalence must be between 1% and 50%.")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

def create_default_config() -> DatasetConfig:
    """Create a default configuration for the dataset generation."""
    return DatasetConfig(
        num_samples=30000,
        asd_prevalence=0.2,
        random_seed=42,
        feature_groups=[
            FeatureGroup(
                name="Social_Interaction",
                features=[f"A{i}_Score" for i in range(1, 11)],
                asd_probability_range=(0.6, 0.85),
                non_asd_probability_range=(0.1, 0.3),
                key_features={"A5_Score": 0.15, "A8_Score": 0.2}
            ),
            # Additional feature groups can be added here
        ],
        demographics={
            "age": {
                "min": 10,
                "max": 36,
                "mean": 24,
                "std": 6
            },
            "gender": {
                "m": 0.8,
                "f": 0.2
            },
            "country": {
                "USA": 0.28,
                "Canada": 0.0152,
                "UK": 0.0175,
                "Ireland": 0.05,
                "Japan": 0.0303,
                "Russia": 0.01,
            },
            "reporter": {
                "Parent": 0.7,
                "Health Care Professional": 0.25,
                "Relative": 0.05
            }
        },
        output_dir="./generated_data",
        output_formats=["csv", "parquet"],
        validation_metrics={
            "max_class_ratio_diff": 0.1,
            "max_missingness": 0.05
        }
    )
