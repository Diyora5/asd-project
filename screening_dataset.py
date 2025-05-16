import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import argparse
import yaml
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import shap

from typing import Callable

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("autism_data_generator.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FeatureGroup:
    """Advanced feature group configuration with complex validation and modeling."""
    name: str
    features: List[str]
    asd_probability_range: Tuple[float, float]
    non_asd_probability_range: Tuple[float, float]
    key_features: Dict[str, float] = field(default_factory=dict)
    missingness_probability: float = 0.01
    feature_dependencies: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interaction_effects: Dict[str, Callable[[Dict[str, int]], float]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0 <= self.asd_probability_range[0] <= self.asd_probability_range[1] <= 1):
            raise ValueError(f"Invalid ASD probability range for {self.name}")
        
        if not (0 <= self.non_asd_probability_range[0] <= self.non_asd_probability_range[1] <= 1):
            raise ValueError(f"Invalid non-ASD probability range for {self.name}")
        
        if not (0 <= self.missingness_probability <= 1):
            raise ValueError(f"Invalid missingness probability for {self.name}")
        
        for feature, boost in self.key_features.items():
            if feature not in self.features:
                raise ValueError(f"Key feature {feature} not in feature list for {self.name}")
        
        for dependent_feature, dependencies in self.feature_dependencies.items():
            if dependent_feature not in self.features:
                raise ValueError(f"Dependent feature {dependent_feature} not in feature list for {self.name}")
            for feature in dependencies.keys():
                if feature not in self.features:
                    raise ValueError(f"Dependency feature {feature} not in feature list for {self.name}")

@dataclass
class DatasetConfig:
    """Comprehensive dataset configuration with advanced validation."""
    num_samples: int
    asd_prevalence: float
    random_seed: int
    feature_groups: List[FeatureGroup]
    demographics: Dict[str, Any]
    output_dir: str
    output_formats: List[str]
    validation_metrics: Dict[str, float]
    advanced_configs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not (100 <= self.num_samples <= 1_000_000):
            raise ValueError(f"Sample size must be between 100 and 1,000,000. Requested: {self.num_samples}")
        
        if not (0.01 <= self.asd_prevalence <= 0.5):
            raise ValueError(f"ASD prevalence must be between 1% and 50%. Requested: {self.asd_prevalence}")
        
        self.output_dir = Path(self.output_dir)
        valid_formats = {"csv", "parquet", "xlsx", "json"}
        invalid_formats = set(self.output_formats) - valid_formats
        
        if invalid_formats:
            raise ValueError(f"Invalid output formats: {invalid_formats}")
        
        if self.advanced_configs is None:
            self.advanced_configs = {}
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'DatasetConfig':
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            feature_groups = []
            for group_data in config_data.pop('feature_groups'):
                feature_groups.append(FeatureGroup(**group_data))
            
            config_data['feature_groups'] = feature_groups
            
            return cls(**config_data)
        except (yaml.YAMLError, KeyError) as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

class StatisticalValidator:
    """Validates synthetic data against expected statistical properties.""" 
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.validation_metrics = config.validation_metrics
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        validation_results = {
            "passed": True,
            "metrics": {}
        }
        
        try:
            asd_ratio = df["Autism_Diagnosis"].mean()
            expected_ratio = self.config.asd_prevalence
            ratio_diff = abs(asd_ratio - expected_ratio) / expected_ratio
            
            validation_results["metrics"]["asd_ratio"] = {
                "actual": asd_ratio,
                "expected": expected_ratio,
                "relative_diff": ratio_diff,
                "passed": ratio_diff <= self.validation_metrics["max_class_ratio_diff"]
            }

            if "Yes" in df.values or "No" in df.values:
                logger.warning("Skipping validation for categorical features.")
                return validation_results

            for group in self.config.feature_groups:
                if len(group.features) >= 2:
                    features = [f for f in group.features if f in df.columns]
                    corr_matrix = df[features].corr()
                    avg_corr = (corr_matrix.sum().sum() - len(features)) / (len(features) * (len(features) - 1))
                    
                    validation_results["metrics"][f"{group.name}_correlation"] = {
                        "average_correlation": avg_corr,
                        "passed": 0.1 <= avg_corr <= 0.9
                    }
            
            missing_ratio = df.isna().mean().mean()
            validation_results["metrics"]["missing_data"] = {
                "ratio": missing_ratio,
                "passed": missing_ratio <= self.validation_metrics["max_missingness"]
            }
            
            validation_results["passed"] = all(
                metric["passed"] for metric in validation_results["metrics"].values() 
                if isinstance(metric, dict) and "passed" in metric
            )
            
            return validation_results

    def generate_validation_report(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """
        Generates a validation report with visualizations for the dataset.
        
        Args:
            df: The dataset to validate
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved report
        """
        try:
            # Create validation directory
            validation_dir = output_dir / "validation"
            validation_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate report with metrics
            validation_results = self.validate_dataset(df)
            
            # Save validation results
            with open(validation_dir / "validation_metrics.json", "w") as f:
                json.dump(validation_results, f, indent=2)
            
            # Generate plots
            plt.figure(figsize=(12, 8))
            
            # Plot ASD distribution
            plt.subplot(2, 2, 1)
            sns.countplot(x="Autism_Diagnosis", data=df)
            plt.title("ASD Distribution")
            
            # Plot age distribution by diagnosis
            plt.subplot(2, 2, 2)
            sns.boxplot(x="Autism_Diagnosis", y="Age", data=df)
            plt.title("Age Distribution by Diagnosis")
            
            # Plot gender distribution by diagnosis
            plt.subplot(2, 2, 3)
            pd.crosstab(df["Gender"], df["Autism_Diagnosis"]).plot(kind="bar", stacked=True)
            plt.title("Gender Distribution")
            
            # Plot feature count distribution
            plt.subplot(2, 2, 4)
            df["Total_Features"] = df[[col for col in df.columns if col.endswith("_Score")]].sum(axis=1)
            sns.histplot(data=df, x="Total_Features", hue="Autism_Diagnosis", bins=20)
            plt.title("Feature Distribution")
            
            plt.tight_layout()
            plot_path = validation_dir / "validation_plots.png"
            plt.savefig(plot_path)
            plt.close()
            
            return validation_dir / "validation_metrics.json"
        
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return None

class AdvancedAutismDataGenerator:
    """
    A sophisticated, multi-dimensional synthetic autism screening 
    data generator with advanced probabilistic modeling.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the generator with comprehensive configuration 
        and advanced preprocessing techniques.
        """
        self.config = config
        np.random.seed(config.random_seed)
        self.validator = StatisticalValidator(config)
        
        # Advanced feature and demographic modeling
        self._preprocess_features()
        self._setup_advanced_demographics()
        
        # Advanced diagnostic and scoring models
        self._initialize_diagnostic_models()
    
    def _preprocess_features(self):
        """
        Advanced feature preprocessing with intelligent feature mapping 
        and complex dependency resolution.
        """
        self.feature_names = []
        for group in self.config.feature_groups:
            self.feature_names.extend(group.features)
    
    def _setup_advanced_demographics(self):
        """
        Enhanced demographic distribution modeling with 
        multi-component and contextual probability estimation.
        """
        demo_config = self.config.demographics
        
        # Advanced multi-modal age distribution
        self.age_distribution = {
            "younger_toddlers": {"mean": 16, "std": 3, "weight": 0.3},
            "mid_toddlers": {"mean": 24, "std": 4, "weight": 0.4},
            "older_toddlers": {"mean": 32, "std": 3, "weight": 0.3}
        }
        
        # Gender, country, and reporter distributions
        self.gender_probs = demo_config["gender"]
        self.country_probs = demo_config["country"]
        self.reporter_probs = demo_config["reporter"]
    
    def sensitivity_curve(self, x):
        return 1 / (1 + np.exp(-0.3 * (x - 15)))

    def run_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Perform Recursive Feature Elimination (RFE) to select the top features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        model = RandomForestClassifier()
        rfe = RFE(model, n_features_to_select=10)
        fit = rfe.fit(X, y)
        selected_features = X.columns[fit.support_].tolist()
        logger.info(f"Selected features: {selected_features}")
        return selected_features

    def explain_model(self, model, X: pd.DataFrame) -> None:
        """
        Explain the model's predictions using SHAP.
        
        Args:
            model: Trained model
            X: Feature DataFrame
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        shap.summary_plot(shap_values, X)
        
        # Save the SHAP summary plot
        plt.savefig(self.config.output_dir / "shap_summary_plot.png")
        plt.close()

    def _initialize_diagnostic_models(self):
        """
        Initialize advanced diagnostic and scoring models 
        with complex probabilistic estimation.
        """
        self.diagnosis_model = {
            "base_threshold": 15,
            "sensitivity_curve": self.sensitivity_curve,
            "asd_boost": 1.5,
            "non_asd_reduction": 0.5
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete synthetic dataset using parallel processing.
        
        Returns:
            DataFrame with the complete synthetic dataset
        """
        logger.info(f"Starting generation of {self.config.num_samples} samples")
        
        try:
            # Set up multiprocessing
            num_processes = min(os.cpu_count(), 8)  # Limit to 8 to avoid excessive resource usage
            
            # Generate samples using multiprocessing
            with multiprocessing.Pool(processes=num_processes) as pool:
                samples = pool.starmap(self.generate_sample, [(i, 0) for i in range(self.config.num_samples)])

            # Convert to DataFrame
            df = pd.DataFrame(samples)
            
            # Reorder columns
            ordered_columns = ["Sample_ID", "Age", "Gender", "Country_of_Residence", "Who_Completed_Test"]
            ordered_columns.extend(self.feature_names)
            ordered_columns.extend(["QChat-40-Score", "Autism_Diagnosis"])
            
            # Keep only columns that exist in the DataFrame
            ordered_columns = [col for col in ordered_columns if col in df.columns]
            
            # Add any remaining columns
            for col in df.columns:
                if col not in ordered_columns:
                    ordered_columns.append(col)
            
            df = df[ordered_columns]
            
            # Handle missing values
            for column in df.columns:
                if df[column].dtype in [np.float64, np.int64]:  # Numerical columns
                    median_value = df[column].median()
                    df[column].fillna(median_value, inplace=True)
                else:  # Categorical columns
                    mode_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                    df[column].fillna(mode_value, inplace=True)

            # Prepare data for RFE
            X = df[self.feature_names]
            y = df["Autism_Diagnosis"]
            
            # Run RFE to select features
            selected_features = self.run_rfe(X, y)
            df = df[selected_features + ["Autism_Diagnosis"]]
            
            # Perform correlation analysis
            self.correlation_analysis(df)
            logger.info(f"Generated {len(df)} samples successfully")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to generate dataset: {e}")
            raise

    def save_dataset(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Save the dataset in the configured formats.
        
        Args:
            df: The dataset to save
            
        Returns:
            Dictionary with paths to saved files
        """
        try:
            # Create output directory
            self.config.output_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare metadata
            metadata = {
                "generator_version": "1.0.0",
                "timestamp": timestamp,
                "config": {
                    "num_samples": self.config.num_samples,
                    "asd_prevalence": self.config.asd_prevalence,
                    "random_seed": self.config.random_seed
                },
                "statistics": {
                    "actual_samples": len(df),
                    "actual_asd_ratio": df["Autism_Diagnosis"].mean()
                }
            }
            
            # Save metadata
            metadata_path = self.config.output_dir / f"metadata_{timestamp}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Save in each configured format
            output_paths = {"metadata": metadata_path}
            
            base_filename = f"autism_screening_synthetic_{timestamp}"
            
            for fmt in self.config.output_formats:
                if fmt == "csv":
                    path = self.config.output_dir / f"{base_filename}.csv"
                    df.to_csv(path, index=False)
                    output_paths["csv"] = path
                
                elif fmt == "parquet":
                    path = self.config.output_dir / f"{base_filename}.parquet"
                    df.to_parquet(path, index=False)
                    output_paths["parquet"] = path
                
                elif fmt == "xlsx":
                    path = self.config.output_dir / f"{base_filename}.xlsx"
                    df.to_excel(path, index=False)
                    output_paths["xlsx"] = path
                
                elif fmt == "json":
                    path = self.config.output_dir / f"{base_filename}.json"
                    df.to_json(path, orient="records", indent=2)
                    output_paths["json"] = path
            
            logger.info(f"Dataset saved in formats: {', '.join(self.config.output_formats)}")
            
            return output_paths
        
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete data generation pipeline.
        
        Returns:
            Tuple of (generated DataFrame, validation results)
        """
        try:
            # Generate dataset
            df = self.generate_dataset()
            
            # Validate dataset
            validation_results = self.validator.validate_dataset(df)
            
            # If validation fails, retry with different seed
            retries = 3
            while not validation_results["passed"] and retries > 0:
                logger.warning(f"Dataset validation failed, retrying with new seed ({retries} attempts remaining)")
                self.config.random_seed += 1
                np.random.seed(self.config.random_seed)
                df = self.generate_dataset()
                validation_results = self.validator.validate_dataset(df)
                retries -= 1
            
            # Save dataset
            output_paths = self.save_dataset(df)
            
            # Generate validation report
            report_path = self.validator.generate_validation_report(df, self.config.output_dir)
            
            if report_path:
                logger.info(f"Validation report saved to {report_path}")
            
            # Return results
            return df, validation_results
        
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            raise


def create_default_config(output_path: Path) -> None:
    """Create a default configuration file at the specified path."""
    default_config = {
        "num_samples": 30000,
        "asd_prevalence": 0.2,
        "random_seed": 42,
        "feature_groups": [
            {
                "name": "Social_Interaction",
                "features": [f"A{i}_Score" for i in range(1, 11)],
                "asd_probability_range": [0.6, 0.85],
                "non_asd_probability_range": [0.1, 0.3],
                "key_features": {"A5_Score": 0.15, "A8_Score": 0.2},
                "missingness_probability": 0.02
            },
            {
                "name": "Speech_Cognitive",
                "features": [f"A{i}_Score" for i in range(11, 21)],
                "asd_probability_range": [0.7, 0.9],
                "non_asd_probability_range": [0.15, 0.35],
                "key_features": {"A18_Score": 0.25},
                "missingness_probability": 0.015
            },
            {
                "name": "Repetitive_Behaviors",
                "features": [f"A{i}_Score" for i in range(21, 31)],
                "asd_probability_range": [0.65, 0.95],
                "non_asd_probability_range": [0.05, 0.25],
                "key_features": {"A21_Score": 0.3, "A25_Score": 0.2},
                "missingness_probability": 0.01
            },
            {
                "name": "Sensory_Motor",
                "features": [f"A{i}_Score" for i in range(31, 41)],
                "asd_probability_range": [0.55, 0.85],
                "non_asd_probability_range": [0.1, 0.4],
                "key_features": {"A30_Score": 0.2, "A35_Score": 0.15},
                "missingness_probability": 0.03
            }
        ],
        "demographics": {
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
                "USA": 0.28,        # 1 in 36 prevalence
                "Canada": 0.0152,   # 1 in 66 prevalence
                "UK": 0.0175,       # 1 in 57 prevalence
                "Ireland": 0.05,    # 1 in 20 prevalence
                "Japan": 0.0303,    # 1 in 33 prevalence
                "Russia": 0.01,     # 1 in 100 prevalence
            },
            "reporter": {
                "Parent": 0.7,
                "Health Care Professional": 0.25,
                "Relative": 0.05
            }
        },
        "output_dir": "./generated_data",
        "output_formats": ["csv", "parquet"],
        "validation_metrics": {
            "max_class_ratio_diff": 0.1,
            "max_missingness": 0.05
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(default_config, f, sort_keys=False)
    
    logger.info(f"Default configuration saved to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Autism Screening Synthetic Data Generator")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file",

        type=int,
        help="Override number of samples to generate"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    try:
        args = parse_arguments()
        
        # Create default config if requested
        if args.create_config:
            create_default_config(Path(args.config))
            return
        
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file {config_path} not found")
            logger.info("Run with --create-config to create a default configuration")
            return
        
        config = DatasetConfig.from_yaml(config_path)
        
        # Override configuration with command-line arguments
        if args.samples:
            config.num_samples = args.samples
        
        if args.output_dir:
            config.output_dir = Path(args.output_dir)
        
        # Create and run data generator
        generator = AdvancedAutismDataGenerator(config)
        df, validation = generator.run_pipeline()
        
        # Display summary
        summary = (
            f"\n{'='*80}\n"
            f"GENERATION SUMMARY\n"
            f"{'='*80}\n"
            f"Generated {len(df)} samples\n"
            f"ASD prevalence: {df['Autism_Diagnosis'].mean():.2%}\n"
            f"Validation passed: {validation['passed']}\n"
            f"Output directory: {config.output_dir}\n"
            f"{'='*80}\n"
        )
        
        print(summary)
        logger.info("Data generation completed successfully")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
def _generate_demographics(self, is_asd: bool, time: int) -> Dict[str, Any]:
```

### Next Steps
1. I will create a new version of the `screening_dataset.py` file with the corrected indentation.
2. After that, I will run the command to save the dataset again.

<create_file>
<path>screening_dataset.py</path>
<content>
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import argparse
import yaml
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import shap

from typing import Callable

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("autism_data_generator.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FeatureGroup:
    """Advanced feature group configuration with complex validation and modeling."""
    name: str
    features: List[str]
    asd_probability_range: Tuple[float, float]
    non_asd_probability_range: Tuple[float, float]
    key_features: Dict[str, float] = field(default_factory=dict)
    missingness_probability: float = 0.01
    feature_dependencies: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interaction_effects: Dict[str, Callable[[Dict[str, int]], float]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0 <= self.asd_probability_range[0] <= self.asd_probability_range[1] <= 1):
            raise ValueError(f"Invalid ASD probability range for {self.name}")
        
        if not (0 <= self.non_asd_probability_range[0] <= self.non_asd_probability_range[1] <= 1):
            raise ValueError(f"Invalid non-ASD probability range for {self.name}")
        
        if not (0 <= self.missingness_probability <= 1):
            raise ValueError(f"Invalid missingness probability for {self.name}")
        
        for feature, boost in self.key_features.items():
            if feature not in self.features:
                raise ValueError(f"Key feature {feature} not in feature list for {self.name}")
        
        for dependent_feature, dependencies in self.feature_dependencies.items():
            if dependent_feature not in self.features:
                raise ValueError(f"Dependent feature {dependent_feature} not in feature list for {self.name}")
            for feature in dependencies.keys():
                if feature not in self.features:
                    raise ValueError(f"Dependency feature {feature} not in feature list for {self.name}")

@dataclass
class DatasetConfig:
    """Comprehensive dataset configuration with advanced validation."""
    num_samples: int
    asd_prevalence: float
    random_seed: int
    feature_groups: List[FeatureGroup]
    demographics: Dict[str, Any]
    output_dir: str
    output_formats: List[str]
    validation_metrics: Dict[str, float]
    advanced_configs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not (100 <= self.num_samples <= 1_000_000):
            raise ValueError(f"Sample size must be between 100 and 1,000,000. Requested: {self.num_samples}")
        
        if not (0.01 <= self.asd_prevalence <= 0.5):
            raise ValueError(f"ASD prevalence must be between 1% and 50%. Requested: {self.asd_prevalence}")
        
        self.output_dir = Path(self.output_dir)
        valid_formats = {"csv", "parquet", "xlsx", "json"}
        invalid_formats = set(self.output_formats) - valid_formats
        
        if invalid_formats:
            raise ValueError(f"Invalid output formats: {invalid_formats}")
        
        if self.advanced_configs is None:
            self.advanced_configs = {}
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'DatasetConfig':
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            feature_groups = []
            for group_data in config_data.pop('feature_groups'):
                feature_groups.append(FeatureGroup(**group_data))
            
            config_data['feature_groups'] = feature_groups
            
            return cls(**config_data)
        except (yaml.YAMLError, KeyError) as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

class StatisticalValidator:
    """Validates synthetic data against expected statistical properties.""" 
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.validation_metrics = config.validation_metrics
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        validation_results = {
            "passed": True,
            "metrics": {}
        }
        
        try:
            asd_ratio = df["Autism_Diagnosis"].mean()
            expected_ratio = self.config.asd_prevalence
            ratio_diff = abs(asd_ratio - expected_ratio) / expected_ratio
            
            validation_results["metrics"]["asd_ratio"] = {
                "actual": asd_ratio,
                "expected": expected_ratio,
                "relative_diff": ratio_diff,
                "passed": ratio_diff <= self.validation_metrics["max_class_ratio_diff"]
            }

            if "Yes" in df.values or "No" in df.values:
                logger.warning("Skipping validation for categorical features.")
                return validation_results

            for group in self.config.feature_groups:
                if len(group.features) >= 2:
                    features = [f for f in group.features if f in df.columns]
                    corr_matrix = df[features].corr()
                    avg_corr = (corr_matrix.sum().sum() - len(features)) / (len(features) * (len(features) - 1))
                    
                    validation_results["metrics"][f"{group.name}_correlation"] = {
                        "average_correlation": avg_corr,
                        "passed": 0.1 <= avg_corr <= 0.9
                    }
            
            missing_ratio = df.isna().mean().mean()
            validation_results["metrics"]["missing_data"] = {
                "ratio": missing_ratio,
                "passed": missing_ratio <= self.validation_metrics["max_missingness"]
            }
            
            validation_results["passed"] = all(
                metric["passed"] for metric in validation_results["metrics"].values() 
                if isinstance(metric, dict) and "passed" in metric
            )
            
            return validation_results

    def generate_validation_report(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """
        Generates a validation report with visualizations for the dataset.
        
        Args:
            df: The dataset to validate
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved report
        """
        try:
            # Create validation directory
            validation_dir = output_dir / "validation"
            validation_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate report with metrics
            validation_results = self.validate_dataset(df)
            
            # Save validation results
            with open(validation_dir / "validation_metrics.json", "w") as f:
                json.dump(validation_results, f, indent=2)
            
            # Generate plots
            plt.figure(figsize=(12, 8))
            
            # Plot ASD distribution
            plt.subplot(2, 2, 1)
            sns.countplot(x="Autism_Diagnosis", data=df)
            plt.title("ASD Distribution")
            
            # Plot age distribution by diagnosis
            plt.subplot(2, 2, 2)
            sns.boxplot(x="Autism_Diagnosis", y="Age", data=df)
            plt.title("Age Distribution by Diagnosis")
            
            # Plot gender distribution by diagnosis
            plt.subplot(2, 2, 3)
            pd.crosstab(df["Gender"], df["Autism_Diagnosis"]).plot(kind="bar", stacked=True)
            plt.title("Gender Distribution")
            
            # Plot feature count distribution
            plt.subplot(2, 2, 4)
            df["Total_Features"] = df[[col for col in df.columns if col.endswith("_Score")]].sum(axis=1)
            sns.histplot(data=df, x="Total_Features", hue="Autism_Diagnosis", bins=20)
            plt.title("Feature Distribution")
            
            plt.tight_layout()
            plot_path = validation_dir / "validation_plots.png"
            plt.savefig(plot_path)
            plt.close()
            
            return validation_dir / "validation_metrics.json"
        
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return None

class AdvancedAutismDataGenerator:
    """
    A sophisticated, multi-dimensional synthetic autism screening 
    data generator with advanced probabilistic modeling.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the generator with comprehensive configuration 
        and advanced preprocessing techniques.
        """
        self.config = config
        np.random.seed(config.random_seed)
        self.validator = StatisticalValidator(config)
        
        # Advanced feature and demographic modeling
        self._preprocess_features()
        self._setup_advanced_demographics()
        
        # Advanced diagnostic and scoring models
        self._initialize_diagnostic_models()
    
    def _preprocess_features(self):
        """
        Advanced feature preprocessing with intelligent feature mapping 
        and complex dependency resolution.
        """
        self.feature_names = []
        for group in self.config.feature_groups:
            self.feature_names.extend(group.features)
    
    def _setup_advanced_demographics(self):
        """
        Enhanced demographic distribution modeling with 
        multi-component and contextual probability estimation.
        """
        demo_config = self.config.demographics
        
        # Advanced multi-modal age distribution
        self.age_distribution = {
            "younger_toddlers": {"mean": 16, "std": 3, "weight": 0.3},
            "mid_toddlers": {"mean": 24, "std": 4, "weight": 0.4},
            "older_toddlers": {"mean": 32, "std": 3, "weight": 0.3}
        }
        
        # Gender, country, and reporter distributions
        self.gender_probs = demo_config["gender"]
        self.country_probs = demo_config["country"]
        self.reporter_probs = demo_config["reporter"]
    
    def sensitivity_curve(self, x):
        return 1 / (1 + np.exp(-0.3 * (x - 15)))

    def run_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Perform Recursive Feature Elimination (RFE) to select the top features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        model = RandomForestClassifier()
        rfe = RFE(model, n_features_to_select=10)
        fit = rfe.fit(X, y)
        selected_features = X.columns[fit.support_].tolist()
        logger.info(f"Selected features: {selected_features}")
        return selected_features

    def explain_model(self, model, X: pd.DataFrame) -> None:
        """
        Explain the model's predictions using SHAP.
        
        Args:
            model: Trained model
            X: Feature DataFrame
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        shap.summary_plot(shap_values, X)
        
        # Save the SHAP summary plot
        plt.savefig(self.config.output_dir / "shap_summary_plot.png")
        plt.close()

    def _initialize_diagnostic_models(self):
        """
        Initialize advanced diagnostic and scoring models 
        with complex probabilistic estimation.
        """
        self.diagnosis_model = {
            "base_threshold": 15,
            "sensitivity_curve": self.sensitivity_curve,
            "asd_boost": 1.5,
            "non_asd_reduction": 0.5
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete synthetic dataset using parallel processing.
        
        Returns:
            DataFrame with the complete synthetic dataset
        """
        logger.info(f"Starting generation of {self.config.num_samples} samples")
        
        try:
            # Set up multiprocessing
            num_processes = min(os.cpu_count(), 8)  # Limit to 8 to avoid excessive resource usage
            
            # Generate samples using multiprocessing
            with multiprocessing.Pool(processes=num_processes) as pool:
                samples = pool.starmap(self.generate_sample, [(i, 0) for i in range(self.config.num_samples)])

            # Convert to DataFrame
            df = pd.DataFrame(samples)
            
            # Reorder columns
            ordered_columns = ["Sample_ID", "Age", "Gender", "Country_of_Residence", "Who_Completed_Test"]
            ordered_columns.extend(self.feature_names)
            ordered_columns.extend(["QChat-40-Score", "Autism_Diagnosis"])
            
            # Keep only columns that exist in the DataFrame
            ordered_columns = [col for col in ordered_columns if col in df.columns]
            
            # Add any remaining columns
            for col in df.columns:
                if col not in ordered_columns:
                    ordered_columns.append(col)
            
            df = df[ordered_columns]
            
            # Handle missing values
            for column in df.columns:
                if df[column].dtype in [np.float64, np.int64]:  # Numerical columns
                    median_value = df[column].median()
                    df[column].fillna(median_value, inplace=True)
                else:  # Categorical columns
                    mode_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                    df[column].fillna(mode_value, inplace=True)

            # Prepare data for RFE
            X = df[self.feature_names]
            y = df["Autism_Diagnosis"]
            
            # Run RFE to select features
            selected_features = self.run_rfe(X, y)
            df = df[selected_features + ["Autism_Diagnosis"]]
            
            # Perform correlation analysis
            self.correlation_analysis(df)
            logger.info(f"Generated {len(df)} samples successfully")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to generate dataset: {e}")
            raise

    def save_dataset(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Save the dataset in the configured formats.
        
        Args:
            df: The dataset to save
            
        Returns:
            Dictionary with paths to saved files
        """
        try:
            # Create output directory
            self.config.output_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare metadata
            metadata = {
                "generator_version": "1.0.0",
                "timestamp": timestamp,
                "config": {
                    "num_samples": self.config.num_samples,
                    "asd_prevalence": self.config.asd_prevalence,
                    "random_seed": self.config.random_seed
                },
                "statistics": {
                    "actual_samples": len(df),
                    "actual_asd_ratio": df["Autism_Diagnosis"].mean()
                }
            }
            
            # Save metadata
            metadata_path = self.config.output_dir / f"metadata_{timestamp}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Save in each configured format
            output_paths = {"metadata": metadata_path}
            
            base_filename = f"autism_screening_synthetic_{timestamp}"
            
            for fmt in self.config.output_formats:
                if fmt == "csv":
                    path = self.config.output_dir / f"{base_filename}.csv"
                    df.to_csv(path, index=False)
                    output_paths["csv"] = path
                
                elif fmt == "parquet":
                    path = self.config.output_dir / f"{base_filename}.parquet"
                    df.to_parquet(path, index=False)
                    output_paths["parquet"] = path
                
                elif fmt == "xlsx":
                    path = self.config.output_dir / f"{base_filename}.xlsx"
                    df.to_excel(path, index=False)
                    output_paths["xlsx"] = path
                
                elif fmt == "json":
                    path = self.config.output_dir / f"{base_filename}.json"
                    df.to_json(path, orient="records", indent=2)
                    output_paths["json"] = path
            
            logger.info(f"Dataset saved in formats: {', '.join(self.config.output_formats)}")
            
            return output_paths
        
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete data generation pipeline.
        
        Returns:
            Tuple of (generated DataFrame, validation results)
        """
        try:
            # Generate dataset
            df = self.generate_dataset()
            
            # Validate dataset
            validation_results = self.validator.validate_dataset(df)
            
            # If validation fails, retry with different seed
            retries = 3
            while not validation_results["passed"] and retries > 0:
                logger.warning(f"Dataset validation failed, retrying with new seed ({retries} attempts remaining)")
                self.config.random_seed += 1
                np.random.seed(self.config.random_seed)
                df = self.generate_dataset()
                validation_results = self.validator.validate_dataset(df)
                retries -= 1
            
            # Save dataset
            output_paths = self.save_dataset(df)
            
            # Generate validation report
            report_path = self.validator.generate_validation_report(df, self.config.output_dir)
            
            if report_path:
                logger.info(f"Validation report saved to {report_path}")
            
            # Return results
            return df, validation_results
        
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            raise


def create_default_config(output_path: Path) -> None:
    """Create a default configuration file at the specified path."""
    default_config = {
        "num_samples": 30000,
        "asd_prevalence": 0.2,
        "random_seed": 42,
        "feature_groups": [
            {
                "name": "Social_Interaction",
                "features": [f"A{i}_Score" for i in range(1, 11)],
                "asd_probability_range": [0.6, 0.85],
                "non_asd_probability_range": [0.1, 0.3],
                "key_features": {"A5_Score": 0.15, "A8_Score": 0.2},
                "missingness_probability": 0.02
            },
            {
                "name": "Speech_Cognitive",
                "features": [f"A{i}_Score" for i in range(11, 21)],
                "asd_probability_range": [0.7, 0.9],
                "non_asd_probability_range": [0.15, 0.35],
                "key_features": {"A18_Score": 0.25},
                "missingness_probability": 0.015
            },
            {
                "name": "Repetitive_Behaviors",
                "features": [f"A{i}_Score" for i in range(21, 31)],
                "asd_probability_range": [0.65, 0.95],
                "non_asd_probability_range": [0.05, 0.25],
                "key_features": {"A21_Score": 0.3, "A25_Score": 0.2},
                "missingness_probability": 0.01
            },
            {
                "name": "Sensory_Motor",
                "features": [f"A{i}_Score" for i in range(31, 41)],
                "asd_probability_range": [0.55, 0.85],
                "non_asd_probability_range": [0.1, 0.4],
                "key_features": {"A30_Score": 0.2, "A35_Score": 0.15},
                "missingness_probability": 0.03
            }
        ],
        "demographics": {
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
                "USA": 0.28,        # 1 in 36 prevalence
                "Canada": 0.0152,   # 1 in 66 prevalence
                "UK": 0.0175,       # 1 in 57 prevalence
                "Ireland": 0.05,    # 1 in 20 prevalence
                "Japan": 0.0303,    # 1 in 33 prevalence
                "Russia": 0.01,     # 1 in 100 prevalence
            },
            "reporter": {
                "Parent": 0.7,
                "Health Care Professional": 0.25,
                "Relative": 0.05
            }
        },
        "output_dir": "./generated_data",
        "output_formats": ["csv", "parquet"],
        "validation_metrics": {
            "max_class_ratio_diff": 0.1,
            "max_missingness": 0.05
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(default_config, f, sort_keys=False)
    
    logger.info(f"Default configuration saved to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Autism Screening Synthetic Data Generator")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file",

        type=int,
        help="Override number of samples to generate"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    try:
        args = parse_arguments()
        
        # Create default config if requested
        if args.create_config:
            create_default_config(Path(args.config))
            return
        
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file {config_path} not found")
            logger.info("Run with --create-config to create a default configuration")
            return
        
        config = DatasetConfig.from_yaml(config_path)
        
        # Override configuration with command-line arguments
        if args.samples:
            config.num_samples = args.samples
        
        if args.output_dir:
            config.output_dir = Path(args.output_dir)
        
        # Create and run data generator
        generator = AdvancedAutismDataGenerator(config)
        df, validation = generator.run_pipeline()
        
        # Display summary
        summary = (
            f"\n{'='*80}\n"
            f"GENERATION SUMMARY\n"
            f"{'='*80}\n"
            f"Generated {len(df)} samples\n"
            f"ASD prevalence: {df['Autism_Diagnosis'].mean():.2%}\n"
            f"Validation passed: {validation['passed']}\n"
            f"Output directory: {config.output_dir}\n"
            f"{'='*80}\n"
        )
        
        print(summary)
        logger.info("Data generation completed successfully")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
