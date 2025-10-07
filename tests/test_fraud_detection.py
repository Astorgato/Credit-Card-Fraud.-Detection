"""
Unit tests for fraud detection models.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection_models import FraudDetectionPipeline


class TestFraudDetectionPipeline:
    """Test suite for FraudDetectionPipeline class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample fraud detection data."""
        np.random.seed(42)
        n_samples = 1000

        # Create synthetic credit card data
        data = {
            'Time': np.random.uniform(0, 172800, n_samples),
            'Amount': np.random.lognormal(3, 1.5, n_samples),
            'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        }

        # Add V1-V10 (simplified PCA components)
        for i in range(1, 11):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)

        return pd.DataFrame(data)

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        return FraudDetectionPipeline(random_state=42)

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.random_state == 42
        assert hasattr(pipeline, 'scaler')
        assert hasattr(pipeline, 'models')
        assert hasattr(pipeline, 'results')

    def test_data_preparation(self, pipeline, sample_data):
        """Test data preparation functionality."""
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']

        X_train_scaled, X_test_scaled, y_train, y_test = pipeline.prepare_data(X, y)

        assert X_train_scaled.shape[1] == X.shape[1]
        assert len(y_train) + len(y_test) == len(y)
        assert abs(X_train_scaled.mean()) < 1e-10  # Should be scaled

    def test_model_training(self, pipeline, sample_data):
        """Test model training process."""
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']

        pipeline.prepare_data(X, y)
        pipeline.train_models()

        assert len(pipeline.models) > 0
        assert len(pipeline.results) > 0

        # Check that models were trained
        for name, result in pipeline.results.items():
            assert 'model' in result
            assert 'metrics' in result


def test_data_quality():
    """Test data quality requirements."""
    # This would test actual dataset quality if available
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
