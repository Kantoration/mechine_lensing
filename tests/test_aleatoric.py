#!/usr/bin/env python3
"""
Unit tests for post-hoc aleatoric uncertainty analysis.

Tests cover:
- Shape consistency and numerical stability
- Temperature scaling effects
- Test-time augmentation indicators
- Selection score strategies
- Ensemble disagreement metrics
- Edge cases and error handling
"""

import pytest
import torch
import numpy as np
from typing import List

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.aleatoric import (
    AleatoricIndicators,
    compute_indicators,
    compute_indicators_with_targets,
    tta_indicators,
    selection_scores,
    topk_indices,
    ensemble_disagreement,
    indicators_to_dataframe_dict,
    _safe_log,
    _safe_sigmoid,
    _logistic_ci
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class TestHelperFunctions:
    """Test helper functions for numerical stability."""
    
    def test_safe_log(self):
        """Test numerically stable logarithm."""
        # Normal values
        x = torch.tensor([0.1, 0.5, 0.9])
        result = _safe_log(x)
        expected = torch.log(x)
        assert torch.allclose(result, expected, atol=1e-6)
        
        # Edge case: very small values
        x_small = torch.tensor([1e-10, 0.0, 1e-8])
        result_small = _safe_log(x_small)
        assert torch.all(torch.isfinite(result_small))
        assert torch.all(result_small < 0)  # All should be negative
    
    def test_safe_sigmoid(self):
        """Test numerically stable sigmoid with clamping."""
        # Normal logits
        logits = torch.tensor([-5.0, 0.0, 5.0])
        result = _safe_sigmoid(logits)
        
        # Should be clamped away from 0 and 1
        assert torch.all(result >= 1e-6)
        assert torch.all(result <= 1.0 - 1e-6)
        
        # Extreme logits
        extreme_logits = torch.tensor([-100.0, 100.0])
        extreme_result = _safe_sigmoid(extreme_logits)
        assert torch.all(torch.isfinite(extreme_result))
        assert torch.all(extreme_result > 0)
        assert torch.all(extreme_result < 1)
    
    def test_logistic_ci(self):
        """Test confidence interval computation."""
        set_seed(42)
        logits = torch.randn(10)
        logit_var = torch.rand(10) * 0.5  # Positive variance
        
        prob_ci_lo, prob_ci_hi, prob_ci_width = _logistic_ci(logits, logit_var)
        
        # Check shapes
        assert prob_ci_lo.shape == logits.shape
        assert prob_ci_hi.shape == logits.shape
        assert prob_ci_width.shape == logits.shape
        
        # Check ordering: lo <= hi
        assert torch.all(prob_ci_lo <= prob_ci_hi)
        
        # Check width is positive
        assert torch.all(prob_ci_width >= 0)
        
        # Check bounds are in [0, 1]
        assert torch.all(prob_ci_lo >= 0)
        assert torch.all(prob_ci_hi <= 1)


class TestAleatoricIndicators:
    """Test AleatoricIndicators dataclass."""
    
    def test_indicators_creation(self):
        """Test creation and basic functionality."""
        # Create with some fields
        probs = torch.tensor([0.2, 0.7, 0.9])
        entropy = torch.tensor([0.5, 0.3, 0.1])
        
        indicators = AleatoricIndicators(probs=probs, pred_entropy=entropy)
        
        assert torch.equal(indicators.probs, probs)
        assert torch.equal(indicators.pred_entropy, entropy)
        assert indicators.logits is None  # Not set
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        probs = torch.tensor([0.3, 0.8])
        indicators = AleatoricIndicators(probs=probs, nll=None)
        
        result_dict = indicators.to_dict()
        
        assert 'probs' in result_dict
        assert torch.equal(result_dict['probs'], probs)
        assert result_dict['nll'] is None
    
    def test_to_numpy_dict(self):
        """Test numpy dictionary conversion."""
        probs = torch.tensor([0.4, 0.6])
        indicators = AleatoricIndicators(probs=probs)
        
        numpy_dict = indicators.to_numpy_dict()
        
        assert 'probs' in numpy_dict
        assert isinstance(numpy_dict['probs'], np.ndarray)
        assert np.allclose(numpy_dict['probs'], [0.4, 0.6])


class TestComputeIndicators:
    """Test main indicator computation functions."""
    
    def test_compute_indicators_basic(self):
        """Test basic indicator computation."""
        set_seed(42)
        logits = torch.randn(5)
        
        indicators = compute_indicators(logits)
        
        # Check all basic fields are computed
        assert indicators.probs is not None
        assert indicators.logits is not None
        assert indicators.pred_entropy is not None
        assert indicators.conf is not None
        assert indicators.margin is not None
        assert indicators.brier is not None
        
        # Check shapes
        assert indicators.probs.shape == logits.shape
        assert indicators.pred_entropy.shape == logits.shape
        
        # Check value ranges
        assert torch.all(indicators.probs >= 0)
        assert torch.all(indicators.probs <= 1)
        assert torch.all(indicators.pred_entropy >= 0)
        assert torch.all(indicators.conf >= 0.5)  # max(p, 1-p) >= 0.5
        assert torch.all(indicators.margin >= 0)
        assert torch.all(indicators.margin <= 0.5)
    
    def test_compute_indicators_with_logit_var(self):
        """Test indicator computation with logit variance."""
        set_seed(42)
        logits = torch.randn(4)
        logit_var = torch.rand(4) * 0.3
        
        indicators = compute_indicators(logits, logit_var=logit_var)
        
        # CI fields should be computed
        assert indicators.prob_ci_lo is not None
        assert indicators.prob_ci_hi is not None
        assert indicators.prob_ci_width is not None
        
        # Check ordering and bounds
        assert torch.all(indicators.prob_ci_lo <= indicators.prob_ci_hi)
        assert torch.all(indicators.prob_ci_width >= 0)
        assert torch.all(indicators.prob_ci_lo >= 0)
        assert torch.all(indicators.prob_ci_hi <= 1)
    
    def test_temperature_scaling_effect(self):
        """Test that higher temperature increases entropy on average."""
        set_seed(42)
        logits = torch.randn(100)  # Larger sample for statistical test
        
        # Compute indicators with different temperatures
        indicators_t1 = compute_indicators(logits, temperature=1.0)
        indicators_t2 = compute_indicators(logits, temperature=2.0)
        
        # Higher temperature should increase entropy on average
        mean_entropy_t1 = indicators_t1.pred_entropy.mean()
        mean_entropy_t2 = indicators_t2.pred_entropy.mean()
        
        assert mean_entropy_t2 > mean_entropy_t1
    
    def test_compute_indicators_with_targets(self):
        """Test indicator computation with targets."""
        set_seed(42)
        logits = torch.randn(6)
        targets = torch.randint(0, 2, (6,)).float()
        
        indicators = compute_indicators_with_targets(logits, targets)
        
        # Should have all fields including NLL
        assert indicators.nll is not None
        assert indicators.brier is not None  # Calibrated version
        
        # Check NLL is positive (it's negative log-likelihood)
        assert torch.all(indicators.nll >= 0)
        
        # Check Brier score bounds
        assert torch.all(indicators.brier >= 0)
        assert torch.all(indicators.brier <= 1)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        # Extreme logits
        extreme_logits = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
        
        indicators = compute_indicators(extreme_logits)
        
        # All outputs should be finite
        assert torch.all(torch.isfinite(indicators.probs))
        assert torch.all(torch.isfinite(indicators.pred_entropy))
        assert torch.all(torch.isfinite(indicators.conf))
        assert torch.all(torch.isfinite(indicators.margin))
        assert torch.all(torch.isfinite(indicators.brier))


class TestTTAIndicators:
    """Test test-time augmentation indicators."""
    
    def test_tta_indicators_basic(self):
        """Test basic TTA indicator computation."""
        set_seed(42)
        # Simulate TTA logits: [MC=5, B=3]
        logits_tta = torch.randn(5, 3)
        
        prob_mean, prob_var = tta_indicators(logits_tta)
        
        # Check shapes
        assert prob_mean.shape == (3,)
        assert prob_var.shape == (3,)
        
        # Check value ranges
        assert torch.all(prob_mean >= 0)
        assert torch.all(prob_mean <= 1)
        assert torch.all(prob_var >= 0)
    
    def test_tta_variance_increases_with_noise(self):
        """Test that TTA variance increases with augmentation diversity."""
        set_seed(42)
        
        # Low noise case: similar logits
        base_logits = torch.randn(1, 4)
        low_noise_logits = base_logits + torch.randn(5, 4) * 0.1
        
        # High noise case: more diverse logits
        high_noise_logits = base_logits + torch.randn(5, 4) * 1.0
        
        _, var_low = tta_indicators(low_noise_logits)
        _, var_high = tta_indicators(high_noise_logits)
        
        # Higher noise should lead to higher variance on average
        assert var_high.mean() > var_low.mean()
    
    def test_tta_identical_inputs(self):
        """Test TTA with identical inputs (no variance)."""
        # All augmentations produce same logits
        identical_logits = torch.ones(5, 3) * 2.0
        
        prob_mean, prob_var = tta_indicators(identical_logits)
        
        # Variance should be very small (numerical precision)
        assert torch.all(prob_var < 1e-6)


class TestSelectionScores:
    """Test active learning selection scores."""
    
    def create_sample_indicators(self) -> AleatoricIndicators:
        """Create sample indicators for testing."""
        return AleatoricIndicators(
            probs=torch.tensor([0.1, 0.5, 0.9, 0.7]),
            pred_entropy=torch.tensor([0.8, 1.0, 0.2, 0.6]),
            margin=torch.tensor([0.4, 0.0, 0.4, 0.2]),
            brier=torch.tensor([0.1, 0.25, 0.1, 0.09]),
            nll=torch.tensor([2.3, 0.7, 0.1, 0.4]),
            prob_ci_width=torch.tensor([0.3, 0.8, 0.1, 0.4])
        )
    
    def test_entropy_selection(self):
        """Test entropy-based selection."""
        indicators = self.create_sample_indicators()
        
        scores = selection_scores(indicators, strategy="entropy")
        
        assert torch.equal(scores, indicators.pred_entropy)
    
    def test_low_margin_selection(self):
        """Test low margin selection."""
        indicators = self.create_sample_indicators()
        
        scores = selection_scores(indicators, strategy="low_margin")
        expected = 1.0 - indicators.margin
        
        assert torch.allclose(scores, expected)
    
    def test_wide_ci_selection(self):
        """Test wide confidence interval selection."""
        indicators = self.create_sample_indicators()
        
        scores = selection_scores(indicators, strategy="wide_ci")
        
        assert torch.equal(scores, indicators.prob_ci_width)
    
    def test_wide_ci_fallback(self):
        """Test fallback to entropy when CI not available."""
        indicators = self.create_sample_indicators()
        indicators.prob_ci_width = None  # Remove CI
        
        scores = selection_scores(indicators, strategy="wide_ci")
        
        assert torch.equal(scores, indicators.pred_entropy)
    
    def test_hybrid_selection(self):
        """Test hybrid selection strategy."""
        indicators = self.create_sample_indicators()
        
        scores = selection_scores(indicators, strategy="hybrid")
        
        # Should be average of normalized indicators
        assert scores.shape == (4,)
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)  # Normalized
    
    def test_selection_error_handling(self):
        """Test error handling for missing indicators."""
        empty_indicators = AleatoricIndicators()
        
        with pytest.raises(ValueError, match="Entropy not available"):
            selection_scores(empty_indicators, strategy="entropy")


class TestTopKIndices:
    """Test top-k sample selection."""
    
    def test_simple_topk(self):
        """Test simple top-k selection without class balancing."""
        scores = torch.tensor([0.1, 0.8, 0.3, 0.9, 0.2])
        k = 3
        
        indices = topk_indices(scores, k)
        
        # Should select indices 3, 1, 2 (highest scores)
        expected = torch.tensor([3, 1, 2])
        assert torch.equal(torch.sort(indices)[0], torch.sort(expected)[0])
    
    def test_class_balanced_topk(self):
        """Test class-balanced top-k selection."""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        class_balance = torch.tensor([1, 0, 1, 0, 1, 0])  # Alternating classes
        k = 4
        pos_frac = 0.5  # 50% positive
        
        indices = topk_indices(scores, k, class_balance=class_balance, pos_frac=pos_frac)
        
        # Should select 2 from each class
        assert len(indices) == 4
        
        # Check class balance
        selected_classes = class_balance[indices]
        pos_count = (selected_classes == 1).sum().item()
        neg_count = (selected_classes == 0).sum().item()
        
        assert pos_count == 2
        assert neg_count == 2
    
    def test_topk_insufficient_samples(self):
        """Test top-k when k is larger than available samples."""
        scores = torch.tensor([0.5, 0.3])
        k = 5  # More than available
        
        indices = topk_indices(scores, k)
        
        # Should return all available samples
        assert len(indices) == 2


class TestEnsembleDisagreement:
    """Test ensemble disagreement metrics."""
    
    def test_ensemble_disagreement_basic(self):
        """Test basic ensemble disagreement computation."""
        # Create 3 ensemble members with different predictions
        prob_members = [
            torch.tensor([0.2, 0.8, 0.5]),
            torch.tensor([0.3, 0.7, 0.6]),
            torch.tensor([0.4, 0.9, 0.4])
        ]
        
        disagreement = ensemble_disagreement(prob_members)
        
        # Check all metrics are computed
        assert 'vote_entropy' in disagreement
        assert 'prob_variance' in disagreement
        assert 'pairwise_kl_mean' in disagreement
        
        # Check shapes
        assert disagreement['vote_entropy'].shape == (3,)
        assert disagreement['prob_variance'].shape == (3,)
        assert disagreement['pairwise_kl_mean'].shape == (3,)
        
        # Check value ranges
        assert torch.all(disagreement['vote_entropy'] >= 0)
        assert torch.all(disagreement['prob_variance'] >= 0)
        assert torch.all(disagreement['pairwise_kl_mean'] >= 0)
    
    def test_ensemble_identical_members(self):
        """Test disagreement when all members are identical."""
        # All members predict the same
        identical_probs = torch.tensor([0.3, 0.7, 0.9])
        prob_members = [identical_probs, identical_probs, identical_probs]
        
        disagreement = ensemble_disagreement(prob_members)
        
        # Variance and KL should be zero (or very small)
        assert torch.all(disagreement['prob_variance'] < 1e-6)
        assert torch.all(disagreement['pairwise_kl_mean'] < 1e-6)
        
        # Vote entropy should equal individual entropy
        expected_entropy = -(identical_probs * torch.log(torch.clamp(identical_probs, min=1e-8)) + 
                           (1 - identical_probs) * torch.log(torch.clamp(1 - identical_probs, min=1e-8)))
        assert torch.allclose(disagreement['vote_entropy'], expected_entropy, atol=1e-6)
    
    def test_ensemble_maximum_disagreement(self):
        """Test disagreement with maximally disagreeing members."""
        # One member predicts 0.1, another predicts 0.9
        prob_members = [
            torch.tensor([0.1, 0.1, 0.1]),
            torch.tensor([0.9, 0.9, 0.9])
        ]
        
        disagreement = ensemble_disagreement(prob_members)
        
        # Should have high variance and KL divergence
        assert torch.all(disagreement['prob_variance'] > 0.1)
        assert torch.all(disagreement['pairwise_kl_mean'] > 0.1)
    
    def test_ensemble_single_member(self):
        """Test disagreement with single member (edge case)."""
        prob_members = [torch.tensor([0.4, 0.6, 0.8])]
        
        disagreement = ensemble_disagreement(prob_members)
        
        # No disagreement with single member
        assert torch.all(disagreement['prob_variance'] < 1e-6)
        assert torch.all(disagreement['pairwise_kl_mean'] < 1e-6)
    
    def test_ensemble_empty_list(self):
        """Test error handling with empty member list."""
        with pytest.raises(ValueError, match="Empty probability list"):
            ensemble_disagreement([])


class TestDataFrameIntegration:
    """Test DataFrame integration utilities."""
    
    def test_indicators_to_dataframe_dict(self):
        """Test conversion to DataFrame-friendly format."""
        indicators = AleatoricIndicators(
            probs=torch.tensor([0.3, 0.7]),
            pred_entropy=torch.tensor([0.8, 0.6]),
            nll=None  # Missing field
        )
        sample_ids = ['sample_1', 'sample_2']
        
        df_dict = indicators_to_dataframe_dict(indicators, sample_ids)
        
        # Check structure
        assert 'sample_id' in df_dict
        assert 'probs' in df_dict
        assert 'pred_entropy' in df_dict
        assert 'nll' not in df_dict  # Should be excluded
        
        # Check values
        assert df_dict['sample_id'] == sample_ids
        assert isinstance(df_dict['probs'], np.ndarray)
        assert np.allclose(df_dict['probs'], [0.3, 0.7])


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from logits to selection."""
        set_seed(42)
        
        # Simulate model outputs
        logits = torch.randn(20)
        targets = torch.randint(0, 2, (20,)).float()
        
        # Compute indicators
        indicators = compute_indicators_with_targets(logits, targets, temperature=1.2)
        
        # Get selection scores
        scores = selection_scores(indicators, strategy="hybrid")
        
        # Select top samples
        selected = topk_indices(scores, k=5)
        
        # Verify pipeline completion
        assert len(selected) == 5
        assert torch.all(selected >= 0)
        assert torch.all(selected < 20)
    
    def test_ensemble_analysis_pipeline(self):
        """Test ensemble analysis pipeline."""
        set_seed(42)
        
        # Simulate ensemble member probabilities
        prob_members = [
            torch.rand(10),
            torch.rand(10),
            torch.rand(10)
        ]
        
        # Compute disagreement
        disagreement = ensemble_disagreement(prob_members)
        
        # Use disagreement as selection criteria
        selection_scores_from_disagreement = disagreement['vote_entropy']
        selected = topk_indices(selection_scores_from_disagreement, k=3)
        
        assert len(selected) == 3
    
    def test_device_consistency(self):
        """Test that outputs match input device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        logits = torch.randn(5, device=device)
        
        indicators = compute_indicators(logits)
        
        # All outputs should be on same device
        assert indicators.probs.device == device
        assert indicators.pred_entropy.device == device


if __name__ == '__main__':
    pytest.main([__file__, "-v"])




