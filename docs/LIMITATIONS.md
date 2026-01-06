# Limitations

## Methodological Limitations

### 1. Unconfoundedness Assumption

**Issue**: We assume all confounders are measured and included in the model.

**Reality**: There may be unmeasured confounders:
- Sales rep quality
- Customer sentiment (not captured in features)
- Competitor actions
- Macroeconomic factors

**Mitigation**:
- A/B test validation provides ground truth
- Sensitivity analysis can assess robustness

### 2. Single Treatment Analysis

**Issue**: Current model estimates effect of binary treatment (intervention vs none).

**Reality**: Different intervention types may have different effects.

**Future Work**: Train separate causal forests for each intervention type.

### 3. Static Analysis

**Issue**: Model captures one-time snapshot, not dynamic customer journey.

**Reality**: Customer risk and intervention effects change over time.

**Future Work**: 
- Time-series features
- Sequential decision making
- Reinforcement learning

## Data Limitations

### 1. Synthetic Interventions

**Issue**: Intervention data is simulated, not from real experiments.

**Reality**: Real intervention effects may differ from simulations.

**Mitigation**: Validate with actual A/B tests before full deployment.

### 2. Sample Size

**Issue**: ~7,000 customers may be insufficient for some segments.

**Reality**: Small segments have high variance in effect estimates.

**Mitigation**: Aggregate small segments or use shrinkage estimators.

### 3. Feature Granularity

**Issue**: Monthly aggregates may miss important daily patterns.

**Future Work**: Add real-time behavioral features.

## Production Limitations

### 1. Cold Start

**Issue**: New customers lack historical features.

**Mitigation**: Use demographic/signup features only for new customers.

### 2. Model Drift

**Issue**: Customer behavior and intervention effects change over time.

**Mitigation**: 
- Monthly model retraining
- Continuous monitoring
- Online/adaptive learning

### 3. Intervention Fatigue

**Issue**: Repeated interventions may have diminishing returns.

**Reality**: Customers may become immune to discounts.

**Future Work**: Track intervention history and model fatigue.

## Recommendations

1. **Start with pilot**: Deploy to subset of at-risk customers
2. **Validate continuously**: Regular A/B tests
3. **Monitor drift**: Automated alerts for performance degradation
4. **Iterate**: Incorporate learnings into model updates
