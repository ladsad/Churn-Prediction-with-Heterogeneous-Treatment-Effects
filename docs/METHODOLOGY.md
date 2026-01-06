# Methodology

## Overview

This project uses **causal machine learning** to predict customer churn and recommend personalized interventions. Unlike traditional churn prediction systems that only predict *who* churns, this system estimates *what intervention works best* for each customer.

## Causal Inference Framework

### The Problem: Confounding

In observational data, we cannot directly compare treated vs untreated customers because:
- High-risk customers are more likely to receive interventions (selection bias)
- This creates spurious correlations that make interventions appear ineffective

### Solution: Doubly Robust Estimation

We use **Doubly Robust (DR) estimation** to obtain unbiased causal effects:

```
ATE_DR = E[ T*(y - mu1(X))/ps - (1-T)*(y - mu0(X))/(1-ps) + mu1(X) - mu0(X) ]
```

Where:
- `T` = treatment indicator
- `y` = observed outcome (churn)
- `mu1(X)` = predicted outcome if treated
- `mu0(X)` = predicted outcome if control
- `ps` = propensity score P(T=1|X)

**Why DR?**
- Consistent if EITHER propensity model OR outcome model is correct
- Provides standard errors and confidence intervals
- More robust than propensity score matching alone

## Heterogeneous Treatment Effects

### Causal Forest (econml)

We use **CausalForestDML** to estimate Conditional Average Treatment Effects (CATE):

- Trees split to maximize variance of treatment effect (not outcome)
- Estimates localized effects: "For customers similar to you, here's the intervention effect"
- Provides uncertainty quantification via confidence intervals

### Why Not Standard ML?

Standard regression only captures additive interactions:
```
E[Y|X,T] = beta0 + beta1*X + beta2*T + beta3*(X*T)
```

Causal Forest captures non-linear, complex interactions automatically.

## Assumptions

1. **Unconfoundedness**: All confounders are measured
   - We address this by including comprehensive customer features
   - Validated partially via A/B testing

2. **Positivity**: All customers have non-zero probability of treatment
   - Verified via propensity score overlap check

3. **SUTVA**: No interference between customers
   - Reasonable for individual churn decisions

## Limitations

1. Unmeasured confounders may bias estimates
2. Treatment effect estimates are for single intervention
3. Temporal dynamics not captured (one-shot intervention)
4. External validity depends on data representativeness

## Validation

- A/B test simulation confirms causal forest predictions
- High-effect segments show larger actual effects
- Statistical significance achieved (p < 0.05)
