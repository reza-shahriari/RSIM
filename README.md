# RSIM – Variance Reduction Simulation Toolkit

RSIM is a comprehensive Python library for running **statistical simulations** with an emphasis on **variance reduction techniques** in Monte Carlo methods.  

---

## 🚀 Features

- Bootstrap confidence intervals (percentile, BC, BCa)
- Cross-validation simulation for regression & classification
- Stratified sampling, antithetic variables, control variates
- Importance sampling & Rao-Blackwellization
- Advanced statistical diagnostics & visualization
- Support for custom statistics and models
- Clean modular architecture with extensible interfaces

---

## 📦 Installation

```bash
pip install RSIM
```

---


## 📚 Integration with RSIM

This module seamlessly integrates with the RSIM ecosystem:

```python
# Access through RSIM package
from RSIM.bootstrap import CrossValidationSimulation

# Follows RSIM base simulation interface
from RSIM.core.base import BaseSimulation
assert issubclass(CrossValidationSimulation, BaseSimulation)

# Compatible with RSIM result format
result = cv_sim.run(X, y)
print(result.simulation_name)
print(result.execution_time)
print(result.statistics)
```

---

### ▶️ `BootstrapConfidenceInterval`
> Perform powerful **distribution-free confidence interval estimation** via bootstrap resampling, supporting multiple methods and statistical metrics.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does
This class implements **non-parametric bootstrap simulation** to compute confidence intervals (CI) for arbitrary statistics — such as mean, median, std, correlation, or any user-defined metric — without relying on normality assumptions.

#### 📐 Supported CI Methods
- **Percentile**: Directly uses quantiles from bootstrap distribution
- **Bias-Corrected (BC)**: Adjusts percentiles based on observed bias
- **Bias-Corrected and Accelerated (BCa)**: Further adjusts for skewness using jackknife estimates
- **Basic Bootstrap**: Reflects quantiles around the original estimate

#### 📊 Supported Statistics
- Mean, Median, Standard Deviation, Variance
- Correlation (for paired 2D data)
- Ratio of means (paired)
- Custom statistic (user-defined function)

#### 📚 Theoretical Background

**Bootstrap Principle:**  
The bootstrap method approximates the sampling distribution of a statistic by **resampling with replacement** from the original dataset. It treats the empirical distribution as a proxy for the population.

Let:
- \( X = \{x_1, x_2, \dots, x_n\} \) be the original data
- \( X_b^* \sim \text{Resample}(X) \), for \( b = 1, \dots, B \)
- \( \hat{\theta}_b^* = T(X_b^*) \) be the statistic on the b-th resample

Then the bootstrap distribution \( \{\hat{\theta}_b^*\} \) is used to construct confidence intervals.

**Percentile CI:**
\[
\text{CI}_{\alpha} = [\hat{\theta}^*_{(\alpha/2)}, \hat{\theta}^*_{(1-\alpha/2)}]
\]

**Bias-Corrected (BC):**
\[
z_0 = \Phi^{-1} \left( \frac{\#(\hat{\theta}^* < \hat{\theta})}{B} \right)
\]

**Bias-Corrected and Accelerated (BCa):**
\[
a = \frac{\sum (\bar{\theta} - \theta_i)^3}{6[\sum (\bar{\theta} - \theta_i)^2]^{3/2}}
\]

#### ✅ Properties
- Distribution-free and robust
- Asymptotically consistent
- Works with small or skewed samples
- Invariant to monotonic transformations

#### 📈 Example – CI for mean
```python
import numpy as np
from RSIM.bootstrap_ci import BootstrapConfidenceInterval

data = np.random.normal(50, 10, 100)
bootstrap = BootstrapConfidenceInterval(data, statistic='mean')
result = bootstrap.run()
print("95% CI for mean:", result.results['confidence_interval'])
```

#### 🎯 Example – Custom statistic (coefficient of variation)
```python
def cv(x):
    return np.std(x) / np.mean(x)

bootstrap_cv = BootstrapConfidenceInterval(data, statistic=cv)
result = bootstrap_cv.run()
print("CV estimate:", result.results['original_statistic'])
print("95% CI:", result.results['confidence_interval'])
```

#### 📉 Example – Correlation (paired data)
```python
x = np.random.normal(0, 1, 50)
y = 0.7 * x + np.random.normal(0, 0.5, 50)
paired = np.column_stack([x, y])

bootstrap_corr = BootstrapConfidenceInterval(paired, statistic='correlation', method='bca')
result = bootstrap_corr.run()
print("95% CI for correlation:", result.results['confidence_interval'])
```

#### 📊 Accuracy Guidelines
| Replications | Purpose                 |
|--------------|--------------------------|
| 1,000        | Quick exploration         |
| 5,000        | General-purpose accuracy  |
| 10,000       | High-confidence results   |
| 50,000+      | Publication-grade         |

#### 📚 References
- Efron & Tibshirani (1993). *An Introduction to the Bootstrap*
- DiCiccio & Efron (1996). *Bootstrap Confidence Intervals*
- Davison & Hinkley (1997). *Bootstrap Methods and Their Applications*

</details>
---

### ▶️ `CrossValidationSimulation`
> Perform comprehensive **machine learning model evaluation** via cross-validation, supporting multiple CV strategies, algorithms, and statistical analysis methods.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does
This class implements **robust cross-validation simulation** to evaluate machine learning model performance, estimate generalization error, and assess model stability across different data splits. It supports both regression and classification tasks with multiple validation schemes and comprehensive performance metrics.

#### 🔄 Supported CV Strategies
- **K-Fold**: Standard approach dividing data into k equal-sized folds
- **Stratified K-Fold**: Maintains class distribution in each fold (essential for classification)
- **Leave-One-Out (LOO)**: Special case where k = n (sample size)
- **Monte Carlo**: Random train/test splits repeated multiple times

#### 🤖 Supported Models
**Regression Models:**
- Linear Regression (`linear_regression`)
- Random Forest Regressor (`random_forest_reg`)
- Support Vector Regressor (`svm_reg`)

**Classification Models:**
- Logistic Regression (`logistic_regression`)
- Random Forest Classifier (`random_forest_clf`)
- Support Vector Classifier (`svm_clf`)

#### 📊 Performance Metrics
**Regression:** MSE, RMSE, MAE, R²  
**Classification:** Accuracy, Precision, Recall, F1-Score

#### 📚 Theoretical Background
**Cross-Validation Error Estimation:**
```
CV Error = (1/k) × Σ(i=1 to k) L(yi, ŷi)
```
Where k = number of folds, L = loss function

**Bias-Variance Decomposition:**
```
E[CV] = Bias² + Variance + Noise
```

**Standard Error:**
```
SE = σ/√k
```

**Statistical Properties:**
- **5-fold CV**: Good bias-variance tradeoff, faster execution
- **10-fold CV**: Lower bias, standard choice for most applications
- **LOO CV**: Nearly unbiased but high variance and computational cost
- **Stratified CV**: Reduces variance in classification, maintains class balance

#### ✅ Key Features
- Multiple CV strategies with automatic parameter validation
- Built-in popular ML algorithms with hyperparameter defaults
- Comprehensive statistical analysis with confidence intervals
- Advanced visualization including feature importance and fold-by-fold analysis
- Model comparison capabilities with statistical significance testing
- Reproducible results with random seed control

#### 📈 Example – Basic Regression Cross-Validation
```python
from RSIM.bootstrap.cross_validation import CrossValidationSimulation
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Configure 10-fold CV
cv_sim = CrossValidationSimulation(
    cv_strategy='kfold',
    n_folds=10,
    model_type='random_forest_reg',
    task_type='regression',
    random_seed=42
)

# Run simulation
result = cv_sim.run(X, y)

# Display results
print(f"CV Score: {result.results['mean_score']:.4f} ± {result.results['std_score']:.4f}")
print(f"95% CI: [{result.results['confidence_interval'][0]:.4f}, {result.results['confidence_interval'][1]:.4f}]")

# Visualize comprehensive results
cv_sim.visualize()
```

#### 🎯 Example – Classification with Stratified CV
```python
from sklearn.datasets import make_classification

# Generate imbalanced classification data
X, y = make_classification(
    n_samples=1000, 
    n_classes=3, 
    weights=[0.6, 0.3, 0.1],  # Imbalanced classes
    random_state=42
)

# Stratified cross-validation (essential for imbalanced data)
cv_sim = CrossValidationSimulation(
    cv_strategy='stratified',
    n_folds=5,
    model_type='random_forest_clf',
    task_type='classification',
    scoring='f1'
)

result = cv_sim.run(X, y)
cv_sim.visualize(show_distributions=True)

print("Classification Results:")
print(f"Mean Accuracy: {result.results['mean_accuracy']:.4f}")
print(f"Mean F1-Score: {result.results['mean_f1_score']:.4f}")
```

#### 🔍 Example – Model Comparison
```python
# Compare multiple regression models
models_to_compare = [
    'linear_regression',
    'random_forest_reg', 
    'svm_reg'
]

# Initialize CV simulation
cv_sim = CrossValidationSimulation(
    cv_strategy='kfold',
    n_folds=5,
    task_type='regression'
)

# Compare models with statistical analysis
comparison_results = cv_sim.compare_models(models_to_compare, X, y)

# Results automatically displayed in comprehensive table and plots
for model, scores in comparison_results.items():
    print(f"{model}: {scores['mean']:.4f} ± {scores['std']:.4f}")
```

#### 🎲 Example – Monte Carlo Cross-Validation
```python
# Monte Carlo CV with custom train/test ratio
cv_mc = CrossValidationSimulation(
    cv_strategy='monte_carlo',
    n_repeats=100,
    test_size=0.3,
    model_type='svm_reg',
    random_seed=123
)

# Run with synthetic data generation
result = cv_mc.run_with_synthetic_data(
    n_samples=500, 
    n_features=8,
    noise=0.2
)

cv_mc.visualize(show_fold_details=True)
```

#### 📊 Computational Guidelines
| CV Strategy | Complexity | Use Case |
|-------------|------------|----------|
| 5-fold | O(5 × training_time) | Quick evaluation, good default |
| 10-fold | O(10 × training_time) | Standard choice, stable estimates |
| LOO | O(n × training_time) | Small datasets only (n < 100) |
| Monte Carlo | O(repeats × training_time) | Custom train/test ratios |

#### 🎯 Statistical Interpretation
- **Mean CV Score**: Expected model performance on unseen data
- **Standard Deviation**: Model stability indicator (lower = more robust)
- **95% Confidence Interval**: [mean - 1.96×SE, mean + 1.96×SE]
- **Coefficient of Variation**: Relative stability measure
  - CV < 0.1: Highly stable model
  - 0.1 ≤ CV < 0.2: Moderately stable
  - CV ≥ 0.2: Potentially unstable

#### 📈 Visualization Features
**Standard Visualizations:**
- Box plot of CV scores across folds
- Score variation by fold with confidence bands
- Score distribution histogram with statistical summaries
- Comprehensive results table with all metrics

**Advanced Visualizations:**
- Feature importance plots (for tree-based models)
- Error metrics by fold (MSE, MAE for regression)
- Classification metrics by fold (Accuracy, F1, Precision, Recall)
- Model comparison charts with statistical significance

#### ⚡ Best Practices
- **Always use stratified CV for classification tasks**
- **Report both mean and standard deviation of CV scores**
- **Use same CV strategy when comparing models**
- **Set random seeds for reproducible results**
- **Validate CV results on independent test set**
- **Consider computational cost vs. accuracy tradeoff**

#### ⚠️ Common Pitfalls to Avoid
- **Data Leakage**: Preprocessing before CV split
- **Temporal Data**: Use time-series CV instead of random splits
- **Small Datasets**: Avoid high k values, consider LOO
- **Imbalanced Classes**: Stratified CV is mandatory
- **Hyperparameter Tuning**: Requires nested CV for unbiased estimates

#### 📚 References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*
- Kohavi, R. (1995). *A study of cross-validation and bootstrap for accuracy estimation*
- Varma, S. & Simon, R. (2006). *Bias in error estimation when using cross-validation*

</details>

---


### ▶️ `JackknifeEstimation`

> Perform **deterministic bias correction and variance estimation** via jackknife resampling, providing robust statistical inference without distributional assumptions.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements the **jackknife resampling method** to estimate bias and variance of statistical estimators using systematic "leave-one-out" sampling. Unlike bootstrap methods, jackknife is deterministic and particularly effective for bias correction of smooth statistics.

#### 📐 Key Features

- **Bias Estimation & Correction**: Reduces estimator bias from O(n⁻¹) to O(n⁻²)
- **Variance Estimation**: Distribution-free variance calculation
- **Influence Analysis**: Identifies influential observations and outliers
- **Deterministic Results**: Same output on repeated runs (no randomness)
- **12+ Built-in Statistics**: Mean, median, std, variance, skewness, kurtosis, etc.
- **Custom Statistics**: Support for user-defined functions

#### 📊 Supported Statistics

**Built-in Statistics:**
- `'mean'`, `'median'`, `'std'`, `'var'`
- `'skewness'`, `'kurtosis'`, `'min'`, `'max'`
- `'range'`, `'iqr'`, `'mad'`, `'cv'` (coefficient of variation)

**Custom Functions:** Any callable accepting numpy array → scalar

#### 📚 Theoretical Background

**Jackknife Principle:**  
For dataset X = {x₁, x₂, ..., xₙ} and statistic θ̂(X), create n "leave-one-out" samples:
- X₍ᵢ₎ = X \ {xᵢ} (dataset without i-th observation)
- θ̂₍ᵢ₎ = θ̂(X₍ᵢ₎) for i = 1, ..., n

**Key Formulas:**

**Jackknife Mean:**
```
θ̂₍·₎ = (1/n) Σᵢ θ̂₍ᵢ₎
```

**Bias Estimate:**
```
bias_jack = (n-1)(θ̂₍·₎ - θ̂)
```

**Bias-Corrected Estimate:**
```
θ̂_bc = θ̂ - bias_jack = nθ̂ - (n-1)θ̂₍·₎
```

**Variance Estimate:**
```
var_jack = ((n-1)/n) Σᵢ (θ̂₍ᵢ₎ - θ̂₍·₎)²
```

**Influence Values:**
```
Iᵢ = (n-1)(θ̂₍·₎ - θ̂₍ᵢ₎)
```

#### ✅ Properties

- **Exact bias correction** for linear statistics
- **Consistent variance estimation** for smooth statistics  
- **Deterministic and reproducible** results
- **Computationally efficient** O(n × T) where T = statistic computation time
- **Outlier detection** via influence analysis
- **Small sample performance** works with n ≥ 3

#### 📈 Example – Basic bias correction for sample mean

```python
import numpy as np
from RSIM.bootstrap import JackknifeEstimation

# Generate sample data
data = np.random.normal(10, 2, 50)

# Jackknife analysis
jack = JackknifeEstimation(data=data, statistic='mean')
result = jack.run()

print(f"Original mean: {result.results['original_estimate']:.4f}")
print(f"Bias estimate: {result.results['bias_estimate']:.6f}")
print(f"Bias-corrected: {result.results['bias_corrected_estimate']:.4f}")
print(f"Standard error: {result.results['standard_error']:.4f}")
print(f"95% CI: {result.results['confidence_interval']}")
```

#### 🎯 Example – Custom statistic (trimmed mean)

```python
def trimmed_mean(x, trim_prop=0.1):
    """10% trimmed mean"""
    n_trim = int(len(x) * trim_prop)
    sorted_x = np.sort(x)
    return np.mean(sorted_x[n_trim:-n_trim]) if n_trim > 0 else np.mean(x)

jack_custom = JackknifeEstimation(data=data, statistic=trimmed_mean)
result = jack_custom.run()
jack_custom.visualize()
```

#### 📉 Example – Variance bias correction

```python
# Sample variance is biased for small samples
small_sample = np.random.normal(5, 1.5, 15)

jack_var = JackknifeEstimation(
    data=small_sample, 
    statistic='var', 
    confidence_level=0.99
)
result = jack_var.run()

print(f"Original variance: {result.results['original_estimate']:.4f}")
print(f"Bias correction: {result.results['bias_estimate']:.6f}")
print(f"Corrected variance: {result.results['bias_corrected_estimate']:.4f}")
```

#### 🔍 Example – Influence analysis and outlier detection

```python
# Add some outliers
data_with_outliers = np.concatenate([
    np.random.normal(10, 1, 45),  # Normal data
    [15, 16, 17, 18, 20]          # Outliers
])

jack_influence = JackknifeEstimation(
    data=data_with_outliers, 
    statistic='mean',
    store_estimates=True
)
result = jack_influence.run()

# Detect influential observations
outliers, scores = jack_influence.detect_outliers(threshold=2.0)
print(f"Influential observations at indices: {outliers}")
print(f"Influence scores: {scores}")

# Visualize with influence analysis
jack_influence.visualize(show_influence=True, show_distribution=True)
```

#### 📊 Comparison with Bootstrap

```python
# Compare jackknife vs bootstrap results
comparison = jack.bootstrap_comparison(n_bootstrap=5000, random_seed=42)

print("Jackknife vs Bootstrap Comparison:")
print(f"Bias estimates - Jack: {comparison['jackknife_bias']:.6f}, "
      f"Boot: {comparison['bootstrap_bias']:.6f}")
print(f"Standard errors - Jack: {comparison['jackknife_se']:.6f}, "
      f"Boot: {comparison['bootstrap_se']:.6f}")
print(f"Methods agree: {comparison['methods_agree']}")
```

#### 🎨 Visualization Features

**Standard Visualization:**
- Summary statistics and confidence intervals
- Comparison of original vs bias-corrected estimates
- Error bar plots with confidence bounds

**Influence Analysis:**
- Scatter plot of influence values by observation
- Highlighting of high-influence points
- Outlier detection thresholds

**Distribution Analysis:**
- Histogram of jackknife estimates
- Overlay of normal approximation
- Convergence assessment

#### ⚡ Performance Guidelines

| Sample Size | Computation Time | Memory Usage |
|-------------|------------------|--------------|
| n < 100     | < 1 second       | Minimal      |
| n = 1,000   | < 10 seconds     | Low          |
| n = 10,000  | < 2 minutes      | Moderate     |
| n > 10,000  | Consider subsampling | High     |

#### 🎯 When to Use Jackknife

**✅ Ideal for:**
- Bias correction of smooth statistics (mean, variance, correlation)
- Small to moderate sample sizes (n < 1000)
- When deterministic results are needed
- Influence analysis and outlier detection
- Linear or quasi-linear statistics

**❌ Less suitable for:**
- Non-smooth statistics (median, quantiles, mode)
- Very large datasets (use bootstrap instead)
- Highly non-linear or discontinuous statistics
- Time series with strong dependencies

#### 📚 References

- Quenouille, M. H. (1949). *Approximate tests of correlation in time-series*
- Tukey, J. W. (1958). *Bias and confidence in not-quite large samples*  
- Miller, R. G. (1974). *The jackknife—a review*. Biometrika, 61(1), 1-15
- Efron, B. (1982). *The Jackknife, the Bootstrap and Other Resampling Plans*
- Shao, J. & Tu, D. (1995). *The Jackknife and Bootstrap*

</details>


### ▶️ `PermutationTest`

> Perform powerful **distribution-free hypothesis testing** via permutation resampling, providing exact p-values without distributional assumptions.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **non-parametric permutation tests** (also known as randomization tests or exact tests) to assess statistical significance of observed differences between groups by randomly reassigning observations and comparing test statistics from permuted data to the original observed statistic.

#### 📐 Supported Test Statistics

- **Mean Difference**: `|mean(X) - mean(Y)|` - Most common location test
- **Median Difference**: `|median(X) - median(Y)|` - Robust location test  
- **Welch's t-statistic**: For unequal variances with normalization
- **Mann-Whitney U**: Rank-based non-parametric comparison
- **Custom Statistics**: User-defined functions taking `(group1, group2) → scalar`

#### 📊 Hypothesis Testing Options

- **Two-sided**: `H₁: groups differ (≠)`
- **Greater**: `H₁: group1 > group2` 
- **Less**: `H₁: group1 < group2`
- **Exact tests**: All possible permutations (small samples)
- **Approximate tests**: Random permutation sampling (large samples)

#### 📚 Theoretical Background

**Permutation Principle:**  
Under the null hypothesis of no group difference, observations are **exchangeable** - any reassignment to groups is equally likely. The permutation test compares the observed test statistic to the distribution of all possible reassignments.

Let:
- `X = {x₁, x₂, ..., xₙ₁}` be group 1 observations
- `Y = {y₁, y₂, ..., yₙ₂}` be group 2 observations  
- `T(X,Y)` be the test statistic
- `T*ᵦ` be the statistic for permutation b

**P-value Calculation:**
```
P-value = #{T*ᵦ ≥ T_observed} / B
```

**Exact vs Approximate:**
- **Exact**: All `C(n₁+n₂, n₁)` possible permutations
- **Approximate**: Random sample of B permutations

#### ✅ Properties

- **Distribution-free**: No normality assumptions required
- **Exact Type I error control**: Under exchangeability assumption
- **Robust**: To outliers and skewed distributions  
- **Flexible**: Works with any test statistic
- **Asymptotically equivalent**: To parametric tests under normality

#### 📈 Example – Basic two-sample test

```python
import numpy as np
from RSIM.bootstrap import PermutationTest

# Treatment vs Control groups
treatment = np.random.normal(12, 3, 50)  # Higher mean
control = np.random.normal(10, 3, 45)    # Lower mean

perm_test = PermutationTest(
    group1=treatment, 
    group2=control,
    test_statistic='mean_diff',
    n_permutations=10000,
    alternative='two-sided'
)

result = perm_test.run()
print(f"P-value: {result.results['p_value']:.4f}")
print(f"Significant: {result.results['is_significant']}")
print(f"Effect size: {result.results['effect_size']:.3f}")
```

#### 🎯 Example – One-tailed test with median

```python
# Test if treatment median > control median
perm_test = PermutationTest(
    group1=treatment,
    group2=control, 
    test_statistic='median_diff',
    alternative='greater',
    alpha=0.01,
    n_permutations=50000
)

result = perm_test.run()
perm_test.visualize(show_effect_size=True)
```

#### 📉 Example – Custom test statistic (variance ratio)

```python
def variance_ratio(g1, g2):
    """Test equality of variances"""
    return np.var(g1, ddof=1) / np.var(g2, ddof=1)

var_test = PermutationTest(
    group1=treatment,
    group2=control,
    test_statistic=variance_ratio,
    alternative='two-sided'
)

result = var_test.run()
print(f"Variance ratio: {result.results['observed_statistic']:.3f}")
```

#### 🔬 Example – Exact test (small samples)

```python
# Small sample exact test
small_treatment = [8.2, 9.1, 7.8, 8.9, 9.5]
small_control = [7.1, 6.8, 7.3, 6.9]

exact_test = PermutationTest(
    group1=small_treatment,
    group2=small_control,
    n_permutations='exact'  # All possible permutations
)

result = exact_test.run()
print(f"Exact p-value: {result.results['p_value']:.6f}")
print(f"Permutations used: {result.results['n_permutations_used']:,}")
```

#### 📊 Accuracy Guidelines

| Permutations | P-value Precision | Use Case |
|--------------|-------------------|----------|
| 1,000        | ±0.01            | Quick screening |
| 10,000       | ±0.003           | Standard analysis |
| 100,000      | ±0.001           | High precision |
| Exact        | Perfect          | Small samples (n₁+n₂ ≤ 20) |

#### 🔍 Comparison with Parametric Tests

```python
# Compare with t-test and Mann-Whitney
comparison = perm_test.compare_with_parametric()

print("Permutation p-value:", comparison['permutation']['p_value'])
print("T-test p-value:", comparison['t_test']['p_value'])  
print("Mann-Whitney p-value:", comparison['mann_whitney']['p_value'])
```

#### ⚡ Performance Characteristics

- **Time complexity**: O(B × n_total) where B = permutations, n_total = n₁ + n₂
- **Space complexity**: O(B) for storing permutation statistics
- **Typical speed**: ~1,000 permutations/second for moderate samples
- **Memory usage**: ~8 bytes per permutation

#### 🎨 Visualization Features

```python
# Comprehensive visualization
perm_test.visualize(
    show_distribution=True,    # Permutation distribution histogram
    show_effect_size=True      # Effect size with confidence intervals
)
```

**Visualization includes:**
- Permutation distribution histogram
- Observed statistic marked with vertical line
- P-value calculation visualization
- Group comparison boxplots
- Effect size with 95% confidence interval
- Statistical significance summary

#### 📈 Power Analysis

```python
# Calculate statistical power
power = perm_test.calculate_power(effect_size=0.5, alpha=0.05)
print(f"Power for Cohen's d=0.5: {power:.3f}")

# Sample size recommendation
n_recommended = perm_test.recommend_sample_size(
    effect_size=0.5, 
    power=0.8, 
    alpha=0.05
)
print(f"Recommended sample size per group: {n_recommended}")
```

#### ⚠️ Assumptions and Limitations

**Assumptions:**
- **Exchangeability** under null hypothesis
- **Independent observations** within and between groups
- **Same population** under H₀

**Limitations:**
- Computationally intensive for large datasets
- Requires sufficient permutations for stable p-values
- May be conservative for discrete test statistics
- Exact tests limited to small sample sizes

#### 🆚 Advantages over Parametric Tests

**Advantages:**
- No distributional assumptions required
- Exact Type I error control
- Robust to outliers and non-normality
- Applicable to any test statistic

**Disadvantages:**
- Computationally more expensive
- May have lower power for normal data
- Requires larger samples for precise p-values

#### 📚 References

- Good, P. I. (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses*
- Ernst, M. D. (2004). *Permutation Methods: A Basis for Exact Inference*
- Edgington, E. S. & Onghena, P. (2007). *Randomization Tests*
- Manly, B. F. J. (2006). *Randomization, Bootstrap and Monte Carlo Methods*

</details>
---

### ▶️ `VasicekModel`
> Simulate **mean-reverting interest rate dynamics** with Gaussian distribution, featuring analytical bond pricing and comprehensive yield curve analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does
This class implements the **Vasicek short-rate model** for simulating interest rate evolution over time. It provides mean-reverting dynamics with analytical tractability for bond pricing and derivatives valuation.

#### 📐 Mathematical Model
The Vasicek model follows the stochastic differential equation:
```
dr(t) = a(b - r(t))dt + σ dW(t)
```

Where:
- `r(t)` = instantaneous interest rate at time t
- `a` = mean reversion speed (a > 0)
- `b` = long-term mean level
- `σ` = volatility parameter
- `dW(t)` = Wiener process (Brownian motion)

#### 🔧 Key Features
- **Mean Reversion**: Rates naturally drift toward long-term mean
- **Analytical Solutions**: Closed-form bond pricing formulas
- **Gaussian Distribution**: Normally distributed rates at any time
- **Yield Curve Generation**: Complete term structure modeling
- **Multiple Visualizations**: Paths, distributions, and statistics

#### 📊 Model Properties
- **Analytical Solution**: `r(t) = r(0)e^(-at) + b(1 - e^(-at)) + σ∫[0,t] e^(-a(t-s)) dW(s)`
- **Theoretical Mean**: `E[r(t)] = b + (r₀ - b)e^(-at)`
- **Theoretical Variance**: `Var[r(t)] = σ²/(2a) × (1 - e^(-2at))`
- **Bond Price**: `P(t,T) = A(t,T) × e^(-B(t,T) × r(t))`

#### ⚠️ Model Limitations
- **Negative Rates**: Can produce negative interest rates
- **Constant Volatility**: Volatility doesn't depend on rate level
- **Single Factor**: Only one source of randomness

#### 📈 Example – Basic Vasicek Simulation
```python
from RSIM.financial import VasicekModel

# Configure Vasicek model
vasicek = VasicekModel(
    initial_rate=0.03,           # 3% starting rate
    mean_reversion_speed=0.5,    # Moderate mean reversion
    long_term_mean=0.04,         # 4% long-term average
    volatility=0.01,             # 1% volatility
    time_horizon=5.0,            # 5-year simulation
    num_paths=1000               # 1000 Monte Carlo paths
)

# Run simulation
result = vasicek.run()

# Display results
print(f"Final mean rate: {result.results['final_mean_rate']:.4f}")
print(f"Theoretical mean: {result.results['theoretical_mean']:.4f}")
print(f"Negative rate probability: {result.results['negative_rate_probability']:.2%}")

# Visualize results
vasicek.visualize()
```

#### 🏦 Example – Bond Pricing
```python
# Calculate bond prices for various maturities
bond_5y = vasicek.calculate_bond_price(maturity=5.0)
bond_10y = vasicek.calculate_bond_price(maturity=10.0)

print(f"5-year bond price: ${bond_5y:.2f}")
print(f"10-year bond price: ${bond_10y:.2f}")

# Generate complete yield curve
maturities = [0.25, 0.5, 1, 2, 5, 10, 20, 30]
yield_curve = vasicek.calculate_yield_curve(maturities)

for mat, yield_rate in yield_curve['yields'].items():
    print(f"{mat:.2f}Y yield: {yield_rate:.4f}")
```

#### 📊 Example – Scenario Analysis
```python
# High volatility scenario
high_vol = VasicekModel(
    initial_rate=0.02,
    volatility=0.03,           # 3% volatility
    num_paths=5000
)
result_high_vol = high_vol.run()

# Fast mean reversion scenario
fast_reversion = VasicekModel(
    mean_reversion_speed=2.0,   # Very fast reversion
    long_term_mean=0.06
)
result_fast = fast_reversion.run()

# Compare scenarios
print("High Vol - Negative Rate Prob:", result_high_vol.results['negative_rate_probability'])
print("Fast Reversion - Final Std:", result_fast.results['final_std_rate'])
```

#### 🎯 Applications
- **Fixed Income**: Bond portfolio valuation and risk management
- **Derivatives**: Interest rate options and swaps pricing
- **ALM**: Asset-liability matching for insurance/pensions
- **Risk Management**: Interest rate scenario generation
- **Central Banking**: Policy rate modeling and forecasting

#### 📚 References
- Vasicek, O. (1977). *An Equilibrium Characterization of the Term Structure*
- Hull, J. (2017). *Options, Futures, and Other Derivatives*
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models*

</details>

---

### ▶️ `CIRModel`
> Implement the **Cox-Ingersoll-Ross model** with guaranteed non-negative rates, stochastic volatility, and advanced discretization schemes.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does
This class implements the **Cox-Ingersoll-Ross (CIR) short-rate model** for simulating interest rate dynamics. It extends Vasicek by ensuring non-negative rates and incorporating stochastic volatility that increases with the rate level.

#### 📐 Mathematical Model
The CIR model follows the stochastic differential equation:
```
dr(t) = a(b - r(t))dt + σ√r(t) dW(t)
```

Where:
- `r(t)` = instantaneous interest rate at time t
- `a` = mean reversion speed (a > 0)
- `b` = long-term mean level (b > 0)
- `σ` = volatility parameter (σ > 0)
- `√r(t)` = square-root diffusion term
- `dW(t)` = Wiener process

#### 🔧 Key Features
- **Non-negative Rates**: Square-root diffusion prevents negative rates
- **Stochastic Volatility**: Volatility proportional to √r(t)
- **Feller Condition**: Automatic checking of `2ab ≥ σ²`
- **Multiple Schemes**: Euler, Milstein, and exact simulation
- **Advanced Analytics**: Chi-squared distribution properties

#### 📊 Feller Condition
**Critical Condition**: `2ab ≥ σ²`
- **If satisfied**: Process stays strictly positive
- **If violated**: Process can touch zero but reflects back
- **Feller Parameter**: `2ab/σ²` (should be ≥ 1)

#### 🎲 Discretization Schemes
1. **Euler-Maruyama**: Simple first-order approximation
2. **Milstein**: Higher-order correction for better accuracy
3. **Exact**: Uses non-central chi-squared distribution (requires scipy)

#### 📈 Example – Basic CIR Simulation
```python
from RSIM.financial import CIRModel

# Configure CIR model
cir = CIRModel(
    initial_rate=0.03,           # 3% starting rate
    mean_reversion_speed=1.0,    # Strong mean reversion
    long_term_mean=0.05,         # 5% long-term average
    volatility=0.15,             # 15% volatility
    time_horizon=10.0,           # 10-year simulation
    num_paths=2000,              # 2000 Monte Carlo paths
    scheme='milstein'            # Higher-order discretization
)

# Check Feller condition
feller_ok = cir.check_feller_condition()
print(f"Feller condition satisfied: {feller_ok}")

# Run simulation
result = cir.run()

# Display results
print(f"Final mean rate: {result.results['final_mean_rate']:.4f}")
print(f"Zero rate probability: {result.results['zero_rate_probability']:.4%}")
print(f"Mean instantaneous volatility: {result.results['mean_instantaneous_volatility']:.4f}")

# Visualize comprehensive results
cir.visualize()
```

#### 🔬 Example – Feller Condition Analysis
```python
# Test different parameter combinations
configs = [
    {'a': 2.0, 'b': 0.04, 'sigma': 0.1},  # Strong Feller
    {'a': 0.5, 'b': 0.03, 'sigma': 0.2},  # Weak Feller
    {'a': 0.1, 'b': 0.02, 'sigma': 0.3}   # Violated Feller
]

for i, config in enumerate(configs):
    cir_test = CIRModel(
        mean_reversion_speed=config['a'],
        long_term_mean=config['b'],
        volatility=config['sigma']
    )
    
    feller_param = 2 * config['a'] * config['b'] / (config['sigma']**2)
    feller_ok = cir_test.check_feller_condition()
    
    print(f"Config {i+1}: Feller parameter = {feller_param:.2f}, Satisfied = {feller_ok}")
    
    result = cir_test.run()
    zero_prob = result.results['zero_rate_probability']
    print(f"  Zero rate probability: {zero_prob:.4%}")
```

#### 📊 Example – Volatility Analysis
```python
# High volatility regime
high_vol_cir = CIRModel(
    initial_rate=0.02,
    volatility=0.3,              # 30% volatility
    num_paths=5000
)

result = high_vol_cir.run()

# Extract volatility statistics
vol_stats = result.statistics['volatility_of_volatility']
mean_vol = result.results['mean_instantaneous_volatility']

print(f"Mean instantaneous volatility: {mean_vol:.4f}")
print(f"Volatility of volatility: {vol_stats:.4f}")

# Rate-volatility relationship analysis
import matplotlib.pyplot as plt
import numpy as np

# Get rate paths and compute instantaneous volatilities
rate_paths = high_vol_cir.rate_paths
vol_paths = 0.3 * np.sqrt(rate_paths)

# Sample for scatter plot
sample_rates = rate_paths.flatten()[::100]
sample_vols = vol_paths.flatten()[::100]

plt.figure(figsize=(10, 6))
plt.scatter(sample_rates, sample_vols, alpha=0.5, s=1)
plt.xlabel('Interest Rate')
plt.ylabel('Instantaneous Volatility')
plt.title('CIR Rate-Volatility Relationship')
plt.grid(True, alpha=0.3)
plt.show()
```

#### 🏦 Example – Bond Pricing Comparison
```python
# Compare CIR vs Vasicek bond prices
vasicek = VasicekModel(initial_rate=0.03, volatility=0.01)
cir = CIRModel(initial_rate=0.03, volatility=0.1)

maturities = [1, 2, 5, 10, 20, 30]

print("Maturity | Vasicek | CIR     | Difference")
print("-" * 40)

for T in maturities:
    vasicek_price = vasicek.calculate_bond_price(T)
    cir_price = cir.calculate_bond_price(T)
    diff = cir_price - vasicek_price
    
    print(f"{T:2d}Y     | {vasicek_price:6.2f}  | {cir_price:6.2f}  | {diff:+6.2f}")
```

#### 🎯 Example – Credit Risk Application
```python
# CIR as intensity process for credit modeling
credit_cir = CIRModel(
    initial_rate=0.02,           # 2% initial default intensity
    mean_reversion_speed=3.0,    # Fast mean reversion
    long_term_mean=0.03,         # 3% long-term default rate
    volatility=0.4,              # High volatility
    time_horizon=5.0,
    num_paths=10000
)

result = credit_cir.run()

# Calculate survival probabilities
rate_paths = credit_cir.rate_paths
dt = credit_cir.dt

# Integrated intensities
integrated_intensities = np.cumsum(rate_paths * dt, axis=1)

# Survival probabilities at each time
survival_probs = np.exp(-integrated_intensities)
mean_survival = np.mean(survival_probs, axis=0)

# 5-year survival probability
survival_5y = mean_survival[-1]
default_prob_5y = 1 - survival_5y

print(f"5-year survival probability: {survival_5y:.4f}")
print(f"5-year default probability: {default_prob_5y:.4f}")
```

#### ⚙️ Parameter Guidelines

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `a` (speed) | 0.1 - 5.0 | Higher = faster mean reversion |
| `b` (mean) | 0.01 - 0.15 | Long-term average rate |
| `σ` (volatility) | 0.05 - 0.5 | Rate volatility coefficient |
| `r₀` (initial) | 0.001 - 0.2 | Starting rate level |

#### 🔍 Model Diagnostics
- **Feller Parameter**: Should be > 1 for strict positivity
- **Zero-touching**: Monitor paths that approach zero
- **Convergence**: Check theoretical vs simulated moments
- **Volatility Clustering**: Examine rate-volatility relationship

#### 🎯 Applications
- **Interest Rate Modeling**: Term structure with positive rates
- **Credit Risk**: Default intensity modeling
- **Commodity Prices**: Mean-reverting commodity dynamics
- **Volatility Modeling**: Stochastic volatility components
- **Insurance**: Mortality/morbidity rate modeling

#### 📚 References
- Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). *A Theory of the Term Structure of Interest Rates*
- Feller, W. (1951). *Two Singular Diffusion Problems*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*
- Kloeden, P.E. & Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*

</details>


### ▶️ `BlackScholesSimulation`

> Perform **Monte Carlo option pricing** under the Black-Scholes framework with comprehensive Greeks calculation and variance reduction techniques.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Monte Carlo simulation** for European option pricing using the Black-Scholes model. It generates thousands of random stock price paths and calculates option payoffs to estimate fair option values, providing both pricing results and risk sensitivities (Greeks).

#### 🎯 Key Features

- **European Call & Put Options**: Standard vanilla option pricing
- **Variance Reduction**: Antithetic variates and control variate methods
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, Rho via finite differences
- **Convergence Analysis**: Track price stability as simulations increase
- **Analytical Comparison**: Validate against closed-form Black-Scholes formula

#### 📐 Mathematical Background

**Stock Price Dynamics:**
Under the Black-Scholes model, stock prices follow geometric Brownian motion:
```
dS = μS dt + σS dW
```

**Risk-Neutral Valuation:**
For option pricing, we use the risk-neutral measure:
```
S(T) = S₀ × exp((r - σ²/2)T + σ√T × Z)
```

Where:
- `S₀`: Initial stock price
- `r`: Risk-free rate
- `σ`: Volatility
- `T`: Time to expiration
- `Z`: Standard normal random variable

**Option Payoffs:**
- **Call Option**: `max(S(T) - K, 0)`
- **Put Option**: `max(K - S(T), 0)`

**Monte Carlo Price:**
```
Option Price ≈ e^(-rT) × (1/N) × Σ[Payoff_i]
```

#### 🔢 The Greeks

**Delta (Δ)**: Price sensitivity to underlying price changes
```
Δ = ∂V/∂S
```

**Gamma (Γ)**: Rate of change of Delta
```
Γ = ∂²V/∂S²
```

**Theta (Θ)**: Time decay
```
Θ = ∂V/∂T
```

**Vega (ν)**: Volatility sensitivity
```
ν = ∂V/∂σ
```

**Rho (ρ)**: Interest rate sensitivity
```
ρ = ∂V/∂r
```

#### 📈 Example – Basic Call Option

```python
from RSIM.financial import BlackScholesSimulation

# Configure call option
bs_sim = BlackScholesSimulation(
    S0=100,           # Current stock price
    K=105,            # Strike price
    T=1.0,            # 1 year to expiration
    r=0.05,           # 5% risk-free rate
    sigma=0.2,        # 20% volatility
    n_simulations=100000
)

result = bs_sim.run()
print(f"Call option price: ${result.results['option_price']:.4f}")
print(f"Analytical price: ${result.results['analytical_price']:.4f}")
print(f"Delta: {result.statistics['delta']:.4f}")
```

#### 📉 Example – Put Option with Variance Reduction

```python
# High-accuracy put option with variance reduction
put_sim = BlackScholesSimulation(
    S0=100, K=100, T=0.5, r=0.03, sigma=0.25,
    option_type='put',
    antithetic=True,        # Use antithetic variates
    control_variate=True,   # Use control variate
    n_simulations=50000
)

result = put_sim.run()
put_sim.visualize()  # Show comprehensive plots
```

#### 🎯 Example – Parameter Sensitivity Analysis

```python
import numpy as np

# Analyze sensitivity to volatility
bs_sim = BlackScholesSimulation(S0=100, K=100, T=1.0, r=0.05)
sweep_results = bs_sim.run_parameter_sweep({
    'sigma': np.linspace(0.1, 0.5, 10)  # 10% to 50% volatility
})

# Extract results
for result in sweep_results[0]['results']:
    vol = result['parameter_value']
    price = result['option_price']
    vega = result['vega']
    print(f"σ={vol:.1%}: Price=${price:.4f}, Vega={vega:.4f}")
```

#### 📊 Simulation Guidelines

| Simulations | Purpose                    | Typical Error |
|-------------|----------------------------|---------------|
| 10,000      | Quick exploration          | ±0.05         |
| 50,000      | Standard accuracy          | ±0.02         |
| 100,000     | High accuracy              | ±0.015        |
| 500,000+    | Publication/trading grade  | ±0.007        |

#### 🎨 Visualization Features

The `visualize()` method creates four comprehensive plots:
1. **Stock Price Distribution**: Final price histogram with strike line
2. **Payoff Distribution**: Option payoff frequency
3. **Convergence Analysis**: Price stability vs. simulation count
4. **Results Summary**: Key metrics and Greeks

#### 📚 References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Hull, J. (2017). *Options, Futures, and Other Derivatives*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*

</details>

---

### ▶️ `AsianOptionSimulation`

> Price **path-dependent Asian options** using Monte Carlo simulation with arithmetic or geometric averaging and comprehensive path analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Monte Carlo simulation** for Asian (average price) options, where the payoff depends on the average price of the underlying asset over the option's life rather than just the final price. This averaging reduces volatility and makes options less expensive than European equivalents.

#### 🎯 Key Features

- **Arithmetic & Geometric Averaging**: Two averaging methods supported
- **Path Generation**: Full price path simulation with configurable time steps
- **Volatility Reduction Analysis**: Compare with European option equivalent
- **Path Visualization**: Display sample price trajectories
- **Convergence Tracking**: Monitor price stability across simulations

#### 📐 Mathematical Background

**Price Path Generation:**
Stock prices follow geometric Brownian motion discretized over time:
```
S(t_{i+1}) = S(t_i) × exp((r - σ²/2)Δt + σ√Δt × Z_i)
```

**Average Price Calculation:**

**Arithmetic Average:**
```
A_arith = (1/n) × Σ S(t_i)
```

**Geometric Average:**
```
A_geom = (Π S(t_i))^(1/n) = exp((1/n) × Σ ln(S(t_i)))
```

**Asian Option Payoffs:**
- **Asian Call**: `max(A - K, 0)`
- **Asian Put**: `max(K - A, 0)`

Where `A` is the calculated average price.

#### 🔄 Volatility Reduction Effect

Asian options exhibit **lower volatility** than European options because:
- Averaging smooths out price fluctuations
- Reduces impact of extreme price movements
- Makes options cheaper (Asian discount)

**Volatility Reduction:**
```
Reduction = 1 - (σ_Asian / σ_European)
```

#### 📈 Example – Arithmetic Asian Call

```python
from RSIM.financial import AsianOptionSimulation

# Configure arithmetic Asian call
asian_sim = AsianOptionSimulation(
    S0=100,              # Initial stock price
    K=105,               # Strike price
    T=1.0,               # 1 year to expiration
    r=0.05,              # 5% risk-free rate
    sigma=0.3,           # 30% volatility
    n_time_steps=252,    # Daily observations
    average_type='arithmetic',
    option_type='call',
    n_simulations=100000
)

result = asian_sim.run()
print(f"Asian call price: ${result.results['option_price']:.4f}")
print(f"European equivalent: ${result.results['european_equivalent_price']:.4f}")
print(f"Asian discount: {result.results['asian_discount']:.2f}%")
```

#### 📉 Example – Geometric Asian Put

```python
# Geometric averaging typically yields lower prices
geo_asian = AsianOptionSimulation(
    S0=100, K=95, T=0.5, r=0.04, sigma=0.25,
    average_type='geometric',
    option_type='put',
    n_time_steps=126,    # Semi-daily observations
    n_simulations=75000
)

result = geo_asian.run()
geo_asian.visualize()  # Show comprehensive analysis
```

#### 🎯 Example – Comparing Averaging Methods

```python
# Compare arithmetic vs geometric for same option
configs = [
    {'average_type': 'arithmetic', 'name': 'Arithmetic'},
    {'average_type': 'geometric', 'name': 'Geometric'}
]

for config in configs:
    sim = AsianOptionSimulation(
        S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
        average_type=config['average_type'],
        n_simulations=50000
    )
    result = sim.run()
    print(f"{config['name']} Asian: ${result.results['option_price']:.4f}")
```

#### 📊 Time Steps Guidelines

| Time Steps | Frequency    | Use Case                |
|------------|--------------|-------------------------|
| 12         | Monthly      | Long-term options       |
| 52         | Weekly       | Standard accuracy       |
| 252        | Daily        | High precision          |
| 1260       | Hourly       | Research/validation     |

**Note**: More time steps increase accuracy but also computational cost.

#### 🎨 Visualization Features

The `visualize()` method creates six comprehensive plots:
1. **Sample Price Paths**: Multiple trajectories with mean path
2. **Average Price Distribution**: Histogram of calculated averages
3. **Payoff Distribution**: Option payoff frequency
4. **Convergence Analysis**: Price stability tracking
5. **Average vs Final Price**: Relationship scatter plot
6. **Results Summary**: Comprehensive metrics display

#### 💡 Applications

**Asian options are ideal for:**
- **Currency Hedging**: Reduce manipulation risk
- **Commodity Trading**: Reflect average market conditions
- **Employee Stock Options**: Fairer average-based valuation
- **Portfolio Insurance**: Smoother protection
- **Volatile Markets**: Reduce extreme price impact

#### 🔍 Key Metrics Explained

**Asian Discount**: Percentage by which Asian option is cheaper than European
```
Asian Discount = (European Price - Asian Price) / European Price × 100%
```

**Volatility Reduction**: How much averaging reduces payoff volatility
```
Vol Reduction = (1 - σ_Asian/σ_European) × 100%
```

**Moneyness**: Ratio of current price to strike
```
Moneyness = S₀ / K
```

#### 📚 References

- Kemna, A. & Vorst, A. (1990). *A Pricing Method for Options Based on Average Asset Values*
- Rogers, L. & Shi, Z. (1995). *The Value of an Asian Option*
- Vecer, J. (2001). *A New PDE Approach for Pricing Arithmetic Average Asian Options*

</details>

---

### ▶️ `PortfolioSimulation`

> Perform comprehensive **multi-asset portfolio performance simulation** with advanced risk analysis, rebalancing strategies, and optimization metrics for investment decision-making.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Monte Carlo portfolio simulation** to analyze multi-asset portfolio performance over time, incorporating correlation structures, various rebalancing strategies, transaction costs, and comprehensive risk-return metrics for investment analysis and optimization.

#### 🔄 Supported Rebalancing Strategies
- **Periodic**: Daily, weekly, monthly, quarterly, annual rebalancing
- **Threshold-based**: Rebalance when weights drift beyond tolerance
- **Buy-and-hold**: No rebalancing (none)
- **Combined**: Both periodic and threshold triggers

#### 📊 Risk & Performance Metrics
- **Return**: Total, annualized, risk-adjusted returns
- **Risk**: Volatility, VaR, CVaR, maximum drawdown, downside deviation
- **Ratios**: Sharpe, Sortino, Calmar ratios
- **Distribution**: Skewness, kurtosis, win rate
- **Trading**: Transaction costs, turnover, rebalancing frequency

#### 📚 Theoretical Background

**Portfolio Mathematics:**
- Portfolio return: \( R_p = \sum_{i=1}^n w_i R_i \)
- Portfolio variance: \( \sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} \)
- Sharpe ratio: \( S = \frac{R_p - R_f}{\sigma_p} \)

**Asset Price Dynamics:**
Assets follow correlated Geometric Brownian Motion:
\[ dS_i = \mu_i S_i dt + \sigma_i S_i dW_i \]

Where correlations are modeled via:
\[ \mathbf{dW} \sim \mathcal{N}(0, \boldsymbol{\Sigma} dt) \]

**Risk Measures:**
- **Value at Risk (VaR)**: \( P(R_p \leq \text{VaR}_\alpha) = \alpha \)
- **Conditional VaR**: \( \text{CVaR}_\alpha = E[R_p | R_p \leq \text{VaR}_\alpha] \)
- **Maximum Drawdown**: \( \text{MDD} = \max_{t} \left( \frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s} \right) \)

**Rebalancing Cost:**
Transaction cost per rebalancing:
\[ C = c \sum_{i=1}^n |w_i^{\text{target}} V - w_i^{\text{current}} V| \]

#### ✅ Key Features
- Multi-asset correlation modeling via Cholesky decomposition
- Flexible rebalancing strategies with cost analysis
- Comprehensive risk-return analytics
- Efficient frontier calculation
- Monte Carlo simulation with variance reduction
- Professional-grade visualization and reporting

#### 📈 Example – Basic 60/40 Portfolio

```python
import numpy as np
from RSIM.financial import PortfolioSimulation

# Create 60/40 stocks/bonds portfolio
portfolio = PortfolioSimulation(
    assets=['Stocks', 'Bonds'],
    initial_weights=[0.6, 0.4],
    expected_returns=[0.10, 0.04],
    volatilities=[0.16, 0.05],
    correlation_matrix=[[1.0, 0.3], [0.3, 1.0]],
    simulation_days=252,  # 1 year
    rebalance_frequency='monthly'
)

result = portfolio.run()
print(f"Total Return: {result.results['total_return']:.2%}")
print(f"Sharpe Ratio: {result.results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {result.results['max_drawdown']:.2%}")

# Visualize results
portfolio.visualize()
```

#### 🎯 Example – Multi-Asset with Custom Correlations

```python
# Three-asset portfolio with custom setup
multi_asset = PortfolioSimulation(
    assets=['US_Stocks', 'Intl_Bonds', 'Commodities'],
    initial_weights=[0.5, 0.3, 0.2],
    expected_returns=[0.08, 0.04, 0.06],
    volatilities=[0.16, 0.05, 0.20],
    correlation_matrix=[
        [1.0,  0.3, 0.1],
        [0.3,  1.0, -0.1],
        [0.1, -0.1, 1.0]
    ],
    initial_capital=500000,
    simulation_days=1260,  # 5 years
    rebalance_frequency='quarterly',
    transaction_cost=0.002
)

result = multi_asset.run()
print(f"Annualized Return: {result.results['annualized_return']:.2%}")
print(f"Volatility: {result.results['portfolio_volatility']:.2%}")
print(f"VaR (95%): {result.results['var_95']:.2%}")
print(f"Transaction Costs: ${result.results['total_transaction_costs']:,.2f}")
```

#### 📉 Example – Threshold-Based Rebalancing

```python
# Active rebalancing strategy
active_portfolio = PortfolioSimulation(
    assets=['Growth', 'Value', 'Bonds'],
    initial_weights=[0.4, 0.4, 0.2],
    expected_returns=[0.12, 0.08, 0.03],
    volatilities=[0.20, 0.15, 0.04],
    rebalance_frequency='weekly',
    rebalance_threshold=0.03,  # 3% drift threshold
    transaction_cost=0.001
)

result = active_portfolio.run()
print(f"Number of Rebalances: {result.results['num_rebalances']}")
print(f"Turnover Rate: {result.results['turnover_rate']:.2%}")
print(f"Sortino Ratio: {result.results['sortino_ratio']:.3f}")
```

#### 🔍 Example – Efficient Frontier Analysis

```python
# Calculate efficient frontier
portfolio = PortfolioSimulation(
    assets=['Stocks', 'Bonds', 'REITs'],
    initial_weights=[0.6, 0.3, 0.1],
    expected_returns=[0.10, 0.04, 0.08],
    volatilities=[0.16, 0.05, 0.18]
)

efficient_frontier = portfolio.calculate_efficient_frontier(num_portfolios=1000)
optimal_weights = efficient_frontier['max_sharpe_weights']
print(f"Optimal Sharpe Portfolio Weights: {optimal_weights}")
print(f"Expected Sharpe Ratio: {efficient_frontier['max_sharpe_portfolio']['sharpe_ratio']:.3f}")
```

#### 📊 Risk Analysis Guidelines

| Metric | Interpretation |
|--------|----------------|
| Sharpe Ratio > 1.0 | Good risk-adjusted performance |
| Max Drawdown < 20% | Acceptable downside risk |
| VaR (95%) | Daily loss exceeded 5% of time |
| Sortino Ratio | Focuses on downside volatility |
| Calmar Ratio | Return per unit of max drawdown |

#### 🎛️ Parameter Optimization

```python
# Compare different rebalancing frequencies
frequencies = ['monthly', 'quarterly', 'annual', 'none']
results = {}

for freq in frequencies:
    portfolio.configure(rebalance_frequency=freq)
    result = portfolio.run()
    results[freq] = {
        'sharpe': result.results['sharpe_ratio'],
        'max_dd': result.results['max_drawdown'],
        'costs': result.results['total_transaction_costs']
    }

# Find optimal frequency
best_freq = max(results.keys(), key=lambda x: results[x]['sharpe'])
print(f"Best rebalancing frequency: {best_freq}")
```

#### 📚 Applications
- **Asset Allocation**: Optimize portfolio weights across asset classes
- **Risk Management**: Stress testing and scenario analysis
- **Strategy Backtesting**: Compare rebalancing and allocation strategies
- **Performance Attribution**: Analyze sources of portfolio returns
- **Client Reporting**: Generate professional investment reports
- **Regulatory Compliance**: Risk measurement for institutional portfolios

#### 📚 References
- Markowitz, H. (1952). *Portfolio Selection*
- Sharpe, W.F. (1966). *Mutual Fund Performance*
- Sortino, F.A. & Price, L.N. (1994). *Performance Measurement in a Downside Risk Framework*
- Bodie, Kane & Marcus. *Investments* (11th Edition)

</details>

### ▶️ `VaRSimulation`

> Comprehensive **Value at Risk (VaR) estimation** using multiple methodologies including parametric, historical simulation, Monte Carlo, and Cornish-Fisher approaches with backtesting and stress testing capabilities.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **multiple VaR estimation methods** to quantify potential portfolio losses at specified confidence levels. It provides comprehensive risk analysis including Expected Shortfall (ES), backtesting validation, and stress scenario generation for robust risk management.

#### 📐 Supported VaR Methods

- **Parametric (Variance-Covariance)**: Assumes normal distribution of returns
- **Historical Simulation**: Uses empirical distribution of historical returns
- **Monte Carlo Simulation**: Generates scenarios based on fitted distributions
- **Cornish-Fisher**: Adjusts parametric VaR for skewness and kurtosis

#### 📊 Risk Metrics Calculated

- Value at Risk (VaR) at multiple confidence levels
- Expected Shortfall (Conditional VaR)
- Sharpe, Sortino, and Calmar ratios
- Maximum drawdown and tail ratios
- Comprehensive backtesting statistics

#### 📚 Theoretical Background

**Value at Risk Definition:**
VaR answers: "What is the maximum loss we can expect with X% confidence over N days?"

Formally: P(Loss ≤ VaR_α) = α, where α is the confidence level

**VaR Methods:**

1. **Parametric VaR:**
   ```
   VaR = μ + σ × Φ⁻¹(α) × √t
   ```
   Where Φ⁻¹ is the inverse normal CDF

2. **Historical Simulation:**
   ```
   VaR = α-th percentile of historical returns
   ```

3. **Monte Carlo VaR:**
   ```
   Generate N scenarios → Calculate α-th percentile
   ```

4. **Cornish-Fisher VaR:**
   ```
   VaR = μ + σ × [z_α + (z_α² - 1)/6 × S + (z_α³ - 3z_α)/24 × K] × √t
   ```
   Where S = skewness, K = excess kurtosis

**Expected Shortfall (ES):**
```
ES_α = E[Loss | Loss > VaR_α]
```
The expected loss given that the loss exceeds VaR

#### ✅ Key Features

- Multiple VaR estimation methods for comparison
- Automatic backtesting with Kupiec and Christoffersen tests
- Stress testing scenarios (market crash, high volatility, black swan)
- Support for normal, t-distribution, and skewed-t distributions
- Comprehensive risk metrics and performance statistics
- Professional-grade visualizations and reporting

#### 📈 Example – Basic VaR Estimation

```python
import numpy as np
from RSIM.financial import VaRSimulation

# Generate sample returns or use your own data
returns = np.random.normal(0.0008, 0.02, 1000)

# Initialize VaR simulation
var_sim = VaRSimulation(
    returns=returns,
    confidence_levels=[0.95, 0.99],
    portfolio_value=1000000,
    time_horizon=1
)

# Run simulation
result = var_sim.run()

# Display results
print(f"95% Parametric VaR: ${result.results['parametric_var_95']:,.0f}")
print(f"99% Parametric VaR: ${result.results['parametric_var_99']:,.0f}")
print(f"95% Expected Shortfall: ${result.results['parametric_es_95']:,.0f}")

# Visualize results
var_sim.visualize()
```

#### 🎯 Example – Multi-Method Comparison

```python
# Compare all VaR methods
var_sim = VaRSimulation(
    mean_return=0.001,
    volatility=0.025,
    portfolio_value=5000000,
    simulation_method='t',  # Use t-distribution
    n_simulations=100000
)

result = var_sim.run()

# Compare methods
comparison = var_sim.compare_methods()
for conf_level, data in comparison.items():
    print(f"\n{conf_level} Confidence Level:")
    print(f"Recommended method: {data['recommended_method']}")
    print(f"Reason: {data['recommendation_reason']}")
    
    for method, info in data['methods'].items():
        print(f"{method}: ${info['value']:,.0f} ({info['relative_diff']:+.1f}%)")
```

#### 📉 Example – Historical Data with Backtesting

```python
# Load your historical returns
returns = np.loadtxt('portfolio_returns.csv')  # Your data

var_sim = VaRSimulation(
    returns=returns,
    confidence_levels=[0.95, 0.99],
    portfolio_value=2000000,
    window_size=252  # 1 year rolling window
)

result = var_sim.run()

# Check backtesting results
for conf_level in [95, 99]:
    bt_result = result.statistics['backtesting_results'][f'var_{conf_level}']
    print(f"\n{conf_level}% VaR Backtesting:")
    print(f"Expected violations: {bt_result['expected_violation_rate']:.1%}")
    print(f"Actual violations: {bt_result['violation_rate']:.1%}")
    print(f"Model adequate: {bt_result['model_adequate']}")
    print(f"Kupiec p-value: {bt_result['kupiec_p_value']:.3f}")
```

#### 🚨 Example – Stress Testing

```python
# Generate stress scenarios
var_sim = VaRSimulation(portfolio_value=10000000)
result = var_sim.run()

# Get stress scenarios
scenarios = var_sim.generate_stress_scenarios()

print("Stress Test Results:")
for name, scenario in scenarios.items():
    print(f"\n{scenario['description']}:")
    print(f"  Impact: ${scenario['portfolio_impact']:,.0f}")
    print(f"  Probability: {scenario['probability']:.2e}")

# Visualize stress scenarios
var_sim.plot_stress_scenarios()
```

#### 📊 Distribution Support

| Distribution | Use Case | Parameters |
|--------------|----------|------------|
| Normal | Standard market conditions | mean, volatility |
| Student's t | Fat-tailed returns | mean, volatility, degrees of freedom |
| Skewed t | Asymmetric return distributions | mean, volatility, skewness, kurtosis |

#### 🎯 Backtesting Guidelines

| Test | Purpose | Interpretation |
|------|---------|----------------|
| Kupiec POF | Tests violation rate | p-value > 0.05 → Model adequate |
| Christoffersen | Tests independence | p-value > 0.05 → No clustering |
| Combined | Overall model validity | Both tests pass → Model acceptable |

#### 📈 Confidence Level Guidelines

| Level | Use Case | Regulatory |
|-------|----------|------------|
| 90% | Internal risk management | - |
| 95% | Standard risk reporting | Some regulations |
| 99% | Conservative risk management | Basel III |
| 99.9% | Extreme risk scenarios | Stress testing |

#### 💡 Best Practices

1. **Method Selection**: Use multiple methods and compare results
2. **Data Quality**: Ensure sufficient historical data (>252 observations)
3. **Backtesting**: Always validate VaR models with out-of-sample testing
4. **Model Updates**: Regularly recalibrate parameters
5. **Stress Testing**: Complement VaR with scenario analysis
6. **Documentation**: Maintain clear methodology documentation

#### 📚 Applications

- **Portfolio Risk Management**: Daily risk monitoring and reporting
- **Regulatory Compliance**: Basel III capital requirements
- **Risk Budgeting**: Allocation of risk across strategies
- **Performance Attribution**: Risk-adjusted return analysis
- **Trading Limits**: Setting position and loss limits
- **Stress Testing**: Regulatory and internal stress scenarios

#### 📚 References

- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk*
- McNeil, A., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*
- Christoffersen, P. (2012). *Elements of Financial Risk Management*
- Basel Committee on Banking Supervision. *Basel III Framework*

</details>


### ▶️ `EOQModel`

> Implement the **Economic Order Quantity (EOQ) model** for optimal inventory management, balancing ordering costs with holding costs to minimize total inventory expenses.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements the **classical EOQ inventory optimization model** that determines the optimal order quantity minimizing total inventory costs. It provides comprehensive analysis including cost breakdowns, sensitivity analysis, inventory simulation, and policy comparisons.

#### 🎯 Key Features

- **Optimal Order Quantity Calculation**: Classic EOQ formula implementation
- **Total Cost Analysis**: Detailed breakdown of ordering vs. holding costs
- **Inventory Simulation**: Multi-cycle inventory level tracking
- **Sensitivity Analysis**: Parameter robustness testing
- **Policy Comparison**: Evaluate alternative order quantities
- **Comprehensive Visualization**: Cost curves, inventory profiles, and summaries

#### 📐 Mathematical Foundation

**EOQ Formula:**
```
Q* = √(2DK/h)
```

**Total Cost Function:**
```
TC(Q) = (D/Q) × K + (Q/2) × h
```

Where:
- `D` = Annual demand rate (units/year)
- `K` = Fixed ordering cost per order
- `h` = Holding cost per unit per year
- `Q` = Order quantity (decision variable)

**Key Relationships:**
- At optimum: Ordering Cost = Holding Cost
- Cycle time: `T* = Q*/D`
- Order frequency: `f* = D/Q*`
- Average inventory: `Q*/2 + Safety Stock`

#### 📊 Performance Metrics

- **Cost Metrics**: Total cost, ordering cost, holding cost
- **Operational Metrics**: Cycle time, order frequency, inventory turnover
- **Efficiency Metrics**: Cost per unit, inventory utilization
- **Risk Metrics**: Safety stock requirements, reorder points

#### 📚 Theoretical Properties

**Optimality Conditions:**
- Convex cost function ensures global minimum
- Square-root relationship provides natural robustness
- Cost relatively insensitive near optimum (robust to estimation errors)

**Assumptions:**
- Constant, deterministic demand
- Fixed ordering and holding costs
- Instantaneous replenishment
- No quantity discounts or stockouts

#### ✅ Applications

- **Manufacturing**: Raw material procurement planning
- **Retail**: Replenishment strategy optimization  
- **Distribution**: Warehouse inventory management
- **Procurement**: Bulk purchasing decisions
- **Supply Chain**: Multi-echelon inventory coordination

#### 📈 Example – Basic EOQ Analysis

```python
from RSIM.inventory import EOQModel

# Configure EOQ model for electronics retailer
eoq = EOQModel(
    demand_rate=1200,        # 1200 units/year
    ordering_cost=25,        # $25 per order
    unit_cost=50,           # $50 per unit
    holding_cost_rate=0.20  # 20% annual holding cost
)

# Run analysis
result = eoq.run()

print(f"Optimal Order Quantity: {result.results['optimal_quantity']:.0f} units")
print(f"Minimum Annual Cost: ${result.results['optimal_cost']:.2f}")
print(f"Order Frequency: {result.results['order_frequency']:.1f} orders/year")
print(f"Cycle Time: {result.results['cycle_time_days']:.1f} days")

# Visualize results
eoq.visualize()
```

#### 🏭 Example – Manufacturing Scenario

```python
# High-volume manufacturing with lead time and safety stock
eoq_mfg = EOQModel(
    demand_rate=50000,       # 50,000 units/year
    ordering_cost=200,       # $200 setup cost
    unit_cost=15,           # $15 per unit
    holding_cost_rate=0.25, # 25% holding cost
    lead_time=14,           # 2 weeks lead time
    safety_stock=500        # 500 units safety stock
)

result = eoq_mfg.run()
print(f"EOQ: {result.results['optimal_quantity']:.0f}")
print(f"Reorder Point: {result.results['reorder_point']:.0f}")
print(f"Total Annual Cost: ${result.results['optimal_cost']:.2f}")
```

#### 📊 Example – Policy Comparison

```python
# Compare different order quantities
eoq_compare = EOQModel(demand_rate=2000, ordering_cost=40, unit_cost=8)
result = eoq_compare.run()

# Compare alternative policies
quantities = [100, 200, 300, 400, 500]
comparison = eoq_compare.compare_policies(quantities)

for policy in comparison['results']:
    print(f"Q={policy['order_quantity']}: Cost=${policy['total_cost']:.2f}, "
          f"Penalty={policy['cost_penalty_percent']:.1f}%")
```

#### 🎯 Example – Sensitivity Analysis

```python
# EOQ with comprehensive sensitivity analysis
eoq_sensitive = EOQModel(
    demand_rate=1000,
    ordering_cost=50,
    unit_cost=20,
    holding_cost_rate=0.15,
    include_sensitivity=True
)

result = eoq_sensitive.run()

# Access sensitivity results
for param, analysis in eoq_sensitive.sensitivity_results.items():
    print(f"\nSensitivity to {param}:")
    for point in analysis[:3]:  # Show first 3 points
        print(f"  Value: {point['parameter_value']:.1f}, "
              f"EOQ: {point['eoq']:.0f}, "
              f"Cost Change: {point['cost_change_percent']:.1f}%")
```

#### 📉 Example – Custom Holding Cost

```python
# Direct specification of holding cost per unit
eoq_direct = EOQModel(
    demand_rate=800,
    ordering_cost=30,
    holding_cost_per_unit=3.50,  # $3.50/unit/year directly
    simulation_periods=12        # Simulate full year
)

result = eoq_direct.run()
print(f"Optimal Quantity: {result.results['optimal_quantity']:.0f}")
print(f"Annual Holding Cost: ${result.results['holding_cost_annual']:.2f}")
```

#### 📊 Parameter Guidelines

| Parameter | Typical Range | Impact on EOQ |
|-----------|---------------|---------------|
| Demand Rate | 100-100,000 units/year | EOQ ∝ √D |
| Ordering Cost | $10-$500 per order | EOQ ∝ √K |
| Holding Cost Rate | 10%-30% per year | EOQ ∝ 1/√h |
| Lead Time | 0-90 days | Affects reorder point |
| Safety Stock | 0-20% of EOQ | Increases holding cost |

#### 🎯 Cost Structure Analysis

**Ordering Costs Include:**
- Setup/changeover costs
- Administrative processing
- Transportation costs
- Quality inspection

**Holding Costs Include:**
- Capital/financing costs (largest component)
- Storage space and handling
- Insurance and taxes
- Obsolescence and deterioration

#### 📈 Optimization Insights

**Square-Root Law Effects:**
- Doubling demand increases EOQ by √2 ≈ 1.41
- Quadrupling ordering cost doubles EOQ
- Cost curve is relatively flat near optimum (±20% quantity → <4% cost increase)

**Practical Considerations:**
- Quantity discounts may override EOQ logic
- Storage capacity constraints
- Supplier minimum order quantities
- Cash flow and working capital limits

#### 📚 Extensions and Variations

- **EOQ with Quantity Discounts**: Price breaks analysis
- **EOQ with Planned Shortages**: Backorder cost optimization
- **Production EOQ**: Finite production rate models
- **Stochastic EOQ**: Uncertain demand/lead time
- **Multi-Product EOQ**: Shared constraint optimization

#### 📊 Validation Checklist

✅ **Parameter Validation:**
- All costs are non-negative
- Demand rate is positive
- Holding cost properly specified
- Lead time and safety stock are reasonable

✅ **Result Validation:**
- EOQ is positive and finite
- Total cost decreases then increases around optimum
- Ordering cost ≈ Holding cost at optimum
- Sensitivity analysis shows expected relationships

#### 📚 References

- Harris, F.W. (1913). *How Many Parts to Make at Once*
- Wilson, R.H. (1934). *A Scientific Routine for Stock Control*
- Hadley, G. & Whitin, T.M. (1963). *Analysis of Inventory Systems*
- Silver, E.A., Pyke, D.F. & Peterson, R. (1998). *Inventory Management and Production Planning*

</details>


### ▶️ `NewsvendorModel`

> Solve the **classic single-period inventory optimization problem** for perishable goods, balancing overstocking and understocking costs to maximize expected profit.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements the **newsvendor model** (also known as the newsboy problem), a fundamental inventory management model for single-period decisions under demand uncertainty. It determines the optimal order quantity that maximizes expected profit by balancing the costs of having too much inventory (overstocking) versus too little (understocking).

#### 🎯 Key Applications
- **Seasonal products** (fashion, holiday items)
- **Perishable goods** (newspapers, food, flowers)
- **Event planning** (catering quantities, seating capacity)
- **Overbooking decisions** (airlines, hotels)
- **Fashion retail buying**
- **One-time purchasing decisions**

#### 📐 Mathematical Foundation

**Decision Variables:**
- Q = Order quantity (decision variable)
- D = Random demand ~ F(d)

**Cost Parameters:**
- c = Unit cost (purchase/production cost per unit)
- p = Selling price (revenue per unit sold)
- s = Salvage value (value per leftover unit)
- π = Shortage penalty (cost per unit of unmet demand)

**Profit Function:**
```
Profit(Q,D) = p·min(Q,D) + s·max(Q-D,0) - π·max(D-Q,0) - c·Q
```

**Optimal Solution (Critical Fractile):**
```
F(Q*) = (p + π - c) / (p + π - s)
```

This represents the **optimal service level** - the probability of not stocking out.

#### 📊 Supported Demand Distributions
- **Normal**: Gaussian distribution with mean and standard deviation
- **Uniform**: Uniform distribution between min and max values
- **Exponential**: Exponential distribution with specified mean
- **Poisson**: Discrete Poisson distribution for count data

#### 🔍 Performance Metrics
- **Expected profit** and profit variance
- **Stockout probability** and fill rate
- **Expected sales, leftover, and shortage**
- **Risk metrics** (5th/95th percentiles, VaR)
- **Service level** and inventory turnover

#### 📚 Theoretical Insights

**Special Cases:**
- No salvage (s=0): `F(Q*) = (p - c) / (p + π)`
- No penalty (π=0): `F(Q*) = (p - c) / (p - s)`
- Classic newsvendor: `F(Q*) = (p - c) / p`

**Economic Intuition:**
- Higher shortage penalty → Higher optimal quantity
- Higher salvage value → Higher optimal quantity
- Higher unit cost → Lower optimal quantity
- More variable demand → Greater profit risk

#### 📈 Example – Basic Newsvendor Problem

```python
from RSIM.inventory import NewsvendorModel

# Fashion retailer scenario
nv = NewsvendorModel(
    unit_cost=20,           # $20 to purchase each item
    selling_price=50,       # $50 selling price
    salvage_value=5,        # $5 clearance value
    shortage_penalty=10,    # $10 goodwill loss per shortage
    demand_distribution='normal',
    demand_mean=100,        # Expected demand: 100 units
    demand_std=25,          # Demand variability: 25 units
    n_simulations=10000
)

result = nv.run()
print(f"Optimal quantity: {result.results['optimal_quantity']:.1f}")
print(f"Expected profit: ${result.results['expected_profit']:.2f}")
print(f"Service level: {result.results['critical_fractile']:.1%}")

# Visualize results
nv.visualize()
```

#### 🍎 Example – Perishable Goods (High Penalty)

```python
# Fresh produce with high shortage penalty
nv_perishable = NewsvendorModel(
    unit_cost=2,            # $2 wholesale cost
    selling_price=8,        # $8 retail price
    salvage_value=0,        # No salvage (spoils)
    shortage_penalty=15,    # High customer dissatisfaction
    demand_distribution='exponential',
    demand_mean=80,
    n_simulations=15000
)

result = nv_perishable.run()
print(f"High-penalty optimal Q: {result.results['optimal_quantity']:.1f}")
print(f"Stockout probability: {result.results['stockout_probability']:.2%}")
```

#### 📊 Example – Sensitivity Analysis

```python
import numpy as np

# Analyze sensitivity to unit cost
nv_sens = NewsvendorModel(
    unit_cost=5, selling_price=12, salvage_value=2, shortage_penalty=4
)

cost_range = np.linspace(2, 10, 20)
sensitivity = nv_sens.sensitivity_analysis('unit_cost', cost_range)

# Plot how optimal quantity changes with cost
import matplotlib.pyplot as plt
costs = [r['parameter_value'] for r in sensitivity['results']]
quantities = [r['optimal_quantity'] for r in sensitivity['results']]
profits = [r['expected_profit'] for r in sensitivity['results']]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(costs, quantities, 'b-o')
plt.xlabel('Unit Cost ($)')
plt.ylabel('Optimal Quantity')
plt.title('Optimal Quantity vs Unit Cost')

plt.subplot(1, 2, 2)
plt.plot(costs, profits, 'r-o')
plt.xlabel('Unit Cost ($)')
plt.ylabel('Expected Profit ($)')
plt.title('Expected Profit vs Unit Cost')
plt.show()
```

#### 🎲 Example – Compare Different Order Quantities

```python
# Evaluate performance at different order levels
quantities_to_test = [80, 100, 120, 140]
results = []

for q in quantities_to_test:
    performance = nv.evaluate_quantity(q)
    results.append(performance)
    print(f"Q={q}: Profit=${performance['expected_profit']:.2f}, "
          f"Stockout={performance['stockout_probability']:.1%}")
```

#### 📋 Parameter Guidelines

| Parameter | Typical Range | Impact |
|-----------|---------------|---------|
| `unit_cost` | 0.1 - 100 | Higher cost → Lower optimal Q |
| `selling_price` | > unit_cost | Higher price → Higher optimal Q |
| `salvage_value` | 0 - unit_cost | Higher salvage → Higher optimal Q |
| `shortage_penalty` | 0 - 50 | Higher penalty → Higher optimal Q |
| `demand_std` | 10-50% of mean | Higher variability → More risk |
| `n_simulations` | 5,000 - 50,000 | More sims → Better accuracy |

#### 🔧 Advanced Features

**Risk Analysis:**
- Value-at-Risk (VaR) at 5th percentile
- Expected shortfall for tail risk
- Coefficient of variation for relative risk

**Model Extensions:**
- Multi-product newsvendor with budget constraints
- Price-dependent demand models
- Supply uncertainty scenarios
- Behavioral biases incorporation

#### 📚 References
- Arrow, K.J., Harris, T., & Marschak, J. (1951). *Optimal Inventory Policy*
- Porteus, E.L. (2002). *Foundations of Stochastic Inventory Theory*
- Cachon, G. & Terwiesch, C. (2017). *Matching Supply with Demand*

</details>

### ▶️ `SQInventoryPolicy`

> Simulate **continuous review (s,Q) inventory systems** with fixed reorder points and order quantities, featuring comprehensive cost analysis and service level optimization.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements a **continuous review inventory policy** where inventory is monitored continuously, and when the inventory position drops to or below the reorder point `s`, a fixed quantity `Q` is ordered. This is one of the most widely used inventory control policies in practice.

#### 🎯 Policy Logic

- **Continuous monitoring**: Inventory position tracked at all times
- **Reorder trigger**: When inventory position ≤ s, place order for Q units
- **Fixed order quantity**: Always order exactly Q units
- **Lead time consideration**: Orders arrive after L periods (deterministic or stochastic)
- **Backorder handling**: Unmet demand is backordered and filled when inventory arrives

#### 📊 Key Performance Metrics

- **Service Levels**: Fill rate, cycle service level
- **Cost Analysis**: Holding, shortage, and ordering costs
- **Inventory Metrics**: Average inventory, turnover ratio
- **Operational Metrics**: Order frequency, stockout probability

#### 📚 Theoretical Background

**Inventory Position:**
```
Inventory Position = On-hand Inventory + On-order Inventory - Backorders
```

**Cost Structure:**
- **Holding Cost**: `h × average positive inventory`
- **Shortage Cost**: `p × average backorders`
- **Ordering Cost**: `K × number of orders`

**Total Cost Function:**
```
TC = h × E[I⁺] + p × E[I⁻] + K × (D/Q)
```

Where:
- `I⁺` = positive inventory level
- `I⁻` = backorder level
- `D` = annual demand
- `Q` = order quantity

**Service Level Metrics:**
- **Fill Rate**: `1 - (Total shortage / Total demand)`
- **Cycle Service Level**: `1 - (Periods with shortage / Total periods)`

#### 🔧 Optimization Considerations

**Economic Order Quantity (EOQ) Component:**
```
Q* = √(2KD/h)
```

**Safety Stock Calculation:**
```
SS = z_α × σ_L
```
Where `σ_L` is demand standard deviation during lead time.

**Optimal Reorder Point:**
```
s* = μ_L + SS = μ_L + z_α × σ_L
```

#### ✅ Applications

- Retail inventory management
- Manufacturing raw materials
- Spare parts inventory
- Distribution center operations
- Supply chain optimization
- Inventory policy comparison

#### 📈 Example – Basic (s,Q) Policy

```python
from RSIM.inventory import SQInventoryPolicy

# Configure basic (s,Q) policy
sq_policy = SQInventoryPolicy(
    s=50,                    # Reorder when inventory ≤ 50
    Q=100,                   # Order 100 units each time
    demand_rate=15,          # 15 units/day average demand
    demand_std=5,            # Standard deviation of 5
    lead_time=7,             # 7 days lead time
    holding_cost=2.0,        # $2/unit/day holding cost
    shortage_cost=25.0,      # $25/unit/day shortage cost
    ordering_cost=100.0,     # $100 per order
    simulation_periods=365   # Simulate 1 year
)

result = sq_policy.run()
print(f"Average inventory: {result.results['avg_inventory']:.2f}")
print(f"Fill rate: {result.results['fill_rate']:.2%}")
print(f"Total cost: ${result.results['total_cost']:.2f}")

# Visualize results
sq_policy.visualize()
```

#### 🎯 Example – High Service Level Policy

```python
# Configure for high service level
high_service = SQInventoryPolicy(
    s=80,                    # Higher reorder point
    Q=150,                   # Larger order quantity
    demand_rate=20,
    demand_std=8,
    lead_time=5,
    lead_time_std=2,         # Variable lead time
    shortage_cost=100.0,     # High shortage penalty
    simulation_periods=1000
)

result = high_service.run()
print(f"Service level: {result.results['cycle_service_level']:.2%}")
print(f"Cost per period: ${result.results['avg_cost_per_period']:.2f}")
```

#### 🔍 Example – Policy Optimization

```python
# Calculate theoretical optimal parameters
sq_optimizer = SQInventoryPolicy(
    demand_rate=25,
    demand_std=7,
    lead_time=6,
    holding_cost=3.0,
    shortage_cost=50.0,
    ordering_cost=200.0
)

optimal = sq_optimizer.calculate_optimal_policy()
print(f"Optimal Q: {optimal['Q_optimal']:.1f}")
print(f"Optimal s: {optimal['s_optimal']:.1f}")
print(f"Safety stock: {optimal['safety_stock']:.1f}")

# Apply optimal parameters
sq_optimizer.configure(
    s=optimal['s_optimal'],
    Q=optimal['Q_optimal']
)
result = sq_optimizer.run()
```

#### 📊 Example – Stochastic Lead Times

```python
# Variable lead times with uncertainty
variable_lt = SQInventoryPolicy(
    s=60,
    Q=120,
    demand_rate=18,
    demand_std=6,
    lead_time=8,             # Average lead time
    lead_time_std=3,         # Lead time variability
    simulation_periods=2000
)

result = variable_lt.run()
print(f"Inventory turnover: {result.statistics['inventory_turnover']:.2f}")
print(f"Stockout probability: {result.statistics['stockout_probability']:.3f}")
```

#### 📉 Example – Cost Sensitivity Analysis

```python
# Compare different shortage cost scenarios
shortage_costs = [10, 25, 50, 100]
results = []

for sc in shortage_costs:
    policy = SQInventoryPolicy(
        s=45, Q=90,
        shortage_cost=sc,
        simulation_periods=1000
    )
    result = policy.run()
    results.append({
        'shortage_cost': sc,
        'service_level': result.results['cycle_service_level'],
        'total_cost': result.results['total_cost']
    })

for r in results:
    print(f"Shortage cost ${r['shortage_cost']}: "
          f"Service {r['service_level']:.2%}, "
          f"Total cost ${r['total_cost']:.0f}")
```

#### 🎛️ Parameter Guidelines

| Parameter | Typical Range | Impact |
|-----------|---------------|---------|
| `s` (reorder point) | 0.5-2.0 × lead time demand | Higher s → Better service, higher holding cost |
| `Q` (order quantity) | EOQ ± 20% | Larger Q → Lower order frequency, higher holding cost |
| `shortage_cost` | 5-50 × holding_cost | Higher penalty → Higher service level |
| `lead_time_std` | 0-50% of mean | Higher variability → Need higher safety stock |

#### 📊 Visualization Features

- **Inventory levels over time**: Shows inventory position vs. physical inventory
- **Cost breakdown**: Pie chart of holding, shortage, and ordering costs
- **Service analysis**: Demand and shortage distributions
- **Performance summary**: Key metrics and policy parameters

#### 📚 References

- Silver, Pyke & Peterson (1998). *Inventory Management and Production Planning*
- Nahmias & Olsen (2015). *Production and Operations Analysis*
- Zipkin (2000). *Foundations of Inventory Management*
- Axsäter (2015). *Inventory Control*

</details>


### ▶️ `SSInventoryPolicy`

> Simulate **continuous review (s,S) inventory management** with variable order quantities, optimizing reorder points and order-up-to levels for cost-effective stock control.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements a **continuous review (s,S) inventory policy** where inventory is monitored continuously, and when the inventory position drops to or below the reorder point `s`, an order is placed to bring the inventory position up to the order-up-to level `S`. Also known as the "min-max" policy, it provides flexible order quantities based on current inventory position.

#### 🔄 Policy Logic

- **Monitor**: Continuous inventory position tracking
- **Trigger**: When inventory position ≤ s (reorder point)  
- **Order**: Variable quantity = S - current inventory position
- **Replenish**: After lead time L periods
- **Satisfy**: Demand from on-hand inventory (backorder if insufficient)

#### 📐 Mathematical Framework

**Key Variables:**
- Inventory Position = On-hand + On-order - Backorders
- Reorder Point: `s` (triggers new orders)
- Order-up-to Level: `S` (target inventory position)
- Lead Time: `L` periods (deterministic or stochastic)
- Demand: `D_t` per period (normal distribution)

**Cost Structure:**
```
Total Cost = Holding Cost + Shortage Cost + Ordering Cost
TC = h·E[I⁺] + p·E[I⁻] + K·E[Orders/period]
```

Where:
- `h` = holding cost per unit per period
- `p` = shortage cost per unit per period  
- `K` = fixed ordering cost per order
- `I⁺` = positive inventory, `I⁻` = backorders

#### 📊 Performance Metrics

**Service Levels:**
- **Fill Rate**: `1 - (Total Shortage / Total Demand)`
- **Cycle Service Level**: `1 - (Shortage Periods / Total Periods)`

**Efficiency Metrics:**
- Average inventory level and position
- Inventory turnover ratio
- Order frequency and average quantity
- Cost breakdown by category

#### 🎯 Applications

- **Retail**: Space-constrained inventory management
- **Manufacturing**: Raw materials with storage limits
- **Healthcare**: Medical supplies and pharmaceuticals  
- **Automotive**: Spare parts inventory
- **Warehousing**: Multi-SKU inventory optimization
- **Perishables**: Items with limited shelf life

#### 📈 Example – Basic (s,S) Policy

```python
from RSIM.inventory import SSInventoryPolicy

# Configure electronics retailer policy
ss_policy = SSInventoryPolicy(
    s=20,                    # Reorder when inventory ≤ 20
    S=60,                    # Order up to 60 units
    demand_rate=12,          # 12 units/day average
    demand_std=4,            # Demand variability
    lead_time=7,             # 7 days lead time
    holding_cost=2.0,        # $2/unit/day holding
    shortage_cost=25.0,      # $25/unit/day shortage
    ordering_cost=100.0,     # $100 per order
    simulation_periods=365   # Simulate 1 year
)

result = ss_policy.run()
print(f"Average inventory: {result.results['avg_inventory']:.1f} units")
print(f"Fill rate: {result.results['fill_rate']:.1%}")
print(f"Total cost: ${result.results['total_cost']:.0f}")
```

#### 🎯 Example – High Service Level Policy

```python
# Configure for 98% service level
ss_high_service = SSInventoryPolicy(
    s=35,                    # Higher reorder point
    S=80,                    # Higher order-up-to level
    demand_rate=15,
    demand_std=5,
    lead_time=5,
    lead_time_std=1,         # Variable lead time
    shortage_cost=100.0,     # High shortage penalty
    simulation_periods=1000
)

result = ss_high_service.run()
ss_high_service.visualize(show_details=True)
```

#### 🔧 Example – Optimal Policy Calculation

```python
# Calculate theoretical optimal parameters
ss_optimizer = SSInventoryPolicy(
    demand_rate=20,
    demand_std=6,
    lead_time=4,
    holding_cost=1.5,
    shortage_cost=30.0,
    ordering_cost=75.0
)

optimal = ss_optimizer.calculate_optimal_policy()
print(f"Optimal s: {optimal['s_optimal']:.1f}")
print(f"Optimal S: {optimal['S_optimal']:.1f}")
print(f"Safety stock: {optimal['safety_stock']:.1f}")
print(f"EOQ reference: {optimal['EOQ_reference']:.1f}")

# Apply optimal parameters
ss_optimizer.configure(
    s=optimal['s_optimal'],
    S=optimal['S_optimal']
)
result = ss_optimizer.run()
```

#### 📊 Example – Sensitivity Analysis

```python
# Analyze sensitivity to reorder point
sensitivity = ss_policy.sensitivity_analysis(
    parameter='s',
    values=[10, 15, 20, 25, 30, 35, 40]
)

ss_policy.plot_sensitivity_analysis(sensitivity)

# Compare cost vs service level trade-offs
for s_val in [15, 20, 25, 30]:
    ss_policy.configure(s=s_val)
    result = ss_policy.run()
    print(f"s={s_val}: Cost=${result.results['avg_cost_per_period']:.2f}, "
          f"Fill Rate={result.results['fill_rate']:.1%}")
```

#### 🆚 Example – Policy Comparison

```python
# Compare (s,S) vs (s,Q) policies
comparison = ss_policy.compare_with_sq_policy({
    's': 20,
    'Q': 40  # Fixed order quantity
})

print("Policy Comparison:")
print(f"(s,S) avg cost: ${comparison['ss_policy']['avg_cost_per_period']:.2f}")
print(f"(s,Q) avg cost: ${comparison['sq_policy']['avg_cost_per_period']:.2f}")
print(f"Cost difference: ${comparison['differences']['cost_diff']:.2f}")
```

#### 📋 Example – Comprehensive Report

```python
# Generate detailed analysis report
result = ss_policy.run()
report = ss_policy.generate_report(result)
print(report)

# Key sections include:
# - Parameter validation
# - Performance metrics  
# - Cost breakdown
# - Service level analysis
# - Theoretical benchmarks
# - Optimization recommendations
```

#### ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `s` | float | 20 | Reorder point (inventory trigger level) |
| `S` | float | 50 | Order-up-to level (target inventory) |
| `demand_rate` | float | 10 | Average demand per period |
| `demand_std` | float | 3 | Demand standard deviation |
| `lead_time` | float | 5 | Average lead time (periods) |
| `lead_time_std` | float | 1 | Lead time standard deviation |
| `holding_cost` | float | 1.0 | Holding cost per unit per period |
| `shortage_cost` | float | 10.0 | Shortage cost per unit per period |
| `ordering_cost` | float | 50.0 | Fixed cost per order |
| `initial_inventory` | float | 40 | Starting inventory level |
| `simulation_periods` | int | 1000 | Number of periods to simulate |

#### 🎯 Policy Design Guidelines

**Reorder Point (s):**
- Higher `s` → Better service, higher holding costs
- Should cover expected demand during lead time + safety stock
- Consider demand variability and lead time uncertainty

**Order-up-to Level (S):**
- Higher `S` → Larger orders, longer cycles
- Should balance ordering frequency with holding costs
- `S - s` determines maximum order quantity

**Cost Trade-offs:**
- High shortage costs → Increase both `s` and `S`
- High holding costs → Decrease `S`, optimize `s`
- High ordering costs → Increase `S - s` (larger orders)

#### 📚 Theoretical Background

The (s,S) policy is optimal for systems with:
- Fixed ordering costs
- Linear holding and shortage costs  
- Stationary demand process
- Single-item inventory

**Optimality Conditions:**
- Reorder point balances holding vs shortage costs
- Order-up-to level balances ordering vs holding costs
- Policy parameters depend on demand distribution during lead time

#### 🔍 Advantages vs Disadvantages

**Advantages:**
- ✅ Variable order quantities adapt to demand
- ✅ Optimal for systems with setup costs
- ✅ Good for items with storage constraints
- ✅ Handles demand uncertainty well

**Disadvantages:**
- ❌ More complex than (s,Q) policy
- ❌ Requires continuous monitoring
- ❌ Order quantities vary (procurement complexity)
- ❌ Parameter optimization is challenging

#### 📚 References

- Zipkin, P. (2000). *Foundations of Inventory Management*
- Silver, E.A., Pyke, D.F., & Peterson, R. (1998). *Inventory Management and Production Planning*
- Porteus, E.L. (2002). *Foundations of Stochastic Inventory Theory*
- Axsäter, S. (2015). *Inventory Control*

</details>


### ▶️ `ContinuousMarkovChain`

> Simulate **continuous-time Markov chains** with finite state spaces, featuring exponential holding times, rate matrix analysis, and comprehensive path visualization.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **continuous-time Markov chain simulation** where the system transitions between discrete states continuously in time. Unlike discrete-time chains, transitions occur at random times following exponential distributions, making it ideal for modeling real-world processes like queueing systems, chemical reactions, and population dynamics.

#### 🎯 Key Features

- **Multiple Process Types**: Birth-death, immigration-death, or custom rate matrices
- **Multi-path Simulation**: Generate multiple independent sample paths
- **Stationary Analysis**: Compute theoretical and empirical stationary distributions
- **Jump Analysis**: Detailed holding time and transition statistics
- **Comprehensive Visualization**: Sample paths, distributions, and rate matrix heatmaps

#### 📐 Mathematical Background

**State Space:** S = {0, 1, 2, ..., n-1} (finite)

**Rate Matrix (Q-matrix):**
- Q[i,j] = transition rate from state i to j (i≠j)
- Q[i,i] = -∑_{j≠i} Q[i,j] (negative exit rate from state i)

**Key Properties:**
- **Holding Times**: Exponentially distributed with rate λᵢ = -Q[i,i]
- **Transition Probabilities**: P(X(t+s) = j | X(t) = i) = [e^{Qt}]ᵢⱼ
- **Stationary Distribution**: πQ = 0, ∑πᵢ = 1
- **Generator Property**: P'(0) = Q

**Simulation Algorithm:**
1. Start in initial state
2. Sample holding time ~ Exponential(-Q[i,i])
3. At jump time, sample next state proportional to Q[i,j]/(-Q[i,i])
4. Repeat until total time T is reached

#### 🔧 Predefined Process Types

**Birth-Death Process:**
```
States: 0 → 1 → 2 → ... → n-1
Birth rate: λ (constant)
Death rate: iμ (proportional to population)
```

**Immigration-Death Process:**
```
States: 0 → 1 → 2 → ... → n-1
Immigration rate: λ (constant)
Death rate: iμ (proportional to population)
```

#### 📊 Applications

- **Queueing Systems**: M/M/k queues, call centers
- **Population Dynamics**: Birth-death processes, epidemics
- **Chemical Reactions**: Reaction networks, enzyme kinetics
- **Reliability Engineering**: System failures and repairs
- **Finance**: Credit rating transitions
- **Biology**: Gene expression, protein folding

#### 📈 Example – Birth-Death Process

```python
from RSIM.markov_chains import ContinuousMarkovChain

# Configure birth-death process
cmc = ContinuousMarkovChain(
    rate_matrix='birth_death',
    n_states=5,
    total_time=20.0,
    n_paths=10,
    birth_rate=1.5,
    death_rate=1.0,
    initial_state='random'
)

# Run simulation
result = cmc.run()

# Display results
print("Stationary distribution:", result.results['stationary_distribution'])
print("Empirical distribution:", result.results['empirical_distribution'])
print("Mean holding time:", result.results['mean_holding_time'])

# Visualize
cmc.visualize()
```

#### 🎯 Example – Custom Rate Matrix

```python
import numpy as np

# Define custom 3-state rate matrix
Q = np.array([
    [-1.0,  0.5,  0.5],  # From state 0
    [ 0.3, -0.8,  0.5],  # From state 1
    [ 0.2,  0.6, -0.8]   # From state 2
])

# Create simulation
cmc_custom = ContinuousMarkovChain(
    rate_matrix=Q,
    total_time=15.0,
    n_paths=100
)

result = cmc_custom.run()
print("Distribution error:", result.statistics['distribution_error'])
```

#### 📉 Example – Single Path Analysis

```python
# Single detailed path
cmc_single = ContinuousMarkovChain(
    rate_matrix='immigration_death',
    n_states=6,
    total_time=50.0,
    n_paths=1,
    birth_rate=2.0,
    death_rate=0.8
)

result = cmc_single.run()

# Extract path data
times, states = result.raw_data['paths'][0]
print(f"Number of jumps: {len(times)-1}")
print(f"Final state: {states[-1]}")

# Visualize single path with detailed analysis
cmc_single.visualize(show_holding_times=True)
```

#### 🔍 Advanced Analysis

```python
# Compute stationary distribution analytically
stationary = cmc.compute_stationary_distribution()
print("Theoretical stationary:", stationary)

# Access detailed statistics
print("Jump count statistics:")
print(f"  Mean jumps per path: {result.results['mean_jump_count']:.2f}")
print(f"  Total jumps: {result.results['total_jumps']}")
print(f"  Average jump rate: {result.statistics['average_rate']:.3f}")

# Holding time analysis
holding_times = result.results['holding_times']
print(f"Holding time stats:")
print(f"  Mean: {np.mean(holding_times):.3f}")
print(f"  Std: {np.std(holding_times):.3f}")
```

#### ⚙️ Parameter Guidelines

| Parameter | Typical Range | Purpose |
|-----------|---------------|---------|
| `n_states` | 2-15 | Number of discrete states |
| `total_time` | 10-100 | Simulation duration |
| `n_paths` | 1-1000 | Statistical accuracy vs. speed |
| `birth_rate` | 0.1-10 | Process intensity |
| `death_rate` | 0.1-10 | Process intensity |

#### 🎨 Visualization Features

**Single Path:**
- Step function sample path with jump markers
- Holding time distribution with exponential overlays
- State occupancy over time
- Rate matrix heatmap

**Multiple Paths:**
- Overlay of sample paths
- Final state distribution vs. stationary
- Empirical vs. theoretical distribution comparison
- Aggregate holding time and jump count distributions

#### ⚠️ Important Notes

- **Stability**: Ensure rate matrix is well-conditioned
- **Performance**: Large state spaces (>20) may be slow
- **Convergence**: Longer simulation times improve stationary distribution accuracy
- **Absorbing States**: States with zero exit rate are handled automatically

#### 📚 References

- Anderson, W. J. (2012). *Continuous-Time Markov Chains*
- Norris, J. R. (1998). *Markov Chains*
- Ross, S. M. (2014). *Introduction to Probability Models*
- Stewart, W. J. (2009). *Probability, Markov Chains, Queues, and Simulation*

</details>

### ▶️ `DiscreteMarkovChain`

> Simulate **discrete-time Markov chains** with finite state spaces, featuring comprehensive analysis of transition dynamics, stationary distributions, and convergence properties.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **discrete-time Markov chain simulation** for systems where the probability of transitioning to the next state depends only on the current state (Markov property). It provides complete analysis including stationary distribution computation, mixing time estimation, and convergence visualization.

#### 🎯 Predefined Chain Types
- **Symmetric Random Walk**: Random walk on line with reflecting boundaries
- **Birth-Death Process**: Population dynamics with birth/death transitions  
- **Random Stochastic**: Randomly generated transition matrix
- **Custom Matrix**: User-defined transition probabilities

#### 📊 Analysis Features
- Multiple chain realizations for statistical robustness
- Stationary distribution computation via eigenvalue decomposition
- Mixing time estimation using total variation distance
- State occupation time analysis and return statistics
- Convergence visualization to equilibrium distribution

#### 📚 Theoretical Background

**Markov Property:**  
The future state depends only on the present state, not the history:
```
P(X_{t+1} = j | X_t = i, X_{t-1}, ..., X_0) = P(X_{t+1} = j | X_t = i)
```

**Transition Matrix:**  
Let **P** be the transition matrix where P[i,j] = P(X_{t+1} = j | X_t = i)
- Each row sums to 1 (stochastic matrix)
- All entries are non-negative probabilities

**Chapman-Kolmogorov Equation:**
```
P^(n) = P^n  (n-step transition probabilities)
```

**Stationary Distribution π:**
```
πP = π  and  Σπ_i = 1
```
Found as the left eigenvector of P corresponding to eigenvalue 1.

**Convergence:**  
For ergodic chains: lim_{n→∞} P^n = ππᵀ

**Key Properties:**
- **Irreducibility**: All states communicate
- **Aperiodicity**: gcd of return times = 1  
- **Ergodicity**: Irreducible + aperiodic → unique stationary distribution

#### ✅ Applications
- Random walks on graphs and networks
- PageRank algorithm for web search
- Population genetics and evolution models
- Queueing systems and service processes
- Financial credit rating transitions
- Weather pattern modeling
- Game theory and strategic behavior

#### 📈 Example – Simple 3-state chain

```python
import numpy as np
from RSIM.markov_chains import DiscreteMarkovChain

# Define custom transition matrix
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3], 
              [0.2, 0.3, 0.5]])

mc = DiscreteMarkovChain(transition_matrix=P, n_steps=1000)
result = mc.run()

print("Stationary distribution:", mc.stationary_dist)
print("Empirical distribution:", result.results['empirical_distribution'])
mc.visualize()
```

#### 🚶 Example – Symmetric random walk

```python
# Random walk on 10 states with reflecting boundaries
mc_walk = DiscreteMarkovChain(
    transition_matrix='symmetric_random_walk', 
    n_states=10, 
    n_steps=2000
)
result = mc_walk.run()

print("Mixing time estimate:", result.statistics['mixing_time_estimate'])
mc_walk.visualize(show_convergence=True)
```

#### 🔄 Example – Multiple chain analysis

```python
# Run 50 independent chains for statistical analysis
mc_multi = DiscreteMarkovChain(
    transition_matrix='birth_death',
    n_states=6,
    n_chains=50,
    n_steps=1500
)
result = mc_multi.run()

print("Distribution error:", result.statistics['distribution_error'])
print("Final state distribution:", result.results['final_state_distribution'])
```

#### 🎲 Example – Custom birth-death process

```python
# Create custom birth-death chain
n_states = 5
P = np.zeros((n_states, n_states))

for i in range(n_states):
    if i == 0:  # Birth only
        P[i, i] = 0.4
        P[i, i+1] = 0.6
    elif i == n_states-1:  # Death only  
        P[i, i-1] = 0.6
        P[i, i] = 0.4
    else:  # Birth or death
        P[i, i-1] = 0.3  # Death
        P[i, i] = 0.4    # Stay
        P[i, i+1] = 0.3  # Birth

mc_bd = DiscreteMarkovChain(P, n_steps=3000)
result = mc_bd.run()
```

#### 📊 Performance Guidelines

| States | Steps | Chains | Purpose                    |
|--------|-------|--------|----------------------------|
| 2-5    | 1000  | 1      | Quick exploration          |
| 5-10   | 2000  | 10     | Standard analysis          |
| 10-20  | 5000  | 50     | Detailed convergence study |
| 20+    | 10000 | 100    | High-precision research    |

#### 🎨 Visualization Features
- **Single Chain**: State sequence, distribution comparison, convergence analysis, transition heatmap
- **Multiple Chains**: Ensemble trajectories, final state histograms, time evolution, error metrics
- **Convergence**: Total variation distance decay, mixing time identification
- **Matrix**: Transition probability heatmap with annotations

#### 📚 References
- Norris, J. R. (1998). *Markov Chains*
- Ross, S. M. (2014). *Introduction to Probability Models*  
- Levin, D. A., et al. (2017). *Markov Chains and Mixing Times*
- Kemeny & Snell (1976). *Finite Markov Chains*

</details>

### ▶️ `HiddenMarkovModel`

> Simulate and analyze **Hidden Markov Models (HMMs)** with parameter estimation, sequence generation, and state inference using Forward-Backward, Viterbi, and Baum-Welch algorithms.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Hidden Markov Model simulation** for modeling sequential data where the underlying states are hidden but generate observable outputs. It supports sequence generation, parameter estimation via EM algorithm, and optimal state path inference.

#### 🔧 Core Algorithms

- **Forward Algorithm**: Compute forward probabilities P(O₁:ₜ, Sₜ = i | λ)
- **Backward Algorithm**: Compute backward probabilities P(Oₜ₊₁:T | Sₜ = i, λ)
- **Viterbi Algorithm**: Find most likely hidden state sequence
- **Baum-Welch Algorithm**: EM-based parameter estimation from observations

#### 🎯 Predefined Models

- **Weather**: Sunny/Rainy states with Clear/Cloudy/Precipitation observations
- **DNA**: AT-rich/GC-rich regions with A/T/G/C nucleotide observations
- **Financial**: Bull/Bear markets with Up/Flat/Down price movements
- **Custom**: User-defined state and observation spaces

#### 📚 Theoretical Background

**HMM Components:**
- Hidden states: S = {s₁, s₂, ..., sₙ}
- Observable symbols: O = {o₁, o₂, ..., oₘ}
- Transition probabilities: A[i,j] = P(sⱼ at t+1 | sᵢ at t)
- Emission probabilities: B[i,k] = P(oₖ | sᵢ)
- Initial distribution: π[i] = P(s₀ = sᵢ)

**Forward Algorithm:**
```
α[t,i] = P(O₁:ₜ, Sₜ = i | λ)
α[0,i] = π[i] × B[i,O₀]
α[t,j] = [Σᵢ α[t-1,i] × A[i,j]] × B[j,Oₜ]
```

**Viterbi Algorithm:**
```
δ[t,i] = max P(S₁:ₜ₋₁, Sₜ = i, O₁:ₜ | λ)
δ[0,i] = π[i] × B[i,O₀]
δ[t,j] = [maxᵢ δ[t-1,i] × A[i,j]] × B[j,Oₜ]
```

**Baum-Welch (EM) Updates:**
```
γ[t,i] = α[t,i] × β[t,i] / P(O|λ)
ξ[t,i,j] = α[t,i] × A[i,j] × B[j,Oₜ₊₁] × β[t+1,j] / P(O|λ)

π'[i] = γ[0,i]
A'[i,j] = Σₜ ξ[t,i,j] / Σₜ γ[t,i]
B'[i,k] = Σₜ:Oₜ=k γ[t,i] / Σₜ γ[t,i]
```

#### ✅ Applications

- Speech recognition and NLP
- Bioinformatics (gene finding, protein structure)
- Financial regime modeling
- Weather prediction
- Signal processing and time series analysis
- Machine learning pattern recognition

#### 📈 Example – Weather Model Simulation

```python
from RSIM.markov_chains import HiddenMarkovModel

# Generate weather sequences
hmm = HiddenMarkovModel(
    model_type='weather',
    sequence_length=50,
    n_sequences=1
)

result = hmm.run()
hmm.visualize()

print("Observation sequence:", result.results['sequences'][0])
print("True hidden states:", result.results['hidden_states'][0])
print("Viterbi path:", result.results['viterbi_paths'][0])
print("Viterbi accuracy:", result.results['viterbi_accuracies'][0])
```

#### 🧬 Example – DNA Sequence Analysis

```python
# DNA model: AT-rich vs GC-rich regions
dna_hmm = HiddenMarkovModel(
    model_type='dna',
    sequence_length=100,
    n_sequences=5
)

result = dna_hmm.run()

# Access estimated parameters (if multiple sequences)
if 'estimated_A' in result.results:
    print("Estimated transition matrix:", result.results['estimated_A'])
    print("Estimated emission matrix:", result.results['estimated_B'])
    print("Log-likelihood:", result.results['log_likelihood'])
```

#### 💰 Example – Financial Regime Detection

```python
# Financial model: Bull vs Bear markets
finance_hmm = HiddenMarkovModel(
    model_type='financial',
    sequence_length=200,
    n_sequences=10
)

result = finance_hmm.run()
finance_hmm.visualize(show_probabilities=True)

print("Mean Viterbi accuracy:", result.results['mean_viterbi_accuracy'])
```

#### 🔧 Example – Custom Model

```python
# Custom 3-state, 4-observation model
custom_hmm = HiddenMarkovModel(
    model_type='custom',
    n_states=3,
    n_observations=4,
    sequence_length=150,
    n_sequences=20
)

result = custom_hmm.run()

# Manual parameter access
print("Transition matrix shape:", custom_hmm.A.shape)
print("Emission matrix shape:", custom_hmm.B.shape)
print("Initial distribution:", custom_hmm.pi)
```

#### 🎯 Example – Parameter Estimation

```python
# Generate training data and estimate parameters
hmm_train = HiddenMarkovModel('weather', sequence_length=100, n_sequences=50)
result = hmm_train.run()

# Compare true vs estimated parameters
true_A = result.statistics['true_A']
est_A = np.array(result.results['estimated_A'])
print("Parameter estimation error:", np.mean(np.abs(true_A - est_A)))
```

#### 📊 Performance Guidelines

| Sequences | Length | Purpose                    |
|-----------|--------|----------------------------|
| 1         | 50-200 | Sequence generation/analysis |
| 5-10      | 100+   | Basic parameter estimation |
| 20-50     | 200+   | Robust parameter learning  |
| 100+      | 500+   | High-accuracy estimation   |

#### 🔍 Key Metrics

- **Viterbi Accuracy**: Fraction of correctly inferred hidden states
- **Log-Likelihood**: Model fit quality for parameter estimation
- **Parameter MSE**: Mean squared error between true and estimated parameters
- **Convergence**: EM algorithm convergence in Baum-Welch

#### 📚 References

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications*
- Baum, L. E., et al. (1970). *A maximization technique occurring in statistical analysis of probabilistic functions*
- Viterbi, A. (1967). *Error bounds for convolutional codes and an asymptotically optimum decoding algorithm*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*

</details>

### ▶️ `MetropolisHastings` & `GibbsSampler`

> Perform advanced **Markov Chain Monte Carlo (MCMC) sampling** from complex probability distributions using Metropolis-Hastings algorithm and Gibbs sampling for Bayesian inference and statistical modeling.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

These classes implement **MCMC algorithms** to generate samples from probability distributions that are difficult to sample from directly. The Metropolis-Hastings algorithm works with any target distribution (up to a normalization constant), while Gibbs sampling is specialized for multivariate distributions with known conditional distributions.

#### 🔧 Available Algorithms

**MetropolisHastings:**
- **Random Walk Metropolis**: Gaussian proposals around current state
- **Independent Metropolis**: Proposals from fixed distribution
- **Adaptive Proposals**: Automatic tuning during burn-in
- **Multiple Target Distributions**: Standard normal, bivariate normal, mixture models, custom

**GibbsSampler:**
- **Bivariate Normal**: Correlated 2D Gaussian sampling
- **Bayesian Linear Regression**: Posterior sampling for regression parameters
- **Mixture Models**: Parameter estimation for Gaussian mixtures

#### 📊 Built-in Target Distributions

- **Standard Normal**: N(0,1) distribution
- **Bivariate Normal**: 2D correlated Gaussian
- **Mixture Gaussian**: Multi-modal distributions
- **Custom Functions**: User-defined log-probability functions

#### 📚 Theoretical Background

**Metropolis-Hastings Algorithm:**
The algorithm constructs a Markov chain with the target distribution π(x) as its stationary distribution.

1. **Proposal Step**: Generate x' ~ q(x'|x_t)
2. **Acceptance Probability**: 
   ```
   α(x,x') = min(1, [π(x')q(x|x')] / [π(x)q(x'|x)])
   ```
3. **Accept/Reject**: Set x_{t+1} = x' with probability α, otherwise x_{t+1} = x_t

**Detailed Balance Condition:**
```
π(x)P(x,x') = π(x')P(x',x)
```

**Gibbs Sampling:**
Special case where proposals are always accepted by sampling from conditional distributions:
```
x_1^{(t+1)} ~ π(x_1 | x_2^{(t)}, ..., x_p^{(t)})
x_2^{(t+1)} ~ π(x_2 | x_1^{(t+1)}, x_3^{(t)}, ..., x_p^{(t)})
...
x_p^{(t+1)} ~ π(x_p | x_1^{(t+1)}, ..., x_{p-1}^{(t+1)})
```

#### ✅ Key Features

- **Convergence Diagnostics**: Effective sample size, autocorrelation time
- **Adaptive Tuning**: Automatic proposal variance adjustment
- **Multiple Chains**: Parallel sampling support
- **Comprehensive Visualization**: Trace plots, marginal distributions, autocorrelation
- **Burn-in and Thinning**: Proper chain preprocessing

#### 📈 Example – Standard Normal Sampling

```python
from RSIM.markov_chains.mcmc import MetropolisHastings

# Sample from standard normal distribution
mh = MetropolisHastings(
    target_distribution='standard_normal',
    n_samples=10000,
    proposal_std=1.0,
    burn_in=1000
)

result = mh.run()
print(f"Sample mean: {result.results['sample_mean']}")
print(f"Acceptance rate: {result.results['acceptance_rate']:.3f}")
print(f"Effective sample size: {result.results['effective_sample_size']:.1f}")

# Visualize results
mh.visualize()
```

#### 🎯 Example – Custom Distribution

```python
# Define custom log-probability function (mixture of normals)
def log_mixture(x):
    p1 = 0.3 * np.exp(-0.5 * (x + 2)**2)
    p2 = 0.7 * np.exp(-0.5 * (x - 2)**2)
    return np.log(p1 + p2)

mh_custom = MetropolisHastings(
    target_distribution=log_mixture,
    n_samples=15000,
    proposal_std=1.5,
    burn_in=2000,
    adapt_proposal=True
)

result = mh_custom.run()
mh_custom.visualize()
```

#### 📉 Example – Gibbs Sampling for Bayesian Regression

```python
from RSIM.markov_chains.mcmc import GibbsSampler

# Sample from Bayesian linear regression posterior
gibbs = GibbsSampler(
    model_type='linear_regression',
    n_samples=5000,
    burn_in=1000,
    thin=2
)

result = gibbs.run()
print("Posterior means:", result.results['sample_mean'])
print("Posterior covariance:", result.results['sample_cov'])

# Visualize parameter traces and posteriors
gibbs.visualize()
```

#### 🔄 Example – Bivariate Normal with Gibbs

```python
# Sample from correlated bivariate normal
gibbs_bvn = GibbsSampler(
    model_type='bivariate_normal',
    n_samples=8000,
    burn_in=1500
)

result = gibbs_bvn.run()
samples = result.raw_data['samples']

# Check correlation
sample_corr = np.corrcoef(samples.T)[0, 1]
print(f"Sample correlation: {sample_corr:.3f} (true: 0.700)")
```

#### ⚙️ Tuning Guidelines

**Acceptance Rates:**
- **Optimal Range**: 20-50% for random walk Metropolis
- **Too Low** (<10%): Decrease proposal_std
- **Too High** (>70%): Increase proposal_std

**Effective Sample Size:**
- **Good**: ESS > 1000 for reliable inference
- **Excellent**: ESS > 5000 for publication quality
- **Rule of Thumb**: ESS ≈ n_samples / (2τ + 1), where τ is autocorrelation time

**Chain Length:**
| Purpose | Burn-in | Samples | Total |
|---------|---------|---------|-------|
| Exploration | 500 | 2,000 | 2,500 |
| Analysis | 1,000 | 10,000 | 11,000 |
| Publication | 2,000 | 50,000 | 52,000 |

#### 🚀 Advanced Features

**Adaptive Proposals:**
```python
mh = MetropolisHastings(
    target_distribution='bivariate_normal',
    adapt_proposal=True,  # Auto-tune during burn-in
    burn_in=2000
)
```

**Multiple Chains (Parallel):**
```python
# Run multiple independent chains
chains = []
for i in range(4):
    mh = MetropolisHastings(random_seed=i)
    result = mh.run()
    chains.append(result.raw_data['samples'])

# Combine for Gelman-Rubin diagnostic
```

#### 📊 Convergence Diagnostics

- **Trace Plots**: Visual inspection of chain mixing
- **Autocorrelation Function**: Measure of sample correlation
- **Effective Sample Size**: Independent samples equivalent
- **Gelman-Rubin Statistic**: Multi-chain convergence (R̂ < 1.1)
- **Running Averages**: Convergence of sample moments

#### 🎯 Applications

- **Bayesian Inference**: Parameter estimation with uncertainty
- **Statistical Physics**: Ising models, spin systems
- **Machine Learning**: Bayesian neural networks, latent variable models
- **Finance**: Option pricing, risk modeling
- **Biology**: Phylogenetic inference, population genetics
- **Image Processing**: Denoising, reconstruction

#### 📚 References

- Metropolis, N., et al. (1953). *Equation of State Calculations by Fast Computing Machines*
- Hastings, W. K. (1970). *Monte Carlo Sampling Methods Using Markov Chains*
- Geman, S. & Geman, D. (1984). *Stochastic Relaxation, Gibbs Distributions*
- Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods*
- Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*

</details>


### ▶️ `MonteCarloIntegration`

> Perform **high-dimensional numerical integration** using Monte Carlo sampling methods with built-in variance reduction techniques and convergence analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Monte Carlo integration** to estimate definite integrals of single and multi-dimensional functions using random sampling. It provides multiple sampling strategies, built-in test functions, and comprehensive convergence analysis for efficient integration of complex functions over specified domains.

#### 🎯 Supported Integration Methods

- **Uniform Sampling**: Random points uniformly distributed over integration domain
- **Importance Sampling**: Sample from probability distribution similar to |f(x)|
- **Stratified Sampling**: Divide domain into subregions for better coverage
- **Hit-or-Miss Method**: Geometric interpretation for positive functions

#### 🔧 Built-in Test Functions

**1D Functions:**
- `'polynomial'`: x² + 2x + 1 over [0,1], analytical = 7/3
- `'sine'`: sin(x) over [0,π], analytical = 2
- `'exponential'`: e^x over [0,1], analytical = e-1
- `'gaussian'`: e^(-x²), analytical = √π
- `'oscillatory'`: sin(10x) over [0,1]

**2D Functions:**
- `'circle'`: √(1-x²-y²) over unit disk, analytical = 2π/3
- `'paraboloid'`: x² + y² over [0,1]×[0,1], analytical = 2/3
- `'gaussian_2d'`: e^(-(x²+y²)), analytical = π

#### 📚 Theoretical Background

**Monte Carlo Principle:**
For a function f(x) over domain D with volume V:

**1D Integration:**
\[I = \int_a^b f(x) dx \approx \frac{b-a}{n} \sum_{i=1}^n f(x_i)\]

**Multi-dimensional Integration:**
\[I = \int_D f(\mathbf{x}) d\mathbf{x} \approx \frac{V}{n} \sum_{i=1}^n f(\mathbf{x}_i)\]

**Key Properties:**
- **Convergence Rate**: O(1/√n) independent of dimension
- **Standard Error**: σ ≈ V × √(Var[f(X)]/n)
- **Central Limit Theorem**: Estimates → Normal distribution
- **Dimension Independence**: Major advantage over quadrature methods

#### ✅ Advantages over Traditional Methods

- Dimension-independent O(1/√n) convergence
- Handles irregular domains naturally
- Robust for discontinuous functions
- Easily parallelizable
- Natural error estimates from sample variance
- Memory requirements independent of dimension

#### 📈 Example – Simple 1D Integration

```python
import numpy as np
from RSIM.monte_carlo import MonteCarloIntegration

# Integrate x² from 0 to 1 (analytical result = 1/3)
integrator = MonteCarloIntegration(
    function=lambda x: x**2,
    domain=(0, 1),
    n_samples=100000,
    analytical_result=1/3
)

result = integrator.run()
print(f"Integral estimate: {result.results['integral_estimate']:.6f}")
print(f"Absolute error: {result.results['absolute_error']:.6f}")
print(f"Relative error: {result.results['relative_error']:.4f}%")

# Visualize results
integrator.visualize()
```

#### 🎯 Example – Built-in Test Function

```python
# Use built-in sine function
integrator = MonteCarloIntegration(
    function='sine',
    domain=(0, np.pi),
    n_samples=1000000,
    show_convergence=True
)

result = integrator.run()
print(f"∫₀^π sin(x)dx ≈ {result.results['integral_estimate']:.6f}")
print(f"True value: 2.0")
print(f"Standard error: {result.results['standard_error']:.6f}")

# Show convergence analysis
integrator.visualize(show_function=True, show_samples=True)
```

#### 🌐 Example – Multi-dimensional Integration

```python
# 2D Gaussian integration
def gaussian_2d(x):
    return np.exp(-(x[0]**2 + x[1]**2))

integrator = MonteCarloIntegration(
    function=gaussian_2d,
    domain=[(-3, 3), (-3, 3)],
    n_samples=500000,
    method='uniform',
    analytical_result=np.pi
)

result = integrator.run()
print(f"2D Gaussian integral: {result.results['integral_estimate']:.6f}")
print(f"95% Confidence Interval: {result.statistics['confidence_interval_95']}")

# Visualize 2D integration
integrator.visualize(show_function=True)
```

#### 🔬 Example – High-dimensional Integration

```python
# 10D hypersphere volume estimation
def hypersphere_indicator(x, dim=10):
    return 1.0 if np.sum(x**2) <= 1 else 0.0

integrator = MonteCarloIntegration(
    function=lambda x: hypersphere_indicator(x, 10),
    domain=[(-1, 1)] * 10,  # 10D unit hypercube
    n_samples=10000000,
    method='uniform'
)

result = integrator.run()
volume_estimate = result.results['integral_estimate']
print(f"10D unit hypersphere volume: {volume_estimate:.8f}")
print(f"Theoretical volume: {np.pi**5 / 120:.8f}")
```

#### 🎨 Example – Custom Function with Importance Sampling

```python
# Add custom peaked function
def peaked_function(x):
    return np.exp(-100 * (x - 0.5)**2)

integrator = MonteCarloIntegration(
    function=peaked_function,
    domain=(0, 1),
    n_samples=1000000,
    method='uniform'
)

result = integrator.run()
print(f"Peaked function integral: {result.results['integral_estimate']:.6f}")
print(f"Standard error: {result.results['standard_error']:.6f}")
```

#### 📊 Performance Guidelines

| Dimension | Recommended Samples | Typical Use Cases |
|-----------|-------------------|-------------------|
| 1D        | 10⁴ - 10⁶        | Simple functions, quick estimates |
| 2D-3D     | 10⁵ - 10⁷        | Engineering applications |
| 4D-10D    | 10⁶ - 10⁸        | Finance, physics simulations |
| >10D      | 10⁷ - 10⁹        | Machine learning, Bayesian inference |

#### 🔍 Error Analysis Features

- **Absolute Error**: |estimate - true_value|
- **Relative Error**: Percentage deviation from true value
- **Standard Error**: Sample standard deviation / √n
- **95% Confidence Intervals**: estimate ± 1.96 × standard_error
- **Convergence Tracking**: Real-time monitoring of estimate evolution

#### 📈 Visualization Features

**1D Functions:**
- Function plot with integration region highlighted
- Sample points overlay showing Monte Carlo sampling
- Convergence plot with confidence bounds
- Error evolution analysis

**2D Functions:**
- 3D surface plots and contour maps
- Sample points projected onto domain
- Convergence analysis with statistical summaries

**Higher Dimensions:**
- Convergence plots and statistical summaries
- Projection plots for visualization
- Comprehensive error analysis

#### 🎯 Applications

- **Finance**: Derivatives pricing, risk assessment, portfolio optimization
- **Physics**: Quantum mechanics, statistical mechanics, particle simulations
- **Engineering**: Reliability analysis, uncertainty quantification
- **Statistics**: Bayesian inference, expectation computations
- **Machine Learning**: High-dimensional probability calculations
- **Computer Graphics**: Global illumination, rendering algorithms

#### 📚 References

- Hammersley & Handscomb (1964). *Monte Carlo Methods*
- Robert & Casella (2004). *Monte Carlo Statistical Methods*
- Glasserman (2003). *Monte Carlo Methods in Financial Engineering*
- Liu (2001). *Monte Carlo Strategies in Scientific Computing*

</details>

### ▶️ `OptionPricingMC`

> Perform **Monte Carlo option pricing** using the Black-Scholes-Merton framework with comprehensive statistical analysis, Greeks calculation, and convergence tracking.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Monte Carlo simulation for European option pricing** by simulating stock price paths using geometric Brownian motion and calculating expected payoffs under risk-neutral valuation. It supports both call and put options with detailed statistical analysis and comparison against analytical Black-Scholes prices.

#### 📐 Supported Option Types
- **Call Options**: Right to buy at strike price
- **Put Options**: Right to sell at strike price
- **European Exercise**: Exercise only at expiration

#### 📊 Key Features
- Monte Carlo price estimation with confidence intervals
- Black-Scholes analytical benchmark comparison
- Option Greeks approximation (Delta, Gamma, Theta, Vega, Rho)
- Convergence analysis and visualization
- Comprehensive error analysis and statistics
- Stock price path visualization
- Sensitivity analysis capabilities

#### 📚 Theoretical Background

**Stock Price Dynamics (Geometric Brownian Motion):**
```
dS(t) = μS(t)dt + σS(t)dW(t)
S(T) = S(0) × exp((r - σ²/2)T + σ√T × Z)
```
Where Z ~ N(0,1) is standard normal random variable.

**Risk-Neutral Valuation:**
Under risk-neutral measure, drift μ = r (risk-free rate):
- **Call price** = e^(-rT) × E[max(S(T) - K, 0)]
- **Put price** = e^(-rT) × E[max(K - S(T), 0)]

**Black-Scholes Formula (Analytical Benchmark):**
- **Call**: C = S₀N(d₁) - Ke^(-rT)N(d₂)
- **Put**: P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
- d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
- d₂ = d₁ - σ√T

**Statistical Properties:**
- Standard error: σ_MC = √(Var[payoff]/n) / e^(rT)
- Convergence rate: O(1/√n)
- 95% confidence interval: price_estimate ± 1.96 × σ_MC

#### ✅ Algorithm Steps
1. Generate n random standard normal variables Z_i
2. Calculate terminal stock prices: S_T,i = S₀ × exp((r - σ²/2)T + σ√T × Z_i)
3. Calculate payoffs: max(S_T,i - K, 0) for calls, max(K - S_T,i, 0) for puts
4. Estimate option price: C ≈ e^(-rT) × (1/n) × Σ Payoff_i
5. Track convergence and calculate confidence intervals

#### 📈 Example – Basic Call Option Pricing

```python
from RSIM.monte_carlo import OptionPricingMC

# Price a call option
option_sim = OptionPricingMC(
    S0=100,           # Current stock price
    K=105,            # Strike price
    T=0.25,           # 3 months to expiration
    r=0.05,           # 5% risk-free rate
    sigma=0.2,        # 20% volatility
    option_type='call',
    n_simulations=100000
)

result = option_sim.run()
print(f"Call price: ${result.results['option_price']:.4f}")
print(f"Black-Scholes: ${result.results['analytical_price']:.4f}")
print(f"Error: ${result.results['absolute_error']:.4f}")
```

#### 🎯 Example – Put Option with Greeks

```python
# Price a put option and calculate Greeks
put_sim = OptionPricingMC(
    S0=100, K=95, T=1.0, r=0.05, sigma=0.3,
    option_type='put',
    n_simulations=500000,
    random_seed=42
)

result = put_sim.run()
greeks = put_sim.calculate_greeks()

print(f"Put price: ${result.results['option_price']:.4f}")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta (daily): {greeks['theta_daily']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

#### 📉 Example – At-the-Money Option with Visualization

```python
# At-the-money option with path visualization
atm_sim = OptionPricingMC(
    S0=100, K=100, T=0.5, r=0.03, sigma=0.25,
    option_type='call',
    n_simulations=200000,
    show_convergence=True
)

result = atm_sim.run()
atm_sim.visualize(show_paths=True, n_display_paths=50)

print(f"Moneyness: {result.results['moneyness_description']}")
print(f"95% CI: [${result.results['confidence_interval_lower']:.4f}, "
      f"${result.results['confidence_interval_upper']:.4f}]")
```

#### 🔍 Example – Sensitivity Analysis

```python
# Analyze sensitivity to volatility
volatilities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
sensitivity = option_sim.sensitivity_analysis('sigma', volatilities)

print("Volatility Sensitivity:")
for vol, price in zip(sensitivity['parameter_values'], sensitivity['option_prices']):
    print(f"σ = {vol:.2f}: Price = ${price:.4f}")
```

#### 🎲 Example – Implied Volatility Calculation

```python
# Calculate implied volatility from market price
market_price = 8.50
implied_vol = option_sim.implied_volatility(market_price)
print(f"Implied volatility: {implied_vol:.4f} ({implied_vol:.2%})")
```

#### 📊 Accuracy Guidelines

| Simulations | Typical Error | Use Case |
|-------------|---------------|----------|
| 10³         | ±$0.10        | Quick estimates |
| 10⁴         | ±$0.03        | Reasonable accuracy |
| 10⁵         | ±$0.01        | Good precision |
| 10⁶         | ±$0.003       | High precision |
| 10⁷         | ±$0.001       | Research quality |

#### 🎯 Option Greeks Approximation

The simulation calculates option sensitivities using finite differences:
- **Delta (Δ)**: ∂V/∂S ≈ [V(S+h) - V(S-h)] / (2h)
- **Gamma (Γ)**: ∂²V/∂S² ≈ [V(S+h) - 2V(S) + V(S-h)] / h²
- **Theta (Θ)**: ∂V/∂T ≈ [V(T+h) - V(T)] / h
- **Vega (ν)**: ∂V/∂σ ≈ [V(σ+h) - V(σ)] / h
- **Rho (ρ)**: ∂V/∂r ≈ [V(r+h) - V(r)] / h

#### 🚀 Applications
- European option pricing and validation
- Risk management and portfolio valuation
- Model validation against analytical solutions
- Educational tool for derivatives pricing
- Benchmark for variance reduction techniques
- Sensitivity analysis and scenario testing

#### 📚 References
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Boyle, P. P. (1977). *Options: A Monte Carlo Approach*
- Hull, J. C. (2017). *Options, Futures, and Other Derivatives*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*

</details>

### ▶️ `PiEstimationMC`

> Estimate **π (pi) using Monte Carlo simulation** with the classic circle-square sampling method, featuring convergence tracking and comprehensive statistical analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements the **Monte Carlo circle-square method** to estimate π by randomly sampling points in a 2×2 square and counting how many fall within the inscribed unit circle. The ratio approximates π/4, providing an elegant demonstration of Monte Carlo integration principles.

#### 🎯 Core Algorithm

1. **Generate** random points (x,y) uniformly in [-1,1] × [-1,1]
2. **Test** if x² + y² ≤ 1 (inside unit circle)
3. **Count** hits inside circle
4. **Estimate** π = 4 × (hits / total_samples)
5. **Track** convergence over sampling progression

#### 📐 Mathematical Background

**Geometric Principle:**
- Unit circle area: A_circle = π × r² = π
- Square area: A_square = (2r)² = 4
- Ratio: A_circle / A_square = π/4
- Point (x,y) is inside circle if: x² + y² ≤ 1

**Statistical Properties:**
- Standard error: σ ≈ √(π(4-π)/n) ≈ 1.64/√n
- Convergence rate: O(1/√n) - typical for Monte Carlo methods
- 95% confidence interval: π_estimate ± 1.96 × σ
- Expected value: E[π_estimate] = π (unbiased estimator)

#### 📊 Accuracy Guidelines

| Samples | Typical Error | Use Case |
|---------|---------------|----------|
| 10³     | ±0.05        | Educational demonstrations |
| 10⁴     | ±0.016       | Quick estimates |
| 10⁶     | ±0.0016      | Good accuracy |
| 10⁸     | ±0.0005      | High precision |
| 10⁹     | ±0.00016     | Research quality |

#### 🎨 Visualization Features

**Standard Mode:**
- Results summary with π estimate and error metrics
- Convergence plot showing approach to true π
- Statistical confidence intervals

**Point Visualization Mode:**
- Scatter plot of sample points (red=inside, blue=outside)
- Unit circle overlay
- Visual demonstration of geometric principle

#### 📈 Example – Basic π estimation

```python
from RSIM.monte_carlo import PiEstimationMC

# Quick π estimation
pi_sim = PiEstimationMC(n_samples=100000, random_seed=42)
result = pi_sim.run()

print(f"π estimate: {result.results['pi_estimate']:.6f}")
print(f"True π: {3.141593:.6f}")
print(f"Absolute error: {result.results['accuracy']:.6f}")
print(f"Relative error: {result.results['relative_error']:.4f}%")
```

#### 🎯 Example – High-precision with convergence

```python
# High-precision estimation with convergence tracking
pi_precise = PiEstimationMC(n_samples=10000000, show_convergence=True)
result = pi_precise.run()

# Visualize results and convergence
pi_precise.visualize()

print(f"Final estimate: {result.results['pi_estimate']:.8f}")
print(f"Execution time: {result.execution_time:.2f} seconds")
```

#### 🎨 Example – Educational visualization

```python
# Visual demonstration with point sampling
pi_visual = PiEstimationMC(n_samples=5000, random_seed=123)
result = pi_visual.run()

# Show the actual sampling points
pi_visual.visualize(show_points=True, n_display_points=2000)
```

#### ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | int | 1,000,000 | Number of random points to generate |
| `show_convergence` | bool | True | Track convergence during simulation |
| `random_seed` | int | None | Seed for reproducible results |

#### 📊 Performance Characteristics

- **Time complexity:** O(n_samples)
- **Space complexity:** O(1) standard, O(n_samples) with convergence
- **Typical speed:** ~1M samples/second on modern hardware
- **Memory usage:** ~80 bytes per sample (only if n_samples ≤ 10,000)
- **Parallelizable:** Independent samples can be distributed

#### 🎓 Educational Applications

- **Monte Carlo method introduction**
- **Law of Large Numbers demonstration**
- **Statistical convergence analysis**
- **Geometric probability illustration**
- **Random number generator testing**
- **Computational geometry examples**

#### 📈 Historical Context

- One of the first Monte Carlo applications (1940s)
- Used in early computer testing and validation
- Classic example in computational physics
- Demonstrates Central Limit Theorem in practice

#### 🔬 Advanced Features

**Error Analysis:**
- Absolute error: |π_estimate - π|
- Relative error: |π_estimate - π| / π × 100%
- Theoretical standard error calculation
- 95% confidence interval bounds

**Convergence Tracking:**
- Real-time estimate progression
- Visual convergence plots
- Statistical error bounds
- Performance timing metrics

#### 📚 References

- Metropolis, N. & Ulam, S. (1949). *The Monte Carlo Method*
- Hammersley, J. M. & Handscomb, D. C. (1964). *Monte Carlo Methods*
- Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods*
- Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*

</details>

### ▶️ `RandomWalk1D` & `RandomWalk2D`

> Simulate **stochastic random walk processes** in 1D and 2D spaces with comprehensive statistical analysis, theoretical comparisons, and rich visualizations for diffusion modeling.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

These classes implement **Monte Carlo random walk simulations** that model the stochastic movement of particles, agents, or processes over discrete time steps. The 1D version simulates movement along a line with configurable bias, while the 2D version models movement on a square lattice with equal probability in four cardinal directions.

#### 🎯 Key Features

**RandomWalk1D:**
- Configurable step probability (symmetric or biased walks)
- Variable step sizes and number of steps
- Multiple independent walks for ensemble statistics
- Real-time convergence analysis

**RandomWalk2D:**
- Equal probability movement in 4 cardinal directions
- Distance tracking from origin over time
- Path visualization with start/end markers
- Mean squared displacement analysis

#### 📐 Mathematical Background

**1D Random Walk:**
- Position after n steps: X_n = X_0 + Σ(i=1 to n) S_i
- Expected displacement: E[X_n] = n × step_size × (2p - 1)
- Variance: Var[X_n] = n × step_size² × 4p(1-p)
- For symmetric walk (p=0.5): E[X_n] = 0, Var[X_n] = n × step_size²

**2D Random Walk:**
- Position: (X_n, Y_n) = (X_0, Y_0) + Σ(i=1 to n) (S_x_i, S_y_i)
- Expected displacement: E[X_n] = E[Y_n] = 0 (symmetric)
- Variance per dimension: Var[X_n] = Var[Y_n] = n × step_size²
- Expected distance²: E[R²_n] = 2n × step_size²
- Expected distance: E[R_n] ≈ step_size × √(πn/2) for large n

#### 🔬 Applications

- **Physics**: Brownian motion, particle diffusion, polymer chains
- **Biology**: Animal foraging patterns, cell migration, ecological dispersal
- **Finance**: Stock price movements, market fluctuations
- **Computer Science**: Search algorithms, network routing
- **Chemistry**: Molecular dynamics, reaction-diffusion processes
- **Economics**: Consumer behavior modeling, market dynamics

#### 📊 Statistical Properties

**1D Properties:**
- Recurrent (returns to origin infinitely often)
- Central Limit Theorem applies for large n
- Displacement variance grows linearly with time
- Configurable bias creates drift

**2D Properties:**
- Recurrent in 2D (will return to origin infinitely often)
- Final positions follow 2D Gaussian distribution for large n
- Distance distribution approaches Rayleigh distribution
- Isotropic (no preferred direction)

#### 🎯 Example – Simple 1D symmetric walk

```python
from RSIM.monte_carlo.random_walk import RandomWalk1D

# Simple symmetric random walk
rw = RandomWalk1D(n_steps=1000, random_seed=42)
result = rw.run()

print(f"Final position: {result.results['final_positions'][0]}")
print(f"Theoretical mean: {result.statistics['theoretical_mean']}")
print(f"Empirical mean: {result.statistics['empirical_mean']}")

# Visualize the walk
rw.visualize()
```

#### 📈 Example – Biased 1D walk (drift)

```python
# Biased random walk with rightward tendency
rw_biased = RandomWalk1D(
    n_steps=1000, 
    step_probability=0.7,  # 70% chance of positive step
    step_size=1.0
)
result = rw_biased.run()

print(f"Expected drift: {result.statistics['theoretical_mean']}")
print(f"Actual final position: {result.results['final_positions'][0]}")
```

#### 🎯 Example – Multiple 1D walks for statistics

```python
# Multiple walks for ensemble analysis
rw_multi = RandomWalk1D(
    n_steps=500, 
    n_walks=100,
    step_probability=0.5
)
result = rw_multi.run()

print(f"Mean final position: {result.results['mean_final_position']:.3f}")
print(f"Std final position: {result.results['std_final_position']:.3f}")
print(f"Theoretical std: {result.statistics['theoretical_std']:.3f}")

# Rich visualization with distributions and statistics
rw_multi.visualize()
```

#### 🌐 Example – 2D random walk

```python
from RSIM.monte_carlo.random_walk import RandomWalk2D

# Single 2D random walk
rw2d = RandomWalk2D(n_steps=2000, random_seed=42)
result = rw2d.run()

final_x = result.results['final_x_positions'][0]
final_y = result.results['final_y_positions'][0]
distance = result.results['final_distances'][0]

print(f"Final position: ({final_x:.2f}, {final_y:.2f})")
print(f"Distance from origin: {distance:.2f}")

# Visualize 2D path
rw2d.visualize()
```

#### 📊 Example – 2D ensemble analysis

```python
# Multiple 2D walks for statistical analysis
rw2d_multi = RandomWalk2D(n_steps=1000, n_walks=100)
result = rw2d_multi.run()

print(f"Mean distance from origin: {result.results['mean_final_distance']:.2f}")
print(f"Theoretical MSD: {result.statistics['theoretical_mean_distance_squared']:.2f}")
print(f"Empirical MSD: {result.statistics['empirical_mean_distance_squared']:.2f}")

# Comprehensive visualization with paths, distributions, and MSD growth
rw2d_multi.visualize()
```

#### 🎨 Visualization Features

**1D Single Walk:**
- Walk trajectory over time
- Step size distribution
- Distance statistics overlay

**1D Multiple Walks:**
- Overlay of all walk paths
- Final position histogram with theoretical comparison
- Maximum distance distributions
- Theoretical vs empirical statistics comparison
- Ensemble mean position evolution
- Variance growth over time

**2D Single Walk:**
- 2D path plot with start/end markers
- Distance from origin over time
- X and Y coordinates vs time

**2D Multiple Walks:**
- Overlay of all walk paths
- Final position scatter plot
- Distance distribution histogram
- X and Y coordinate distributions
- Mean squared displacement growth analysis

#### ⚡ Performance Guidelines

| Parameter | 1D Recommended | 2D Recommended |
|-----------|----------------|----------------|
| n_steps   | ≤ 100,000     | ≤ 50,000      |
| n_walks   | ≤ 1,000       | ≤ 500         |
| Memory    | O(n_steps × n_walks) | O(n_steps × n_walks) |

#### 🔧 Parameter Configuration

```python
# Get parameter information for UI generation
param_info = rw.get_parameter_info()

# Validate parameters
errors = rw.validate_parameters()
if errors:
    print("Parameter errors:", errors)
```

#### 📚 References

- Weiss, G. H. (1994). *Aspects and Applications of the Random Walk*
- Spitzer, F. (2001). *Principles of Random Walk*  
- Hughes, B. D. (1995). *Random Walks and Random Environments*
- Berg, H. C. (1993). *Random Walks in Biology*
- Redner, S. (2001). *A Guide to First-Passage Processes*

</details>

### ▶️ `SIRModel` & `SEIRModel`

> Simulate **epidemic spread on networks** using compartmental models (SIR/SEIR) to analyze disease transmission dynamics, outbreak patterns, and intervention strategies.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

These classes implement **network-based epidemic simulations** using compartmental models that track disease progression through populations. The SIR model divides individuals into Susceptible-Infected-Recovered states, while SEIR adds an Exposed (incubation) compartment for diseases with latency periods.

#### 🦠 Epidemic Models

**SIR Model:**
- **S (Susceptible)**: Can become infected
- **I (Infected)**: Infectious and can transmit disease  
- **R (Recovered)**: Immune to reinfection

**SEIR Model:**
- **S (Susceptible)**: Can become infected
- **E (Exposed)**: Infected but not yet infectious (incubation)
- **I (Infected)**: Infectious and can transmit disease
- **R (Recovered)**: Immune to reinfection

#### 📐 Mathematical Background

**SIR Transitions:**
- S → I: Infection probability = 1 - (1-β)^(infected neighbors)
- I → R: Recovery probability = γ per time step

**SEIR Transitions:**
- S → E: Exposure probability = 1 - (1-β)^(infected neighbors)  
- E → I: Incubation probability = σ per time step
- I → R: Recovery probability = γ per time step

**Key Parameters:**
- β: Infection rate (transmission probability)
- γ: Recovery rate (1/infectious period)
- σ: Incubation rate (1/incubation period) [SEIR only]
- R₀: Basic reproduction number = β/γ × ⟨k⟩

**Epidemic Threshold:**
- R₀ > 1: Epidemic spreads
- R₀ < 1: Epidemic dies out
- R₀ = 1: Critical threshold

#### 🌐 Network Effects

- **Degree heterogeneity**: High-degree nodes drive transmission
- **Clustering**: Local clustering slows spread
- **Path length**: Short paths accelerate spread
- **Network structure**: Scale-free networks more vulnerable than random

#### 📊 Key Metrics

- **Attack rate**: Final proportion infected
- **Peak infected**: Maximum simultaneous infections
- **Epidemic duration**: Time until no active cases
- **R₀ estimate**: Effective reproduction number
- **Final epidemic size**: Total recovered individuals

#### ✅ Applications

- **Disease outbreaks**: COVID-19, influenza, measles
- **Computer viruses**: Malware propagation
- **Information spread**: Rumors, news, social contagion
- **Innovation adoption**: Technology diffusion
- **Marketing**: Viral marketing campaigns
- **Biological systems**: Protein folding, gene expression

#### 📈 Example – Basic SIR Epidemic

```python
import networkx as nx
from RSIM.networks.epidemic_models import SIRModel

# Create network (scale-free for realistic contact patterns)
network = nx.barabasi_albert_graph(500, 3)

# Configure SIR model
sir = SIRModel(
    network=network,
    infection_rate=0.1,      # 10% transmission per contact
    recovery_rate=0.05,      # 5% recovery per day
    initial_infected=5,      # Start with 5 infected
    max_time_steps=200
)

# Run simulation
result = sir.run()

# Display results
print(f"Attack rate: {result.results['attack_rate']:.1%}")
print(f"Peak infected: {result.results['peak_infected']}")
print(f"R₀ estimate: {result.statistics['r0_estimate']:.2f}")

# Visualize epidemic curves and network
sir.visualize()
```

#### 🦠 Example – SEIR with Incubation Period

```python
from RSIM.networks.epidemic_models import SEIRModel

# Configure SEIR for disease with 5-day incubation
seir = SEIRModel(
    network=network,
    infection_rate=0.12,     # Higher transmission rate
    incubation_rate=0.2,     # 1/5 = 5-day incubation
    recovery_rate=0.1,       # 10-day infectious period
    initial_infected=3,
    max_time_steps=300
)

result = seir.run()

print(f"Peak exposed: {result.results['peak_exposed']}")
print(f"Peak infected: {result.results['peak_infected']}")
print(f"Incubation period: {result.statistics['incubation_period']:.1f} days")
print(f"Infectious period: {result.statistics['infectious_period']:.1f} days")

seir.visualize()
```

#### 🎯 Example – Intervention Analysis

```python
# Compare different intervention scenarios
scenarios = [
    {"name": "No intervention", "infection_rate": 0.15},
    {"name": "Social distancing", "infection_rate": 0.08},
    {"name": "Lockdown", "infection_rate": 0.03}
]

results = {}
for scenario in scenarios:
    sir = SIRModel(
        network=network,
        infection_rate=scenario["infection_rate"],
        recovery_rate=0.1,
        initial_infected=10,
        random_seed=42  # For fair comparison
    )
    results[scenario["name"]] = sir.run()

# Compare attack rates
for name, result in results.items():
    attack_rate = result.results['attack_rate']
    peak = result.results['peak_infected']
    print(f"{name}: {attack_rate:.1%} attack rate, peak {peak}")
```

#### 📊 Example – Network Structure Impact

```python
# Compare epidemic spread on different network types
networks = {
    "Random": nx.erdos_renyi_graph(500, 0.012),
    "Small-world": nx.watts_strogatz_graph(500, 6, 0.3),
    "Scale-free": nx.barabasi_albert_graph(500, 3),
    "Regular": nx.random_regular_graph(6, 500)
}

for net_type, net in networks.items():
    sir = SIRModel(
        network=net,
        infection_rate=0.1,
        recovery_rate=0.05,
        initial_infected=5,
        random_seed=42
    )
    result = sir.run()
    
    print(f"{net_type:12} | Attack rate: {result.results['attack_rate']:.1%} | "
          f"R₀: {result.statistics['r0_estimate']:.2f}")
```

#### 🔬 Advanced Features

**Comprehensive Visualization:**
- Epidemic curves (S, I, R over time)
- Network visualization with final states
- Attack rate vs node degree analysis
- Phase space plots (S vs I)
- Instantaneous infection rates

**Statistical Analysis:**
- R₀ estimation from network structure
- Peak timing and magnitude
- Epidemic duration and final size
- Degree-stratified attack rates

**Parameter Validation:**
- Automatic bounds checking
- Stability analysis (ρ < 1 requirement)
- Network connectivity verification

#### 📚 Model Extensions

- **Vaccination**: Pre-immune individuals
- **Waning immunity**: R → S transitions
- **Multiple strains**: Competing variants
- **Behavioral changes**: Dynamic contact rates
- **Spatial structure**: Geographic networks
- **Age structure**: Heterogeneous mixing

#### 🎯 Parameter Guidelines

| Parameter | Typical Range | Disease Example |
|-----------|---------------|-----------------|
| β (infection_rate) | 0.01-0.3 | 0.05 (seasonal flu), 0.15 (COVID-19) |
| γ (recovery_rate) | 0.05-0.2 | 0.1 (10-day infectious period) |
| σ (incubation_rate) | 0.1-0.5 | 0.2 (5-day incubation) |
| Initial infected | 1-10 | Depends on outbreak size |

#### 📚 References

- Keeling & Rohani (2008). *Modeling Infectious Diseases*
- Pastor-Satorras & Vespignani (2001). *Epidemic Spreading in Scale-Free Networks*
- Newman (2002). *Spread of Epidemic Disease on Networks*
- Hethcote (2000). *The Mathematics of Infectious Diseases*

</details>

### ▶️ `ErdosRenyiGraph`
> Generate and analyze **Erdős-Rényi random graphs** with comprehensive network metrics, theoretical comparisons, and rich visualizations for network science applications.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does
This class implements the **G(n,p) Erdős-Rényi random graph model** where each possible edge between n vertices is included independently with probability p. It provides complete network analysis including connectivity, clustering, component structure, and comparison with theoretical predictions.

#### 🔗 Network Properties Analyzed
- **Basic Metrics**: Nodes, edges, density, degree distribution
- **Connectivity**: Connected components, diameter, average path length
- **Clustering**: Local and global clustering coefficients
- **Components**: Size distribution and largest component analysis
- **Degree Statistics**: Mean, variance, and distribution shape

#### 📚 Theoretical Background
**Erdős-Rényi Model G(n,p):**
- n vertices, each edge exists independently with probability p
- Expected edges: E[m] = p × n(n-1)/2
- Expected degree: E[k] = p(n-1)
- Degree distribution: Binomial(n-1, p) → Poisson(λ) for large n

**Phase Transitions:**
- **Subcritical (p < 1/n)**: Only small components exist
- **Critical (p ≈ 1/n)**: Giant component emerges
- **Supercritical (p > 1/n)**: Giant component dominates

**Key Properties:**
- Clustering coefficient: C = p (independent edges)
- Diameter: O(log n) in connected regime
- Degree distribution: Poisson for large n with λ = p(n-1)

#### ✅ Applications
- **Social Networks**: Friendship modeling and analysis
- **Internet Topology**: Network structure studies
- **Epidemic Models**: Disease spread on random contacts
- **Percolation Theory**: Connectivity threshold analysis
- **Benchmarking**: Baseline for complex network comparisons
- **Communication Networks**: Random connectivity patterns

#### 📈 Example – Basic Random Graph
```python
from RSIM.networks.random_graphs import ErdosRenyiGraph

# Create Erdős-Rényi graph
er_graph = ErdosRenyiGraph(n_nodes=100, edge_probability=0.05)
result = er_graph.run()

print(f"Nodes: {result.results['n_nodes']}")
print(f"Edges: {result.results['n_edges']}")
print(f"Connected: {result.results['is_connected']}")
print(f"Components: {result.results['n_components']}")

# Visualize results
er_graph.visualize()
```

#### 🎯 Example – Phase Transition Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

n = 100
probabilities = np.linspace(0.005, 0.05, 20)
largest_components = []

for p in probabilities:
    er = ErdosRenyiGraph(n_nodes=n, edge_probability=p)
    result = er.run()
    largest_components.append(result.results['largest_component_size'])

plt.plot(probabilities, largest_components, 'bo-')
plt.axvline(1/n, color='red', linestyle='--', label='Critical p = 1/n')
plt.xlabel('Edge Probability')
plt.ylabel('Largest Component Size')
plt.legend()
plt.show()
```

#### 📊 Example – Theoretical vs Empirical Comparison
```python
er = ErdosRenyiGraph(n_nodes=200, edge_probability=0.1)
result = er.run()

print("Theoretical vs Empirical:")
print(f"Edges: {result.statistics['theoretical_edges']:.1f} vs {result.statistics['empirical_edges']}")
print(f"Degree: {result.statistics['theoretical_degree']:.2f} vs {result.statistics['empirical_degree']:.2f}")
print(f"Clustering: {result.statistics['theoretical_clustering']:.3f} vs {result.statistics['empirical_clustering']:.3f}")
```

</details>

---

### ▶️ `BarabasiAlbertGraph`
> Generate and analyze **scale-free networks** using preferential attachment, with power-law degree distributions and comprehensive hub analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does
This class implements the **Barabási-Albert preferential attachment model** that generates scale-free networks where new nodes preferentially connect to high-degree nodes, resulting in power-law degree distributions and hub formation.

#### 🌟 Scale-Free Properties Analyzed
- **Power-Law Distribution**: Degree distribution P(k) ∝ k^(-γ)
- **Hub Analysis**: Identification and characterization of high-degree nodes
- **Preferential Attachment**: Growth mechanism verification
- **Scale-Free Metrics**: Power-law fitting and exponent estimation
- **Network Resilience**: Hub-based connectivity patterns

#### 📚 Theoretical Background
**Barabási-Albert Model:**
- Start with m₀ initial nodes
- At each step: add one node with m edges (m ≤ m₀)
- Connection probability: P(kᵢ) ∝ kᵢ (preferential attachment)

**Asymptotic Properties:**
- Degree distribution: P(k) ∝ k^(-3) for large k
- Average degree: ⟨k⟩ = 2m
- No clustering in basic model
- Small-world property: diameter ∝ log log n

**Growth Mechanism:**
- "Rich get richer" phenomenon
- Matthew effect in network growth
- Emergence of highly connected hubs
- Scale-free topology across size ranges

#### ✅ Applications
- **Social Networks**: Citation networks, collaboration graphs
- **World Wide Web**: Link structure and page connectivity
- **Biological Networks**: Protein interactions, metabolic pathways
- **Economic Networks**: Trade relationships, financial connections
- **Infrastructure**: Internet topology, transportation networks
- **Scientific Networks**: Research collaboration patterns

#### 📈 Example – Basic Scale-Free Network
```python
from RSIM.networks.random_graphs import BarabasiAlbertGraph

# Create Barabási-Albert graph
ba_graph = BarabasiAlbertGraph(n_nodes=200, m_edges=3)
result = ba_graph.run()

print(f"Nodes: {result.results['n_nodes']}")
print(f"Edges: {result.results['n_edges']}")
print(f"Max degree: {result.results['max_degree']}")
print(f"Hub fraction: {result.results['hub_fraction']:.2%}")

# Visualize with hub highlighting
ba_graph.visualize()
```

#### 🎯 Example – Power-Law Analysis
```python
import numpy as np
from scipy import stats

ba = BarabasiAlbertGraph(n_nodes=1000, m_edges=2)
result = ba.run()

degrees, counts = result.statistics['degree_distribution']

# Log-log plot for power-law detection
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.loglog(degrees, counts, 'bo', alpha=0.7, label='Data')

# Fit power law
log_degrees = np.log(degrees[degrees > 0])
log_counts = np.log(counts[degrees > 0])
slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_counts)

x_fit = np.logspace(np.log10(min(degrees)), np.log10(max(degrees)), 50)
y_fit = np.exp(intercept) * x_fit ** slope
plt.loglog(x_fit, y_fit, 'r-', linewidth=2, label=f'Power law (γ≈{-slope:.2f})')

plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend()
plt.title('Power-Law Degree Distribution')
plt.show()
```

#### 📊 Example – Hub Analysis
```python
ba = BarabasiAlbertGraph(n_nodes=500, m_edges=2)
result = ba.run()

# Analyze top hubs
hub_degrees = result.results['hub_degrees']
print("Top 10 Hub Degrees:")
for i, degree in enumerate(hub_degrees[:10], 1):
    print(f"Hub {i}: {degree} connections")

# Hub connectivity impact
graph = result.raw_data['graph']
degree_sequence = result.raw_data['degree_sequence']

print(f"\nNetwork Statistics:")
print(f"Average degree: {np.mean(degree_sequence):.2f}")
print(f"Degree variance: {np.var(degree_sequence):.2f}")
print(f"Degree heterogeneity: {np.var(degree_sequence)/np.mean(degree_sequence):.2f}")
```

#### 🔬 Example – Growth Mechanism Verification
```python
# Compare different m values
m_values = [1, 2, 3, 5]
results = {}

for m in m_values:
    ba = BarabasiAlbertGraph(n_nodes=300, m_edges=m)
    result = ba.run()
    results[m] = {
        'avg_degree': result.results['avg_degree'],
        'max_degree': result.results['max_degree'],
        'clustering': result.results['clustering_coefficient']
    }

print("Effect of m on network properties:")
for m, props in results.items():
    print(f"m={m}: avg_degree={props['avg_degree']:.1f}, "
          f"max_degree={props['max_degree']}, "
          f"clustering={props['clustering']:.4f}")
```

#### 📚 References
- Barabási & Albert (1999). *Emergence of scaling in random networks*
- Newman (2003). *The structure and function of complex networks*
- Albert & Barabási (2002). *Statistical mechanics of complex networks*

</details>

### ▶️ `SocialNetworkDiffusion`

> Model **information, innovation, and behavior spread** through social networks using threshold models, cascade processes, and complex contagion mechanisms.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **social network diffusion simulation** to study how information, innovations, behaviors, or opinions spread through social systems. It incorporates social influence, network effects, and individual adoption thresholds to model realistic diffusion processes.

#### 🔄 Supported Diffusion Models

- **Threshold Model**: Individuals adopt when fraction of adopter neighbors exceeds threshold
- **Independent Cascade**: Each adopter has one chance to influence each neighbor with fixed probability
- **Complex Contagion**: Requires multiple exposures/reinforcement for adoption

#### 🌐 Network Structure Support

- Custom NetworkX graphs (any topology)
- Auto-generated small-world networks (Watts-Strogatz)
- Configurable network parameters (nodes, connections, rewiring probability)

#### 📚 Theoretical Background

**Social Influence Models:**
Social diffusion combines individual decision-making with network structure effects. Key mechanisms include:

- **Social proof**: People adopt based on neighbors' behavior
- **Network effects**: Value increases with adoption
- **Homophily**: Similar individuals cluster together
- **Weak ties**: Bridges between communities enable spread

**Threshold Model:**
Individual *i* adopts at time *t* if:
```
fraction_adopter_neighbors ≥ adoption_threshold
```

**Independent Cascade:**
For each adopter-neighbor pair, adoption probability per time step:
```
P(adoption) = influence_probability
```

**Complex Contagion:**
Requires multiple reinforcing signals:
```
P(adoption) = min(1.0, num_adopter_neighbors × influence_probability)
```

**Key Metrics:**
- **Final adoption rate**: Fraction of population that eventually adopts
- **Diffusion speed**: Rate of new adoptions over time
- **Time to adoption**: Individual adoption timing
- **Network effects**: How structure influences spread

#### ✅ Applications

- **Innovation adoption** (new technologies, products)
- **Information spreading** (news, rumors, viral content)
- **Behavior change** (health behaviors, social movements)
- **Opinion dynamics** and polarization
- **Marketing campaigns** and viral strategies
- **Social movements** and collective action

#### 📈 Example – Technology adoption with threshold model

```python
import networkx as nx
from RSIM.networks import SocialNetworkDiffusion

# Create social network (small-world structure)
network = nx.watts_strogatz_graph(n=200, k=6, p=0.1)

# Configure diffusion simulation
diffusion = SocialNetworkDiffusion(
    network=network,
    diffusion_model='threshold',
    adoption_threshold=0.3,        # Need 30% of neighbors to adopt
    initial_adopters=5,            # Start with 5 early adopters
    max_time_steps=50
)

result = diffusion.run()

print(f"Final adoption rate: {result.results['final_adoption_rate']:.1%}")
print(f"Diffusion duration: {result.results['diffusion_duration']} steps")
print(f"Peak adoption speed: {result.results['peak_adoption_speed']} adopters/step")

# Visualize diffusion process
diffusion.visualize()
```

#### 🎯 Example – Viral marketing with cascade model

```python
# Simulate viral marketing campaign
viral_campaign = SocialNetworkDiffusion(
    diffusion_model='cascade',
    influence_probability=0.15,    # 15% chance to influence neighbor
    initial_adopters=10,           # 10 influencers as seeds
    network_params={'n_nodes': 500, 'k': 8, 'p': 0.05},
    max_time_steps=30
)

result = viral_campaign.run()

# Analyze campaign effectiveness
time_series = result.statistics['time_series']
peak_time = time_series['diffusion_speed'].index(max(time_series['diffusion_speed']))
print(f"Campaign peaked at time {peak_time}")
print(f"Total reach: {result.results['total_adopters']} people")
```

#### 📊 Example – Complex contagion for behavior change

```python
# Model health behavior adoption (requires social reinforcement)
health_behavior = SocialNetworkDiffusion(
    diffusion_model='complex_contagion',
    adoption_threshold=0.4,        # Higher threshold for behavior change
    influence_probability=0.08,    # Lower individual influence
    initial_adopters=15,           # Multiple health advocates
    max_time_steps=100
)

result = health_behavior.run()

# Analyze adoption patterns
print(f"Behavior adoption rate: {result.results['final_adoption_rate']:.1%}")
print(f"Average adoption time: {result.results['avg_adoption_time']:.1f} steps")
print(f"Network clustering: {result.results['clustering_coefficient']:.3f}")
```

#### 🔍 Example – Compare diffusion models

```python
models = ['threshold', 'cascade', 'complex_contagion']
results = {}

for model in models:
    sim = SocialNetworkDiffusion(
        diffusion_model=model,
        adoption_threshold=0.25,
        influence_probability=0.12,
        initial_adopters=8,
        random_seed=42  # For fair comparison
    )
    results[model] = sim.run()

# Compare final adoption rates
for model, result in results.items():
    rate = result.results['final_adoption_rate']
    duration = result.results['diffusion_duration']
    print(f"{model:18}: {rate:.1%} adoption in {duration} steps")
```

#### 📊 Key Output Metrics

| Metric | Description |
|--------|-------------|
| `final_adoption_rate` | Fraction of population that adopted |
| `total_adopters` | Absolute number of adopters |
| `diffusion_duration` | Time steps until saturation |
| `avg_adoption_time` | Mean time to adoption |
| `peak_adoption_speed` | Maximum new adopters per step |
| `network_size` | Total nodes in network |
| `clustering_coefficient` | Network clustering measure |
| `adopter_density` | Connectivity among adopters |

#### 🎨 Visualization Features

- **Network layout** with adoption states (color-coded by timing)
- **Diffusion curve** (S-shaped adoption over time)
- **Adoption speed** (new adopters per time step)
- **Time distribution** (when people adopted)
- **Network analysis** (degree vs adoption time)
- **Summary statistics** panel

#### ⚙️ Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|---------|
| `adoption_threshold` | 0.1 - 0.5 | Higher = slower, more clustered diffusion |
| `influence_probability` | 0.05 - 0.3 | Higher = faster cascade spread |
| `initial_adopters` | 1 - 5% of network | More seeds = faster start |
| `network_params['k']` | 4 - 10 | Higher connectivity = faster spread |
| `network_params['p']` | 0.01 - 0.3 | Higher rewiring = more global spread |

#### 📚 References

- Granovetter (1978). *Threshold Models of Collective Behavior*
- Watts & Strogatz (1998). *Collective Dynamics of Small-World Networks*
- Centola & Macy (2007). *Complex Contagions and the Weakness of Long Ties*
- Jackson (2008). *Social and Economic Networks*
- Easley & Kleinberg (2010). *Networks, Crowds, and Markets*

</details>



### ▶️ `NetworkTrafficSimulation`

> Simulate **network traffic flow and congestion** with packet routing, analyzing throughput, latency, and packet loss in complex network topologies.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **network traffic flow simulation** using packet routing and congestion modeling. It simulates data packet transmission through networks where nodes represent routers/switches and edges represent communication links, including realistic congestion effects, packet loss, and routing algorithms.

#### 🌐 Network Components

- **Nodes**: Routers/switches with finite buffer capacity
- **Links**: Communication channels with capacity limits and delays
- **Packets**: Data units with source, destination, size, and routing paths
- **Buffers**: FIFO queues at each node for packet storage
- **Routing**: Shortest path algorithm with congestion awareness

#### 📊 Performance Metrics

- **Throughput**: Successful packet delivery rate (packets/time)
- **Latency**: Average packet transmission time
- **Packet Loss Rate**: Fraction of dropped packets due to congestion
- **Link Utilization**: Traffic load relative to capacity
- **Network Utilization**: Overall system load
- **Hop Count Distribution**: Path length statistics

#### 📚 Theoretical Background

**Traffic Model:**
- Packets generated with Poisson arrival process: λ packets/node/time
- Each packet has random size from specified range
- Links have finite capacity C (packets per time unit)

**Congestion Model:**
- Congestion occurs when traffic exceeds link capacity
- Buffer overflow leads to packet drops
- Queuing delays increase with utilization

**Routing Algorithm:**
- Shortest path routing using Dijkstra's algorithm
- Static routing tables precomputed
- No adaptive routing (can be extended)

**Mathematical Formulation:**
Let:
- G = (V, E) be the network graph
- λᵢ = packet generation rate at node i
- Cₑ = capacity of edge e
- Bᵢ = buffer size at node i

**Performance Equations:**
```
Throughput = Packets_delivered / Simulation_time
Packet_loss_rate = Packets_dropped / Packets_generated  
Average_latency = Σ(delivery_time - creation_time) / Packets_delivered
Link_utilization = Current_load / Link_capacity
```

#### ✅ Key Features

- **Realistic Network Modeling**: Finite buffers, link capacities, routing delays
- **Congestion Effects**: Packet drops, queuing delays, throughput degradation
- **Flexible Topologies**: Custom networks or auto-generated (Erdős-Rényi)
- **Comprehensive Metrics**: Throughput, latency, loss rates, utilizations
- **Time Series Analysis**: Performance evolution over simulation time
- **Visualization**: Network topology with utilization heatmaps

#### 🚀 Applications

- Internet traffic analysis and capacity planning
- Data center network design and optimization
- Telecommunication network performance evaluation
- Quality of Service (QoS) analysis
- Network congestion control algorithm testing
- Cloud computing resource allocation

#### 📈 Example – Basic Network Traffic Simulation

```python
import networkx as nx
from RSIM.networks import NetworkTrafficSimulation

# Create a small network topology
network = nx.erdos_renyi_graph(10, 0.3)

# Configure traffic simulation
traffic_sim = NetworkTrafficSimulation(
    network=network,
    packet_generation_rate=0.15,  # 15% chance per node per time step
    link_capacity=8.0,            # 8 packets per link per time step
    packet_size_range=(1, 4),     # Packet sizes 1-4 units
    simulation_time=200,          # 200 time steps
    buffer_size=12                # 12 packet buffer per node
)

# Run simulation
result = traffic_sim.run()

# Display key metrics
print(f"Final Throughput: {result.results['final_throughput']:.3f} packets/time")
print(f"Packet Loss Rate: {result.results['final_packet_loss_rate']:.2%}")
print(f"Average Latency: {result.results['final_avg_latency']:.2f} time steps")
print(f"Network Utilization: {result.results['avg_link_utilization']:.2%}")
```

#### 🎯 Example – High-Load Congestion Analysis

```python
# Test network under heavy load
heavy_traffic = NetworkTrafficSimulation(
    packet_generation_rate=0.3,   # High generation rate
    link_capacity=5.0,            # Limited capacity
    buffer_size=8,                # Small buffers
    simulation_time=300,
    network_params={'n_nodes': 15, 'edge_probability': 0.25}
)

result = heavy_traffic.run()

# Analyze congestion effects
print(f"Packets Generated: {result.results['packets_generated']}")
print(f"Packets Delivered: {result.results['packets_delivered']}")
print(f"Packets Dropped: {result.results['packets_dropped']}")
print(f"Delivery Ratio: {result.statistics['delivery_ratio']:.2%}")
print(f"Max Link Utilization: {result.results['max_link_utilization']:.2%}")
```

#### 📊 Example – Network Topology Impact

```python
# Compare different network topologies
topologies = {
    'Dense': nx.erdos_renyi_graph(12, 0.5),
    'Sparse': nx.erdos_renyi_graph(12, 0.2),
    'Star': nx.star_graph(11),
    'Ring': nx.cycle_graph(12)
}

results = {}
for name, network in topologies.items():
    sim = NetworkTrafficSimulation(
        network=network,
        packet_generation_rate=0.2,
        link_capacity=6.0,
        simulation_time=150
    )
    results[name] = sim.run()
    
# Compare performance
for name, result in results.items():
    print(f"{name:8s}: Throughput={result.results['final_throughput']:.3f}, "
          f"Loss={result.results['final_packet_loss_rate']:.2%}, "
          f"Latency={result.results['final_avg_latency']:.2f}")
```

#### 🔧 Parameter Guidelines

| Parameter | Low Load | Medium Load | High Load |
|-----------|----------|-------------|-----------|
| Generation Rate | 0.05-0.1 | 0.1-0.2 | 0.2-0.4 |
| Link Capacity | 3-5 | 5-10 | 10-20 |
| Buffer Size | 5-8 | 8-15 | 15-25 |
| Network Density | 0.2-0.3 | 0.3-0.4 | 0.4-0.6 |

#### 📈 Visualization Features

- **Network Topology**: Nodes and edges with utilization color-coding
- **Performance Time Series**: Throughput, loss rate, utilization over time
- **Latency Distribution**: Histogram of packet delivery times
- **Hop Count Analysis**: Path length statistics
- **Link Utilization Heatmap**: Congestion hotspots identification

#### 🎛️ Advanced Configuration

```python
# Custom network with specific properties
custom_network = nx.barabasi_albert_graph(20, 3)  # Scale-free network

# Advanced simulation setup
advanced_sim = NetworkTrafficSimulation(
    network=custom_network,
    packet_generation_rate=0.12,
    link_capacity=7.5,
    packet_size_range=(1, 5),
    simulation_time=500,
    buffer_size=15,
    random_seed=42  # For reproducibility
)

# Run and visualize
result = advanced_sim.run()
advanced_sim.visualize(result)
```

#### 📚 References

- Bertsekas & Gallager (1992). *Data Networks*
- Tanenbaum & Wetherall (2010). *Computer Networks*
- Kurose & Ross (2016). *Computer Networking: A Top-Down Approach*
- Peterson & Davie (2011). *Computer Networks: A Systems Approach*

</details>

### ▶️ `GeneticAlgorithm`

> Implement **evolutionary optimization** using genetic algorithms with selection, crossover, and mutation operators for global optimization of complex functions.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **genetic algorithm optimization** that evolves a population of candidate solutions through natural selection principles — selection, crossover, and mutation — to find optimal or near-optimal solutions for complex optimization problems.

#### 🧬 Genetic Operators

- **Tournament Selection**: Select parents based on fitness competition
- **Uniform Crossover**: Combine genetic material from two parents
- **Gaussian Mutation**: Introduce random variations with adaptive strength
- **Elitism**: Preserve best individuals across generations

#### 🎯 Key Features

- Population-based global search
- Handles multimodal and discontinuous functions
- Configurable genetic operators and parameters
- Comprehensive convergence tracking
- Diversity monitoring and analysis

#### 📚 Theoretical Background

**Genetic Algorithm Principle:**  
GA mimics natural evolution by maintaining a population of candidate solutions and applying genetic operators to evolve better solutions over generations.

**Algorithm Steps:**
1. Initialize random population within bounds
2. Evaluate fitness of all individuals
3. Select parents using tournament selection
4. Apply crossover to create offspring
5. Apply mutation to maintain diversity
6. Replace population with new generation
7. Repeat until convergence

**Mathematical Framework:**
- **Population**: \( P(t) = \{x_1^{(t)}, x_2^{(t)}, \ldots, x_N^{(t)}\} \)
- **Fitness**: \( f(x_i) \) for individual \( x_i \)
- **Selection Pressure**: Tournament size controls selection intensity
- **Crossover**: \( P(\text{crossover}) = p_c \)
- **Mutation**: \( P(\text{mutation}) = p_m \)

**Tournament Selection:**
```
Select k individuals randomly
Return fittest individual from tournament
```

**Uniform Crossover:**
```
For each gene position:
  If random() < 0.5: child1[i] = parent2[i], child2[i] = parent1[i]
  Else: child1[i] = parent1[i], child2[i] = parent2[i]
```

**Gaussian Mutation:**
```
If random() < mutation_rate:
  gene = gene + N(0, σ²)
  gene = clip(gene, bounds)
```

#### ✅ Properties

- Global optimization capability
- Handles non-differentiable functions
- Robust to noise and discontinuities
- Population diversity maintains exploration
- Elitism ensures monotonic improvement

#### 📈 Example – Function optimization

```python
import numpy as np
from RSIM.optimization import GeneticAlgorithm

# Rosenbrock function (global minimum at [1,1] = 0)
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Configure GA
bounds = [(-2, 2), (-1, 3)]
ga = GeneticAlgorithm(
    objective_function=rosenbrock,
    bounds=bounds,
    population_size=50,
    max_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1
)

# Run optimization
result = ga.run()
print("Best solution:", result.results['best_solution'])
print("Best fitness:", result.results['best_fitness'])
```

#### 🎯 Example – Multi-dimensional optimization

```python
# Rastrigin function (multimodal test function)
def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# 5D optimization
bounds = [(-5.12, 5.12)] * 5
ga_multi = GeneticAlgorithm(
    objective_function=rastrigin,
    bounds=bounds,
    population_size=100,
    max_generations=200,
    elitism_rate=0.2,
    tournament_size=5
)

result = ga_multi.run()
ga_multi.visualize()  # Shows convergence plots
```

#### 📉 Example – Custom configuration

```python
# Engineering design optimization
def engineering_objective(x):
    # Complex engineering function
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1])

ga_custom = GeneticAlgorithm(
    objective_function=engineering_objective,
    bounds=[(-3, 3), (-3, 3)],
    population_size=80,
    max_generations=150,
    crossover_rate=0.9,
    mutation_rate=0.05,
    elitism_rate=0.15,
    tournament_size=4,
    random_seed=42
)

result = ga_custom.run()
```

#### ⚙️ Parameter Guidelines

| Parameter | Typical Range | Purpose |
|-----------|---------------|---------|
| `population_size` | 30-200 | Larger for complex problems |
| `max_generations` | 50-500 | Until convergence plateau |
| `crossover_rate` | 0.6-0.9 | High for exploitation |
| `mutation_rate` | 0.01-0.2 | Low to moderate for diversity |
| `elitism_rate` | 0.05-0.2 | Preserve best solutions |
| `tournament_size` | 2-7 | Higher = more selection pressure |

#### 🎨 Visualization Features

- **Fitness Evolution**: Best, mean, and diversity over generations
- **Population Distribution**: Final population scatter (2D problems)
- **Convergence Analysis**: Improvement tracking
- **Parameter Evolution**: Best solution trajectory

#### 📊 Performance Metrics

- **Best Fitness**: Global optimum found
- **Convergence Generation**: When best solution was discovered
- **Population Diversity**: Genetic diversity maintenance
- **Improvement Rate**: Fitness improvement per generation

#### 🔧 Advanced Usage

```python
# Access detailed results
result = ga.run()
print("Convergence generation:", result.results['convergence_generation'])
print("Final diversity:", result.statistics['final_diversity'])
print("Total improvement:", result.statistics['improvement'])

# Get evolution history
fitness_history = result.raw_data['best_fitness_history']
solutions_history = result.raw_data['best_solutions_history']
```

#### 📚 References

- Holland, J.H. (1992). *Adaptation in Natural and Artificial Systems*
- Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*
- Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*

</details>

### ▶️ `ParticleSwarmOptimization`

> Implement **bio-inspired swarm intelligence optimization** using particle swarm dynamics to find global optima in continuous search spaces with adaptive exploration and exploitation.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Particle Swarm Optimization (PSO)**, a population-based metaheuristic inspired by the social behavior of bird flocking and fish schooling. Particles navigate the search space by balancing personal experience with collective intelligence to locate global optima.

#### 🔧 Key Features

- **Adaptive velocity control** with inertia weight and acceleration coefficients
- **Boundary constraint handling** for position and velocity limits
- **Comprehensive tracking** of swarm dynamics and convergence history
- **Multi-dimensional visualization** for trajectory analysis
- **Parameter validation** and automatic configuration

#### 📐 Algorithm Components

- **Swarm**: Population of particles exploring the search space
- **Personal Best (pbest)**: Best position found by each individual particle
- **Global Best (gbest)**: Best position discovered by the entire swarm
- **Velocity**: Direction and magnitude of particle movement
- **Inertia Weight (w)**: Controls exploration vs exploitation balance

#### 📚 Theoretical Background

**PSO Update Equations:**

Velocity update:
```
v(t+1) = w·v(t) + c1·r1·(pbest - x(t)) + c2·r2·(gbest - x(t))
```

Position update:
```
x(t+1) = x(t) + v(t+1)
```

Where:
- `w` = inertia weight (exploration/exploitation balance)
- `c1` = cognitive coefficient (personal best attraction)
- `c2` = social coefficient (global best attraction)
- `r1, r2` = random numbers ∈ [0,1]

**Algorithm Steps:**
1. Initialize particles with random positions and velocities
2. Evaluate fitness of each particle
3. Update personal best and global best positions
4. Update velocities using PSO equation
5. Update positions and apply boundary constraints
6. Repeat until convergence or maximum iterations

#### ✅ Properties

- **Global optimization capability** for multimodal functions
- **Fast convergence** on unimodal landscapes
- **Parallelizable** particle evaluations
- **Parameter robustness** across diverse problems
- **Memory efficiency** with moderate population sizes

#### 📈 Example – Rosenbrock Function Optimization

```python
import numpy as np
from RSIM.optimization import ParticleSwarmOptimization

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Define search bounds
bounds = [(-2, 2), (-1, 3)]

# Configure PSO
pso = ParticleSwarmOptimization(
    objective_function=rosenbrock,
    bounds=bounds,
    n_particles=30,
    max_iterations=100,
    w=0.7,
    c1=2.0,
    c2=2.0
)

# Run optimization
result = pso.run()

print("Best solution:", result.results['best_solution'])
print("Best fitness:", result.results['best_fitness'])
print("Convergence iteration:", result.results['convergence_iteration'])

# Visualize optimization process
pso.visualize()
```

#### 🎯 Example – High-Dimensional Sphere Function

```python
def sphere(x):
    return np.sum(x**2)

# 10-dimensional optimization
bounds = [(-5, 5)] * 10

pso_hd = ParticleSwarmOptimization(
    objective_function=sphere,
    bounds=bounds,
    n_particles=50,
    max_iterations=200,
    random_seed=42
)

result = pso_hd.run()
print(f"10D Sphere minimum: {result.results['best_fitness']:.6f}")
```

#### 🔧 Example – Parameter Tuning

```python
# Configure with custom parameters
pso.configure(
    n_particles=40,
    max_iterations=150,
    w=0.9,          # Higher exploration
    c1=1.5,         # Reduced personal attraction
    c2=2.5,         # Increased social attraction
    v_max=0.3       # 30% of search space velocity limit
)

result = pso.run()
```

#### 📊 Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|---------|
| `n_particles` | 20-50 | More particles = better exploration, slower convergence |
| `w` | 0.4-0.9 | Higher = more exploration, lower = faster convergence |
| `c1` | 1.0-3.0 | Personal best attraction strength |
| `c2` | 1.0-3.0 | Global best attraction strength |
| `v_max` | 0.1-0.5 | Fraction of search space for velocity limit |

#### 🎨 Visualization Features

- **Fitness evolution** with global best and mean fitness tracking
- **Swarm diversity** analysis over iterations
- **Particle trajectories** for 2D problems
- **Velocity magnitude** dynamics
- **Convergence statistics** and performance metrics

#### 🚀 Applications

- **Engineering design** optimization
- **Neural network** training and hyperparameter tuning
- **Portfolio optimization** in finance
- **Feature selection** in machine learning
- **Process parameter** optimization
- **Antenna design** and electromagnetic problems

#### ⚙️ Advanced Configuration

```python
# Custom velocity limits and seeded optimization
pso_advanced = ParticleSwarmOptimization(
    objective_function=your_function,
    bounds=your_bounds,
    n_particles=25,
    max_iterations=300,
    w=0.8,
    c1=2.1,
    c2=1.9,
    v_max=0.25,
    random_seed=123
)

# Validate parameters before running
errors = pso_advanced.validate_parameters()
if not errors:
    result = pso_advanced.run()
```

#### 📚 References

- Kennedy & Eberhart (1995). *Particle Swarm Optimization*
- Shi & Eberhart (1998). *A Modified Particle Swarm Optimizer*
- Clerc & Kennedy (2002). *The Particle Swarm - Explosion, Stability, and Convergence*
- Poli et al. (2007). *Particle Swarm Optimization: An Overview*

</details>

### ▶️ `ResponseSurfaceMethodology`

> Perform **systematic optimization using polynomial response surface models** with sequential design of experiments for efficient exploration of complex objective functions.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Response Surface Methodology (RSM)** - a collection of mathematical and statistical techniques for modeling and analyzing optimization problems where a response is influenced by several variables. It uses polynomial approximations and design of experiments to efficiently find optimal conditions.

#### 🔬 Algorithm Components

- **Central Composite Design**: Factorial + axial + center points for efficient sampling
- **Polynomial Fitting**: 1st, 2nd, or 3rd degree response surface models
- **Sequential Optimization**: Iterative movement toward predicted optimum
- **Adaptive Search**: Dynamic radius adjustment based on improvement
- **Boundary Handling**: Constraint satisfaction for bounded optimization

#### 📊 Design Point Generation

- **Factorial Points**: 2^k corner points of design space
- **Axial Points**: Star points along coordinate axes
- **Center Points**: Replicated center for error estimation
- **Rotatable Design**: Equal prediction variance at equal distances

#### 📚 Theoretical Background

**Second-Order Response Surface Model:**
```
y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + ΣΣβᵢⱼxᵢxⱼ + ε
```

**Central Composite Design Points:**
- Factorial: 2^k points at (±1, ±1, ..., ±1)
- Axial: 2k points at (±α, 0, ..., 0), (0, ±α, 0, ...), etc.
- Center: Multiple points at (0, 0, ..., 0)
- Total points: 2^k + 2k + n_center

**Sequential Strategy:**
1. Generate design points around current center
2. Evaluate objective function at design points
3. Fit polynomial model using least squares
4. Find optimum of fitted model
5. Move toward predicted optimum
6. Repeat until convergence

**Rotatable Design Parameter:**
```
α = (2^k)^(1/4)  # For rotatability
```

#### ✅ Properties

- **Model-based**: Uses surrogate models for efficiency
- **Sequential**: Iterative improvement with learning
- **Adaptive**: Adjusts search radius based on progress
- **Robust**: Handles noisy and expensive functions
- **Flexible**: Works with various polynomial degrees

#### 📈 Example – Basic RSM Optimization

```python
import numpy as np
from RSIM.optimization import ResponseSurfaceMethodology

# Define Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Set up RSM optimizer
bounds = [(-2, 2), (-1, 3)]
rsm = ResponseSurfaceMethodology(
    objective_function=rosenbrock,
    bounds=bounds,
    polynomial_degree=2,
    max_iterations=20,
    step_size=0.3
)

# Run optimization
result = rsm.run()
print("Best solution:", result.results['best_solution'])
print("Best fitness:", result.results['best_fitness'])
print("Converged:", result.results['converged'])
```

#### 🎯 Example – High-Dimensional Optimization

```python
# 5D sphere function
def sphere_5d(x):
    return np.sum(x**2)

bounds_5d = [(-5, 5)] * 5
rsm_5d = ResponseSurfaceMethodology(
    objective_function=sphere_5d,
    bounds=bounds_5d,
    n_initial_points=50,  # More points for higher dimensions
    polynomial_degree=2,
    max_iterations=30
)

result = rsm_5d.run()
rsm_5d.visualize()  # Shows parameter evolution and convergence
```

#### 🔧 Example – Custom Configuration

```python
# Manufacturing process optimization
def process_yield(x):
    temp, pressure, time = x
    # Complex process model
    return -(0.8*temp + 0.6*pressure + 0.4*time - 
             0.01*temp**2 - 0.02*pressure**2 - 0.005*time**2 +
             0.001*temp*pressure + np.random.normal(0, 0.1))

bounds = [(150, 250), (10, 50), (30, 120)]  # temp, pressure, time
initial_center = [200, 30, 75]

rsm_process = ResponseSurfaceMethodology(
    objective_function=process_yield,
    bounds=bounds,
    initial_center=initial_center,
    polynomial_degree=2,
    max_iterations=25,
    step_size=0.2,
    convergence_tol=1e-5,
    random_seed=42
)

result = rsm_process.run()
```

#### 📊 Parameter Guidelines

| Parameter | Recommended Range | Purpose |
|-----------|------------------|---------|
| `polynomial_degree` | 1-3 | Model complexity vs. overfitting |
| `n_initial_points` | 2^k + 2k + 1 to 3×(2^k + 2k + 1) | Design richness |
| `max_iterations` | 10-50 | Exploration vs. computation |
| `step_size` | 0.1-0.5 | Conservative vs. aggressive moves |
| `convergence_tol` | 1e-6 to 1e-3 | Precision vs. efficiency |

#### 🎨 Visualization Features

- **Fitness Evolution**: Best fitness and model predictions over iterations
- **Search Radius**: Adaptive radius adjustment visualization
- **Parameter Trajectory**: Movement of design center through space
- **Design Points**: Distribution of experimental points per iteration

#### 🏭 Applications

- **Process Optimization**: Manufacturing parameter tuning
- **Chemical Engineering**: Reaction condition optimization
- **Quality Improvement**: Six Sigma and robust design
- **Agricultural Research**: Crop yield optimization
- **Pharmaceutical**: Drug formulation optimization
- **Engineering Design**: Component parameter optimization

#### ⚡ Performance Characteristics

| Problem Size | Typical Iterations | Function Evaluations |
|--------------|-------------------|---------------------|
| 2D | 10-20 | 100-300 |
| 3-5D | 15-30 | 200-600 |
| 6-10D | 20-40 | 400-1000 |

#### 📚 References

- Box & Wilson (1951). *On the Experimental Attainment of Optimum Conditions*
- Myers, Montgomery & Anderson-Cook (2016). *Response Surface Methodology*
- Khuri & Mukhopadhyay (2010). *Response Surface Methodology*
- Box & Draper (2007). *Response Surfaces, Mixtures, and Ridge Analyses*

</details>


### ▶️ `SimulatedAnnealing`

> Perform powerful **global optimization** using the simulated annealing metaheuristic, inspired by the metallurgical annealing process to escape local optima and find near-optimal solutions.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **simulated annealing optimization** - a probabilistic technique that explores the solution space by accepting both improving and worsening moves with decreasing probability over time, allowing escape from local optima to find global solutions.

#### 🔥 Key Features

- **Metropolis Criterion**: Probabilistic acceptance of worse solutions
- **Adaptive Temperature Scheduling**: Geometric cooling with customizable rates
- **Flexible Perturbation**: Gaussian neighborhood generation with bounds handling
- **Multi-dimensional Support**: Works with vector optimization problems
- **Comprehensive Tracking**: Records costs, temperatures, and solution paths

#### 🎯 Applications

- **Combinatorial Optimization**: Traveling Salesman Problem (TSP), scheduling
- **Continuous Function Optimization**: Non-convex, multimodal functions
- **Neural Network Training**: Weight optimization in deep learning
- **Image Processing**: Segmentation, restoration, feature extraction
- **VLSI Design**: Circuit layout and component placement
- **Protein Folding**: Molecular structure prediction

#### 📚 Theoretical Background

**Core Algorithm:**
The algorithm mimics the physical annealing process where materials are heated and slowly cooled to reach minimum energy states.

**Acceptance Probability:**
For a move with cost increase ΔE > 0:
```
P(accept) = exp(-ΔE / T)
```
Where T is the current temperature.

**Temperature Schedule:**
Geometric cooling:
```
T(t) = T₀ × α^t
```
Where:
- T₀ = initial temperature
- α = cooling rate (0 < α < 1)
- t = iteration number

**Convergence Conditions:**
- Temperature reaches minimum threshold
- Maximum iterations exceeded
- No improvement for extended periods

#### 🔧 Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `initial_temperature` | Starting temperature T₀ | 10.0 - 1000.0 |
| `cooling_rate` | Temperature reduction factor α | 0.8 - 0.999 |
| `min_temperature` | Stopping temperature | 1e-8 - 1e-3 |
| `max_iterations` | Maximum iterations | 1000 - 100000 |
| `step_size` | Perturbation standard deviation | 0.01 - 10.0 |

#### 📈 Example – Minimize Rosenbrock Function

```python
import numpy as np
from RSIM.optimization import SimulatedAnnealing

def rosenbrock(x):
    """Classic optimization test function"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Initialize optimization
initial_point = np.array([-1.0, 1.0])
bounds = [(-2, 2), (-1, 3)]

sa = SimulatedAnnealing(
    objective_function=rosenbrock,
    initial_solution=initial_point,
    bounds=bounds,
    initial_temperature=100.0,
    cooling_rate=0.95,
    max_iterations=10000
)

# Run optimization
result = sa.run()

print(f"Best solution: {result.results['best_solution']}")
print(f"Best cost: {result.results['best_cost']:.6f}")
print(f"Improvement: {result.results['improvement']:.6f}")
```

#### 🎯 Example – Multi-dimensional Sphere Function

```python
def sphere(x):
    """N-dimensional sphere function"""
    return np.sum(x**2)

# 5-dimensional optimization
initial_solution = np.random.uniform(-5, 5, 5)
bounds = [(-5, 5)] * 5

sa_sphere = SimulatedAnnealing(
    objective_function=sphere,
    initial_solution=initial_solution,
    bounds=bounds,
    initial_temperature=50.0,
    cooling_rate=0.98,
    step_size=0.5
)

result = sa_sphere.run()
sa_sphere.visualize()  # Shows parameter evolution for high-dim problems
```

#### 🔄 Example – Custom Cooling Schedule

```python
# Fine-tuned parameters for difficult optimization
sa_custom = SimulatedAnnealing(
    objective_function=your_function,
    initial_solution=start_point,
    initial_temperature=200.0,    # High initial temperature
    cooling_rate=0.999,           # Very slow cooling
    min_temperature=1e-10,        # Very low final temperature
    max_iterations=50000,         # Many iterations
    step_size=2.0                 # Larger exploration steps
)

result = sa_custom.run()
```

#### 📊 Performance Guidelines

| Problem Type | Temperature | Cooling Rate | Iterations |
|--------------|-------------|--------------|------------|
| **Smooth Functions** | 10-50 | 0.9-0.95 | 1K-5K |
| **Multimodal** | 50-200 | 0.95-0.99 | 5K-20K |
| **High-Dimensional** | 100-500 | 0.98-0.999 | 10K-50K |
| **Discrete/Combinatorial** | 1-10 | 0.85-0.95 | 1K-10K |

#### 📈 Visualization Features

The `visualize()` method provides comprehensive analysis:
- **Cost Evolution**: Current vs. best cost over time
- **Temperature Schedule**: Cooling curve visualization  
- **Acceptance Rate**: Moving average of accepted moves
- **Solution Path**: 2D trajectory or parameter evolution

#### ⚡ Algorithm Properties

**Advantages:**
- Guaranteed to find global optimum (infinite time)
- Escapes local optima effectively
- Simple to implement and understand
- Works on discrete and continuous problems
- No gradient information required

**Considerations:**
- Slow convergence compared to local methods
- Parameter tuning affects performance significantly
- No convergence guarantees in finite time
- May require many function evaluations

#### 📚 References

- Kirkpatrick et al. (1983). *Optimization by Simulated Annealing*
- Metropolis et al. (1953). *Equation of State Calculations*
- Aarts & Korst (1989). *Simulated Annealing and Boltzmann Machines*
- Van Laarhoven & Aarts (1987). *Simulated Annealing: Theory and Applications*

</details>

### ▶️ `MG1Queue`

> Simulate **M/G/1 queueing systems** with general service time distributions, implementing the Pollaczek-Khinchine formula for theoretical validation and comprehensive performance analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **M/G/1 queue simulation** where arrivals follow a Poisson process (Markovian), service times follow any general distribution (General), and there is a single server. It provides both empirical simulation results and theoretical predictions using the Pollaczek-Khinchine formula.

#### 📐 Supported Service Distributions

- **Exponential**: Traditional M/M/1 case with memoryless service
- **Deterministic**: Constant service times (M/D/1)
- **Uniform**: Service times uniformly distributed over an interval
- **Normal**: Gaussian service times (truncated at zero)
- **Gamma**: Flexible shape parameter for various coefficient of variation
- **Lognormal**: Right-skewed distribution for highly variable services
- **Custom**: User-defined distribution function

#### 📊 Key Performance Metrics

- Average customers in system (L) and queue (Lq)
- Average waiting time (Wq) and system time (W)
- Server utilization (ρ)
- Service time statistics and variability analysis
- Theoretical vs empirical comparison

#### 📚 Theoretical Background

**M/G/1 Queue Components:**
- **M**: Markovian arrivals with rate λ (Poisson process)
- **G**: General service time distribution with mean E[S] and variance Var[S]
- **1**: Single server

**Key Parameters:**
- Traffic intensity: ρ = λ × E[S] (must be < 1 for stability)
- Coefficient of variation: Cs = √(Var[S]) / E[S]

**Pollaczek-Khinchine Formula:**
For a stable M/G/1 queue (ρ < 1):

Average customers in queue:
```
Lq = ρ² × (1 + Cs²) / (2(1-ρ))
```

Average customers in system:
```
L = Lq + ρ
```

Average waiting time:
```
Wq = Lq / λ
```

Average time in system:
```
W = Wq + E[S]
```

**Impact of Service Variability:**
- Higher Cs² → Longer queues and waiting times
- Deterministic service (Cs² = 0) minimizes queue length
- Exponential service (Cs² = 1) is the M/M/1 baseline

#### ✅ Applications

- **File Transfer Systems**: Variable file sizes and transfer times
- **Manufacturing**: Processing times that vary by job complexity
- **Human Services**: Variable service times at counters or help desks
- **Computer Systems**: Jobs with different computational requirements
- **Medical Procedures**: Variable treatment times by patient condition
- **Network Packet Processing**: Variable packet sizes and processing

#### 📈 Example – Basic M/G/1 with Exponential Service

```python
from RSIM.queueing import MG1Queue

# M/M/1 queue (exponential service)
mg1 = MG1Queue(
    arrival_rate=0.8,
    service_mean=1.0,
    service_distribution='exponential',
    simulation_time=10000,
    warmup_time=1000
)

result = mg1.run()
mg1.visualize()

print(f"Traffic intensity: {result.results['traffic_intensity']:.3f}")
print(f"Average queue length: {result.results['avg_queue_length']:.3f}")
print(f"Average waiting time: {result.results['avg_waiting_time']:.3f}")
```

#### 🎯 Example – M/D/1 with Deterministic Service

```python
# M/D/1 queue (constant service times)
md1 = MG1Queue(
    arrival_rate=0.8,
    service_mean=1.0,
    service_distribution='deterministic',
    simulation_time=10000
)

result = md1.run()
print(f"Deterministic service CV²: {result.statistics['theoretical_cs_squared']:.3f}")
print(f"Queue length (M/D/1): {result.statistics['theoretical_Lq']:.3f}")

# Compare with M/M/1
mm1_lq = (0.8**2) / (1 - 0.8)  # M/M/1 formula
improvement = ((mm1_lq - result.statistics['theoretical_Lq']) / mm1_lq) * 100
print(f"Improvement over M/M/1: {improvement:.1f}%")
```

#### 📉 Example – High Variability Service (Lognormal)

```python
# M/G/1 with highly variable service times
mg1_variable = MG1Queue(
    arrival_rate=0.7,
    service_mean=1.0,
    service_distribution='lognormal',
    service_params={'cv': 2.0},  # High coefficient of variation
    simulation_time=10000
)

result = mg1_variable.run()
mg1_variable.visualize()

print(f"Service CV²: {result.statistics['theoretical_cs_squared']:.3f}")
print(f"Queue length with high variability: {result.statistics['theoretical_Lq']:.3f}")
```

#### 🔧 Example – Custom Service Distribution

```python
import numpy as np

def custom_service_time():
    """Bimodal service time: 80% fast, 20% slow"""
    if np.random.random() < 0.8:
        return np.random.exponential(0.5)  # Fast service
    else:
        return np.random.exponential(3.0)   # Slow service

# Note: For custom distributions, theoretical calculations use provided mean/variance
mg1_custom = MG1Queue(
    arrival_rate=0.6,
    service_mean=1.0,  # Theoretical mean for calculations
    service_distribution='custom'
)

# Override the service time generator
mg1_custom._generate_service_time = custom_service_time

result = mg1_custom.run()
mg1_custom.visualize()
```

#### 📊 Variability Impact Analysis

```python
# Compare different service distributions with same mean
distributions = ['deterministic', 'exponential', 'gamma', 'lognormal']
cv_values = [0.0, 1.0, 1.5, 2.0]
service_params_list = [{}, {}, {'cv': 1.5}, {'cv': 2.0}]

results = {}
for dist, cv, params in zip(distributions, cv_values, service_params_list):
    mg1 = MG1Queue(
        arrival_rate=0.8,
        service_mean=1.0,
        service_distribution=dist,
        service_params=params,
        simulation_time=5000
    )
    
    result = mg1.run()
    results[f"{dist} (CV²={cv:.1f})"] = {
        'Lq': result.statistics['theoretical_Lq'],
        'Wq': result.statistics['theoretical_Wq']
    }

for name, metrics in results.items():
    print(f"{name}: Lq={metrics['Lq']:.3f}, Wq={metrics['Wq']:.3f}")
```

#### 📈 Stability Analysis

```python
# Analyze system behavior near stability boundary
arrival_rates = [0.7, 0.8, 0.9, 0.95, 0.99]

for rate in arrival_rates:
    mg1 = MG1Queue(
        arrival_rate=rate,
        service_mean=1.0,
        service_distribution='exponential',
        simulation_time=5000
    )
    
    result = mg1.run()
    rho = result.results['traffic_intensity']
    
    if rho < 1:
        print(f"ρ={rho:.3f}: L={result.statistics['theoretical_L']:.2f}, "
              f"W={result.statistics['theoretical_W']:.2f}")
    else:
        print(f"ρ={rho:.3f}: System unstable!")
```

#### 📊 Performance Guidelines

| Traffic Intensity (ρ) | System Behavior |
|----------------------|-----------------|
| ρ < 0.7             | Light traffic, minimal queuing |
| 0.7 ≤ ρ < 0.9       | Moderate traffic, stable performance |
| 0.9 ≤ ρ < 0.95      | Heavy traffic, significant queuing |
| 0.95 ≤ ρ < 1.0      | Very heavy traffic, long delays |
| ρ ≥ 1.0             | Unstable system, infinite queues |

#### 🎯 Service Distribution Selection Guide

| Distribution | When to Use | CV² Value |
|-------------|-------------|-----------|
| Deterministic | Automated systems, constant processing | 0.0 |
| Exponential | Memoryless service, baseline comparison | 1.0 |
| Gamma | Flexible variability, positive skew | Adjustable |
| Normal | Symmetric variability around mean | Adjustable |
| Lognormal | High variability, heavy right tail | > 1.0 |
| Uniform | Bounded service times | < 1.0 |

#### 📚 References

- Pollaczek, F. (1930). *Über eine Aufgabe der Wahrscheinlichkeitstheorie*
- Khintchine, A. Y. (1932). *Mathematical theory of a stationary queue*
- Gross, D. & Harris, C. M. (2008). *Fundamentals of Queueing Theory*
- Kleinrock, L. (1975). *Queueing Systems Volume 1: Theory*

</details>


### ▶️ `MM1Queue`

> Simulate the **fundamental M/M/1 queueing system** with Poisson arrivals and exponential service times, providing comprehensive performance analysis and theoretical validation.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **discrete event simulation** of the M/M/1 queue - the most fundamental queueing model in operations research. It simulates customer arrivals, service processes, and queue dynamics while collecting detailed performance statistics for comparison with analytical results.

#### 🏗️ System Components

- **M**: Markovian (Poisson) arrival process with rate λ
- **M**: Markovian (exponential) service times with rate μ  
- **1**: Single server with unlimited queue capacity
- **FIFO**: First-In-First-Out service discipline

#### 📊 Key Performance Metrics

- **System Size (L)**: Average number of customers in system
- **Queue Length (Lq)**: Average number waiting in queue
- **System Time (W)**: Average time spent in system
- **Waiting Time (Wq)**: Average time spent waiting
- **Server Utilization (ρ)**: Fraction of time server is busy

#### 📚 Theoretical Background

**Traffic Intensity:**
```
ρ = λ/μ (must be < 1 for stability)
```

**Steady-State Formulas:**
- Utilization: `ρ = λ/μ`
- Customers in system: `L = ρ/(1-ρ)`
- Customers in queue: `Lq = ρ²/(1-ρ)`
- Time in system: `W = 1/(μ-λ)`
- Waiting time: `Wq = ρ/(μ-λ)`
- Probability of n customers: `P(n) = ρⁿ(1-ρ)`

**Little's Law Relationships:**
```
L = λW    (customers in system)
Lq = λWq  (customers in queue)
W = Wq + 1/μ  (system time = wait + service)
```

#### ✅ Simulation Features

- **Discrete Event Engine**: Efficient priority queue-based simulation
- **Warmup Period**: Eliminates transient effects for steady-state analysis
- **Statistical Collection**: Comprehensive data gathering during simulation
- **Theoretical Validation**: Direct comparison with analytical results
- **Visualization**: Multi-panel performance analysis plots

#### 🎯 Applications

- **Computer Systems**: CPU scheduling, network packet processing
- **Call Centers**: Agent utilization and customer wait times
- **Manufacturing**: Single-machine processing systems
- **Healthcare**: Single-server service points (registration, triage)
- **Banking**: Single teller operations
- **Web Servers**: Request processing analysis

#### 📈 Example – Basic M/M/1 Simulation

```python
from RSIM.queueing import MM1Queue

# Configure M/M/1 queue
queue = MM1Queue(
    arrival_rate=0.8,      # 0.8 customers/time unit
    service_rate=1.0,      # 1.0 customers/time unit  
    simulation_time=1000,  # Simulate for 1000 time units
    warmup_time=100,       # 100 time units warmup
    random_seed=42         # For reproducibility
)

# Run simulation
result = queue.run()

# Display key results
print(f"Traffic Intensity (ρ): {result.results['traffic_intensity']:.3f}")
print(f"Average customers in system: {result.results['avg_customers_in_system']:.2f}")
print(f"Average waiting time: {result.results['avg_waiting_time']:.2f}")
print(f"Server utilization: {result.results['server_utilization']:.3f}")

# Visualize results
queue.visualize()
```

#### 🔥 Example – High Utilization Analysis

```python
# High traffic scenario (ρ = 0.95)
busy_queue = MM1Queue(
    arrival_rate=0.95,
    service_rate=1.0,
    simulation_time=5000,  # Longer simulation for stability
    warmup_time=500
)

result = busy_queue.run()
busy_queue.visualize()

print(f"High utilization results:")
print(f"Average queue length: {result.results['avg_queue_length']:.1f}")
print(f"Average waiting time: {result.results['avg_waiting_time']:.1f}")
```

#### 📊 Example – Theoretical vs Empirical Comparison

```python
# Compare simulation with theory
queue = MM1Queue(arrival_rate=0.7, service_rate=1.0, simulation_time=2000)
result = queue.run()

# Theoretical calculations
rho = 0.7
theoretical_L = rho / (1 - rho)  # 2.33
theoretical_W = 1 / (1.0 - 0.7)  # 3.33

# Empirical results
empirical_L = result.results['avg_customers_in_system']
empirical_W = result.results['avg_system_time']

print(f"Customers in System - Theory: {theoretical_L:.2f}, Simulation: {empirical_L:.2f}")
print(f"System Time - Theory: {theoretical_W:.2f}, Simulation: {empirical_W:.2f}")
print(f"Relative Error: {abs(theoretical_L - empirical_L)/theoretical_L*100:.1f}%")
```

#### ⚠️ Example – Unstable System Detection

```python
# Unstable system (ρ ≥ 1)
unstable_queue = MM1Queue(
    arrival_rate=1.2,      # λ > μ
    service_rate=1.0,
    simulation_time=1000
)

result = unstable_queue.run()
# Warning: System is unstable (ρ ≥ 1)! Queue will grow without bound.
```

#### 🎛️ Parameter Guidelines

| Parameter | Typical Range | Purpose |
|-----------|---------------|---------|
| `arrival_rate` | 0.1 - 0.95 | Lower values for stable systems |
| `service_rate` | 1.0+ | Usually normalized to 1.0 |
| `simulation_time` | 1000+ | Longer for accurate steady-state |
| `warmup_time` | 10-20% of sim_time | Remove transient effects |

#### 📈 Visualization Features

- **Queue Length**: Time series of customers waiting
- **System Size**: Total customers over time  
- **Server Utilization**: Busy/idle periods
- **Waiting Time Distribution**: Histogram with theoretical overlay
- **System Time Distribution**: Complete service time analysis
- **Theoretical Comparison**: Side-by-side metric comparison

#### 🔬 Validation Methods

- **Steady-State Convergence**: Long-run averages approach theory
- **Little's Law Verification**: L = λW relationships hold
- **Distribution Fitting**: Service times follow exponential distribution
- **Confidence Intervals**: Statistical significance testing

#### 📚 References

- Gross, D., et al. (2008). *Fundamentals of Queueing Theory*
- Kleinrock, L. (1975). *Queueing Systems Volume 1: Theory*  
- Ross, S. M. (2014). *Introduction to Probability Models*
- Law, A. M. (2015). *Simulation Modeling and Analysis*

</details>


### ▶️ `MMKQueue`

> Simulate **multi-server queueing systems** with Poisson arrivals and exponential service times, providing comprehensive performance analysis and theoretical validation.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **M/M/k queue simulation** - a fundamental multi-server queueing model where customers arrive according to a Poisson process, are served by k identical servers with exponential service times, and wait in a single queue when all servers are busy.

#### 🏗️ System Architecture
- **M**: Markovian (Poisson) arrival process
- **M**: Markovian (exponential) service times  
- **k**: k parallel identical servers
- **∞**: Unlimited queue capacity
- **FIFO**: First-In-First-Out service discipline

#### 📊 Key Performance Metrics
- **System utilization**: ρ = λ/(kμ)
- **Average customers in system**: L
- **Average customers in queue**: Lq
- **Average time in system**: W
- **Average waiting time**: Wq
- **Server utilization**: Individual and overall

#### 📚 Theoretical Background

**Traffic Intensity:**
```
ρ = λ/(kμ) < 1 (stability condition)
```

**Steady-State Probabilities:**
```
P₀ = [Σ(n=0 to k-1) (λ/μ)ⁿ/n! + (λ/μ)ᵏ/(k!(1-ρ))]⁻¹
```

**Performance Measures:**
```
Lq = P₀(λ/μ)ᵏρ/(k!(1-ρ)²)  (customers in queue)
L = Lq + λ/μ                 (customers in system)
Wq = Lq/λ                    (waiting time)
W = Wq + 1/μ                 (time in system)
```

**Little's Law Relationships:**
- L = λW (customers in system)
- Lq = λWq (customers in queue)

#### ✅ Model Properties
- **Memoryless**: Exponential inter-arrival and service times
- **Steady-state**: Long-run equilibrium behavior
- **Birth-death process**: State transitions only ±1
- **Erlang-C formula**: Exact analytical solutions available

#### 🎯 Applications
- **Call centers** with multiple agents
- **Bank branches** with multiple tellers
- **Hospital emergency** departments
- **Airport check-in** counters
- **Manufacturing systems** with parallel machines
- **Computer systems** with multiple processors

#### 📈 Example – Basic M/M/3 Queue
```python
from RSIM.queueing import MMKQueue

# Configure 3-server system
mmk = MMKQueue(
    arrival_rate=2.0,      # 2 customers/hour
    service_rate=1.0,      # 1 customer/hour per server
    num_servers=3,         # 3 parallel servers
    simulation_time=1000,  # 1000 time units
    warmup_time=100       # 100 time units warmup
)

result = mmk.run()
mmk.visualize()

print(f"Traffic intensity: {result.results['traffic_intensity']:.3f}")
print(f"Average customers in system: {result.results['avg_customers_in_system']:.2f}")
print(f"Average waiting time: {result.results['avg_waiting_time']:.2f}")
```

#### 🏥 Example – Hospital Emergency Department
```python
# Model emergency department with 5 doctors
hospital_ed = MMKQueue(
    arrival_rate=4.5,      # 4.5 patients/hour
    service_rate=1.2,      # 1.2 patients/hour per doctor
    num_servers=5,         # 5 doctors on duty
    simulation_time=2000,
    warmup_time=200
)

result = hospital_ed.run()

print(f"System utilization: {result.results['traffic_intensity']:.1%}")
print(f"Average patients waiting: {result.results['avg_queue_length']:.1f}")
print(f"Average wait time: {result.results['avg_waiting_time']:.1f} hours")
```

#### 📞 Example – Call Center Optimization
```python
import numpy as np

# Compare different staffing levels
staffing_levels = [3, 4, 5, 6]
arrival_rate = 15  # calls per hour
service_rate = 4   # calls per hour per agent

for k in staffing_levels:
    call_center = MMKQueue(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        num_servers=k,
        simulation_time=1000
    )
    
    result = call_center.run()
    
    print(f"Servers: {k}")
    print(f"  Utilization: {result.results['traffic_intensity']:.1%}")
    print(f"  Avg wait time: {result.results['avg_waiting_time']:.2f} hours")
    print(f"  Service level: {1 - result.results['avg_waiting_time']/0.25:.1%}")
    print()
```

#### 📊 Visualization Features
- **Queue length** over time with theoretical benchmarks
- **System size** evolution and steady-state comparison
- **Server utilization** tracking across all servers
- **Waiting time** and **system time** distributions
- **Theoretical vs empirical** performance comparison
- **Stability analysis** and traffic intensity visualization

#### ⚖️ Capacity Planning Guidelines

| Traffic Intensity (ρ) | System Status | Recommendation |
|----------------------|---------------|----------------|
| ρ < 0.7 | Under-utilized | Consider reducing servers |
| 0.7 ≤ ρ < 0.85 | Well-balanced | Optimal range |
| 0.85 ≤ ρ < 0.95 | High utilization | Monitor closely |
| ρ ≥ 0.95 | Near saturation | Add servers immediately |
| ρ ≥ 1.0 | **Unstable** | System will collapse |

#### 🔧 Parameter Tuning
- **arrival_rate**: Customer arrival intensity (λ)
- **service_rate**: Service capacity per server (μ)
- **num_servers**: Number of parallel servers (k)
- **simulation_time**: Total simulation duration
- **warmup_time**: Transient period to exclude
- **random_seed**: For reproducible results

#### 📚 References
- Gross, D. & Harris, C.M. (1998). *Fundamentals of Queueing Theory*
- Kleinrock, L. (1975). *Queueing Systems Volume 1: Theory*
- Cooper, R.B. (1981). *Introduction to Queueing Theory*
- Bolch, G. et al. (2006). *Queueing Networks and Markov Chains*

</details>


### ▶️ `QueueNetwork`

> Simulate **complex multi-station queueing networks** with customer routing, feedback loops, and comprehensive performance analysis for manufacturing, service, and computer systems.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **discrete-event simulation** of interconnected queueing systems where customers move between multiple service stations according to routing probabilities. It models complex systems like manufacturing networks, computer networks, hospital patient flows, and multi-stage service systems.

#### 🏗️ Network Structure

- **External arrivals** to specified stations (Poisson processes)
- **Internal routing** between stations (probabilistic)
- **External departures** from any station
- **Feedback loops** allowed (customers can return to previous stations)
- **Multiple servers** per station supported

#### 📊 Key Performance Metrics

- **Queue lengths** and system sizes over time
- **Server utilizations** for each station
- **Throughput rates** and total customers served
- **Traffic intensities** and stability analysis
- **Customer flow** tracking and routing statistics

#### 📚 Theoretical Background

**Network Structure:**
- Stations: \( i = 1, 2, \ldots, N \)
- External arrival rates: \( \lambda_i \) (Poisson)
- Service rates: \( \mu_i \) (exponential)
- Routing probabilities: \( P_{ij} \) (station \( i \) to \( j \))

**Traffic Equations:**
\[
\gamma_i = \lambda_i + \sum_{j=1}^{N} \gamma_j P_{ji}
\]
where \( \gamma_i \) is the total arrival rate to station \( i \).

**Stability Condition:**
\[
\rho_i = \frac{\gamma_i}{\mu_i} < 1 \quad \forall i
\]

**Jackson Networks:**
For open networks with exponential service times, steady-state probabilities have product form:
\[
\pi(n_1, n_2, \ldots, n_N) = \prod_{i=1}^{N} \pi_i(n_i)
\]

#### 🎯 Applications

- **Manufacturing systems** with multiple workstations
- **Computer networks** with packet routing
- **Hospital patient flow** through departments
- **Call center routing** systems
- **Supply chain networks**
- **Airport terminal operations**

#### ⚙️ Default Configuration

```python
# Simple 3-station tandem queue
QueueNetwork(
    num_stations=3,
    arrival_rates=[1.0, 0.0, 0.0],    # Only external arrivals to station 1
    service_rates=[2.0, 1.5, 1.8],    # Different service rates
    routing_matrix=[[0.0, 0.8, 0.0],  # 80% from station 1 to 2
                    [0.0, 0.0, 0.8],  # 80% from station 2 to 3
                    [0.0, 0.0, 0.0]]  # 100% external departure from station 3
)
```

#### 📈 Example – Manufacturing System

```python
from RSIM.queueing import QueueNetwork
import numpy as np

# 4-station manufacturing line with rework loops
manufacturing = QueueNetwork(
    num_stations=4,
    arrival_rates=[5.0, 0.0, 0.0, 0.0],  # Raw materials arrive at station 1
    service_rates=[6.0, 5.5, 4.8, 5.2],  # Processing rates
    routing_matrix=[
        [0.0, 0.9, 0.0, 0.1],  # 90% to next, 10% rework
        [0.0, 0.0, 0.85, 0.15], # 85% forward, 15% rework
        [0.0, 0.0, 0.0, 0.95],  # 95% to final station
        [0.0, 0.0, 0.0, 0.0]    # 100% finished products
    ],
    simulation_time=2000.0,
    warmup_time=200.0
)

result = manufacturing.run()
manufacturing.visualize()

print("Station Utilizations:", result.results['utilizations'])
print("Throughput Rates:", result.results['throughput_rates'])
```

#### 🏥 Example – Hospital Patient Flow

```python
# Hospital emergency department with 3 stages
hospital = QueueNetwork(
    num_stations=3,
    arrival_rates=[8.0, 0.0, 0.0],      # Patients arrive at triage
    service_rates=[12.0, 6.0, 4.0],     # Triage, treatment, discharge
    routing_matrix=[
        [0.0, 0.7, 0.3],   # 70% to treatment, 30% discharged
        [0.1, 0.0, 0.9],   # 10% back to triage, 90% to discharge
        [0.0, 0.0, 0.0]    # 100% leave system
    ],
    simulation_time=1440.0,  # 24 hours in minutes
    warmup_time=120.0
)

result = hospital.run()
print("Average queue lengths:", result.results['avg_queue_lengths'])
print("Patient waiting times by stage")
```

#### 🌐 Example – Computer Network Routing

```python
# 5-node network with load balancing
network = QueueNetwork(
    num_stations=5,
    arrival_rates=[10.0, 5.0, 0.0, 0.0, 0.0],  # External traffic
    service_rates=[15.0, 12.0, 18.0, 14.0, 16.0],  # Processing capacities
    routing_matrix=[
        [0.0, 0.3, 0.3, 0.2, 0.2],  # Load balancing from node 1
        [0.0, 0.0, 0.4, 0.3, 0.3],  # Routing from node 2
        [0.0, 0.0, 0.0, 0.5, 0.5],  # Split traffic
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Exit points
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ]
)

result = network.run()
# Check for network stability
unstable = [i for i, rho in enumerate(result.statistics['traffic_intensities']) if rho >= 1.0]
if unstable:
    print(f"⚠️ Unstable stations: {unstable}")
```

#### 📊 Visualization Features

- **Queue lengths** over time (all stations)
- **System sizes** evolution
- **Performance metrics** comparison bar charts
- **Throughput rates** by station
- **Routing matrix** heatmap
- **Traffic intensities** with stability indicators

#### 🔧 Advanced Configuration

```python
# Complex network with feedback and multiple entry points
complex_network = QueueNetwork(
    num_stations=6,
    arrival_rates=[3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
    service_rates=[4.5, 3.8, 5.2, 4.0, 3.5, 6.0],
    routing_matrix=[
        [0.0, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.1, 0.0, 0.0, 0.5, 0.4, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.3, 0.5],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.9],
        [0.2, 0.0, 0.0, 0.3, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    simulation_time=5000.0,
    warmup_time=500.0,
    random_seed=42
)
```

#### ⚡ Performance Guidelines

| Stations | Simulation Time | Typical Runtime |
|----------|----------------|-----------------|
| 2-5      | 1,000-5,000    | < 1 second      |
| 6-10     | 5,000-10,000   | 1-5 seconds     |
| > 10     | Use carefully  | > 10 seconds    |

#### 🎯 Stability Analysis

The simulation automatically checks:
- **Traffic intensities** ρᵢ = γᵢ/μᵢ for each station
- **Stability warnings** when ρᵢ ≥ 1.0
- **Color-coded visualization** (green: stable, red: unstable)

#### 📚 References

- Jackson, J.R. (1957). *Networks of Waiting Lines*
- Baskett, F. et al. (1975). *Open, Closed, and Mixed Networks*
- Bolch, G. et al. (2006). *Queueing Networks and Markov Chains*
- Gross, D. & Harris, C.M. (2008). *Fundamentals of Queueing Theory*

</details>


### ▶️ `ComponentReliability`

> Perform comprehensive **component reliability analysis** using multiple failure distributions, Monte Carlo simulation, and advanced reliability metrics for engineering applications.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **parametric reliability analysis** for individual components using various failure distribution models. It calculates key reliability metrics, performs Monte Carlo lifetime simulations, and provides comprehensive statistical analysis of failure patterns and maintenance strategies.

#### 📐 Supported Distributions

- **Exponential**: Constant failure rate (memoryless property)
- **Weibull**: Flexible bathtub curve modeling (early, random, wear-out failures)
- **Normal**: Wear-out failure mechanisms
- **Lognormal**: Multiplicative damage processes (fatigue, corrosion)

#### 📊 Key Reliability Metrics

- Reliability function R(t) and unreliability F(t)
- Failure rate (hazard function) λ(t)
- Mean Time To Failure (MTTF)
- Percentile life (B₁₀, B₅₀, B₉₀)
- Mission reliability at specified time
- Confidence intervals for all estimates

#### 📚 Theoretical Background

**Fundamental Relationships:**
- Reliability Function: R(t) = P(T > t) = 1 - F(t)
- Failure Rate: λ(t) = f(t) / R(t)
- Mean Time To Failure: MTTF = ∫₀^∞ R(t) dt

**Distribution Models:**

**Exponential Distribution:**
- PDF: f(t) = λe^(-λt)
- Reliability: R(t) = e^(-λt)
- MTTF: 1/λ
- Constant failure rate (λ)

**Weibull Distribution:**
- PDF: f(t) = (β/η)(t/η)^(β-1)e^(-(t/η)^β)
- Reliability: R(t) = e^(-(t/η)^β)
- Shape parameter β: <1 (decreasing), =1 (constant), >1 (increasing)
- Scale parameter η: characteristic life

**Normal Distribution:**
- Used for wear-out mechanisms
- Parameters: μ (mean), σ (standard deviation)

**Lognormal Distribution:**
- Used for multiplicative damage processes
- Parameters: μ (log-scale), σ (log-shape)

#### ✅ Applications

- Electronic component reliability assessment
- Mechanical system failure analysis
- Preventive maintenance planning
- Warranty analysis and cost estimation
- Life testing data analysis
- System design optimization

#### 📈 Example – Exponential reliability analysis

```python
from RSIM.reliability import ComponentReliability

# Configure exponential reliability model
exp_rel = ComponentReliability(
    distribution='exponential',
    parameters={'lambda': 0.001},  # 0.001 failures/hour
    mission_time=1000,             # 1000 hours mission
    n_simulations=50000
)

result = exp_rel.run()
print(f"Reliability at mission time: {result.results['mission_reliability']:.4f}")
print(f"MTTF: {result.results['mttf']:.2f} hours")

# Visualize results
exp_rel.visualize()
```

#### 🎯 Example – Weibull bathtub curve analysis

```python
# Weibull model for wear-out failures
weibull_rel = ComponentReliability(
    distribution='weibull',
    parameters={'beta': 2.5, 'eta': 5000},  # Increasing failure rate
    mission_time=3000,
    confidence_level=0.90
)

result = weibull_rel.run()
print(f"B10 life: {result.results['b10_life']:.2f} hours")
print(f"Mission reliability: {result.results['mission_reliability']:.4f}")

# Calculate failure rate at different times
failure_rate_1000 = weibull_rel.calculate_failure_rate(1000)
failure_rate_3000 = weibull_rel.calculate_failure_rate(3000)
print(f"Failure rate at 1000h: {failure_rate_1000:.6f}")
print(f"Failure rate at 3000h: {failure_rate_3000:.6f}")
```

#### 📉 Example – Parameter estimation from failure data

```python
# Estimate parameters from observed failure times
failure_data = [1200, 1850, 2100, 2400, 2900, 3200, 3800, 4100]

normal_rel = ComponentReliability(distribution='normal')
estimated_params = normal_rel.estimate_parameters_from_data(failure_data)
print("Estimated parameters:", estimated_params)

# Configure with estimated parameters
normal_rel.configure('normal', estimated_params, mission_time=2000)
result = normal_rel.run()
print(f"Estimated MTTF: {result.results['mttf']:.2f}")
```

#### 🔧 Example – Maintenance optimization

```python
# Calculate optimal maintenance intervals
maintenance_metrics = weibull_rel.calculate_maintenance_metrics(
    maintenance_cost=500,    # $500 per maintenance
    failure_cost=5000       # $5000 per failure
)

print(f"Optimal replacement interval: {maintenance_metrics['optimal_replacement_interval']:.1f} hours")
print(f"Total cost rate: ${maintenance_metrics['total_cost_rate']:.2f}/hour")
```

#### 📊 Example – Sensitivity analysis

```python
# Analyze sensitivity to parameter variations
sensitivity_results = weibull_rel.perform_sensitivity_analysis({
    'beta': [1.5, 2.0, 2.5, 3.0, 3.5],
    'eta': [4000, 4500, 5000, 5500, 6000]
})

# Results show how reliability metrics change with parameters
for param, results in sensitivity_results.items():
    print(f"\nSensitivity to {param}:")
    for result in results:
        print(f"  {param}={result['parameter_value']}: R={result['mission_reliability']:.4f}")
```

#### 📊 Simulation Guidelines

| Simulations | Purpose                    |
|-------------|----------------------------|
| 1,000       | Quick exploration          |
| 10,000      | Standard analysis          |
| 50,000      | High-precision results     |
| 100,000+    | Publication-grade accuracy |

#### 🎨 Visualization Features

- Reliability function R(t) over time
- Failure rate λ(t) curves (bathtub, increasing, decreasing)
- Probability density and cumulative distribution functions
- Monte Carlo lifetime histograms with theoretical overlay
- Convergence analysis for simulation accuracy
- Comprehensive metrics summary tables

#### 📚 References

- Barlow, R. E. & Proschan, F. (1975). *Statistical Theory of Reliability*
- Nelson, W. (1982). *Applied Life Data Analysis*
- Meeker, W. Q. & Escobar, L. A. (1998). *Statistical Methods for Reliability Data*
- Rausand, M. & Høyland, A. (2004). *System Reliability Theory*

</details>


### ▶️ `FailureAnalysis`

> Perform comprehensive **failure mode analysis and reliability assessment** for Monte Carlo π estimation simulations, detecting statistical anomalies, convergence issues, and implementation errors.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **comprehensive failure analysis** for Monte Carlo π estimation simulations by running multiple independent simulations with different configurations and systematically analyzing failure modes, statistical properties, convergence behavior, and reliability metrics.

#### 🔍 Failure Categories Analyzed

- **Convergence Failures**: Slow convergence, oscillatory behavior, premature plateaus
- **Statistical Anomalies**: Bias detection, variance issues, confidence interval violations
- **Numerical Precision Issues**: Floating-point errors, accumulation errors, catastrophic cancellation
- **Random Number Generator Problems**: Poor randomness quality, correlation, periodicity
- **Parameter-Related Failures**: Insufficient samples, memory issues, timeouts
- **Implementation Errors**: Algorithm correctness, boundary conditions, resource management

#### 📊 Analysis Methods

- **Multiple Independent Runs**: Different seeds and sample sizes
- **Statistical Hypothesis Testing**: Bias tests, normality tests, variance analysis
- **Convergence Rate Analysis**: Power law fitting, monotonicity checks
- **Error Distribution Characterization**: Outlier detection, distribution fitting
- **Performance Profiling**: Timing analysis, scalability assessment

#### 📚 Theoretical Background

**Monte Carlo Error Analysis:**
For π estimation using circle-square method:
- Theoretical error: σ ≈ √(π(4-π)/n) ≈ 1.64/√n
- Expected convergence rate: O(n^(-0.5))
- Bias should approach zero as n → ∞

**Statistical Tests:**
- **Bias Test**: One-sample t-test against π
- **Variance Test**: Compare observed vs theoretical variance
- **Normality Test**: Shapiro-Wilk or Anderson-Darling
- **Confidence Interval Test**: Check if CI contains true π

**Reliability Score:**
Weighted failure rate considering severity:
```
Reliability = max(0, 1 - Σ(failure_count × weight) / total_runs)
```

#### ✅ Diagnostic Outputs

- **Failure Probability Estimates**: By failure type and sample size
- **Error Distribution Analysis**: Outliers, bias, variance anomalies
- **Convergence Quality Metrics**: Rates, oscillation, plateau detection
- **Statistical Test Results**: p-values, confidence intervals
- **Performance Benchmarks**: Timing, scalability, efficiency
- **Actionable Recommendations**: Specific improvement suggestions

#### 📈 Example – Basic Failure Analysis

```python
from RSIM.monte_carlo import PiEstimationMC
from RSIM.reliability import FailureAnalysis

# Create π estimation simulation
pi_sim = PiEstimationMC(n_samples=100000)

# Set up failure analysis
analyzer = FailureAnalysis(
    base_simulation=pi_sim,
    n_runs=50,
    confidence_level=0.95,
    failure_threshold=0.01  # 1% error threshold
)

# Run comprehensive analysis
results = analyzer.run_analysis()

print(f"Reliability Score: {analyzer.reliability_score:.3f}")
print(f"Total Failures: {len(analyzer.failure_modes)}")

# Generate detailed report
report = analyzer.generate_report()
print(report)
```

#### 🎯 Example – Multi-Scale Analysis

```python
# Test multiple sample sizes
analyzer = FailureAnalysis(
    pi_sim,
    n_runs=100,
    sample_sizes=[1000, 10000, 100000, 1000000],
    failure_threshold=0.005,
    timeout_seconds=120
)

results = analyzer.run_analysis()

# Visualize failure patterns
analyzer.visualize_failures()

# Export results
analyzer.export_results("pi_analysis_report.txt")
```

#### 📉 Example – Convergence Analysis

```python
# Focus on convergence issues
analyzer = FailureAnalysis(pi_sim, n_runs=200)
results = analyzer.run_analysis()

# Check convergence-specific failures
conv_analysis = results['convergence_analysis']
print(f"Slow convergence cases: {conv_analysis['slow_convergence_count']}")
print(f"Oscillatory behavior: {conv_analysis['oscillatory_behavior_count']}")
print(f"Premature plateaus: {conv_analysis['premature_plateau_count']}")

# Get recommendations
for rec in analyzer.recommendations:
    print(f"• {rec}")
```

#### 🔧 Example – Custom Failure Thresholds

```python
# Strict quality requirements
strict_analyzer = FailureAnalysis(
    pi_sim,
    n_runs=500,
    confidence_level=0.99,
    failure_threshold=0.001,  # 0.1% error threshold
    timeout_seconds=300
)

results = strict_analyzer.run_analysis()

# Check production readiness
summary = strict_analyzer.get_failure_summary()
if summary['production_ready']:
    print("✅ Simulation is production-ready")
else:
    print("❌ Simulation needs improvement")
    print(f"Critical failures: {summary['critical_failures']}")
```

#### 📊 Reliability Assessment Scale

| Score Range | Assessment | Description |
|-------------|------------|-------------|
| 0.90 - 1.00 | Excellent | Production ready, minimal issues |
| 0.70 - 0.89 | Good | Minor issues, fine-tuning recommended |
| 0.50 - 0.69 | Moderate | Needs attention, review required |
| 0.00 - 0.49 | Poor | Major issues, complete review needed |

#### 🎨 Visualization Features

- **Reliability Metrics Dashboard**: Overall scores and failure rates
- **Failure Type Distribution**: Pie charts and bar plots
- **Performance Scaling**: Log-log plots of timing vs sample size
- **Error Analysis**: Scatter plots, histograms, Q-Q plots
- **Convergence Behavior**: Detailed convergence trajectories
- **Statistical Test Results**: Test statistics and p-values

#### 📋 Common Failure Types

| Failure Type | Description | Typical Cause |
|--------------|-------------|---------------|
| `accuracy_failure` | Error exceeds threshold | Insufficient samples |
| `slow_convergence` | Poor convergence rate | RNG quality issues |
| `significant_bias` | Systematic error | Implementation bug |
| `extreme_outlier` | Statistical anomaly | Numerical precision |
| `timeout` | Execution too slow | Performance issue |
| `floating_point_error` | Numerical instability | Precision loss |

#### 🚀 Performance Guidelines

| Sample Size | Expected Time | Recommended Runs |
|-------------|---------------|------------------|
| 1,000 | < 1ms | 100+ |
| 10,000 | < 10ms | 100+ |
| 100,000 | < 100ms | 50+ |
| 1,000,000 | < 1s | 20+ |

#### 📚 References

- Gentle, J.E. (2003). *Random Number Generation and Monte Carlo Methods*
- Robert, C.P. & Casella, G. (2004). *Monte Carlo Statistical Methods*
- Fishman, G.S. (1995). *Monte Carlo: Concepts, Algorithms, and Applications*
- L'Ecuyer, P. (2012). *Random Number Generation*

</details>


### ▶️ `PreventiveMaintenance`

> Optimize **preventive maintenance schedules** using Monte Carlo simulation to minimize total costs while maximizing system reliability and availability.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Monte Carlo simulation for preventive maintenance optimization** that determines the optimal maintenance interval by balancing preventive maintenance costs, corrective maintenance costs, and downtime expenses. It uses Weibull failure modeling to simulate realistic component lifetimes and provides comprehensive economic analysis.

#### 🔧 Key Features

- **Multi-interval optimization** with cost-benefit analysis
- **Weibull failure modeling** with configurable shape and scale parameters
- **Economic analysis** including all cost components (preventive, corrective, downtime)
- **Sensitivity analysis** for robust decision making
- **Availability and reliability metrics** calculation
- **Visual comparison** of maintenance strategies
- **Monte Carlo uncertainty quantification**

#### 📊 Optimization Metrics

- **Cost Rate**: Total cost per unit time ($/hour, $/day)
- **System Availability**: Percentage of uptime
- **Maintenance Frequency**: Preventive vs corrective events
- **Reliability**: Probability of survival at given times
- **Economic Indicators**: ROI, break-even analysis

#### 📚 Theoretical Background

**Weibull Failure Model:**
- Failure rate function: λ(t) = (β/η) × (t/η)^(β-1)
- Reliability function: R(t) = exp(-(t/η)^β)
- Mean Time To Failure: MTTF = η × Γ(1 + 1/β)

**Cost Optimization:**
- Expected cost per cycle: E[C] = C_pm + C_fail × P(failure) + C_downtime × E[downtime]
- Cost rate: CR(T) = E[C(T)] / E[cycle_length(T)]
- Optimal interval: T* = argmin(CR(T))

**Key Parameters:**
- β: Shape parameter (β > 1 = wear-out, β < 1 = early failures, β = 1 = constant rate)
- η: Scale parameter (characteristic life)
- T: Maintenance interval

#### ✅ Applications

- Industrial equipment maintenance planning
- Fleet management optimization
- Manufacturing system reliability
- Power plant maintenance scheduling
- Aircraft maintenance optimization
- Medical equipment servicing
- IT system maintenance planning

#### 📈 Example – Basic Maintenance Optimization

```python
from RSIM.reliability import PreventiveMaintenance

# Configure maintenance simulation
maint_sim = PreventiveMaintenance(
    n_simulations=5000,
    weibull_shape=2.5,              # Wear-out failure pattern
    weibull_scale=800,              # 800 hours characteristic life
    maintenance_intervals=[100, 200, 300, 400, 500],
    cost_preventive=1200,           # $1200 per preventive maintenance
    cost_corrective=6000,           # $6000 per failure repair
    cost_downtime_per_hour=500,     # $500/hour downtime cost
    preventive_maintenance_time=4,   # 4 hours planned downtime
    corrective_maintenance_time=24   # 24 hours unplanned downtime
)

# Run optimization
result = maint_sim.run()

# Display results
print(f"Optimal interval: {result.results['optimal_interval']} hours")
print(f"Minimum cost rate: ${result.results['minimum_cost_rate']:.2f}/hour")
print(f"System availability: {result.results['optimal_availability']:.2%}")

# Visualize results
maint_sim.visualize(show_sensitivity=True)
```

#### 🎯 Example – High-Reliability System

```python
# Critical system with high downtime costs
critical_system = PreventiveMaintenance(
    weibull_shape=1.8,
    weibull_scale=2000,
    cost_downtime_per_hour=2000,    # High downtime penalty
    maintenance_intervals=range(50, 500, 25),  # Fine-grained search
    n_simulations=10000
)

result = critical_system.run()
print(f"Critical system optimal interval: {result.results['optimal_interval']} hours")
print(f"Availability: {result.results['optimal_availability']:.3%}")
```

#### 📉 Example – Fleet Maintenance Planning

```python
# Vehicle fleet maintenance (mileage-based)
fleet_maint = PreventiveMaintenance(
    weibull_shape=2.2,
    weibull_scale=5000,             # 5000 miles characteristic life
    maintenance_intervals=[500, 1000, 1500, 2000, 2500],
    cost_preventive=800,            # Scheduled service cost
    cost_corrective=3500,           # Breakdown repair cost
    simulation_horizon=50000        # 50,000 miles simulation
)

result = fleet_maint.run()
print(f"Optimal service interval: {result.results['optimal_interval']} miles")
```

#### 🔍 Example – Sensitivity Analysis

```python
# Analyze parameter sensitivity
maint_sim = PreventiveMaintenance(
    weibull_shape=2.0,
    weibull_scale=1000,
    maintenance_intervals=[100, 200, 300, 400, 500],
    cost_preventive=1000,
    cost_corrective=5000
)

result = maint_sim.run()

# Check sensitivity results
for param, sensitivity in result.results['sensitivity_analysis'].items():
    max_change = max(abs(s['relative_change']) for s in sensitivity)
    print(f"{param}: Max cost impact = ±{max_change:.1f}%")
```

#### 📊 Weibull Shape Parameter Guide

| β Value | Failure Pattern | Typical Applications |
|---------|----------------|---------------------|
| β < 1   | Decreasing failure rate | Electronics, early failures |
| β = 1   | Constant failure rate | Random failures, exponential |
| β = 2   | Linear increase (Rayleigh) | Wear-out, fatigue |
| β > 3   | Rapid wear-out | Mechanical systems, aging |

#### 💡 Decision Guidelines

- **Cost-driven**: Choose interval minimizing total cost rate
- **Reliability-driven**: Choose interval maximizing availability
- **Risk-averse**: Choose interval with lowest cost variance
- **Practical**: Consider maintenance resource constraints
- **Regulatory**: Ensure compliance with safety requirements

#### 📈 Performance Characteristics

- **Time complexity**: O(n_simulations × n_intervals × avg_cycles)
- **Typical runtime**: 1-10 seconds for standard problems
- **Memory usage**: ~1MB for typical parameter sets
- **Recommended simulations**: 5,000+ for reliable results, 10,000+ for precision

#### 📚 References

- Barlow, R. E. & Proschan, F. (1965). *Mathematical Theory of Reliability*
- Nakagawa, T. (2005). *Maintenance Theory of Reliability*
- Jardine, A. K. S. & Tsang, A. H. C. (2013). *Maintenance, Replacement, and Reliability*
- Rausand, M. & Høyland, A. (2004). *System Reliability Theory*

</details>


### ▶️ `SystemReliability`

> Perform comprehensive **Monte Carlo system reliability analysis** with support for multiple architectures, component failure modeling, and importance measures.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **Monte Carlo simulation for system reliability analysis** using component failure modeling. It estimates the reliability of complex systems by modeling individual component failures and their impact on overall system functionality, supporting various architectures including series, parallel, k-out-of-n, and complex network topologies with redundancy and repair mechanisms.

#### 🏗️ Supported System Architectures

- **Series Systems**: All components must function for system success
- **Parallel Systems**: At least one component must function (redundancy)
- **k-out-of-n Systems**: At least k out of n components must function
- **Complex Networks**: Arbitrary component interconnections
- **Standby Systems**: Active components with backup configurations

#### 📊 Key Reliability Metrics

- **Point Reliability**: R(t) at mission time
- **Mean Time To Failure (MTTF)**: Expected time until first failure
- **Availability**: A(t) for repairable systems
- **Component Importance Measures**: Birnbaum, Fussell-Vesely, RAW, RRW
- **Failure Rate**: λ(t) = f(t) / R(t)
- **Confidence Intervals**: Statistical uncertainty bounds

#### 📚 Theoretical Background

**System Reliability Formulas:**

For systems with n components:
- **Series System**: R_sys(t) = ∏(i=1 to n) R_i(t)
- **Parallel System**: R_sys(t) = 1 - ∏(i=1 to n) [1 - R_i(t)]
- **k-out-of-n System**: R_sys(t) = Σ(i=k to n) C(n,i) * [R(t)]^i * [1-R(t)]^(n-i)

**Component Reliability:**
R_i(t) = e^(-λ_i * t) where λ_i is the failure rate of component i

**Statistical Properties:**
- Standard error: σ ≈ √(R(1-R)/n) where n is simulation runs
- Convergence rate: O(1/√n) - typical for Monte Carlo methods
- 95% confidence interval: R_estimate ± 1.96 × σ

**Importance Measures:**
- **Birnbaum Importance**: ∂R_sys/∂R_i (marginal reliability improvement)
- **Fussell-Vesely**: Contribution to system unreliability
- **Risk Achievement Worth (RAW)**: Risk increase if component fails
- **Risk Reduction Worth (RRW)**: Risk decrease if component perfect

#### ✅ Key Features

- Multi-component system modeling with configurable architectures
- Time-dependent reliability analysis with failure/repair cycles
- Sensitivity analysis for critical component identification
- Uncertainty quantification and confidence intervals
- Performance optimization for large-scale systems
- Comprehensive visualization and reporting

#### 📈 Example – Series System Analysis

```python
from RSIM.reliability import SystemReliability

# Configure series system (all components must work)
series_sim = SystemReliability(
    system_type='series',
    n_components=5,
    failure_rates=0.001,  # Same rate for all components
    mission_time=1000,
    n_simulations=50000
)

result = series_sim.run()
print(f"System reliability: {result.results['reliability']:.4f}")
print(f"MTTF: {result.results['mttf']:.2f}")
print(f"95% CI: [{result.results['reliability_ci_lower']:.4f}, {result.results['reliability_ci_upper']:.4f}]")

# Visualize results
series_sim.visualize(show_importance=True)
```

#### 🔄 Example – Parallel Redundant System

```python
# Configure parallel system with different component reliabilities
parallel_sim = SystemReliability(
    system_type='parallel',
    n_components=3,
    failure_rates=[0.002, 0.001, 0.0015],  # Different rates
    mission_time=2000,
    n_simulations=100000
)

result = parallel_sim.run()
print(f"System reliability: {result.results['reliability']:.4f}")

# Calculate component importance
importance = parallel_sim.calculate_importance_measures()
print("\nComponent Importance (Birnbaum):")
for i, imp in importance['birnbaum'].items():
    print(f"  Component {i+1}: {imp:.4f}")
```

#### 🗳️ Example – k-out-of-n Voting System

```python
# Configure 4-out-of-7 voting system (e.g., fault-tolerant computer)
voting_sim = SystemReliability(
    system_type='k_out_of_n',
    n_components=7,
    k_value=4,  # Need at least 4 working
    failure_rates=0.0005,
    mission_time=5000,
    random_seed=42
)

result = voting_sim.run()
print(f"Voting system reliability: {result.results['reliability']:.4f}")
print(f"System can tolerate {7-4} component failures")
```

#### 🔧 Example – Repairable System with Maintenance

```python
# System with repair capability
repairable_sim = SystemReliability(
    system_type='parallel',
    n_components=4,
    failure_rates=0.01,     # Higher failure rate
    repair_rates=0.1,       # Repair rate
    include_repair=True,
    mission_time=1000
)

result = repairable_sim.run()
print(f"System reliability: {result.results['reliability']:.4f}")
print(f"System availability: {result.results['availability']:.4f}")
print(f"MTTR: {result.results['mttr']:.2f}")
print(f"MTBF: {result.results['mtbf']:.2f}")
```

#### 🔍 Example – Sensitivity Analysis

```python
# Perform sensitivity analysis on failure rates
sensitivity = series_sim.sensitivity_analysis(parameter_range=0.5)

print("Sensitivity Analysis Results:")
for component, curve in sensitivity.items():
    print(f"\n{component}:")
    for multiplier, reliability in curve[:3]:  # Show first 3 points
        print(f"  Rate × {multiplier:.1f}: R = {reliability:.4f}")
```

#### 🎯 Applications

- **Aerospace**: Aircraft system design and certification
- **Nuclear**: Power plant safety analysis
- **Manufacturing**: Production system optimization
- **Networks**: Infrastructure reliability planning
- **Medical**: Device reliability assessment
- **Automotive**: Safety system validation
- **Software**: Fault tolerance analysis
- **Supply Chain**: Risk analysis

#### 📊 Performance Guidelines

| Components | Simulations | Typical Speed    | Memory Usage |
|------------|-------------|------------------|--------------|
| 5          | 10,000      | ~1 sec          | ~10 MB       |
| 10         | 50,000      | ~5 sec          | ~50 MB       |
| 20         | 100,000     | ~15 sec         | ~200 MB      |
| 50         | 100,000     | ~30 sec         | ~500 MB      |

#### 📈 Visualization Features

**Standard Mode:**
- System reliability over time with confidence intervals
- Component failure distribution and statistics
- Reliability metrics summary
- System state timeline for sample runs

**Component Analysis Mode:**
- Individual component reliability curves
- Component failure time distributions
- Component state correlation analysis

**Importance Analysis Mode:**
- Component importance measure rankings
- Sensitivity analysis results
- Critical component identification
- Redundancy effectiveness assessment

#### 📚 References

- Rausand, M. & Høyland, A. (2004). *System Reliability Theory*
- Barlow, R. E. & Proschan, F. (1975). *Statistical Theory of Reliability*
- Kuo, W. & Zuo, M. J. (2003). *Optimal Reliability Modeling*
- O'Connor, P. & Kleyner, A. (2012). *Practical Reliability Engineering*

</details>


### ▶️ `CoxProportionalHazards`

> Perform **semi-parametric survival analysis** using the Cox Proportional Hazards model to estimate covariate effects on hazard rates without distributional assumptions.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements the **Cox Proportional Hazards model**, the gold standard for survival analysis that estimates how covariates affect the hazard rate while leaving the baseline hazard unspecified. It handles censored data and provides hazard ratios with confidence intervals.

#### 📐 Key Features

- **Maximum Likelihood Estimation**: Uses Newton-Raphson optimization for coefficient estimation
- **Tie Handling Methods**: Breslow (fast), Efron (accurate), Exact (precise)
- **Statistical Tests**: Wald, likelihood ratio, and score tests
- **Diagnostic Tools**: Proportional hazards testing, residual analysis
- **Prediction**: Survival curves and hazard ratios for new observations
- **Robust Standard Errors**: Optional sandwich estimator

#### 📊 Model Components

- **Hazard Function**: h(t|x) = h₀(t) × exp(β₁x₁ + β₂x₂ + ... + βₚxₚ)
- **Baseline Hazard**: h₀(t) - unspecified function of time
- **Linear Predictor**: βᵀx - log hazard ratio
- **Hazard Ratio**: exp(β) - multiplicative effect on hazard

#### 📚 Theoretical Background

**Cox Model Specification:**
The hazard at time t given covariates x is:
```
h(t|x) = h₀(t) × exp(βᵀx)
```

**Partial Likelihood:**
The key innovation is using partial likelihood that eliminates the baseline hazard:
```
L(β) = ∏ᵢ [exp(βᵀxᵢ) / Σⱼ∈R(tᵢ) exp(βᵀxⱼ)]^δᵢ
```

Where:
- δᵢ = 1 if event occurred, 0 if censored
- R(tᵢ) = risk set at time tᵢ (all subjects still at risk)

**Statistical Properties:**
- Hazard ratios are constant over time (proportional hazards assumption)
- Semi-parametric: no distributional assumptions for baseline hazard
- Asymptotically normal coefficient estimates
- Transformation invariant

#### ✅ Model Assumptions

1. **Proportional Hazards**: Hazard ratios constant over time
2. **Log-linearity**: Log hazard linear in covariates
3. **Independence**: Observations are independent
4. **Multiplicative Effects**: Covariates multiply baseline hazard
5. **Correct Functional Form**: Appropriate covariate transformations

#### 📈 Example – Basic Cox Regression

```python
import pandas as pd
from RSIM.survival import CoxProportionalHazards

# Survival data
data = pd.DataFrame({
    'time': [5, 6, 6, 2, 4, 4, 7, 8, 3, 5],
    'event': [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    'age': [65, 72, 55, 60, 68, 70, 58, 75, 62, 69],
    'treatment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'stage': [1, 2, 1, 3, 2, 2, 1, 3, 2, 1]
})

# Fit Cox model
cox = CoxProportionalHazards()
result = cox.run(data, 'time', 'event', ['age', 'treatment', 'stage'])

print("Hazard Ratios:")
for i, var in enumerate(['age', 'treatment', 'stage']):
    hr = result.results['hazard_ratios'][i]
    ci = result.results['confidence_intervals'][i]
    p_val = result.results['p_values'][i]
    print(f"{var}: HR = {hr:.3f}, 95% CI = ({np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}), p = {p_val:.3f}")
```

#### 🎯 Example – Advanced Analysis with Diagnostics

```python
# Advanced Cox model with robust standard errors
cox_advanced = CoxProportionalHazards(
    tie_method='efron',
    robust_se=True,
    include_baseline=True
)

result = cox_advanced.run(data, 'time', 'event', ['age', 'treatment', 'stage'])

# Model summary with visualization
cox_advanced.visualize(plot_type='summary')

# Diagnostic plots
cox_advanced.visualize(plot_type='diagnostics')

# Test proportional hazards assumption
ph_test = cox_advanced.test_proportional_hazards()
print(f"Proportional Hazards Test p-value: {ph_test['global_p_value']:.4f}")
print(f"Assumption: {ph_test['interpretation']}")
```

#### 📉 Example – Survival Prediction

```python
# Predict survival for new patients
new_patients = pd.DataFrame({
    'age': [60, 70, 65],
    'treatment': [0, 1, 0],
    'stage': [1, 2, 3]
})

# Survival probabilities at specific times
times = [1, 2, 3, 4, 5]
survival_probs = cox_advanced.predict_survival(new_patients, times)

# Hazard ratios (relative to baseline)
hazard_ratios = cox_advanced.predict_hazard_ratio(new_patients)

print("Survival Probabilities:")
for i, time in enumerate(times):
    print(f"Time {time}: {survival_probs[:, i]}")

print(f"Hazard Ratios: {hazard_ratios}")
```

#### 🔧 Example – Model Comparison and Selection

```python
# Compare different covariate combinations
models = [
    ['age'],
    ['age', 'treatment'],
    ['age', 'treatment', 'stage'],
    ['age', 'treatment', 'stage', 'age*treatment']  # interaction
]

results = []
for covariates in models:
    cox_temp = CoxProportionalHazards()
    result = cox_temp.run(data, 'time', 'event', covariates)
    results.append({
        'covariates': covariates,
        'aic': result.results['aic'],
        'bic': result.results['bic'],
        'concordance': result.results['concordance_index'],
        'lr_p_value': result.results['lr_p_value']
    })

# Display model comparison
for i, res in enumerate(results):
    print(f"Model {i+1}: {res['covariates']}")
    print(f"  AIC: {res['aic']:.2f}, BIC: {res['bic']:.2f}")
    print(f"  C-index: {res['concordance']:.3f}, p-value: {res['lr_p_value']:.4f}")
```

#### 📊 Tie Handling Methods

| Method   | Speed | Accuracy | Best For |
|----------|-------|----------|----------|
| Breslow  | Fast  | Good     | Few ties, large datasets |
| Efron    | Medium| Better   | Moderate ties (recommended) |
| Exact    | Slow  | Best     | Many ties, small datasets |

#### 🎯 Model Diagnostics

**Proportional Hazards Testing:**
- Schoenfeld residuals correlation with time
- Global and individual covariate tests
- Graphical assessment

**Goodness of Fit:**
- Martingale residuals for functional form
- Deviance residuals for outliers
- Concordance index (C-statistic)

**Influence Diagnostics:**
- DFBETA statistics
- Score residuals
- Leverage measures

#### 📈 Performance Guidelines

| Sample Size | Events Needed | Covariates | Recommendation |
|-------------|---------------|------------|----------------|
| < 100       | 20-50         | 2-5        | Simple models only |
| 100-500     | 50-100        | 5-10       | Standard analysis |
| 500-1000    | 100-200       | 10-20      | Complex models OK |
| > 1000      | 200+          | 20+        | Full diagnostics |

#### 📚 Applications

- **Clinical Trials**: Treatment effect estimation
- **Epidemiology**: Risk factor identification
- **Reliability**: Failure time analysis
- **Economics**: Duration modeling
- **Marketing**: Customer churn analysis
- **Insurance**: Actuarial modeling

#### 📚 References

- Cox, D.R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society*
- Therneau, T.M. & Grambsch, P.M. (2000). *Modeling Survival Data*
- Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis*
- Collett, D. (2015). *Modelling Survival Data in Medical Research*

</details>


### ▶️ `ExponentialSurvival`

> Perform comprehensive **exponential survival analysis** with parameter estimation, censoring simulation, and goodness-of-fit testing for reliability and medical applications.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **parametric survival analysis** using the exponential distribution, the simplest survival model with constant hazard rate. It generates survival times, applies various censoring mechanisms, estimates parameters via maximum likelihood, and provides extensive diagnostic tools.

#### 🔧 Key Features

- **Survival Time Generation**: Exponential random variates with configurable rate parameter
- **Censoring Mechanisms**: Random, administrative, or no censoring
- **Parameter Estimation**: Maximum likelihood estimation with confidence intervals
- **Goodness-of-Fit**: Kolmogorov-Smirnov and coefficient of variation tests
- **Comprehensive Visualization**: Survival curves, hazard plots, Q-Q plots, and diagnostics

#### 📊 Censoring Types

- **None**: All events observed (complete data)
- **Random**: Censoring times from exponential distribution
- **Administrative**: Fixed study end time (Type I censoring)

#### 📚 Theoretical Background

**Exponential Distribution:**
The exponential distribution is characterized by a single rate parameter λ (lambda):

- **PDF**: f(t) = λe^(-λt) for t ≥ 0
- **Survival Function**: S(t) = e^(-λt)
- **Hazard Function**: h(t) = λ (constant)
- **Mean Survival Time**: E[T] = 1/λ
- **Median Survival Time**: t₅₀ = ln(2)/λ ≈ 0.693/λ

**Key Properties:**
- **Memoryless**: P(T > s+t | T > s) = P(T > t)
- **Constant hazard rate** (no aging effect)
- **Scale parameter**: θ = 1/λ

**Maximum Likelihood Estimation:**
For censored data with d events and total time T:
- **MLE**: λ̂ = d/T
- **Standard Error**: SE(λ̂) = λ̂/√d
- **95% CI**: λ̂ ± 1.96 × SE(λ̂)
- **Log-likelihood**: ℓ(λ) = d×ln(λ) - λ×T

#### ✅ Applications

- **Medical**: Baseline survival model, clinical trials
- **Reliability**: Electronic component failure analysis
- **Engineering**: Maintenance scheduling, risk analysis
- **Queueing**: Service time modeling
- **Insurance**: Claim occurrence modeling

#### 📈 Example – Basic Exponential Survival

```python
from RSIM.survival import ExponentialSurvival

# Basic exponential survival simulation
exp_sim = ExponentialSurvival(
    lambda_rate=0.5,      # Rate parameter (failures per unit time)
    n_samples=1000,       # Sample size
    random_seed=42        # Reproducibility
)

result = exp_sim.run()
print(f"True λ: 0.5, Estimated λ: {result.results['estimated_lambda']:.4f}")
print(f"Mean survival time: {result.results['mean_survival_time']:.2f}")
print(f"95% CI: {result.results['confidence_interval']}")

# Visualize results
exp_sim.visualize()
```

#### 🎯 Example – Survival Analysis with Censoring

```python
# Simulate realistic censored survival data
exp_cens = ExponentialSurvival(
    lambda_rate=0.2,           # Lower failure rate
    n_samples=500,
    censoring_rate=0.3,        # 30% censoring
    censoring_type='random'    # Random censoring
)

result = exp_cens.run()
print(f"Events observed: {result.results['n_events']}")
print(f"Censoring proportion: {result.results['censoring_proportion']:.2f}")
print(f"Estimated λ: {result.results['estimated_lambda']:.4f}")

# Detailed visualization with diagnostics
exp_cens.visualize(show_details=True)
```

#### 📉 Example – Administrative Censoring (Clinical Trial)

```python
# Simulate clinical trial with fixed study duration
clinical_trial = ExponentialSurvival(
    lambda_rate=0.1,               # Low event rate
    n_samples=200,
    censoring_type='administrative',
    study_time=10.0                # 10-year follow-up
)

result = clinical_trial.run()
print(f"Study duration: 10 years")
print(f"Events observed: {result.results['n_events']}")
print(f"Patients censored: {result.results['n_censored']}")
```

#### 🔍 Example – Fit Model to External Data

```python
import numpy as np

# Your survival data
times = np.array([1.2, 2.5, 0.8, 3.1, 1.9, 4.2, 0.5])
events = np.array([1, 1, 0, 1, 1, 1, 0])  # 1=event, 0=censored

# Fit exponential model
exp_model = ExponentialSurvival()
fit_results = exp_model.fit(times, events)

print(f"Fitted λ: {fit_results['lambda_estimate']:.4f}")
print(f"Mean survival: {fit_results['mean_survival_time']:.2f}")
print(f"95% CI: {fit_results['confidence_interval']}")

# Goodness-of-fit tests
if 'kolmogorov_smirnov' in fit_results['goodness_of_fit']:
    ks_test = fit_results['goodness_of_fit']['kolmogorov_smirnov']
    print(f"KS test p-value: {ks_test['p_value']:.4f}")
    print(f"Reject exponential: {ks_test['reject_exponential']}")
```

#### 📊 Visualization Outputs

**Standard Plots:**
- Survival function comparison (Kaplan-Meier vs. Exponential)
- Constant hazard function visualization
- Histogram with fitted exponential density
- Parameter estimates with confidence intervals

**Detailed Diagnostics (show_details=True):**
- Q-Q plot for exponential distribution
- Log-survival plot (linearity check)
- Cox-Snell residual analysis
- Goodness-of-fit test results

#### 🎯 Parameter Guidelines

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `lambda_rate` | 0.01 - 10.0 | Higher = more failures |
| `n_samples` | 100 - 10,000 | Larger = better estimates |
| `censoring_rate` | 0.0 - 0.8 | Proportion censored |
| `study_time` | 1.0 - 100.0 | Administrative cutoff |

#### ⚠️ Model Assumptions

- **Constant hazard rate** over time
- **Independence** of survival times
- **Exponential distribution** assumption
- **Non-informative censoring**
- **Homogeneous population**

#### 📚 References

- Lawless, J.F. (2003). *Statistical Models and Methods for Lifetime Data*
- Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis*
- Collett, D. (2015). *Modelling Survival Data in Medical Research*
- Cox, D.R. & Oakes, D. (1984). *Analysis of Survival Data*

</details>


### ▶️ `WeibullSurvival`

> Perform comprehensive **Weibull survival analysis** with parameter estimation, reliability metrics, and hazard modeling for lifetime data analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **parametric survival analysis** using the Weibull distribution, one of the most versatile models in reliability engineering and survival analysis. It provides parameter estimation, survival function computation, hazard analysis, and lifetime prediction with support for censored data.

#### 📐 Supported Estimation Methods

- **Maximum Likelihood Estimation (MLE)**: Most efficient for complete and censored data
- **Method of Moments**: Simple closed-form solutions for initial estimates  
- **Least Squares (LSQ)**: Graphical Weibull probability plot method
- **Bootstrap Confidence Intervals**: Non-parametric uncertainty quantification

#### 📊 Key Features

- **Censoring Support**: Right, left, and interval censoring handling
- **Goodness-of-Fit**: Kolmogorov-Smirnov, Anderson-Darling tests
- **Reliability Metrics**: MTTF, percentiles, characteristic life
- **Hazard Analysis**: Increasing, decreasing, or constant hazard patterns
- **Visualization**: Survival curves, hazard plots, diagnostic plots

#### 📚 Theoretical Background

**Weibull Distribution:**
The Weibull distribution is characterized by shape parameter *k* and scale parameter *λ*:

**Probability Density Function:**
```
f(t) = (k/λ) × (t/λ)^(k-1) × exp(-(t/λ)^k)
```

**Survival Function:**
```
S(t) = exp(-(t/λ)^k)
```

**Hazard Function:**
```
h(t) = (k/λ) × (t/λ)^(k-1)
```

**Shape Parameter Interpretation:**
- k < 1: Decreasing hazard (early failures, infant mortality)
- k = 1: Constant hazard (exponential distribution, random failures)  
- k > 1: Increasing hazard (wear-out failures, aging)
- k = 2: Rayleigh distribution (special case)

**Statistical Properties:**
- Mean: μ = λ × Γ(1 + 1/k)
- Variance: σ² = λ² × [Γ(1 + 2/k) - Γ²(1 + 1/k)]
- Median: λ × (ln(2))^(1/k)
- Characteristic life: λ (63.2% failure probability)

#### ✅ Applications

- **Reliability Engineering**: Component lifetime modeling, failure analysis
- **Medical Research**: Survival times, treatment efficacy studies
- **Quality Control**: Product failure analysis, warranty modeling
- **Materials Science**: Fatigue life, strength modeling
- **Environmental Studies**: Time-to-event modeling
- **Manufacturing**: Equipment maintenance planning

#### 📈 Example – Basic Weibull Simulation

```python
from RSIM.survival import WeibullSurvival

# Configure Weibull survival analysis
weibull_sim = WeibullSurvival(
    shape_param=2.5,      # Increasing hazard (wear-out)
    scale_param=100,      # Characteristic life = 100 time units
    n_samples=1000,       # Sample size
    censoring_rate=0.2    # 20% censored observations
)

# Run simulation
result = weibull_sim.run()

# Display results
print(f"True shape: {weibull_sim.shape_param}")
print(f"Estimated shape: {result.results['estimated_shape']:.3f}")
print(f"95% CI: [{result.results['shape_ci_lower']:.3f}, {result.results['shape_ci_upper']:.3f}]")
print(f"MTTF: {result.statistics['mttf']:.2f}")

# Visualize results
weibull_sim.visualize()
```

#### 🎯 Example – Fit External Data

```python
# Survival times with censoring indicators
times = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
censored = [False, False, True, False, True, False, False, True, False, False]

# Fit Weibull model
weibull_fit = WeibullSurvival(estimation_method='mle')
result = weibull_fit.fit_data(times, censored)

print(f"Estimated parameters:")
print(f"Shape (k): {result.results['estimated_shape']:.3f}")
print(f"Scale (λ): {result.results['estimated_scale']:.3f}")
print(f"Log-likelihood: {result.statistics['log_likelihood']:.2f}")

# Goodness-of-fit
print(f"KS test p-value: {result.statistics['ks_p_value']:.4f}")
```

#### 📉 Example – Reliability Prediction

```python
# Configure for reliability analysis
weibull_pred = WeibullSurvival(shape_param=3.0, scale_param=1000)
result = weibull_pred.run()

# Predict survival at specific times
times_of_interest = [500, 750, 1000, 1250]
survival_probs = weibull_pred.predict_survival(times_of_interest)

for t, s in zip(times_of_interest, survival_probs):
    print(f"Survival at t={t}: {s:.3f} ({s*100:.1f}%)")

# Calculate reliability percentiles
percentiles = weibull_pred.calculate_percentiles([0.1, 0.5, 0.9])
print(f"B10 life (10% failure): {percentiles[0]:.1f}")
print(f"Median life (50% failure): {percentiles[1]:.1f}")
print(f"B90 life (90% failure): {percentiles[2]:.1f}")
```

#### 🔧 Example – Hazard Analysis

```python
# Different hazard patterns
hazard_patterns = [
    (0.8, "Decreasing hazard - Early failures"),
    (1.0, "Constant hazard - Random failures"), 
    (2.5, "Increasing hazard - Wear-out failures")
]

for shape, description in hazard_patterns:
    weibull = WeibullSurvival(shape_param=shape, scale_param=100)
    result = weibull.run()
    
    print(f"\n{description}")
    print(f"Shape parameter: {shape}")
    print(f"MTTF: {result.statistics['mttf']:.2f}")
    
    # Visualize hazard function
    weibull.visualize(plot_type='hazard')
```

#### 📊 Parameter Guidelines

| Parameter | Range | Description |
|-----------|-------|-------------|
| shape_param | 0.1 - 10.0 | Weibull shape parameter (k) |
| scale_param | 0.1 - 1000+ | Weibull scale parameter (λ) |
| n_samples | 50 - 10,000 | Sample size for simulation |
| censoring_rate | 0.0 - 0.9 | Proportion of censored observations |
| confidence_level | 0.8 - 0.99 | Confidence level for intervals |

#### 📈 Visualization Options

- **'survival'**: Kaplan-Meier vs fitted survival curves
- **'hazard'**: Hazard function with pattern interpretation
- **'density'**: PDF with histogram overlay and key statistics
- **'diagnostic'**: Weibull probability plot for goodness-of-fit
- **'all'**: Comprehensive 4-panel visualization

#### 📚 References

- Weibull, W. (1951). *A Statistical Distribution Function of Wide Applicability*
- Lawless, J.F. (2003). *Statistical Models and Methods for Lifetime Data*
- Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis*
- Meeker, W.Q. & Escobar, L.A. (1998). *Statistical Methods for Reliability Data*
- Nelson, W. (2004). *Applied Life Data Analysis*

</details>

---

### ▶️ `ExponentialSurvival`

> Perform **exponential survival analysis** for constant hazard rate modeling, ideal for random failure processes and memoryless systems.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **exponential survival analysis**, a special case of Weibull distribution with shape parameter k=1. The exponential distribution is characterized by a constant hazard rate and memoryless property, making it ideal for modeling random failures and Poisson processes.

#### 📐 Key Characteristics

- **Constant Hazard Rate**: No aging effect, failures occur randomly
- **Memoryless Property**: Past survival doesn't affect future probability
- **Single Parameter**: Only rate parameter λ needs estimation
- **Maximum Entropy**: Optimal distribution for given mean constraint
- **Poisson Connection**: Models inter-arrival times in Poisson processes

#### 📊 Mathematical Properties

**Probability Density Function:**
```
f(t) = λ × exp(-λt)
```

**Survival Function:**
```
S(t) = exp(-λt)
```

**Hazard Function:**
```
h(t) = λ (constant)
```

**Statistical Properties:**
- Mean: μ = 1/λ
- Variance: σ² = 1/λ²
- Median: ln(2)/λ
- Mode: 0
- Memoryless: P(T > s+t | T > s) = P(T > t)

#### ✅ Applications

- **Electronics**: Random component failures
- **Software**: Bug discovery processes
- **Telecommunications**: Call inter-arrival times
- **Nuclear Physics**: Radioactive decay
- **Queueing Theory**: Service time modeling
- **Reliability**: Baseline constant failure rate analysis

#### 📈 Example – Basic Exponential Analysis

```python
from RSIM.survival import ExponentialSurvival

# Configure exponential survival analysis
exp_survival = ExponentialSurvival(
    rate_param=0.02,      # Failure rate = 0.02 per time unit
    n_samples=1000,       # Sample size
    censoring_rate=0.15   # 15% censored observations
)

# Run simulation
result = exp_survival.run()

# Display results
print(f"True rate: {exp_survival.rate_param}")
print(f"Estimated rate: {result.results['estimated_rate']:.4f}")
print(f"95% CI: [{result.results['rate_ci_lower']:.4f}, {result.results['rate_ci_upper']:.4f}]")
print(f"Mean lifetime: {result.statistics['mean_lifetime']:.2f}")
print(f"Median lifetime: {result.statistics['median_lifetime']:.2f}")

# Visualize results
exp_survival.visualize()
```

#### 🎯 Example – Reliability Metrics

```python
# Electronic component reliability
component_failure_rate = 0.001  # 0.1% per hour

exp_model = ExponentialSurvival(rate_param=component_failure_rate)
result = exp_model.run()

# Calculate key reliability metrics
mean_time_to_failure = 1 / component_failure_rate
median_life = np.log(2) / component_failure_rate

print(f"Component Reliability Analysis:")
print(f"Failure rate: {component_failure_rate} failures/hour")
print(f"MTTF: {mean_time_to_failure:.0f} hours")
print(f"Median life: {median_life:.0f} hours")

# Survival probabilities at key times
times = [100, 500, 1000, 2000]  # hours
survival_probs = exp_model.predict_survival(times)

for t, prob in zip(times, survival_probs):
    print(f"Survival at {t} hours: {prob:.3f} ({prob*100:.1f}%)")
```

#### 📉 Example – Memoryless Property Demonstration

```python
# Demonstrate memoryless property
rate = 0.05
exp_demo = ExponentialSurvival(rate_param=rate)

# P(T > 20) 
prob_20 = np.exp(-rate * 20)

# P(T > 30 | T > 10) should equal P(T > 20)
prob_conditional = np.exp(-rate * 20)  # Due to memoryless property

print(f"Memoryless Property Verification:")
print(f"P(T > 20) = {prob_20:.4f}")
print(f"P(T > 30 | T > 10) = {prob_conditional:.4f}")
print(f"Equal? {np.isclose(prob_20, prob_conditional)}")
```

#### 🔧 Example – Comparing with Weibull

```python
from RSIM.survival import WeibullSurvival

# Compare exponential (k=1) with Weibull
rate = 0.03
scale = 1/rate  # Convert rate to scale parameter

# Exponential model
exp_model = ExponentialSurvival(rate_param=rate, n_samples=1000)
exp_result = exp_model.run()

# Equivalent Weibull model (shape=1)
weibull_model = WeibullSurvival(shape_param=1.0, scale_param=scale, n_samples=1000)
weibull_result = weibull_model.run()

print("Model Comparison:")
print(f"Exponential - Rate: {exp_result.results['estimated_rate']:.4f}")
print(f"Weibull - Shape: {weibull_result.results['estimated_shape']:.4f}")
print(f"Weibull - Scale: {weibull_result.results['estimated_scale']:.2f}")
print(f"Equivalent rate: {1/weibull_result.results['estimated_scale']:.4f}")
```

#### 📊 Parameter Guidelines

| Parameter | Range | Description |
|-----------|-------|-------------|
| rate_param | 0.01 - 10.0 | Exponential rate parameter (λ) |
| n_samples | 50 - 10,000 | Sample size for simulation |
| censoring_rate | 0.0 - 0.9 | Proportion of censored observations |
| confidence_level | 0.8 - 0.99 | Confidence level for intervals |

#### 📈 When to Use Exponential vs Weibull

**Use Exponential when:**
- Constant failure rate is reasonable
- Memoryless property applies
- Simple baseline model needed
- Random failures (no wear-out or burn-in)

**Use Weibull when:**
- Hazard rate changes over time
- Burn-in or wear-out effects present
- More flexible modeling needed
- Shape of hazard function is important

#### 📚 References

- Ross, S.M. (2014). *Introduction to Probability Models*
- Lawless, J.F. (2003). *Statistical Models and Methods for Lifetime Data*
- Barlow, R.E. & Proschan, F. (1975). *Statistical Theory of Reliability*
- Nelson, W. (2004). *Applied Life Data Analysis*

</details>


### ▶️ `WeibullSurvival`

> Perform comprehensive **Weibull survival analysis** with parameter estimation, reliability metrics, and hazard modeling for lifetime data analysis.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **parametric survival analysis** using the Weibull distribution, one of the most versatile models in reliability engineering and survival analysis. It provides parameter estimation, survival function computation, hazard analysis, and lifetime prediction with support for censored data.

#### 📐 Supported Estimation Methods

- **Maximum Likelihood Estimation (MLE)**: Most efficient for complete and censored data
- **Method of Moments**: Simple closed-form solutions for initial estimates  
- **Least Squares (LSQ)**: Graphical Weibull probability plot method
- **Bootstrap Confidence Intervals**: Non-parametric uncertainty quantification

#### 📊 Key Features

- **Censoring Support**: Right, left, and interval censoring handling
- **Goodness-of-Fit**: Kolmogorov-Smirnov, Anderson-Darling tests
- **Reliability Metrics**: MTTF, percentiles, characteristic life
- **Hazard Analysis**: Increasing, decreasing, or constant hazard patterns
- **Visualization**: Survival curves, hazard plots, diagnostic plots

#### 📚 Theoretical Background

**Weibull Distribution:**
The Weibull distribution is characterized by shape parameter *k* and scale parameter *λ*:

**Probability Density Function:**
```
f(t) = (k/λ) × (t/λ)^(k-1) × exp(-(t/λ)^k)
```

**Survival Function:**
```
S(t) = exp(-(t/λ)^k)
```

**Hazard Function:**
```
h(t) = (k/λ) × (t/λ)^(k-1)
```

**Shape Parameter Interpretation:**
- k < 1: Decreasing hazard (early failures, infant mortality)
- k = 1: Constant hazard (exponential distribution, random failures)  
- k > 1: Increasing hazard (wear-out failures, aging)
- k = 2: Rayleigh distribution (special case)

**Statistical Properties:**
- Mean: μ = λ × Γ(1 + 1/k)
- Variance: σ² = λ² × [Γ(1 + 2/k) - Γ²(1 + 1/k)]
- Median: λ × (ln(2))^(1/k)
- Characteristic life: λ (63.2% failure probability)

#### ✅ Applications

- **Reliability Engineering**: Component lifetime modeling, failure analysis
- **Medical Research**: Survival times, treatment efficacy studies
- **Quality Control**: Product failure analysis, warranty modeling
- **Materials Science**: Fatigue life, strength modeling
- **Environmental Studies**: Time-to-event modeling
- **Manufacturing**: Equipment maintenance planning

#### 📈 Example – Basic Weibull Simulation

```python
from RSIM.survival import WeibullSurvival

# Configure Weibull survival analysis
weibull_sim = WeibullSurvival(
    shape_param=2.5,      # Increasing hazard (wear-out)
    scale_param=100,      # Characteristic life = 100 time units
    n_samples=1000,       # Sample size
    censoring_rate=0.2    # 20% censored observations
)

# Run simulation
result = weibull_sim.run()

# Display results
print(f"True shape: {weibull_sim.shape_param}")
print(f"Estimated shape: {result.results['estimated_shape']:.3f}")
print(f"95% CI: [{result.results['shape_ci_lower']:.3f}, {result.results['shape_ci_upper']:.3f}]")
print(f"MTTF: {result.statistics['mttf']:.2f}")

# Visualize results
weibull_sim.visualize()
```

#### 🎯 Example – Fit External Data

```python
# Survival times with censoring indicators
times = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
censored = [False, False, True, False, True, False, False, True, False, False]

# Fit Weibull model
weibull_fit = WeibullSurvival(estimation_method='mle')
result = weibull_fit.fit_data(times, censored)

print(f"Estimated parameters:")
print(f"Shape (k): {result.results['estimated_shape']:.3f}")
print(f"Scale (λ): {result.results['estimated_scale']:.3f}")
print(f"Log-likelihood: {result.statistics['log_likelihood']:.2f}")

# Goodness-of-fit
print(f"KS test p-value: {result.statistics['ks_p_value']:.4f}")
```

#### 📉 Example – Reliability Prediction

```python
# Configure for reliability analysis
weibull_pred = WeibullSurvival(shape_param=3.0, scale_param=1000)
result = weibull_pred.run()

# Predict survival at specific times
times_of_interest = [500, 750, 1000, 1250]
survival_probs = weibull_pred.predict_survival(times_of_interest)

for t, s in zip(times_of_interest, survival_probs):
    print(f"Survival at t={t}: {s:.3f} ({s*100:.1f}%)")

# Calculate reliability percentiles
percentiles = weibull_pred.calculate_percentiles([0.1, 0.5, 0.9])
print(f"B10 life (10% failure): {percentiles[0]:.1f}")
print(f"Median life (50% failure): {percentiles[1]:.1f}")
print(f"B90 life (90% failure): {percentiles[2]:.1f}")
```

#### 🔧 Example – Hazard Analysis

```python
# Different hazard patterns
hazard_patterns = [
    (0.8, "Decreasing hazard - Early failures"),
    (1.0, "Constant hazard - Random failures"), 
    (2.5, "Increasing hazard - Wear-out failures")
]

for shape, description in hazard_patterns:
    weibull = WeibullSurvival(shape_param=shape, scale_param=100)
    result = weibull.run()
    
    print(f"\n{description}")
    print(f"Shape parameter: {shape}")
    print(f"MTTF: {result.statistics['mttf']:.2f}")
    
    # Visualize hazard function
    weibull.visualize(plot_type='hazard')
```

#### 📊 Parameter Guidelines

| Parameter | Range | Description |
|-----------|-------|-------------|
| shape_param | 0.1 - 10.0 | Weibull shape parameter (k) |
| scale_param | 0.1 - 1000+ | Weibull scale parameter (λ) |
| n_samples | 50 - 10,000 | Sample size for simulation |
| censoring_rate | 0.0 - 0.9 | Proportion of censored observations |
| confidence_level | 0.8 - 0.99 | Confidence level for intervals |

#### 📈 Visualization Options

- **'survival'**: Kaplan-Meier vs fitted survival curves
- **'hazard'**: Hazard function with pattern interpretation
- **'density'**: PDF with histogram overlay and key statistics
- **'diagnostic'**: Weibull probability plot for goodness-of-fit
- **'all'**: Comprehensive 4-panel visualization

#### 📚 References

- Weibull, W. (1951). *A Statistical Distribution Function of Wide Applicability*
- Lawless, J.F. (2003). *Statistical Models and Methods for Lifetime Data*
- Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis*
- Meeker, W.Q. & Escobar, L.A. (1998). *Statistical Methods for Reliability Data*
- Nelson, W. (2004). *Applied Life Data Analysis*

</details>

---

### ▶️ `ExponentialSurvival`

> Perform **exponential survival analysis** for constant hazard rate modeling, ideal for random failure processes and memoryless systems.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **exponential survival analysis**, a special case of Weibull distribution with shape parameter k=1. The exponential distribution is characterized by a constant hazard rate and memoryless property, making it ideal for modeling random failures and Poisson processes.

#### 📐 Key Characteristics

- **Constant Hazard Rate**: No aging effect, failures occur randomly
- **Memoryless Property**: Past survival doesn't affect future probability
- **Single Parameter**: Only rate parameter λ needs estimation
- **Maximum Entropy**: Optimal distribution for given mean constraint
- **Poisson Connection**: Models inter-arrival times in Poisson processes

#### 📊 Mathematical Properties

**Probability Density Function:**
```
f(t) = λ × exp(-λt)
```

**Survival Function:**
```
S(t) = exp(-λt)
```

**Hazard Function:**
```
h(t) = λ (constant)
```

**Statistical Properties:**
- Mean: μ = 1/λ
- Variance: σ² = 1/λ²
- Median: ln(2)/λ
- Mode: 0
- Memoryless: P(T > s+t | T > s) = P(T > t)

#### ✅ Applications

- **Electronics**: Random component failures
- **Software**: Bug discovery processes
- **Telecommunications**: Call inter-arrival times
- **Nuclear Physics**: Radioactive decay
- **Queueing Theory**: Service time modeling
- **Reliability**: Baseline constant failure rate analysis

#### 📈 Example – Basic Exponential Analysis

```python
from RSIM.survival import ExponentialSurvival

# Configure exponential survival analysis
exp_survival = ExponentialSurvival(
    rate_param=0.02,      # Failure rate = 0.02 per time unit
    n_samples=1000,       # Sample size
    censoring_rate=0.15   # 15% censored observations
)

# Run simulation
result = exp_survival.run()

# Display results
print(f"True rate: {exp_survival.rate_param}")
print(f"Estimated rate: {result.results['estimated_rate']:.4f}")
print(f"95% CI: [{result.results['rate_ci_lower']:.4f}, {result.results['rate_ci_upper']:.4f}]")
print(f"Mean lifetime: {result.statistics['mean_lifetime']:.2f}")
print(f"Median lifetime: {result.statistics['median_lifetime']:.2f}")

# Visualize results
exp_survival.visualize()
```

#### 🎯 Example – Reliability Metrics

```python
# Electronic component reliability
component_failure_rate = 0.001  # 0.1% per hour

exp_model = ExponentialSurvival(rate_param=component_failure_rate)
result = exp_model.run()

# Calculate key reliability metrics
mean_time_to_failure = 1 / component_failure_rate
median_life = np.log(2) / component_failure_rate

print(f"Component Reliability Analysis:")
print(f"Failure rate: {component_failure_rate} failures/hour")
print(f"MTTF: {mean_time_to_failure:.0f} hours")
print(f"Median life: {median_life:.0f} hours")

# Survival probabilities at key times
times = [100, 500, 1000, 2000]  # hours
survival_probs = exp_model.predict_survival(times)

for t, prob in zip(times, survival_probs):
    print(f"Survival at {t} hours: {prob:.3f} ({prob*100:.1f}%)")
```

#### 📉 Example – Memoryless Property Demonstration

```python
# Demonstrate memoryless property
rate = 0.05
exp_demo = ExponentialSurvival(rate_param=rate)

# P(T > 20) 
prob_20 = np.exp(-rate * 20)

# P(T > 30 | T > 10) should equal P(T > 20)
prob_conditional = np.exp(-rate * 20)  # Due to memoryless property

print(f"Memoryless Property Verification:")
print(f"P(T > 20) = {prob_20:.4f}")
print(f"P(T > 30 | T > 10) = {prob_conditional:.4f}")
print(f"Equal? {np.isclose(prob_20, prob_conditional)}")
```

#### 🔧 Example – Comparing with Weibull

```python
from RSIM.survival import WeibullSurvival

# Compare exponential (k=1) with Weibull
rate = 0.03
scale = 1/rate  # Convert rate to scale parameter

# Exponential model
exp_model = ExponentialSurvival(rate_param=rate, n_samples=1000)
exp_result = exp_model.run()

# Equivalent Weibull model (shape=1)
weibull_model = WeibullSurvival(shape_param=1.0, scale_param=scale, n_samples=1000)
weibull_result = weibull_model.run()

print("Model Comparison:")
print(f"Exponential - Rate: {exp_result.results['estimated_rate']:.4f}")
print(f"Weibull - Shape: {weibull_result.results['estimated_shape']:.4f}")
print(f"Weibull - Scale: {weibull_result.results['estimated_scale']:.2f}")
print(f"Equivalent rate: {1/weibull_result.results['estimated_scale']:.4f}")
```

#### 📊 Parameter Guidelines

| Parameter | Range | Description |
|-----------|-------|-------------|
| rate_param | 0.01 - 10.0 | Exponential rate parameter (λ) |
| n_samples | 50 - 10,000 | Sample size for simulation |
| censoring_rate | 0.0 - 0.9 | Proportion of censored observations |
| confidence_level | 0.8 - 0.99 | Confidence level for intervals |

#### 📈 When to Use Exponential vs Weibull

**Use Exponential when:**
- Constant failure rate is reasonable
- Memoryless property applies
- Simple baseline model needed
- Random failures (no wear-out or burn-in)

**Use Weibull when:**
- Hazard rate changes over time
- Burn-in or wear-out effects present
- More flexible modeling needed
- Shape of hazard function is important

#### 📚 References

- Ross, S.M. (2014). *Introduction to Probability Models*
- Lawless, J.F. (2003). *Statistical Models and Methods for Lifetime Data*
- Barlow, R.E. & Proschan, F. (1975). *Statistical Theory of Reliability*
- Nelson, W. (2004). *Applied Life Data Analysis*

</details>


### ▶️ `AntitheticVariables`

> Implement **antithetic variables variance reduction** for Monte Carlo simulations, achieving up to 50% variance reduction through negatively correlated sampling pairs.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements the **antithetic variables method**, a powerful variance reduction technique that generates pairs of negatively correlated random samples to reduce Monte Carlo estimation variance. For each uniform random variable U, it creates an antithetic pair (U, 1-U) and averages the function evaluations.

#### 🎯 Key Features

- **Variance Reduction**: Up to 50% variance reduction for monotonic functions
- **Unbiased Estimation**: Maintains same expected value as standard Monte Carlo
- **Automatic Comparison**: Built-in comparison with standard Monte Carlo
- **Correlation Analysis**: Measures and reports correlation between antithetic pairs
- **Domain Transformation**: Supports integration over arbitrary intervals
- **Convergence Tracking**: Real-time monitoring of estimation accuracy

#### 📐 Mathematical Background

**Standard Monte Carlo:**
```
θ̂ = (1/n) Σ f(Uᵢ)
Var[θ̂] = σ²/n
```

**Antithetic Variables:**
```
θ̂ₐ = (1/2n) Σ [f(Uᵢ) + f(1-Uᵢ)]
Var[θ̂ₐ] = (σ² + σ'² + 2Cov[f(U), f(1-U)])/(2n)
```

**Variance Reduction Ratio:**
```
Efficiency = Var[θ̂]/Var[θ̂ₐ] = 2/(1 + ρ)
```
where ρ is the correlation coefficient between f(U) and f(1-U).

**Optimal Case:** When ρ = -1 (perfect negative correlation), variance reduces by 50%.

#### ✅ When Antithetic Variables Work Best

- **Monotonic functions** (strictly increasing/decreasing)
- **Smooth functions** with consistent curvature
- **Financial option pricing** (European options, Asian options)
- **Reliability analysis** and survival functions
- **Integration problems** over bounded domains

#### ❌ When They May Not Help

- **Oscillatory functions** with multiple extrema
- **Discontinuous** or highly irregular functions
- **Symmetric problems** where f(u) ≈ f(1-u)
- **Functions with positive correlation** between pairs

#### 📈 Example – Simple Integration

```python
import numpy as np
from RSIM.variance_reduction import AntitheticVariables

# Integrate x² from 0 to 1 (true value = 1/3)
def quadratic(x):
    return x**2

av_sim = AntitheticVariables(
    target_function=quadratic,
    n_samples=100000,
    domain=(0, 1)
)

result = av_sim.run()
print(f"Estimate: {result.results['antithetic_estimate']:.6f}")
print(f"True value: {1/3:.6f}")
print(f"Variance reduction: {result.results['variance_reduction_ratio']:.2f}x")
print(f"Correlation: {result.results['correlation_coefficient']:.4f}")
```

#### 🎯 Example – Financial Option Pricing

```python
from scipy.stats import norm

def european_call_payoff(x):
    """Black-Scholes European call option"""
    # Transform uniform to normal
    z = norm.ppf(x)
    # Stock price at maturity (S₀=100, r=0.05, σ=0.2, T=1)
    S_T = 100 * np.exp(0.05 - 0.5*0.2**2 + 0.2*z)
    # Call payoff with strike K=105, discounted
    return np.maximum(S_T - 105, 0) * np.exp(-0.05)

option_sim = AntitheticVariables(
    target_function=european_call_payoff,
    n_samples=500000,
    random_seed=42
)

result = option_sim.run()
option_sim.visualize()

print(f"Option price: ${result.results['antithetic_estimate']:.4f}")
print(f"95% CI: [${result.results['antithetic_ci_lower']:.4f}, "
      f"${result.results['antithetic_ci_upper']:.4f}]")
print(f"Efficiency gain: {result.results['efficiency_gain_percent']:.1f}%")
```

#### 📊 Example – Custom Domain Integration

```python
# Integrate e^x from -1 to 1 (true value = e - 1/e)
def exponential(x):
    return np.exp(x)

exp_sim = AntitheticVariables(
    target_function=exponential,
    domain=(-1, 1),
    n_samples=200000
)

result = exp_sim.run()
true_value = np.exp(1) - np.exp(-1)
error = abs(result.results['antithetic_estimate'] - true_value)

print(f"Estimate: {result.results['antithetic_estimate']:.6f}")
print(f"True value: {true_value:.6f}")
print(f"Absolute error: {error:.6f}")
print(f"Relative error: {100*error/true_value:.4f}%")
```

#### 🔧 Advanced Configuration

```python
# Configure with all options
av_advanced = AntitheticVariables()
av_advanced.configure(
    target_function=your_function,
    n_samples=1000000,          # Must be even
    domain=(-5, 5),             # Integration bounds
    compare_standard=True,       # Compare with standard MC
    show_convergence=True        # Track convergence
)

result = av_advanced.run()

# Detailed analysis
av_advanced.visualize(
    show_correlation=True,       # Scatter plot of pairs
    show_convergence=True        # Convergence comparison
)
```

#### 📊 Visualization Features

- **Results Summary**: Estimates, confidence intervals, correlation
- **Convergence Plots**: Antithetic vs standard Monte Carlo
- **Correlation Analysis**: Scatter plot of f(U) vs f(1-U)
- **Variance Comparison**: Bar charts of variance reduction
- **Statistical Analysis**: Detailed performance metrics

#### 📈 Performance Guidelines

| Sample Size | Use Case                    | Expected Accuracy |
|-------------|-----------------------------|--------------------|
| 10,000      | Quick exploration           | ±0.01             |
| 100,000     | Standard applications       | ±0.003            |
| 1,000,000   | High-precision needs        | ±0.001            |
| 10,000,000  | Publication-grade results   | ±0.0003           |

#### 🎯 Efficiency Expectations

| Function Type | Typical Variance Reduction | Correlation Range |
|---------------|----------------------------|-------------------|
| Monotonic     | 1.5x - 2.0x               | -0.8 to -0.3      |
| Smooth        | 1.2x - 1.8x               | -0.6 to -0.2      |
| Financial     | 1.3x - 1.7x               | -0.5 to -0.3      |
| Oscillatory   | 0.9x - 1.1x               | -0.2 to +0.2      |

#### 📚 Applications

- **Financial Engineering**: Option pricing, risk assessment
- **Reliability Engineering**: System failure probabilities  
- **Physics Simulations**: Particle transport, diffusion processes
- **Queueing Theory**: Waiting times, service level analysis
- **Bayesian Statistics**: Posterior expectation estimation
- **Engineering Design**: Optimization under uncertainty

#### 📚 References

- Hammersley & Morton (1956). *A New Monte Carlo Technique*
- Ross (2012). *Simulation, 5th Edition*
- Glasserman (2003). *Monte Carlo Methods in Financial Engineering*
- Asmussen & Glynn (2007). *Stochastic Simulation*

</details>



### ▶️ `ControlVariates`

> Achieve **dramatic variance reduction** in Monte Carlo simulations using correlated auxiliary variables with known expectations as control variates.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements the **Control Variates variance reduction technique** for Monte Carlo estimation. It exploits correlation between your target estimator and auxiliary random variables (controls) with known expected values to create new estimators with significantly lower variance while maintaining unbiasedness.

#### 🎯 Key Features

- **Built-in Examples**: Pre-configured problems with optimal control variates
- **Custom Functions**: Support for user-defined target and control functions  
- **Multiple Controls**: Regression-based combination of several control variates
- **Real-time Tracking**: Convergence monitoring and variance reduction analysis
- **Statistical Analysis**: Bootstrap CIs, sensitivity analysis, efficiency metrics

#### 📐 Control Variate Types

- **Natural Controls**: Arise from problem structure (e.g., underlying asset for options)
- **Antithetic Controls**: Use negatively correlated variables
- **Multiple Controls**: Linear combinations of several correlated variables
- **Regression-based**: Optimal coefficients via least squares

#### 📚 Theoretical Background

**Core Principle:**  
For target variable X and control Y with known E[Y] = μ_Y, the control variate estimator is:

```
X_cv = X - c(Y - μ_Y)
```

**Optimal Coefficient:**
```
c* = Cov(X,Y) / Var(Y)
```

**Variance Reduction:**
```
Var(X_cv) = Var(X)(1 - ρ²)
```

where ρ is the correlation between X and Y.

**Efficiency Gains:**
- ρ = 0.5 → 25% variance reduction
- ρ = 0.8 → 64% variance reduction  
- ρ = 0.9 → 81% variance reduction
- ρ = 0.95 → 90% variance reduction

#### ✅ Properties

- **Unbiased**: E[X_cv] = E[X] regardless of coefficient choice
- **Consistent**: Converges to true value as sample size increases
- **Efficient**: Can achieve dramatic variance reduction with high correlation
- **Robust**: Works with any correlation structure

#### 🔧 Built-in Examples

| Example | Description | Control Variate | Expected Reduction |
|---------|-------------|-----------------|-------------------|
| `exponential_integral` | ∫₀¹ eˣ dx | Linear function | ~50% |
| `asian_option` | Asian call option | Geometric mean | ~70% |
| `normal_cdf` | Φ(x) estimation | Linear approximation | ~40% |
| `gamma_function` | Γ(x) calculation | Stirling approximation | ~60% |
| `portfolio_var` | Portfolio VaR | Individual VaRs | ~80% |

#### 📈 Example – Built-in exponential integral

```python
from RSIM.variance_reduction import ControlVariates

# Built-in example with automatic control variate
cv_sim = ControlVariates('exponential_integral', n_samples=50000)
result = cv_sim.run()

print(f"Standard MC: {result.results['standard_estimate']:.6f}")
print(f"Control Variates: {result.results['cv_estimate']:.6f}")
print(f"Variance Reduction: {result.results['variance_reduction']:.2f}x")
print(f"Correlation: {result.results['correlation']:.4f}")

# Visualize results
cv_sim.visualize()
```

#### 🎯 Example – Custom function with control

```python
import numpy as np

# Target: E[e^(X²)] for X~N(0,1)
def target_func(x):
    return np.exp(x**2)

# Control: E[X²] = 1 for X~N(0,1)  
def control_func(x):
    return x**2

cv_custom = ControlVariates(
    target_function=target_func,
    control_function=control_func,
    control_mean=1.0,
    distribution='normal',
    n_samples=100000
)

result = cv_custom.run()
print(f"Efficiency gain: {result.results['efficiency_gain']:.2f}x")
```

#### 📊 Example – Multiple control variates

```python
# Start with primary control
cv_multi = ControlVariates('portfolio_var', multiple_controls=True)

# Add additional controls
def secondary_control(x):
    return np.sum(x**2, axis=1)  # Sum of squares

cv_multi.add_control_variate(secondary_control, known_mean=3.0)

result = cv_multi.run()
print(f"Multiple CV reduction: {result.results['variance_reduction']:.2f}x")
```

#### 🔍 Example – Advanced analysis

```python
# Run simulation
cv_sim = ControlVariates('asian_option', n_samples=100000)
result = cv_sim.run()

# Sensitivity analysis
sensitivity = cv_sim.sensitivity_analysis()
print("Optimal coefficient:", sensitivity['optimal_coefficient'])

# Bootstrap confidence intervals  
bootstrap_ci = cv_sim.bootstrap_confidence_intervals(n_bootstrap=1000)
print("CV estimate CI:", bootstrap_ci['control_variates']['ci_lower'], 
      "to", bootstrap_ci['control_variates']['ci_upper'])

# Sample size recommendations
sample_analysis = cv_sim.estimate_required_samples(target_error=0.01)
print(f"Samples needed - Standard: {sample_analysis['standard_mc_samples']}")
print(f"Samples needed - CV: {sample_analysis['control_variates_samples']}")
```

#### 📉 Variance Reduction Guidelines

| Correlation | Reduction | Interpretation |
|-------------|-----------|----------------|
| \|ρ\| > 0.8 | >60% | Excellent - High efficiency gain |
| 0.5 < \|ρ\| < 0.8 | 25-60% | Good - Worthwhile improvement |
| 0.3 < \|ρ\| < 0.5 | 9-25% | Moderate - Some benefit |
| \|ρ\| < 0.3 | <9% | Minimal - Consider alternatives |

#### 🎛️ Configuration Options

```python
cv_sim = ControlVariates(
    target_function='exponential_integral',  # Built-in or custom function
    control_function='auto',                 # 'auto' or custom function
    control_mean=None,                       # Known E[Y] (auto for built-ins)
    n_samples=100000,                        # Monte Carlo sample size
    distribution='uniform',                  # Sampling distribution
    distribution_params={},                  # Distribution parameters
    multiple_controls=False,                 # Use multiple control variates
    show_convergence=True,                   # Track convergence
    random_seed=42                          # Reproducibility
)
```

#### 📊 Visualization Features

The `visualize()` method provides comprehensive analysis:

- **Results Summary**: Estimates, variance reduction, correlation
- **Scatter Plot**: Target vs control values with regression line
- **Convergence Plot**: Standard MC vs CV estimates over time
- **Variance Comparison**: Side-by-side variance reduction
- **Distribution Comparison**: Histograms of estimate distributions

#### ⚡ Performance Characteristics

- **Time Complexity**: O(n) - same as standard Monte Carlo
- **Space Complexity**: O(n) - stores samples and control values
- **Computational Overhead**: 10-20% vs standard MC
- **Memory Overhead**: 2-3x for storing control evaluations
- **Efficiency Gain**: Up to 1/(1-ρ²) effective sample increase

#### 🔧 Advanced Methods

```python
# Export results
filename = cv_sim.export_results(format='json')

# Parameter validation
errors = cv_sim.validate_parameters()

# Built-in examples info
examples = cv_sim.get_builtin_examples_info()

# Theoretical variance reduction
theoretical_reduction = cv_sim.calculate_theoretical_variance_reduction(correlation=0.8)
```

#### 💡 Best Practices

1. **Choose Correlated Controls**: Use domain knowledge to select highly correlated auxiliary variables
2. **Verify Control Means**: Ensure known expectations are accurate
3. **Multiple Controls**: Combine several moderately correlated controls for better reduction
4. **Sample Size**: Use sufficient samples to stabilize coefficient estimation
5. **Validation**: Always verify unbiasedness and variance reduction

#### 🚨 Common Pitfalls

- **Poor Control Choice**: Low correlation provides minimal benefit
- **Unknown Control Mean**: Estimation errors can introduce bias
- **Coefficient Instability**: Small samples may give unreliable coefficients
- **Nonlinear Relationships**: Linear controls may miss nonlinear correlations

#### 📚 Applications

- **Finance**: Option pricing, risk management, portfolio optimization
- **Engineering**: Reliability analysis, system performance evaluation  
- **Physics**: Particle simulations, quantum Monte Carlo
- **Operations Research**: Queueing systems, inventory management
- **Statistics**: Integration problems, expectation estimation

#### 📖 References

- Hammersley & Handscomb (1964). *Monte Carlo Methods*
- Glasserman (2003). *Monte Carlo Methods in Financial Engineering*
- Asmussen & Glynn (2007). *Stochastic Simulation*
- Owen (2013). *Monte Carlo theory, methods and examples*

</details>


### ▶️ `ImportanceSampling`

> Perform **advanced Monte Carlo variance reduction** using importance sampling to efficiently estimate integrals, rare event probabilities, and complex expectations with dramatic efficiency gains.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **importance sampling Monte Carlo simulation** to estimate integrals and expectations by sampling from an alternative proposal distribution rather than the original distribution. This technique can achieve **10x-1000x variance reduction** for problems involving rare events, tail probabilities, or regions of high importance that are poorly sampled by standard Monte Carlo.

#### 🎯 Built-in Problem Types

- **Exponential Tail Integration**: Estimate ∫ exp(-x²/2) dx for x > threshold (large threshold)
- **Rare Event Probability**: P(X > threshold) where X ~ N(0,1) and threshold is large  
- **Option Pricing**: European call option valuation using Black-Scholes model
- **Custom Integration**: User-defined integrand with custom proposal distributions

#### 📐 Mathematical Foundation

**Standard Monte Carlo:**
```
I = ∫ f(x) p(x) dx = E_p[f(X)] ≈ (1/n) Σ f(X_i), X_i ~ p(x)
```

**Importance Sampling:**
```
I = ∫ f(x) p(x)/q(x) q(x) dx = E_q[f(X) w(X)]
Estimator: Î = (1/n) Σ f(X_i) w(X_i), where w(X_i) = p(X_i)/q(X_i)
```

Where:
- `p(x)`: Original probability density function
- `q(x)`: Importance (proposal) density function  
- `w(x) = p(x)/q(x)`: Importance weight
- **Optimal proposal**: `q*(x) ∝ |f(x)| p(x)` minimizes variance

#### 🔬 Key Properties

- **Unbiased estimator**: E_q[Î] = I for any valid q(x)
- **Variance reduction**: Can be orders of magnitude better than standard MC
- **Effective sample size**: n_eff = (Σw_i)² / Σw_i²
- **Weight diagnostics**: CV(weights) should be < 5 for good efficiency
- **Requires**: q(x) > 0 wherever f(x)p(x) ≠ 0

#### 📊 Efficiency Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Variance Reduction Factor | >10x | 2-10x | <2x |
| Effective Sample Size Ratio | >50% | 10-50% | <10% |
| Weight Coefficient of Variation | <1.0 | 1.0-5.0 | >5.0 |

#### 🎯 Example – Exponential Tail Integration

```python
from RSIM.variance_reduction import ImportanceSampling

# Estimate ∫ exp(-x²/2) dx for x > 3.0 (very small probability)
is_sim = ImportanceSampling(
    problem_type='exponential_tail',
    n_samples=10000,
    problem_params={'threshold': 3.0}
)

result = is_sim.run()
print(f"Estimate: {result.results['importance_sampling_estimate']:.8f}")
print(f"Variance reduction: {result.results['variance_reduction_factor']:.1f}x")
print(f"Effective samples: {result.results['effective_sample_size']:.0f}")

# Visualize results with convergence comparison
is_sim.visualize()
```

#### 🎲 Example – Rare Event Probability

```python
# P(X > 4.0) where X ~ N(0,1) - extremely rare event
rare_sim = ImportanceSampling(
    problem_type='rare_event',
    n_samples=5000,
    problem_params={'threshold': 4.0}
)

result = rare_sim.run()
print(f"Rare event probability: {result.results['importance_sampling_estimate']:.2e}")
print(f"True value: {result.results['true_value']:.2e}")
print(f"Relative error: {result.results['is_relative_error']:.4f}%")
```

#### 💰 Example – Option Pricing

```python
# European call option with importance sampling for out-of-the-money options
option_sim = ImportanceSampling(
    problem_type='option_pricing',
    n_samples=20000,
    problem_params={
        'strike': 120.0,      # Strike price
        'spot': 100.0,        # Current stock price  
        'rate': 0.05,         # Risk-free rate
        'volatility': 0.2,    # Volatility
        'maturity': 1.0       # Time to maturity
    }
)

result = option_sim.run()
print(f"Option value: ${result.results['importance_sampling_estimate']:.4f}")
print(f"Black-Scholes: ${result.results['true_value']:.4f}")
print(f"Variance reduction: {result.results['variance_reduction_factor']:.1f}x")
```

#### 🔧 Example – Custom Integration Problem

```python
import numpy as np

# Define custom integration problem: ∫ exp(-x²) sin(x) p(x) dx
def my_integrand(x):
    return np.exp(-x**2) * np.sin(x)

def standard_normal_pdf(x):
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)

def laplace_pdf(x):  # Laplace proposal distribution
    return np.exp(-np.abs(x)) / 2

def laplace_sampler(n):
    return np.random.laplace(0, 1, n)

# Setup custom problem
custom_sim = ImportanceSampling(problem_type='custom')
custom_sim.set_custom_problem(
    integrand=my_integrand,
    original_pdf=standard_normal_pdf,
    importance_pdf=laplace_pdf,
    importance_sampler=laplace_sampler
)

result = custom_sim.run(n_samples=15000)
print(f"Custom integral estimate: {result.results['importance_sampling_estimate']:.6f}")
```

#### 📈 Convergence Analysis

```python
# Compare convergence with standard Monte Carlo
is_sim = ImportanceSampling(
    problem_type='exponential_tail',
    n_samples=50000,
    comparison_samples=50000,  # Same number for fair comparison
    show_convergence=True,
    problem_params={'threshold': 2.5}
)

result = is_sim.run()

# Automatic visualization shows:
# - Convergence comparison plot
# - Importance weight distribution  
# - Sample distribution analysis
# - Efficiency metrics summary
# - Effective sample size evolution
# - Weight coefficient of variation tracking
is_sim.visualize()
```

#### ⚠️ Common Pitfalls and Solutions

**1. Poor Proposal Choice:**
- **Symptom**: High weight variance, low effective sample size
- **Solution**: Choose q(x) closer to optimal |f(x)|p(x)

**2. Weight Degeneracy:**
- **Symptom**: Few samples dominate the estimate  
- **Solution**: Improve proposal or use adaptive importance sampling

**3. Infinite Variance:**
- **Symptom**: Weights have theoretically infinite variance
- **Solution**: Ensure q(x) has heavier tails than f(x)p(x)

#### 🎛️ Advanced Configuration

```python
# High-precision rare event simulation
precision_sim = ImportanceSampling(
    problem_type='rare_event',
    n_samples=100000,           # Large sample size
    comparison_samples=100000,   # Fair comparison
    problem_params={'threshold': 5.0},  # Very rare event
    show_convergence=True,
    random_seed=42              # Reproducible results
)

result = precision_sim.run()

# Check efficiency metrics
print(f"Effective sample size: {result.results['effective_sample_size']:.0f}")
print(f"ESS ratio: {result.results['effective_sample_size']/100000:.1%}")
print(f"Weight CV: {result.results['weight_coefficient_variation']:.3f}")
```

#### 📊 Applications

- **Finance**: Option pricing, VaR estimation, credit risk modeling
- **Reliability**: Failure probability estimation, system safety analysis  
- **Physics**: Particle transport, quantum Monte Carlo simulations
- **Engineering**: Structural reliability, rare failure analysis
- **Statistics**: Tail probability estimation, extreme value analysis
- **Machine Learning**: Variational inference, reinforcement learning

#### 📚 Theoretical References

- Rubinstein, R. Y. & Kroese, D. P. (2016). *Simulation and the Monte Carlo Method*
- Owen, A. B. (2013). *Monte Carlo theory, methods and examples*  
- Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods*
- Glynn, P. W. & Iglehart, D. L. (1989). *Importance sampling for stochastic simulations*
- Bucklew, J. A. (2004). *Introduction to Rare Event Simulation*

</details>


### ▶️ `StratifiedSampling`

> Achieve **dramatic variance reduction** in Monte Carlo integration through intelligent domain stratification and optimal sample allocation.

<details>
  <summary>Click to expand – Full explanation, theory, and examples</summary>

#### 📘 What it does

This class implements **stratified sampling** for Monte Carlo integration, dividing the integration domain into non-overlapping strata and sampling from each independently. This variance reduction technique can achieve **2-100x efficiency gains** over simple Monte Carlo, especially for smooth functions.

#### 🎯 Stratification Methods
- **Equal-width**: Divide domain into equal intervals
- **Equal-probability**: Each stratum has equal probability mass  
- **Adaptive**: Based on function variation and gradient analysis

#### 📊 Sample Allocation Strategies
- **Equal**: Same number of samples per stratum
- **Proportional**: Samples proportional to stratum width
- **Optimal (Neyman)**: Minimize variance using n_i ∝ w_i × σ_i

#### 🧮 Built-in Test Functions
- **Polynomial**: f(x) = x³ + 2x² - x + 1, ∫[0,1] = 7/4
- **Exponential**: f(x) = eˣ, ∫[0,1] = e - 1  
- **Trigonometric**: f(x) = sin(πx), ∫[0,1] = 2/π
- **Oscillatory**: f(x) = sin(10πx), ∫[0,1] = 0
- **Peak**: f(x) = exp(-100(x-0.5)²), sharp Gaussian peak
- **Custom**: User-defined function

#### 📚 Theoretical Background

**Stratified Estimator:**
For integral I = ∫[a,b] f(x)dx, divide [a,b] into k strata:

```
I_strat = Σ(i=1 to k) w_i × (1/n_i) × Σ(j=1 to n_i) f(X_ij)
```

Where:
- w_i = stratum width
- n_i = samples in stratum i  
- X_ij = j-th sample in stratum i

**Variance Reduction:**
```
Var[I_strat] = Σ(i=1 to k) (w_i²/n_i) × Var[f(X)|X ∈ stratum_i]
Var[I_MC] = Var[f(X)] / n_total
```

**Efficiency Gain:**
```
Efficiency = Var[I_MC] / Var[I_strat] ≥ 1
```

**Optimal Allocation (Neyman):**
```
n_i ∝ w_i × σ_i
```
Where σ_i is the standard deviation within stratum i.

#### ✅ Properties
- **Unbiased estimator**: E[I_strat] = I
- **Variance reduction**: Always ≤ simple MC variance
- **Convergence**: O(1/√n) with smaller constant
- **Robustness**: Works for any integrable function

#### 📈 Example – Basic Integration

```python
from RSIM.variance_reduction import StratifiedSampling

# Integrate polynomial function
strat = StratifiedSampling(
    n_samples=10000,
    n_strata=20,
    test_function='polynomial'
)

result = strat.run()
print(f"Estimate: {result.results['integral_estimate']:.6f}")
print(f"True value: 1.750000")
print(f"Efficiency gain: {result.results['efficiency_gain']:.2f}x")
```

#### 🎯 Example – Custom Function with Optimal Allocation

```python
import numpy as np

def gaussian_like(x):
    return np.exp(-x**2)

strat_custom = StratifiedSampling(
    n_samples=50000,
    n_strata=50,
    domain=(-3, 3),
    test_function='custom',
    custom_function=gaussian_like,
    allocation_method='optimal',
    stratification_method='adaptive'
)

result = strat_custom.run()
strat_custom.visualize(show_strata=True, show_samples=True)
```

#### 📊 Example – Comparative Analysis

```python
# Compare different allocation methods
methods = ['equal', 'proportional', 'optimal']
results = {}

for method in methods:
    strat = StratifiedSampling(
        n_samples=20000,
        n_strata=25,
        allocation_method=method,
        test_function='peak'  # Challenging function
    )
    result = strat.run()
    results[method] = {
        'estimate': result.results['integral_estimate'],
        'efficiency': result.results['efficiency_gain']
    }

for method, res in results.items():
    print(f"{method:12}: {res['estimate']:.6f} (gain: {res['efficiency']:.1f}x)")
```

#### 🔬 Example – Monte Carlo Comparison Study

```python
# Statistical comparison over multiple runs
strat = StratifiedSampling(n_samples=10000, n_strata=30)
comparison = strat.compare_with_simple_mc(n_runs=200)

print(f"Stratified mean: {comparison['stratified_mean']:.6f}")
print(f"Simple MC mean: {comparison['simple_mc_mean']:.6f}")
print(f"Variance reduction: {comparison['variance_reduction_factor']:.2f}x")
print(f"Efficiency gain: {comparison['avg_efficiency_gain']:.2f}x")
```

#### ⚡ Performance Guidelines

| Strata Count | Best For | Efficiency Gain |
|--------------|----------|-----------------|
| 5-10 | Simple functions | 2-5x |
| 10-30 | General purpose | 3-10x |
| 30-100 | Complex/oscillatory | 5-50x |
| 100+ | Highly variable | 10-100x |

#### 🎛️ Parameter Tuning

**Optimal Strata Count:**
- Rule of thumb: k ≈ √n for many functions
- More strata for highly variable functions
- Fewer strata for smooth functions

**Sample Allocation:**
- Use 'optimal' for maximum efficiency
- Use 'proportional' for balanced approach
- Use 'equal' for uniform exploration

**Stratification Method:**
- 'equal_width' for smooth functions
- 'adaptive' for functions with sharp features
- 'equal_probability' for uniform distributions

#### 📊 Visualization Features

```python
# Comprehensive visualization
strat.visualize(
    show_strata=True,      # Show stratum boundaries
    show_samples=True,     # Show individual sample points
)
```

Displays:
- Function plot with colored strata
- Sample allocation bar chart
- Variance contribution pie chart
- Convergence comparison with simple MC

#### 📈 Advanced Features

**Export Results:**
```python
strat.export_results('integration_results.json')
```

**Stratum Analysis:**
```python
stratum_info = strat.get_stratum_info()
for name, info in stratum_info.items():
    print(f"{name}: bounds={info['bounds']}, samples={info['n_samples']}")
```

**Parameter Validation:**
```python
errors = strat.validate_parameters()
if errors:
    print("Parameter issues:", errors)
```

#### 🔧 Applications

- **Financial Engineering**: Option pricing, risk assessment
- **Physics Simulations**: Cross-section integration, Monte Carlo transport
- **Bayesian Statistics**: Posterior integration, evidence computation
- **Engineering**: Reliability analysis, uncertainty quantification
- **Machine Learning**: Model evaluation, hyperparameter optimization

#### 📚 References

- Cochran, W. G. (1977). *Sampling Techniques, 3rd Edition*
- Owen, A. B. (2013). *Monte Carlo theory, methods and examples*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*
- Rubinstein, R. Y. & Kroese, D. P. (2016). *Simulation and the Monte Carlo Method*

</details>



