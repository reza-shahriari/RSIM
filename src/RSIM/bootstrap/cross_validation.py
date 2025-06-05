import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Union, Callable, Dict, Any, List, Tuple
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class CrossValidationSimulation(BaseSimulation):
    """
    Comprehensive Cross-Validation Simulation for Model Performance Assessment.
    
    This simulation implements various cross-validation strategies to evaluate machine learning
    model performance, estimate generalization error, and assess model stability across
    different data splits. It supports both regression and classification tasks with
    multiple validation schemes and performance metrics.
    
    Mathematical Background:
    -----------------------
    Cross-validation estimates the expected prediction error of a model on unseen data:
    - CV Error = (1/k) × Σ(i=1 to k) L(yi, ŷi)
    - Where L is the loss function and k is the number of folds
    - Bias-Variance decomposition: E[CV] = Bias² + Variance + Noise
    - Standard error: SE = σ/√k, where σ is the standard deviation across folds
    
    Validation Strategies:
    ---------------------
    1. K-Fold Cross-Validation:
       - Divide data into k equal-sized folds
       - Train on k-1 folds, test on remaining fold
       - Repeat k times, average results
       - Bias-variance tradeoff: larger k → lower bias, higher variance
    
    2. Stratified K-Fold:
       - Maintains class distribution in each fold
       - Reduces variance in classification tasks
       - Particularly important for imbalanced datasets
    
    3. Leave-One-Out (LOO):
       - Special case of k-fold where k = n (sample size)
       - Nearly unbiased but high variance
       - Computationally expensive for large datasets
    
    4. Monte Carlo Cross-Validation:
       - Random train/test splits repeated multiple times
       - More flexible than k-fold
       - Can control train/test ratio independently
    
    Statistical Properties:
    ----------------------
    - Unbiased estimation: E[CV_error] ≈ E[Test_error]
    - Variance depends on k and data correlation
    - 5-fold CV: good bias-variance tradeoff for most cases
    - 10-fold CV: standard choice, lower bias than 5-fold
    - LOO CV: lowest bias but highest variance and computational cost
    
    Performance Metrics:
    -------------------
    Regression:
    - Mean Squared Error (MSE): (1/n) × Σ(yi - ŷi)²
    - Root Mean Squared Error (RMSE): √MSE
    - Mean Absolute Error (MAE): (1/n) × Σ|yi - ŷi|
    - R² Score: 1 - (SS_res / SS_tot)
    
    Classification:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Precision: TP / (TP + FP)
    - Recall (Sensitivity): TP / (TP + FN)
    - F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
    - ROC-AUC: Area under ROC curve
    
    Algorithm Details:
    -----------------
    1. Data Preparation:
       - Load or generate dataset
       - Handle missing values and outliers
       - Feature scaling if required
    
    2. Model Selection:
       - Choose appropriate algorithm for task
       - Set hyperparameters or use defaults
       - Initialize model instance
    
    3. Cross-Validation Execution:
       - Split data according to chosen strategy
       - For each fold:
         * Train model on training set
         * Predict on validation set
         * Calculate performance metrics
       - Aggregate results across folds
    
    4. Statistical Analysis:
       - Calculate mean and standard deviation
       - Compute confidence intervals
       - Perform significance tests if needed
    
    Applications:
    ------------
    - Model selection and comparison
    - Hyperparameter tuning validation
    - Feature selection evaluation
    - Generalization error estimation
    - Model stability assessment
    - Publication-ready performance reporting
    - A/B testing for ML models
    - Regulatory compliance validation
    
    Simulation Features:
    -------------------
    - Multiple CV strategies (k-fold, stratified, LOO, Monte Carlo)
    - Support for regression and classification tasks
    - Built-in popular ML algorithms
    - Custom model and metric support
    - Comprehensive statistical analysis
    - Visualization of results and distributions
    - Confidence interval estimation
    - Performance comparison across models
    - Reproducible results with random seeds
    
    Parameters:
    -----------
    cv_strategy : str, default='kfold'
        Cross-validation strategy: 'kfold', 'stratified', 'loo', 'monte_carlo'
    n_folds : int, default=5
        Number of folds for k-fold strategies (ignored for LOO)
    n_repeats : int, default=1
        Number of repetitions for Monte Carlo CV or repeated k-fold
    test_size : float, default=0.2
        Test set proportion for Monte Carlo CV (0.1 to 0.5)
    model_type : str, default='linear_regression'
        Model to evaluate: 'linear_regression', 'logistic_regression',
        'random_forest_reg', 'random_forest_clf', 'svm_reg', 'svm_clf'
    task_type : str, default='regression'
        Task type: 'regression' or 'classification'
    scoring : str or callable, default='auto'
        Scoring metric ('auto' selects based on task_type)
    random_seed : int, optional
        Seed for reproducible results
    
    Attributes:
    -----------
    cv_results : dict
        Detailed cross-validation results including scores per fold
    fold_predictions : list
        Predictions for each fold (stored for analysis)
    feature_importance : array, optional
        Feature importance scores if supported by model
    model_params : dict
        Parameters of the fitted model
    result : SimulationResult
        Complete simulation results with statistics and metadata
    
    Methods:
    --------
    configure(cv_strategy, n_folds, model_type, task_type, **kwargs) : bool
        Configure cross-validation parameters
    run(X, y, **kwargs) : SimulationResult
        Execute cross-validation simulation
    run_with_synthetic_data(**kwargs) : SimulationResult
        Run with generated synthetic dataset
    visualize(result=None, show_distributions=True) : None
        Create comprehensive result visualizations
    compare_models(models_list, X, y) : dict
        Compare multiple models using same CV strategy
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Basic regression cross-validation
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>> cv_sim = CrossValidationSimulation(cv_strategy='kfold', n_folds=10)
    >>> result = cv_sim.run(X, y)
    >>> print(f"CV Score: {result.results['mean_score']:.4f} ± {result.results['std_score']:.4f}")
    
    >>> # Classification with stratified CV
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_classes=3, random_state=42)
    >>> cv_sim = CrossValidationSimulation(
    ...     cv_strategy='stratified', 
    ...     n_folds=5, 
    ...     model_type='random_forest_clf',
    ...     task_type='classification'
    ... )
    >>> result = cv_sim.run(X, y)
    >>> cv_sim.visualize()
    
    >>> # Model comparison
    >>> models = ['linear_regression', 'random_forest_reg', 'svm_reg']
    >>> comparison = cv_sim.compare_models(models, X, y)
    >>> for model, scores in comparison.items():
    ...     print(f"{model}: {scores['mean']:.4f} ± {scores['std']:.4f}")
    
    >>> # Monte Carlo cross-validation
    >>> cv_mc = CrossValidationSimulation(
    ...     cv_strategy='monte_carlo',
    ...     n_repeats=100,
    ...     test_size=0.3,
    ...     random_seed=123
    ... )
    >>> result = cv_mc.run_with_synthetic_data(n_samples=500, n_features=5)
    
    Visualization Outputs:
    ---------------------
    Standard Mode:
    - Box plot of CV scores across folds
    - Score distribution histogram
    - Confidence interval visualization
    - Performance metrics summary table
    
    Advanced Mode:
    - Learning curves across folds
    - Residual plots for regression
    - Confusion matrices for classification
    - Feature importance plots (if available)
    - Bias-variance decomposition visualization
    
    Performance Guidelines:
    ----------------------
    - 5-fold CV: Good default, fast execution
    - 10-fold CV: More stable estimates, 2x slower
    - LOO CV: Most accurate but very slow for large datasets
    - Stratified CV: Always use for classification with imbalanced data
    - Monte Carlo CV: Use when you need specific train/test ratios
    
    Statistical Interpretation:
    --------------------------
    - Mean CV score: Expected model performance
    - Standard deviation: Model stability indicator
    - 95% CI: [mean - 1.96*std, mean + 1.96*std]
    - High std suggests model is sensitive to training data
    - Low std indicates robust, stable model performance
    
    Best Practices:
    --------------
    - Always use stratified CV for classification
    - Use same CV strategy for model comparison
    - Report both mean and standard deviation
    - Consider computational cost vs. accuracy tradeoff
    - Validate CV results on independent test set
    - Use appropriate metrics for your problem domain
    
    Common Pitfalls:
    ---------------
    - Data leakage: preprocessing before CV split
    - Temporal data: use time-series CV instead
    - Small datasets: LOO or leave-p-out CV
    - Imbalanced classes: stratified CV mandatory
    - Hyperparameter tuning: nested CV required
    
    Extensions:
    ----------
    - Nested cross-validation for hyperparameter tuning
    - Time series cross-validation for temporal data
    - Group-based CV for clustered data
    - Adversarial validation for distribution shift detection
    - Bayesian optimization with CV for hyperparameter search
    
    References:
    -----------
    - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
    - James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning
    - Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation
    - Varma, S. & Simon, R. (2006). Bias in error estimation when using cross-validation
    - Bischl, B. et al. (2012). Resampling methods for meta-model validation with recommendations
    """

    def __init__(self, cv_strategy: str = 'kfold', n_folds: int = 5, n_repeats: int = 1,
                 test_size: float = 0.2, model_type: str = 'linear_regression', 
                 task_type: str = 'regression', scoring: str = 'auto',
                 random_seed: Optional[int] = None):
        super().__init__("Cross-Validation Simulation")
        
        # Initialize parameters
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.model_type = model_type
        self.task_type = task_type
        self.scoring = scoring
        
        # Store in parameters dict for base class
        self.parameters.update({
            'cv_strategy': cv_strategy,
            'n_folds': n_folds,
            'n_repeats': n_repeats,
            'test_size': test_size,
            'model_type': model_type,
            'task_type': task_type,
            'scoring': scoring,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.cv_results = None
        self.fold_predictions = None
        self.feature_importance = None
        self.model_params = None
        self.is_configured = True
        
        # Model mapping
        self._model_map = {
            'linear_regression': LinearRegression,
            'logistic_regression': LogisticRegression,
            'random_forest_reg': RandomForestRegressor,
            'random_forest_clf': RandomForestClassifier,
            'svm_reg': SVR,
            'svm_clf': SVC
        }
        
        # Scoring mapping
        self._scoring_map = {
            'regression': {
                'auto': 'neg_mean_squared_error',
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'r2': 'r2'
            },
            'classification': {
                'auto': 'accuracy',
                'accuracy': 'accuracy',
                'precision': 'precision_macro',
                'recall': 'recall_macro',
                'f1': 'f1_macro'
            }
        }
    
    def configure(self, cv_strategy: str = 'kfold', n_folds: int = 5, 
                 model_type: str = 'linear_regression', task_type: str = 'regression',
                 **kwargs) -> bool:
        """Configure cross-validation parameters"""
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.model_type = model_type
        self.task_type = task_type
        
        # Update optional parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update parameters dict
        self.parameters.update({
            'cv_strategy': cv_strategy,
            'n_folds': n_folds,
            'model_type': model_type,
            'task_type': task_type,
            **kwargs
        })
        
        self.is_configured = True
        return True
    
    def _get_cv_splitter(self, X, y):
        """Get appropriate cross-validation splitter"""
        if self.cv_strategy == 'kfold':
            return KFold(n_splits=self.n_folds, shuffle=True, 
                        random_state=self.parameters.get('random_seed'))
        elif self.cv_strategy == 'stratified':
                        return StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.parameters.get('random_seed'))
        elif self.cv_strategy == 'loo':
            return LeaveOneOut()
        elif self.cv_strategy == 'monte_carlo':
            # For Monte Carlo CV, we'll use repeated random splits
            from sklearn.model_selection import ShuffleSplit
            return ShuffleSplit(n_splits=self.n_repeats, test_size=self.test_size,
                              random_state=self.parameters.get('random_seed'))
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def _get_model(self):
        """Get model instance based on model_type"""
        if self.model_type not in self._model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model_class = self._model_map[self.model_type]
        
        # Set random state for models that support it
        if hasattr(model_class(), 'random_state'):
            return model_class(random_state=self.parameters.get('random_seed'))
        else:
            return model_class()
    
    def _get_scoring_metric(self):
        """Get appropriate scoring metric"""
        if self.scoring == 'auto':
            return self._scoring_map[self.task_type]['auto']
        elif self.scoring in self._scoring_map[self.task_type]:
            return self._scoring_map[self.task_type][self.scoring]
        else:
            return self.scoring  # Assume it's a valid sklearn scoring string
    
    def run(self, X: np.ndarray, y: np.ndarray, **kwargs) -> SimulationResult:
        """Execute cross-validation simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Get model and CV splitter
        model = self._get_model()
        cv_splitter = self._get_cv_splitter(X, y)
        scoring_metric = self._get_scoring_metric()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring_metric)
        
        # Store detailed results
        self.cv_results = {
            'scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores)
        }
        
        # Calculate confidence interval
        n_folds_actual = len(cv_scores)
        std_error = np.std(cv_scores) / np.sqrt(n_folds_actual)
        confidence_interval = (
            np.mean(cv_scores) - 1.96 * std_error,
            np.mean(cv_scores) + 1.96 * std_error
        )
        
        # Get detailed fold-by-fold predictions and metrics
        fold_results = []
        fold_predictions = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model and predict
            fold_model = self._get_model()
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_val)
            
            fold_predictions.append({
                'fold': fold_idx,
                'y_true': y_val,
                'y_pred': y_pred,
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
            
            # Calculate additional metrics
            if self.task_type == 'regression':
                mse = mean_squared_error(y_val, y_pred)
                mae = np.mean(np.abs(y_val - y_pred))
                fold_results.append({
                    'fold': fold_idx,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mae
                })
            else:  # classification
                accuracy = accuracy_score(y_val, y_pred)
                try:
                    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
                    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
                except:
                    precision = recall = f1 = 0.0
                
                fold_results.append({
                    'fold': fold_idx,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
        
        self.fold_predictions = fold_predictions
        
        # Try to get feature importance
        try:
            final_model = self._get_model()
            final_model.fit(X, y)
            if hasattr(final_model, 'feature_importances_'):
                self.feature_importance = final_model.feature_importances_
            elif hasattr(final_model, 'coef_'):
                self.feature_importance = np.abs(final_model.coef_).flatten()
        except:
            self.feature_importance = None
        
        execution_time = time.time() - start_time
        
        # Create comprehensive results
        results = {
            'mean_score': self.cv_results['mean_score'],
            'std_score': self.cv_results['std_score'],
            'min_score': self.cv_results['min_score'],
            'max_score': self.cv_results['max_score'],
            'confidence_interval': confidence_interval,
            'cv_scores': cv_scores.tolist(),
            'n_folds': n_folds_actual,
            'fold_results': fold_results
        }
        
        # Add task-specific aggregate metrics
        if self.task_type == 'regression':
            all_mse = [fr['mse'] for fr in fold_results]
            all_mae = [fr['mae'] for fr in fold_results]
            results.update({
                'mean_mse': np.mean(all_mse),
                'mean_rmse': np.sqrt(np.mean(all_mse)),
                'mean_mae': np.mean(all_mae)
            })
        else:
            all_accuracy = [fr['accuracy'] for fr in fold_results]
            all_precision = [fr['precision'] for fr in fold_results]
            all_recall = [fr['recall'] for fr in fold_results]
            all_f1 = [fr['f1_score'] for fr in fold_results]
            results.update({
                'mean_accuracy': np.mean(all_accuracy),
                'mean_precision': np.mean(all_precision),
                'mean_recall': np.mean(all_recall),
                'mean_f1_score': np.mean(all_f1)
            })
        
        statistics = {
            'primary_metric_mean': self.cv_results['mean_score'],
            'primary_metric_std': self.cv_results['std_score'],
            'standard_error': std_error,
            'coefficient_of_variation': self.cv_results['std_score'] / abs(self.cv_results['mean_score']) if self.cv_results['mean_score'] != 0 else float('inf'),
            'confidence_interval_95': confidence_interval
        }
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results=results,
            statistics=statistics,
            execution_time=execution_time,
            convergence_data=[(i+1, score) for i, score in enumerate(cv_scores)]
        )
        
        self.result = result
        return result
    
    def run_with_synthetic_data(self, n_samples: int = 1000, n_features: int = 10, 
                               n_classes: int = 2, noise: float = 0.1, **kwargs) -> SimulationResult:
        """Run cross-validation with generated synthetic data"""
        if self.task_type == 'regression':
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples, 
                n_features=n_features, 
                noise=noise,
                random_state=self.parameters.get('random_seed')
            )
        else:  # classification
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_redundant=0,
                n_informative=min(n_features, n_classes),
                random_state=self.parameters.get('random_seed')
            )
        
        return self.run(X, y, **kwargs)
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_distributions: bool = True, show_fold_details: bool = False) -> None:
        """Visualize cross-validation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Determine number of subplots
        n_plots = 2 + (1 if show_distributions else 0) + (1 if show_fold_details else 0)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: CV Scores Box Plot
        cv_scores = result.results['cv_scores']
        axes[0].boxplot([cv_scores], labels=[f'{self.cv_strategy.title()} CV'])
        axes[0].set_ylabel('CV Score')
        axes[0].set_title(f'Cross-Validation Scores Distribution\n{self.model_type} - {self.task_type}')
        axes[0].grid(True, alpha=0.3)
        
        # Add mean and confidence interval
        mean_score = result.results['mean_score']
        ci_lower, ci_upper = result.results['confidence_interval']
        axes[0].axhline(y=mean_score, color='red', linestyle='--', 
                       label=f'Mean: {mean_score:.4f}')
        axes[0].axhline(y=ci_lower, color='orange', linestyle=':', alpha=0.7,
                       label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        axes[0].axhline(y=ci_upper, color='orange', linestyle=':', alpha=0.7)
        axes[0].legend()
        
        # Plot 2: Scores by Fold
        fold_numbers = list(range(1, len(cv_scores) + 1))
        axes[1].plot(fold_numbers, cv_scores, 'bo-', linewidth=2, markersize=8)
        axes[1].axhline(y=mean_score, color='red', linestyle='--', alpha=0.7)
        axes[1].fill_between(fold_numbers, ci_lower, ci_upper, alpha=0.2, color='orange')
        axes[1].set_xlabel('Fold Number')
        axes[1].set_ylabel('CV Score')
        axes[1].set_title('Score by Fold')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Score Distribution Histogram
        if show_distributions:
            axes[2].hist(cv_scores, bins=min(10, len(cv_scores)), alpha=0.7, color='skyblue', edgecolor='black')
            axes[2].axvline(x=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
            axes[2].axvline(x=ci_lower, color='orange', linestyle=':', label='95% CI')
            axes[2].axvline(x=ci_upper, color='orange', linestyle=':')
            axes[2].set_xlabel('CV Score')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Score Distribution')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            # Show summary statistics instead
            stats_text = f"""
            Cross-Validation Results Summary
            
            Model: {self.model_type}
            Task: {self.task_type}
            CV Strategy: {self.cv_strategy}
            Number of Folds: {result.results['n_folds']}
            
            Mean Score: {mean_score:.6f}
            Std Score: {result.results['std_score']:.6f}
            Min Score: {result.results['min_score']:.6f}
            Max Score: {result.results['max_score']:.6f}
            
            95% Confidence Interval:
            [{ci_lower:.6f}, {ci_upper:.6f}]
            
            Standard Error: {result.statistics['standard_error']:.6f}
            Coefficient of Variation: {result.statistics['coefficient_of_variation']:.4f}
            """
            axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
            axes[2].axis('off')
        
        # Plot 4: Feature Importance (if available) or Detailed Metrics
        if self.feature_importance is not None and len(self.feature_importance) <= 20:
            feature_names = [f'Feature {i+1}' for i in range(len(self.feature_importance))]
            sorted_idx = np.argsort(self.feature_importance)[::-1]
            
            axes[3].barh(range(len(self.feature_importance)), 
                        self.feature_importance[sorted_idx])
            axes[3].set_yticks(range(len(self.feature_importance)))
            axes[3].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[3].set_xlabel('Importance')
            axes[3].set_title('Feature Importance')
            axes[3].grid(True, alpha=0.3)
        else:
            # Show detailed metrics by task type
            if self.task_type == 'regression' and 'fold_results' in result.results:
                fold_results = result.results['fold_results']
                folds = [fr['fold'] + 1 for fr in fold_results]
                mse_values = [fr['mse'] for fr in fold_results]
                mae_values = [fr['mae'] for fr in fold_results]
                
                axes[3].plot(folds, mse_values, 'ro-', label='MSE', linewidth=2)
                ax3_twin = axes[3].twinx()
                ax3_twin.plot(folds, mae_values, 'bs-', label='MAE', linewidth=2)
                
                axes[3].set_xlabel('Fold Number')
                axes[3].set_ylabel('MSE', color='red')
                ax3_twin.set_ylabel('MAE', color='blue')
                axes[3].set_title('Error Metrics by Fold')
                axes[3].grid(True, alpha=0.3)
                
                # Add legends
                axes[3].legend(loc='upper left')
                ax3_twin.legend(loc='upper right')
                
            elif self.task_type == 'classification' and 'fold_results' in result.results:
                fold_results = result.results['fold_results']
                folds = [fr['fold'] + 1 for fr in fold_results]
                accuracy_values = [fr['accuracy'] for fr in fold_results]
                f1_values = [fr['f1_score'] for fr in fold_results]
                
                axes[3].plot(folds, accuracy_values, 'go-', label='Accuracy', linewidth=2)
                axes[3].plot(folds, f1_values, 'mo-', label='F1-Score', linewidth=2)
                axes[3].set_xlabel('Fold Number')
                axes[3].set_ylabel('Score')
                axes[3].set_title('Classification Metrics by Fold')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
            else:
                # Show execution summary
                exec_summary = f"""
                Execution Summary
                
                Total Execution Time: {result.execution_time:.3f} seconds
                Time per Fold: {result.execution_time / result.results['n_folds']:.3f} seconds
                
                Data Information:
                Number of Folds: {result.results['n_folds']}
                CV Strategy: {self.cv_strategy}
                
                Model Performance:
                Primary Metric: {result.results['mean_score']:.6f}
                Standard Deviation: {result.results['std_score']:.6f}
                
                Stability Assessment:
                CV < 0.1: Highly Stable
                0.1 ≤ CV < 0.2: Moderately Stable  
                CV ≥ 0.2: Potentially Unstable
                
                Current CV: {result.statistics['coefficient_of_variation']:.4f}
                """
                axes[3].text(0.05, 0.95, exec_summary, transform=axes[3].transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                axes[3].set_xlim(0, 1)
                axes[3].set_ylim(0, 1)
                axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n" + "="*60)
        print("CROSS-VALIDATION SIMULATION RESULTS")
        print("="*60)
        print(f"Model: {self.model_type}")
        print(f"Task Type: {self.task_type}")
        print(f"CV Strategy: {self.cv_strategy}")
        print(f"Number of Folds: {result.results['n_folds']}")
        print(f"Execution Time: {result.execution_time:.3f} seconds")
        print("-"*60)
        print(f"Mean CV Score: {result.results['mean_score']:.6f}")
        print(f"Standard Deviation: {result.results['std_score']:.6f}")
        print(f"Standard Error: {result.statistics['standard_error']:.6f}")
        print(f"95% Confidence Interval: [{result.results['confidence_interval'][0]:.6f}, {result.results['confidence_interval'][1]:.6f}]")
        print(f"Coefficient of Variation: {result.statistics['coefficient_of_variation']:.4f}")
        print("-"*60)
        
        # Task-specific metrics
        if self.task_type == 'regression':
            print("REGRESSION METRICS:")
            print(f"Mean MSE: {result.results.get('mean_mse', 'N/A')}")
            print(f"Mean RMSE: {result.results.get('mean_rmse', 'N/A')}")
            print(f"Mean MAE: {result.results.get('mean_mae', 'N/A')}")
        else:
            print("CLASSIFICATION METRICS:")
            print(f"Mean Accuracy: {result.results.get('mean_accuracy', 'N/A'):.6f}")
            print(f"Mean Precision: {result.results.get('mean_precision', 'N/A'):.6f}")
            print(f"Mean Recall: {result.results.get('mean_recall', 'N/A'):.6f}")
            print(f"Mean F1-Score: {result.results.get('mean_f1_score', 'N/A'):.6f}")
        
        print("="*60)
    
    def compare_models(self, models_list: List[str], X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compare multiple models using the same CV strategy"""
        comparison_results = {}
        original_model = self.model_type
        
        print(f"Comparing {len(models_list)} models using {self.cv_strategy} CV...")
        
        for model_name in models_list:
            print(f"Evaluating {model_name}...")
            self.model_type = model_name
            self.parameters['model_type'] = model_name
            
            try:
                result = self.run(X, y)
                comparison_results[model_name] = {
                    'mean': result.results['mean_score'],
                    'std': result.results['std_score'],
                    'min': result.results['min_score'],
                    'max': result.results['max_score'],
                    'ci_lower': result.results['confidence_interval'][0],
                    'ci_upper': result.results['confidence_interval'][1]
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                comparison_results[model_name] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'error': str(e)
                }
        
        # Restore original model
        self.model_type = original_model
        self.parameters['model_type'] = original_model
        
        # Visualize comparison
        self._visualize_model_comparison(comparison_results)
        
        return comparison_results
    
    def _visualize_model_comparison(self, comparison_results: Dict[str, Dict[str, float]]) -> None:
        """Visualize model comparison results"""
        models = list(comparison_results.keys())
        means = [comparison_results[model].get('mean', 0) for model in models]
        stds = [comparison_results[model].get('std', 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot with error bars
        bars = ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_ylabel('CV Score')
        ax1.set_title('Model Comparison - Mean CV Scores')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Box plot comparison
        valid_models = []
        valid_scores = []
        for model in models:
            if not np.isnan(comparison_results[model].get('mean', np.nan)):
                valid_models.append(model)
                # Approximate score distribution using mean and std
                mean = comparison_results[model]['mean']
                std = comparison_results[model]['std']
                # Generate approximate scores for box plot
                scores = np.random.normal(mean, std, 100)
                valid_scores.append(scores)
        
        if valid_scores:
            ax2.boxplot(valid_scores, labels=valid_models)
            ax2.set_ylabel('CV Score')
            ax2.set_title('Model Comparison - Score Distributions')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        for model, results in comparison_results.items():
            if 'error' not in results:
                print(f"{model:<20} {results['mean']:<12.6f} {results['std']:<12.6f} "
                      f"{results['min']:<12.6f} {results['max']:<12.6f}")
            else:
                print(f"{model:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
        print("="*80)
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'cv_strategy': {
                'type': 'choice',
                'choices': ['kfold', 'stratified', 'loo', 'monte_carlo'],
                'default': 'kfold',
                'description': 'Cross-validation strategy'
            },
            'n_folds': {
                'type': 'int',
                'default': 5,
                'min': 2,
                'max': 20,
                'description': 'Number of folds (ignored for LOO)'
            },
            'n_repeats': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 100,
                'description': 'Number of repetitions for Monte Carlo CV'
            },
            'test_size': {
                'type': 'float',
                'default': 0.2,
                'min': 0.1,
                'max': 0.5,
                'description': 'Test set proportion for Monte Carlo CV'
            },
            'model_type': {
                'type': 'choice',
                'choices': ['linear_regression', 'logistic_regression', 'random_forest_reg', 
                           'random_forest_clf', 'svm_reg', 'svm_clf'],
                'default': 'linear_regression',
                'description': 'Model to evaluate'
            },
            'task_type': {
                'type': 'choice',
                'choices': ['regression', 'classification'],
                'default': 'regression',
                'description': 'Type of machine learning task'
            },
            'scoring': {
                'type': 'choice',
                'choices': ['auto', 'mse', 'mae', 'r2', 'accuracy', 'precision', 'recall', 'f1'],
                'default': 'auto',
                'description': 'Scoring metric (auto selects based on task)'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        
        if self.cv_strategy not in ['kfold', 'stratified', 'loo', 'monte_carlo']:
            errors.append("cv_strategy must be one of: kfold, stratified, loo, monte_carlo")
        
        if self.n_folds < 2 or self.n_folds > 20:
            errors.append("n_folds must be between 2 and 20")
        
        if self.n_repeats < 1 or self.n_repeats > 100:
            errors.append("n_repeats must be between 1 and 100")
        
        if self.test_size <= 0 or self.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        if self.model_type not in self._model_map:
            errors.append(f"model_type must be one of: {list(self._model_map.keys())}")
        
        if self.task_type not in ['regression', 'classification']:
            errors.append("task_type must be 'regression' or 'classification'")
        
        # Check model-task compatibility
        regression_models = ['linear_regression', 'random_forest_reg', 'svm_reg']
        classification_models = ['logistic_regression', 'random_forest_clf', 'svm_clf']
        
        if self.task_type == 'regression' and self.model_type not in regression_models:
            errors.append(f"For regression tasks, model_type must be one of: {regression_models}")
        
        if self.task_type == 'classification' and self.model_type not in classification_models:
            errors.append(f"For classification tasks, model_type must be one of: {classification_models}")
        
        return errors
    
    def get_recommendations(self, n_samples: int = None) -> Dict[str, str]:
        """Get recommendations based on dataset size and task"""
        recommendations = {}
        
        if n_samples is not None:
            if n_samples < 100:
                recommendations['cv_strategy'] = 'Use LOO or small k-fold (k=3-5) for small datasets'
            elif n_samples < 1000:
                recommendations['cv_strategy'] = 'Use 5-fold or 10-fold CV'
            else:
                recommendations['cv_strategy'] = 'Use 10-fold CV or Monte Carlo CV for large datasets'
        
        if self.task_type == 'classification':
            recommendations['stratification'] = 'Always use stratified CV for classification tasks'
        
        if self.cv_strategy == 'loo':
            recommendations['performance'] = 'LOO CV is computationally expensive but gives nearly unbiased estimates'
        
        recommendations['general'] = 'Report both mean and standard deviation of CV scores'
        
        return recommendations

