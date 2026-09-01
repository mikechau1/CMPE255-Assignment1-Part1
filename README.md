# CMPE255-Assignment1-Part 1

Medium article: https://medium.com/@m.chau2021/predicting-house-prices-with-machine-learning-what-i-learned-from-the-process-6cbc4ad3dde5?sharedUserId=m.chau2021

Video: 

## Project overview

This repository contains a complete, professional, end-to-end data science workflow for house-price prediction using the **CRISP-DM** methodology and scikit-learn. The project begins with raw housing transaction data and progresses through business understanding, exploratory analysis, cleaning, leakage control, preprocessing, feature engineering, outlier analysis, feature selection, clustering, regression benchmarking, final locked-holdout evaluation, and production deployment.

The project is designed to be both **educational** and **reproducible**. The primary entry point is a polished Google Colab notebook that reconstructs the full analysis in one place, while the chunk-level artifacts preserve detailed diagnostics, intermediate datasets, plots, reports, scripts, model comparisons, and deployment assets.

---

## Executive summary

### Final modeling recommendation

The final recommended model is a **Huber regression trained on `log1p(price)`** using the selected nine conceptual feature groups:

1. Location
2. Living area
3. Room layout
4. Property age
5. Lot and land intensity
6. View
7. Basement characteristics
8. Waterfront status
9. Condition

The target-derived `price_per_sqft` field is excluded from all predictive models because it directly leaks the target.

### Key final results

| Result | Value |
|---|---:|
| Valid labeled observations | 4,551 |
| Development observations | 3,588 |
| Locked future holdout observations | 963 |
| Final holdout MAE | **$112,606** |
| Final holdout Median Absolute Error | **$46,064** |
| Final holdout WAPE | **19.07%** |
| Final holdout RMSLE | **0.260** |
| Mainstream-market MAE (≤ $895K) | **$60,245** |
| Final recommended estimator | **Log-target Huber regression** |
| Best nonlinear challenger | HistGradientBoosting |
| Selected property clusters | 3 |

A single **$26.59M** transaction materially inflates RMSE and R² sensitivity. The official holdout retains the record; a separate sensitivity analysis documents its influence rather than removing it post hoc.

---

# Quick start

## Recommended: Google Colab

Use the clean notebook:

```text
House_Price_CRISP_DM_Professional_Colab.ipynb
```

1. Open Google Colab.
2. Upload the notebook.
3. Upload `House Price.csv` when prompted, or place it at `/content/House Price.csv`.
4. Choose **Runtime → Run all**.
5. The notebook creates figures, tables, diagnostics, predictions, and a serialized model.
6. The last notebook section packages generated outputs for download.

A verified notebook with embedded outputs is also provided:

```text
House_Price_CRISP_DM_Professional_Colab_Verified.ipynb
```

---

# Primary downloadable artifacts

These are the highest-level deliverables for most users.

| Artifact | Purpose |
|---|---|
| `House_Price_CRISP_DM_Professional_Colab.ipynb` | Clean, professional Colab notebook for end-to-end execution |
| `House_Price_CRISP_DM_Professional_Colab_Verified.ipynb` | Executed notebook with representative outputs embedded |
| `House_Price_CRISP_DM_Colab_Package.zip` | Turnkey package containing notebook(s), data, README, requirements, and example outputs |
| `house_price_crispdm_verified_outputs_results.zip` | Consolidated tables, figures, predictions, diagnostics, and fitted-model outputs from a verified run |
| `house_price_chunk12_package.zip` | Complete deployment package with serialized production model and governance assets |
| `House Price.csv` | Original source dataset |

---

# Master Colab output structure

The consolidated notebook writes artifacts under a structure similar to:

```text
house_price_crispdm_outputs/
├── figures/
├── models/
├── tables/
└── run_manifest.json
```

## Consolidated figures

| File | Description |
|---|---|
| `figures/01_target_raw_and_log.png` | Raw and log-transformed target distributions |
| `figures/02_univariate_feature_overview.png` | Univariate feature distribution overview |
| `figures/03_target_correlations.png` | Target association overview |
| `figures/04_log_price_vs_log_living.png` | Log-price versus log-living-area relationship |
| `figures/05_heteroscedasticity_comparison.png` | Raw-versus-log residual variance comparison |
| `figures/06_cluster_pca_projection.png` | PCA visualization of target-free property clusters |
| `figures/07_model_mae_ranking.png` | Development-stage model MAE ranking |
| `figures/08_holdout_model_mae.png` | Final locked-holdout model comparison |
| `figures/09_huber_predicted_vs_actual.png` | Final Huber predicted-versus-actual diagnostic |

## Consolidated tables

| File | Description |
|---|---|
| `tables/01_initial_data_audit.csv` | Initial schema, validity, and integrity audit |
| `tables/02_temporal_split_report.csv` | Development/holdout split definition |
| `tables/03_target_summary.csv` | Target summary statistics |
| `tables/04_univariate_numeric_summary.csv` | Numeric univariate profile |
| `tables/05_target_associations.csv` | Numerical target associations |
| `tables/06_size_decile_summary.csv` | Price by living-area decile |
| `tables/07_partial_correlations.csv` | Marginal and adjusted feature relationships |
| `tables/08_raw_residual_spread.csv` | Raw-scale heteroscedasticity summary |
| `tables/09_log_residual_spread.csv` | Log-scale residual spread summary |
| `tables/10_data_quality_report.csv` | Formal quality and cleaning report |
| `tables/11_canonical_development.csv` | Canonical development dataset |
| `tables/12_locked_holdout_features.csv` | Locked holdout predictor table |
| `tables/13_transformed_matrix_audit.csv` | Transformed feature-matrix verification |
| `tables/14_univariate_outlier_screens.csv` | Outlier-screen thresholds and counts |
| `tables/15_structural_anomaly_scores.csv` | Predictor-space anomaly scores |
| `tables/16_influence_diagnostics.csv` | Leverage and influence diagnostics |
| `tables/17_filter_feature_group_evidence.csv` | Filter-based feature-selection evidence |
| `tables/18_embedded_feature_group_stability.csv` | Embedded feature-selection stability |
| `tables/19_selected_feature_groups.csv` | Final selected conceptual feature groups |
| `tables/20_cluster_count_evaluation.csv` | Candidate K-means cluster evaluation |
| `tables/21_target_free_cluster_profiles.csv` | Target-free cluster profiles |
| `tables/22_posthoc_cluster_price_overlay.csv` | Price overlay after cluster construction |
| `tables/23_temporal_fold_model_metrics.csv` | Fold-level regression metrics |
| `tables/24_temporal_oof_predictions.csv` | Development out-of-fold predictions |
| `tables/25_aggregate_model_metrics.csv` | Aggregate benchmark results |
| `tables/26_price_segment_metrics.csv` | Mainstream/luxury performance metrics |
| `tables/27_frozen_model_decision.json` | Model decision frozen before holdout opening |
| `tables/28_holdout_predictions_before_target_open.csv` | Holdout predictions saved before target opening |
| `tables/29_final_holdout_metrics.csv` | Final locked-holdout model metrics |
| `tables/30_holdout_price_segment_metrics.csv` | Final price-segment diagnostics |
| `tables/31_extreme_record_sensitivity.csv` | Extreme-transaction sensitivity analysis |
| `tables/32_production_refit_summary.json` | Final full-data production-refit summary |
| `tables/33_example_canonical_inference_output.csv` | Example canonical-model output |
| `tables/34_example_raw_inference_input.csv` | Example raw production input |
| `tables/34_monitoring_policy.csv` | Monitoring rules and alert thresholds |
| `tables/35_example_raw_inference_output.csv` | Example production-service output |
| `tables/_LOCKED_future_holdout_labels.csv` | Holdout labels kept isolated until final evaluation |

## Consolidated model artifact

```text
models/house_price_huber_service.pkl
```

This is the serialized end-to-end historical Huber model generated by the notebook workflow.

---

# CRISP-DM artifact catalog by phase

## Phase 1 — Business Understanding

Business objectives, target definition, prediction boundary, baseline design, metrics, limited-compute constraints, and the temporal validation protocol are documented directly in the Colab notebook and the narrative study. No separate Chunk 1 artifact package is required.

---

## Phase 2 — Data Understanding

### Chunk 2 — Univariate EDA

**Package**

```text
house_price_chunk2_package.zip
```

**Primary artifacts**

```text
house_price_chunk2_eda.py
house_price_chunk2_numeric_summary.csv
house_price_chunk2/01_price_raw_full.png
house_price_chunk2/02_price_log1p.png
house_price_chunk2/03_sqft_living.png
house_price_chunk2/06_city_counts_top15.png
house_price_chunk2/08_condition_counts.png
```

**Purpose**

- Raw and log target distributions
- Numeric distribution summaries
- Skewness analysis
- Categorical frequency analysis
- Rare-level diagnostics
- Early transformation hypotheses

### Chunk 3 — Bivariate and multivariate EDA

**Package**

```text
house_price_chunk3_package.zip
```

**Primary artifacts**

```text
house_price_chunk3_final.py
house_price_chunk3_final/01_numeric_target_associations.csv
house_price_chunk3_final/03_marginal_and_partial_correlations.csv
house_price_chunk3_final/14_top_city_median_bootstrap_intervals.csv
house_price_chunk3_final/18_exploratory_size_model_diagnostics.csv
house_price_chunk3_final/02_log_price_vs_log_living_area.png
house_price_chunk3_final/03_price_by_living_area_decile.png
house_price_chunk3_final/04_city_median_price_bootstrap_ci.png
house_price_chunk3_final/09_marginal_and_partial_correlations.png
house_price_chunk3_final/10_raw_model_residual_spread.png
house_price_chunk3_final/11_log_model_residual_spread.png
```

**Purpose**

- Pearson and Spearman relationships
- Partial correlations
- Location-price analysis
- Living-area response shape
- Heteroscedasticity diagnosis
- Early multicollinearity assessment

---

# Phase 3 — Data Preparation

## Chunk 4 — Cleaning, validation, and canonical data

**Package**

```text
house_price_chunk4_final_package.zip
```

**Primary artifacts**

```text
house_price_chunk4/house_price_chunk4_canonical_export.py
house_price_chunk4/01_validation_rule_report.csv
house_price_chunk4/02_before_after_quality_report.csv
house_price_chunk4/04_invalid_renovation_patterns.csv
house_price_chunk4/18_model_development_canonical.csv
house_price_chunk4/19_locked_holdout_features_canonical.csv
house_price_chunk4/21_default_predictor_manifest.json
house_price_chunk4/22_address_overlap_summary.csv
house_price_chunk4/15_before_after_validity.png
house_price_chunk4/16_invalid_renovation_patterns.png
house_price_chunk4/17_record_disposition.png
```

**Purpose**

- Target quarantine
- Semantic missingness
- Renovation chronology treatment
- ZIP/location parsing
- Exact redundancy removal
- Leakage-safe canonical modeling schema

## Chunk 5 — Preprocessing and feature engineering

**Package**

```text
house_price_chunk5_final_package.zip
```

**Primary artifacts**

```text
house_price_chunk5_final/house_price_chunk5_pipeline.py
house_price_chunk5_final/house_price_chunk5_analysis.py
house_price_chunk5_final/02_engineered_feature_dictionary.csv
house_price_chunk5_final/05_full_matrix_audit.csv
house_price_chunk5_final/11_temporal_fold_matrix_audit.csv
house_price_chunk5_final/24_preprocessing_verification.json
house_price_chunk5_final/16_skewness_before_after.png
house_price_chunk5_final/19_location_category_pooling.png
house_price_chunk5_final/20_fold_feature_count_stability.png
house_price_chunk5_final/23_area_representation_conditioning.png
```

**Purpose**

- Fold-fitted imputation
- Robust scaling
- Spline transformations
- Rare-category handling
- City/ZIP encoding
- Ratio features
- Matrix integrity and leakage checks

## Chunk 6 — Outliers, anomalies, leverage, and influence

**Package**

```text
house_price_chunk6_package.zip
```

**Primary artifacts**

```text
house_price_chunk6/house_price_chunk6_outlier_analysis.py
house_price_chunk6/02_univariate_thresholds.csv
house_price_chunk6/05_structural_anomaly_scores_development.csv
house_price_chunk6/06_structural_anomaly_scores_holdout_features.csv
house_price_chunk6/10_record_level_influence.csv
house_price_chunk6/15_treatment_scenario_aggregate_metrics.csv
house_price_chunk6/19_oof_conditional_residuals_huber_log.csv
house_price_chunk6/22_outlier_review_queue.csv
house_price_chunk6/25_recommended_outlier_policy.csv
house_price_chunk6/26_verification_results.json
house_price_chunk6/03_predictor_anomaly_agreement.png
house_price_chunk6/08_log_influence_map.png
house_price_chunk6/09_oof_conditional_residuals.png
house_price_chunk6/10_treatment_scenario_mae.png
```

**Purpose**

- Tukey/MAD screening
- Isolation Forest
- Robust Mahalanobis distance
- Leverage and Cook's distance
- Out-of-fold conditional residuals
- Outlier-treatment scenario comparison
- Human-review prioritization

## Chunk 7 — Feature selection and stability

**Package**

```text
house_price_chunk7_package.zip
```

**Primary artifacts**

```text
house_price_chunk7_final/house_price_chunk7_feature_selection.py
house_price_chunk7_final/01_feature_group_dictionary.csv
house_price_chunk7_final/04_filter_group_aggregate.csv
house_price_chunk7_final/09_embedded_group_stability.csv
house_price_chunk7_final/15_permutation_importance_aggregate.csv
house_price_chunk7_final/17_ablation_importance_aggregate.csv
house_price_chunk7_final/20_recipe_comparison_aggregate.csv
house_price_chunk7_final/22_feature_selection_consensus.csv
house_price_chunk7_final/25_parsimony_confirmation_aggregate.csv
house_price_chunk7_final/27_selected_feature_groups.csv
house_price_chunk7_final/28_selected_transformed_feature_manifest.csv
house_price_chunk7_final/32_selected_feature_group_manifest.json
house_price_chunk7_final/01_filter_group_ranking.png
house_price_chunk7_final/02_elasticnet_group_selection_stability.png
house_price_chunk7_final/04_group_permutation_importance.png
house_price_chunk7_final/05_drop_group_ablation_importance.png
house_price_chunk7_final/07_preprocessing_recipe_comparison.png
house_price_chunk7_final/08_consensus_feature_evidence.png
house_price_chunk7_final/09_group_parsimony_curve.png
```

**Purpose**

- Filter selection
- Elastic Net embedded selection
- Group permutation importance
- Drop-group ablation
- Feature-selection stability
- Parsimonious nine-group feature set

---

# Phase 4 — Modeling

## Chunk 8 — Target-free clustering

**Package**

```text
house_price_chunk8_package.zip
```

**Primary artifacts**

```text
house_price_chunk8_analysis.py
house_price_chunk8/01_kmeans_k_evaluation.csv
house_price_chunk8/02_development_cluster_assignments.csv
house_price_chunk8/03_holdout_feature_cluster_assignments.csv
house_price_chunk8/04_target_free_cluster_profiles.csv
house_price_chunk8/05_standardized_cluster_profiles.csv
house_price_chunk8/06_top_cities_by_cluster.csv
house_price_chunk8/07_top_zips_by_cluster.csv
house_price_chunk8/08_price_overlay_by_cluster.csv
house_price_chunk8/09_selected_k_resampling_stability.csv
house_price_chunk8/10_cluster_mix_development_vs_holdout_features.csv
house_price_chunk8/11_pca_coordinates_for_visualization.csv
house_price_chunk8/12_pca_loadings.csv
house_price_chunk8/13_cluster_names.csv
house_price_chunk8/14_named_target_free_cluster_profiles.csv
house_price_chunk8/15_named_price_overlay_by_cluster.csv
house_price_chunk8/24_verification_results.json
house_price_chunk8/25_clustering_summary.json
house_price_chunk8/16_silhouette_by_k.png
house_price_chunk8/17_davies_bouldin_by_k.png
house_price_chunk8/18_stability_by_k.png
house_price_chunk8/19_min_cluster_share_by_k.png
house_price_chunk8/20_cluster_pca_projection.png
house_price_chunk8/21_target_free_profile_heatmap.png
house_price_chunk8/22_price_overlay_by_cluster.png
house_price_chunk8/23_cluster_mix_shift.png
```

**Selected structural segments**

1. Compact older non-basement homes
2. Older basement-oriented homes
3. Newer larger multi-story homes

**Purpose**

- Target-free structural segmentation
- K selection using multiple validity metrics
- Resampling stability
- PCA interpretation
- Post-hoc price overlay
- Future-population segment monitoring

## Chunk 9 — Linear-family benchmark

**Package**

```text
house_price_chunk9_package.zip
```

**Primary artifacts**

```text
house_price_chunk9_benchmark.py
house_price_chunk9/01_model_catalog.csv
house_price_chunk9/02_outer_fold_metrics.csv
house_price_chunk9/03_aggregate_model_metrics.csv
house_price_chunk9/04_oof_predictions.csv
house_price_chunk9/05_price_segment_metrics.csv
house_price_chunk9/06_fold_model_ranks.csv
house_price_chunk9/07_rank_stability_summary.csv
house_price_chunk9/08_key_model_comparisons.csv
house_price_chunk9/09_inner_tuning_detail.csv
house_price_chunk9/10_inner_tuning_summary.csv
house_price_chunk9/11_selected_hyperparameters.csv
house_price_chunk9/16_verification_results.json
house_price_chunk9/17_chunk9_summary.json
house_price_chunk9/12_model_mae_ranking.png
house_price_chunk9/13_mae_rmse_tradeoff.png
house_price_chunk9/14_top_model_fold_mae.png
house_price_chunk9/15_price_segment_mae.png
```

**Models benchmarked**

- Mean dummy
- Median dummy
- Size-only raw regression
- Size-only log regression
- OLS
- Ridge
- Lasso
- Elastic Net
- Huber regression

**Development-stage leader**

```text
Log-target Huber, selected 9 groups
Development temporal MAE ≈ $103,939
```

## Chunk 10 — Tree ensembles and boosting

**Package**

```text
house_price_chunk10_package.zip
```

**Primary artifacts**

```text
house_price_chunk10_benchmark.py
house_price_chunk10/01_tree_model_catalog.csv
house_price_chunk10/02_outer_fold_metrics.csv
house_price_chunk10/03_aggregate_tree_metrics.csv
house_price_chunk10/04_oof_tree_predictions.csv
house_price_chunk10/05_price_segment_metrics.csv
house_price_chunk10/06_cluster_segment_metrics.csv
house_price_chunk10/07_inner_tuning_detail.csv
house_price_chunk10/08_inner_tuning_summary.csv
house_price_chunk10/09_selected_hyperparameters.csv
house_price_chunk10/10_linear_nonlinear_frontier.csv
house_price_chunk10/11_fold_model_ranks.csv
house_price_chunk10/12_rank_stability_summary.csv
house_price_chunk10/13_best_model_group_permutation_detail.csv
house_price_chunk10/14_best_model_group_permutation_aggregate.csv
house_price_chunk10/15_best_family_full12_fold_metrics.csv
house_price_chunk10/16_best_family_full12_aggregate.csv
house_price_chunk10/17_best_model_price_segment_summary.csv
house_price_chunk10/18_best_model_cluster_summary.csv
house_price_chunk10/24_verification_results.json
house_price_chunk10/26_key_model_comparisons.csv
house_price_chunk10/27_huber_vs_best_nonlinear_price_segments.csv
house_price_chunk10/28_huber_vs_best_nonlinear_cluster_metrics.csv
house_price_chunk10/19_nonlinear_mae_ranking.png
house_price_chunk10/20_linear_nonlinear_tradeoff.png
house_price_chunk10/21_best_model_group_permutation.png
house_price_chunk10/22_best_model_price_segment_mae.png
house_price_chunk10/23_best_model_cluster_mae.png
```

**Models benchmarked**

- Random Forest
- Extra Trees
- GradientBoostingRegressor
- HistGradientBoostingRegressor

**Best nonlinear result**

```text
Log-target HistGradientBoosting
Development temporal MAE ≈ $115,473
```

The nonlinear models did not outperform the Huber benchmark.

---

# Phase 5 — Final Evaluation

## Chunk 11 — Frozen model decision and one-time holdout evaluation

**Package**

```text
house_price_chunk11_package.zip
```

**Primary artifacts**

```text
house_price_chunk11_final_evaluation.py
house_price_chunk11/01_development_oof_blend_grid_summary.csv
house_price_chunk11/02_crossfit_blend_summary.csv
house_price_chunk11/03_crossfit_blend_weights.csv
house_price_chunk11/04_development_calibration_summary.csv
house_price_chunk11/05_crossfit_calibration_factors.csv
house_price_chunk11/06_development_calibration_stats.json
house_price_chunk11/07_development_calibration_by_decile.csv
house_price_chunk11/08_frozen_model_decision_BEFORE_HOLDOUT.json
house_price_chunk11/09_locked_holdout_predictions_BEFORE_TARGET_OPEN.csv
house_price_chunk11/10_final_fit_hyperparameters_BEFORE_TARGET_OPEN.json
house_price_chunk11/11_final_holdout_labels_OPENED_ONCE.csv
house_price_chunk11/12_final_holdout_model_metrics.csv
house_price_chunk11/13_holdout_mae_date_cluster_bootstrap_ci.csv
house_price_chunk11/14_paired_mae_difference_vs_huber.csv
house_price_chunk11/15_holdout_calibration_stats.json
house_price_chunk11/16_holdout_calibration_by_decile.csv
house_price_chunk11/17_holdout_price_segment_metrics.csv
house_price_chunk11/18_holdout_cluster_metrics.csv
house_price_chunk11/19_holdout_novelty_segment_metrics_huber.csv
house_price_chunk11/20_holdout_daily_metrics_huber.csv
house_price_chunk11/21_holdout_residuals_huber.csv
house_price_chunk11/22_prespecified_replacement_rule_review.csv
house_price_chunk11/23_final_model_recommendation.json
house_price_chunk11/24_verification_results.json
house_price_chunk11/32_extreme_record_sensitivity.csv
house_price_chunk11/33_top_20_holdout_errors_huber.csv
house_price_chunk11/34_development_vs_holdout_primary_metrics.csv
house_price_chunk11/35_extreme_record_impact_summary.json
house_price_chunk11/25_holdout_model_mae.png
house_price_chunk11/26_huber_predicted_vs_actual_log.png
house_price_chunk11/27_huber_residuals_vs_prediction.png
house_price_chunk11/28_huber_calibration_by_decile.png
house_price_chunk11/29_huber_price_segment_mae.png
house_price_chunk11/30_huber_cluster_mae.png
house_price_chunk11/31_huber_daily_mae.png
```

**Governance significance**

The following artifacts prove that the holdout prediction was generated before the labels were opened:

```text
08_frozen_model_decision_BEFORE_HOLDOUT.json
09_locked_holdout_predictions_BEFORE_TARGET_OPEN.csv
11_final_holdout_labels_OPENED_ONCE.csv
```

This preserves the integrity of the final model-selection process.

---

# Phase 6 — Deployment

## Chunk 12 — Production model, contracts, monitoring, and retraining

**Package**

```text
house_price_chunk12_package.zip
```

**Primary artifacts**

```text
house_price_chunk12/house_price_huber_service.joblib
house_price_chunk12/house_price_deployment.py
house_price_chunk12/house_price_chunk5_pipeline.py
house_price_chunk12/example_load_and_predict.py
house_price_chunk12/requirements.txt
house_price_chunk12/01_production_selected_feature_manifest.csv
house_price_chunk12/02_input_output_contract.json
house_price_chunk12/03_monitoring_policy.csv
house_price_chunk12/04_production_reference_metrics.csv
house_price_chunk12/05_frozen_holdout_performance_record.csv
house_price_chunk12/06_model_card.md
house_price_chunk12/07_retraining_policy.md
house_price_chunk12/08_deployment_checklist.md
house_price_chunk12/09_example_inference_input.csv
house_price_chunk12/10_example_inference_output.csv
house_price_chunk12/11_verification_results.json
house_price_chunk12/12_production_refit_summary.json
```

**Deployment capabilities**

- Raw-input schema validation
- Target-leakage protection
- Deterministic cleaning
- End-to-end Huber prediction
- Structural K-means segmentation
- Isolation Forest novelty flag
- Robust-distance novelty flag
- Unseen city/ZIP indicators
- Luxury-property review flags
- Manual-review routing
- Monitoring thresholds
- Retraining and rollback policy

---

# Final model behavior

## Strengths

- Strong temporal MAE relative to naive and size-only baselines
- Robust to heavy-tailed price errors
- No negative predictions under log-target inverse transformation
- Interpretable additive structure
- Strong location and living-area signal
- Better performance than tested tree ensembles under the same temporal protocol
- Explicit outlier and novelty governance
- Reproducible sklearn architecture

## Primary weakness

The model systematically underpredicts very expensive properties, especially the extreme luxury tail. The available dataset contains too few such transactions and lacks several high-value determinants such as exact coordinates, school district, finish quality, architectural significance, and transaction-specific information.

Luxury and strongly novel properties should therefore be treated as **manual-review cases**, not routine automated valuations.

---

# Important historical limitation

The source transactions are from **2014**. The project is a complete historical modeling and productionization demonstration, but the fitted artifact should not be used as a direct 2026 market-value estimator without retraining on recent representative transactions.

A single appreciation multiplier is not sufficient because market appreciation and price relationships differ across geography, property type, size, luxury tier, and amenities.

---

# Recommended environment

The project is designed around Python and scikit-learn and is intended to run in Google Colab or a comparable Python environment.

Typical libraries include:

```text
pandas
numpy
matplotlib
scikit-learn
scipy
joblib
```

Use the packaged `requirements.txt` where available for reproducible installation.

---

# Reproducibility principles used throughout

1. **Temporal validation rather than shuffled validation** for model selection.
2. **Fold-local preprocessing** so validation rows never influence imputation, scaling, spline knots, or categorical vocabularies.
3. **No target leakage** from `price_per_sqft`.
4. **No target imputation** for zero-price records.
5. **No automatic deletion of legitimate statistical outliers**.
6. **Outlier and anomaly flags are separated from deterministic data errors**.
7. **Cluster formation excludes price**.
8. **Final model is frozen before opening the locked holdout target**.
9. **Post-holdout production refit does not alter the recorded unbiased holdout metrics**.
10. **Model complexity must demonstrate out-of-time value** rather than being preferred by default.

---

# Suggested artifact reading order

For a reviewer who does not want to inspect hundreds of individual files, use this order:

1. `House_Price_CRISP_DM_Professional_Colab_Verified.ipynb`
2. `house_price_chunk11/23_final_model_recommendation.json`
3. `house_price_chunk11/12_final_holdout_model_metrics.csv`
4. `house_price_chunk11/32_extreme_record_sensitivity.csv`
5. `house_price_chunk12/06_model_card.md`
6. `house_price_chunk12/03_monitoring_policy.csv`
7. `house_price_chunk12/07_retraining_policy.md`
8. `house_price_chunk12/house_price_huber_service.joblib`

For technical review, proceed backward through Chunks 10 → 2 as needed.

---

# Package map

```text
House Price.csv
│
├── House_Price_CRISP_DM_Professional_Colab.ipynb
├── House_Price_CRISP_DM_Professional_Colab_Verified.ipynb
├── House_Price_CRISP_DM_Colab_Package.zip
├── house_price_crispdm_verified_outputs_results.zip
│
├── house_price_chunk2_package.zip     # Univariate EDA
├── house_price_chunk3_package.zip     # Bivariate / multivariate EDA
├── house_price_chunk4_final_package.zip  # Cleaning / validation
├── house_price_chunk5_final_package.zip  # Preprocessing / feature engineering
├── house_price_chunk6_package.zip     # Outlier / anomaly / influence analysis
├── house_price_chunk7_package.zip     # Feature selection
├── house_price_chunk8_package.zip     # Target-free clustering
├── house_price_chunk9_package.zip     # Linear-family benchmark
├── house_price_chunk10_package.zip    # Tree / boosting benchmark
├── house_price_chunk11_package.zip    # Locked-holdout final evaluation
└── house_price_chunk12_package.zip    # Deployment and productionization
```

---

# Internal and temporary files

The working environment also contains build scripts, notebook execution logs, Python bytecode caches, failed/intermediate notebook executions, and patch utilities used during artifact construction and verification.

Examples include:

```text
__pycache__/
*.log
*_FAILED.ipynb
build_*.py
patch_*.py
execute_*.py
```

These are **not supported project deliverables** and are intentionally excluded from the curated artifact catalog above.

---

# Final recommendation

For this dataset and evaluation design, the most defensible model is:

```text
Log-target Huber regression
+ selected nine conceptual feature groups
+ city and ZIP location effects
+ leakage-safe preprocessing
+ manual-review flags for luxury and structural novelty
```

The project demonstrates that disciplined validation, preprocessing, target transformation, robust loss, and governance produced more value than simply choosing a more complex model family.

---

## Project status

**CRISP-DM lifecycle: COMPLETE**

```text
Business Understanding  ✓
Data Understanding      ✓
Data Preparation        ✓
Modeling                ✓
Evaluation              ✓
Deployment              ✓
```
