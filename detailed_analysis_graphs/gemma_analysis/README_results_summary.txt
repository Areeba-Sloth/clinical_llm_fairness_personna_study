Outputs saved in this folder:
- data_with_dimensions.csv
- Q1_descriptives_by_training.csv, Q1_mixedlm_coefficients.csv, Q1_total_by_training_bar.png
- Q2_mixedlm_coefficients.csv, Q2_dimension_effects_forest.png
- Q3_mixedlm_training_x_run.csv, Q3_run_stability_lines.png, Q3_run_correlations.csv
- Q4_mixedlm_coefficients_all_qs.csv, Q4_training_Pakistan_vs_US_pvalues_FDR.csv
- Q4_qk_by_training_bar.png (for k=1..5), Q4_rubric_sensitivity_heatmap.png

Key models:
Q1: total_score ~ training + (random: persona_id) + vc(clinical_id, run)
Q2: total_score ~ training + resources + reasoning + epi + socioeconomic + guidelines + interactions
Q3: total_score ~ training * run + random persona + vc(clinical_id, run)
Q4: qk ~ training + dimensions + random persona + vc(clinical_id, run)
