# Towards Equitable Medical AI: Testing the Generalizability of LLMs Across Diverse Clinical Training Backgrounds

**Research Project | Lahore University of Management Sciences (LUMS)**

---

## 📌 Overview

This research examines the **fairness and generalizability of Large Language Models (LLMs)** in clinical decision-making. We investigate whether diagnostic performance varies when models are prompted to assume **clinician personas from different training backgrounds and resource settings**, including **low- and middle-income countries (LMICs)** versus **high-resource environments**.

By systematically varying persona attributes while keeping clinical content constant, this study evaluates whether **identity-based prompts influence model reasoning, knowledge application, or diagnostic quality**.

---

## ❓ Research Questions

1. Does diagnostic performance differ across simulated clinician personas?
2. Which persona dimensions (e.g., resources, reasoning style) most strongly influence outcomes?
3. Are persona effects stable across multiple independent model runs?
4. Which components of clinical reasoning are most sensitive to persona variation?

---

## 🧪 Methodology

### Experimental Design

* **Models Evaluated:** GPT-OSS, Gemini 2.5 Flash, LLaMA-3.1, Gemma-3, MedGemma
* **Clinical Vignettes:** 13 physician-validated cases across multiple medical specialties
* **Persona Dimensions:**

  * Training background: US-trained, Pakistani-trained, Neutral
  * Diagnostic resources: Advanced vs. Limited
  * Reasoning style: Step-by-step vs. Unconstrained
  * Context awareness: Epidemiological and socioeconomic considerations
* **Prompting Setup:** Each model was prompted using structured persona descriptions with identical clinical vignettes.

### Evaluation Protocol

Responses were graded using a rubric designed by licensed physicians familiar with clinical practice in resource-limited settings, across five components (maximum 23 points):

1. Diagnosis list (0–3)
2. Supporting evidence (0–6)
3. Opposing evidence (0–6)
4. Final diagnosis (0–2)
5. Additional steps (0–6)

* Evaluation was performed using **LLaMA-3.3-70B as an automated judge**.
* Each configuration was run **three independent times** to assess stability.

### Statistical Analysis

* Linear Mixed-Effects Models (LMMs) to account for repeated measures and nested structure
* **Total observations:** 5,070 (13 vignettes × 26 personas × 5 models × 3 runs)
* False Discovery Rate (FDR) correction applied for multiple comparisons

---

## 📊 Key Findings

* **Training background neutrality:** No significant performance differences between US-trained, Pakistani-trained, or Neutral personas
* **Operational constraints matter:** Limited diagnostic resources significantly reduced diagnostic quality, measured against a physician rubric that already reflects practice in resource-limited settings
* **Reasoning style effects:** Step-by-step reasoning consistently improved diagnostic quality
* **Model-specific behavior:** Different architectures showed varying sensitivity and stability
* **The “Hollow Persona” Phenomenon:** Identity labels alone do not meaningfully alter diagnostic knowledge or reasoning strategies in current LLMs

---

## 📁 Repository Structure

```
.
├── detailed_analysis_graphs/     # Model-specific analysis plots and visualizations
├── model_outputs/                # Raw LLM responses under different clinician personas
├── evaluation_outputs/           # Rubric-based evaluation scores and summaries
├── final_report/                 # Final report and supplementary material
├── personnas.txt                 # List of clinician personas used for prompting
└── README.md                     # Project overview, methodology, and findings
```

---

## 📝 Notes on Reproducibility

* All model outputs and evaluations are included for transparency
* Large pretrained models were accessed via external APIs or checkpoints and are **not redistributed**
* Random seeds and multiple runs were used to ensure robustness

---

## 📬 Contact

For questions regarding methodology or results, please contact the project authors via **LUMS**.

---

