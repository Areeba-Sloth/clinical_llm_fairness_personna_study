Towards Equitable Medical AI: Testing the Generalizability of LLMs Across Diverse Clinical Training Backgrounds

Research Project | Lahore University of Management Sciences (LUMS)

📌 Overview

This research examines the fairness and generalizability of Large Language Models (LLMs) in clinical decision-making contexts. Specifically, we investigate whether diagnostic performance varies when models are prompted to assume clinician personas from different training backgrounds and resource settings, including low- and middle-income countries (LMICs) versus high-resource environments.

By systematically varying persona attributes while holding clinical content constant, this study evaluates whether identity-based prompts meaningfully influence model reasoning, knowledge application, or diagnostic quality.

🎯 Research Questions

Does diagnostic performance differ across simulated clinician personas?

Which persona dimensions (e.g., resources, reasoning style) most strongly influence outcomes?

Are persona effects stable across multiple independent model runs?

Which components of clinical reasoning are most sensitive to persona variation?

🧪 Methodology
Experimental Design

Models Evaluated:
GPT-OSS, Gemini 2.5 Flash, LLaMA-3.1, Gemma-3, MedGemma

Clinical Vignettes:
13 physician-validated cases spanning multiple medical specialties

Persona Dimensions:

Training background: US-trained, Pakistani-trained, Neutral

Diagnostic resources: Advanced vs. Limited

Reasoning style: Step-by-step vs. Unconstrained

Context awareness: Epidemiological and socioeconomic considerations

Prompting Setup:
Each model was prompted using structured persona descriptions while keeping clinical vignettes identical across conditions.

Evaluation Protocol

Responses were graded using a physician-designed rubric with five components:

Diagnostic accuracy

Clinical reasoning

Differential diagnosis quality

Contextual awareness

Management recommendations

Evaluation was performed using LLaMA-3.3-70B as an automated judge.

Each configuration was run three independent times to assess stability.

Statistical Analysis

Linear Mixed-Effects Models (LMMs) to account for repeated measures and nested structure

Total observations: 5,070

13 vignettes × 26 personas × 5 models × 3 runs

False Discovery Rate (FDR) correction applied for multiple comparisons

📊 Key Findings
Main Results

Training background neutrality:
No statistically significant performance differences were observed between US-trained, Pakistani-trained, or Neutral personas.

Operational constraints matter:
Limited diagnostic resources significantly reduced performance scores.

Reasoning style effects:
Step-by-step reasoning prompts consistently improved diagnostic quality.

Model-specific behavior:
Different architectures exhibited varying levels of sensitivity and stability.

The “Hollow Persona” Phenomenon

Identity-based persona labels alone do not meaningfully alter diagnostic knowledge or reasoning strategies in current LLMs. Despite surface-level persona variation, models appear to draw from a shared underlying knowledge distribution.

📁 Repository Structure
.
├── detailed_analysis_graphs/
│   ├── Model-specific analysis plots
│   ├── Cross-model comparison visualizations
│   └── Statistical figures used in analysis and reporting
│
├── model_outputs/
│   ├── Raw LLM responses to clinical vignettes
│   ├── Outputs generated under different clinician personas
│   └── Multiple independent runs per model
│
├── evaluation_outputs/
│   ├── Rubric-based evaluation scores
│   ├── Component-wise grading results
│   └── Aggregated evaluation summaries
│
├── final_report/
│   ├── Final written report / paper
│   └── Supplementary material (if applicable)
│
├── personnas.txt
│   └── Complete list of clinician personas used for prompting
│
└── README.md
    └── Project overview, methodology, and findings

📄 Notes on Reproducibility

All model outputs and evaluations are included to ensure transparency.

Large pretrained models were accessed via external APIs or checkpoints and are not redistributed here.

Random seeds and multiple runs were used where applicable to assess robustness.

📬 Contact

For questions regarding the methodology or results, please contact the project authors via LUMS.
