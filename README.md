# Machine Learning for RNA Secondary Structure Stability Prediction: Applications in Regulatory Genomics

This project investigates whether RNA secondary structure features alone can predict structural stability, aiming to assess the potential of feature-based machine learning models as a rapid screening tool for regulatory RNA analysis. The study focuses on regulatory RNAs (miRNA, riboswitches, tRNA, rRNA, ribozymes) to demonstrate the model's applications in gene regulation research and RNA engineering.

---

## Project Overview

This project implements a hypothesis-driven computational workflow combining RNA structural analysis, feature-based machine learning, and regulatory RNA characterization:

- **Dataset generation:** Collected 200 RNA sequences across 5 regulatory RNA types from Rfam & miRBase, and generated features using ViennaRNA (RNAfold).
- **Feature engineering:** Extracted secondary structure features including length, GC content, stems, loops, bulges, and minimum free energy (MFE)
- **Machine learning:** Trained a Random Forest regression model to predict RNA structural stability
- **Regulatory RNA analysis:**
  - Identified tRNAs as most stable regulatory RNAs (mean stability: 0.686).
  - Characterized structural patterns across different RNA classes
  - Demonstrated length and MFE as primary stability predictors
- **Interactive application:** Developed end-to-end pipeline accepting RNA sequences and outputting stability predictions

All analyses and results are documented in the `notebooks/` folder.

---
## Research Question

Can RNA secondary structure features reliably predict structural stability, and can this approach guide RNA engineering and regulatory genomics research by identifying stable functional RNAs?

---
## Model Performance

| Metric           | Value       |
|------------------|-------------|
| R² (test)        | 0.968       |
| RMSE             | 0.044       |
| MAE              | 0.033       |
| CV R² (5-fold)   | 0.885       |

The model demonstrates that structural features can capture substantial variance in RNA stability, supporting their predictive utility for regulatory RNA analysis and synthetic biology applications.

---

## Interactive Application

**Streamlit App:** https://rna-stability-prediction.streamlit.app/

I developed a Streamlit application demonstrating the complete RNA stability prediction workflow:

- **Input:** RNA sequence (AUGC format)
- **Processing:** Automatic calculation of secondary structure features using approximate folding thermodynamics calculations OR user enter them manually
- **Output:** 
  - Predicted stability score (0-1 scale)
  - All structural features (stems, loops, bulges, MFE) [If user chose tha automaric mode]
  - Color-coded stability assessment
  - Biological interpretation

This workflow emphasizes the practical application of ML in regulatory genomics and RNA engineering.

---
## Repository Structure

```text
rna-structure-prediction/
├── data/
│   └──raw/                          # raw RNA sequences and their IDs in FASTA form
│   └── rna_structure_data.csv       # Dataset (200 RNA sequences)
├── models/
│   ├── rna_model.joblib             # Trained Random Forest model
│   └── rna_scaler.joblib            # Feature scaler
├── notebooks/
│   └── rna_analysis.ipynb           # Complete EDA and ML pipeline
├── results/
│   ├── rna_eda_analysis.png         # EDA visualizations
│   ├── rna_model_comparison.png     # Model performance
│   ├── rna_predictions.png          # Prediction analysis
│   └── rna_feature_importance.png   # Feature importance plot
├── src/
│   └── rna_app.py                   # Streamlit application
├── README.md
├── requirements.txt                 # Python dependencies
└── runtime.txt
```

---
## Sustainable Development Goals Alignment

This project contributes to **SDG 3: Good Health and Well-being** (Target 3.4: Reduce premature mortality from non-communicable diseases) by enabling rapid stability prediction for RNA therapeutics, including:
- **mRNA vaccines** for infectious diseases (e.g., COVID-19 and future pandemic preparedness)
- **siRNA and miRNA therapeutics** for cancer and genetic disorders
- **Gene therapy** applications using regulatory RNAs

By streamlining RNA engineering workflows, this tool supports the development of next-generation therapeutics for diseases that disproportionately affect low- and middle-income countries.

---
## Limitations and Future Work

  - Limited to secondary structure features (excludes tertiary interactions)
  - Validation & Data expansion 

---
## Author

Ali Kawar  
Bioinformatics Student, Lebanese American University  
