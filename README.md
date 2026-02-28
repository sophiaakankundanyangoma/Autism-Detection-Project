# Autism Detection Project
## Overview
This project aims to detect autism using a structured dataset from kaggle titled "Early Autism Screening Dataset for Toddlers by Mukesh Dilip"
It includes;
- Preprocessing: missing values, encoding, class imbalance
- Statistical EDA: T-tests, Chi-Square, visualizations
## Dataset
The raw dataset is **not included** for privacy reasons.
- Modeling: Random Forest
## Folder Structure
- 'src/main.py' - main workflow
## Construct Mapping Table

| Raw Dataset Feature | Theoretical Construct | Domain Framework Category | Justification for Impact on ASD Diagnosis |
|---------------------|----------------------|---------------------------|------------------------------------------|
| Age                 | Developmental Factor | Developmental Psychology  | ASD symptoms often appear in early childhood and developmental delays become more noticeable as a child grows. Age helps clinicians evaluate whether behavioral patterns are developmentally appropriate or indicative of ASD. |
| Gender              | Biological Factor    | Neurodevelopmental Risk   | Research shows ASD is more commonly diagnosed in males than females due to biological and neurological differences. Gender therefore influences the probability and expression of ASD traits. |
| Family_ASD          | Genetic Predisposition| Genetic Risk Framework    | ASD has a strong genetic component. Individuals with a family history of ASD are at a higher risk of diagnosis due to inherited genetic factors affecting brain development. |
| Screening Score     | Behavioral Assessment | Clinical Diagnostic Framework | Screening scores directly measure behavioral symptoms such as communication difficulties and social interaction problems, which are core diagnostic criteria for ASD. Higher scores strongly indicate likelihood of ASD diagnosis. |
| Autism Traits Score | Behavioral Severity   | Clinical Psychology Framework | Higher trait scores reflect more pronounced ASD-related behaviors, making this a direct predictor of diagnosis outcome. This construct aligns with standardized diagnostic tools used in clinical evaluation. |

