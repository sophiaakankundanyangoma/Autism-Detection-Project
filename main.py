import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Loading dataset
DATA_PATH = "data/asd.csv"
TARGET_COLUMN = "Diagnosed_ASD"

df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns
df = df.drop(columns=["Child_ID", "Unnamed: 10"], errors='ignore')

# Testing the dataset
print("First 5 rows:\n", df.head())
print("\nDataset info:")
print(df.info())

# Preprocessing with rigor
# Step 1: Handle missing values
print("\nMissing values per column:\n", df.isnull().sum())

# For simplicity, assume:
# - MCAR: can use mode/median imputation

# Impute categorical features
categorical_cols = ['Gender', 'Jaundice', 'Family_ASD_History', 'Language_Delay']
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Imputed {col} with mode: {df[col].mode()[0]}")

# Impute numeric features
numeric_cols = ['Age', 'Social_Interaction_Score', 'Communication_Score', 'Repetitive_Behavior_Score']
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Imputed {col} with median: {df[col].median()}")

# Keep a copy of categorical features for Chi-square before encoding
df_cat_original = df[categorical_cols + [TARGET_COLUMN]].copy()

# Step 2b: Encode categorical variables for modeling
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
    print(f"Encoded {col} to numeric")

# Step 2c: Check class balance
print("\nTarget value counts before resampling:")
print(df[TARGET_COLUMN].value_counts())

# Upsample minority class
majority = df[df[TARGET_COLUMN] == df[TARGET_COLUMN].value_counts().idxmax()]
minority = df[df[TARGET_COLUMN] != df[TARGET_COLUMN].value_counts().idxmax()]
minority_upsampled = resample(minority,
                              replace=True,
                              n_samples=len(majority),
                              random_state=42)
df = pd.concat([majority, minority_upsampled])

print("\nTarget value counts after resampling:")
print(df[TARGET_COLUMN].value_counts())

# Separate features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Step 2d: Scale numeric features
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Statistical EDA

# T-tests for numeric features
print("\nT-Test Results (Numeric Features Only):")
for col in numeric_cols:
    class0 = df[df[TARGET_COLUMN]=="No"][col]
    class1 = df[df[TARGET_COLUMN]=="Yes"][col]
    t_stat, p_val = stats.ttest_ind(class0, class1, equal_var=False)
    print(f"{col}: t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"--> {col} differs significantly between ASD and non-ASD")

# Chi-square for categorical features using original values
print("\nChi-Square Results (Categorical Features Only):")
for col in categorical_cols:
    contingency_table = pd.crosstab(df_cat_original[col], df_cat_original[TARGET_COLUMN])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"{col}: chi2={chi2:.3f}, p={p:.4f}")
    if p < 0.05:
        print(f"--> {col} is significantly associated with ASD")

# ----------------------------
# Visualizations
# ----------------------------
# Target distribution
sns.countplot(data=df, x=TARGET_COLUMN)
plt.title("Target Class Distribution")
plt.show()

# Numeric features by target
for col in numeric_cols:
    sns.boxplot(x=TARGET_COLUMN, y=col, data=df)
    plt.title(f"{col} by Target")
    plt.show()

# ----------------------------
# 4. Train Random Forest
# ----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
