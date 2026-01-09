import pandas as pd

# Load your CSV
df = pd.read_csv("rna_structure_data.csv")  # replace with your file name if different

# Define stability classes
def classify_stability(score):
    if score >= 0.7:
        return "Stable"
    elif score >= 0.5:
        return "Moderate"
    else:
        return "Unstable"

# Apply function to create new column
df['Stability_Class'] = df['Stability_Score'].apply(classify_stability)

# Save updated CSV (overwrite or use a new name)
df.to_csv("rna_structure_data.csv", index=False)

print("Added Stability_Class column and saved updated CSV.")
