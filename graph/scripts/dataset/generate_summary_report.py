import pandas as pd
import numpy as np
import os

def generate_simulation_report():
    # 1. Fetch Open Source / Simulation Dataset
    # Using a known dataset like Iris from UCI or from a local CSV
    url = "https://githubusercontent.com"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Process / Simulate Metrics (Example: statistical summary)
    mean_sepal_length = df['sepal_length'].mean()
    mean_sepal_width = df['sepal_width'].mean()
    max_petal_length = df['petal_length'].max()

    # 3. Write content to a text document
    output_filename = "simulation_report.txt"
    
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("=== SIMULATION REPORT ===\n")
        file.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("=" * 35 + "\n\n")
        
        file.write("Dataset Overview:\n")
        file.write(f"- Total Records Analyzed: {len(df)}\n")
        file.write(f"- Average Sepal Length: {mean_sepal_length:.2f} cm\n")
        file.write(f"- Average Sepal Width: {mean_sepal_width:.2f} cm\n")
        file.write(f"- Maximum Petal Length: {max_petal_length:.2f} cm\n\n")
        
        file.write("--- Raw Data Sample ---\n")
        # Write the first 5 rows to the text file
        file.write(df.head(5).to_string(index=False))
        
    print(f"Successfully generated text document at: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    generate_simulation_report()
