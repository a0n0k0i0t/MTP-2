import pandas as pd
import numpy as np

def generate_points(input_file="adult.data", output_file="points.txt", start_idx=0, sample_size=500):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
        "hours-per-week", "native-country", "income"
    ]
    
    # Read adult.data
    df = pd.read_csv(input_file, names=columns, skipinitialspace=True)
    
    # Map Female -> 1, Male -> 0
    df['sex'] = df['sex'].map({'Female': 1, 'Male': 2}).fillna(0).astype(int)
    
    # Handle categorical columns by replacing them with categorical codes (label encoding)
    for col in columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    # Save full numerical data
    df.to_csv("adult_numerical.csv", index=False)
    print("Generated intermediate data: adult_numerical.csv")
    
    # Slice sequentially instead of random sampling
    df = df.iloc[start_idx : start_idx + sample_size].reset_index(drop=True)
    
    # Separate color (sex)
    colors = df['sex'].tolist()
    
    # Remove 'sex' from coordinates calculation
    # We have 14 features remaining
    feature_cols = [c for c in columns if c != "sex"]
            
    # Calculate a query bounding box for our 14 dimensions
    # E.g., from 10th percentile to 90th percentile within the sample
    qbox_bounds = []
    for col in feature_cols:
        min_v = np.percentile(df[col], 10)
        max_v = np.percentile(df[col], 90)
        qbox_bounds.append((min_v, max_v))
        
    d = len(feature_cols)
    t = 0
    N = len(df)
    
    # Prepare coordinates
    coords = df[feature_cols].values.tolist()
    
    # Write to file
    with open(output_file, 'w') as f:
        # N
        f.write(f"{N}\n")
        # d t
        f.write(f"{d} {t}\n")
        
        # Points
        for i in range(N):
            point_line = " ".join(map(str, coords[i]))
            f.write(f"{point_line}\n")
            
        # Colors (space separated)
        color_line = " ".join(map(str, colors))
        f.write(f"{color_line}\n")
        
        # Query box bounds
        for i in range(d):
            min_v, max_v = qbox_bounds[i]
            f.write(f"{min_v} {max_v}\n")
            
    print(f"Successfully generated {output_file} heavily based on {input_file}")
    print(f"Sample size: {N}, Dimensions: {d}, Query Box: {qbox_bounds}")

if __name__ == "__main__":
    generate_points()
