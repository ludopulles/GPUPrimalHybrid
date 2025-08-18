import pandas as pd

df = pd.read_csv("attack_results.csv")

# Garder seulement les runs avec success == True
df_true = df[df["success"] == True]

for col in ["iterations",]:
    values = df_true[ "estimated_time"]
    print(f"\n=== {col} ===")
    print(f"Iterations needed for each :")
    print( df_true[ "iterations_used"])
    print(f"Time taken each (in hours) :")
    print((values/ 60 /60))
    print(f"Mean   : {(values.mean()/60/60):,.2f} h")
    print(f"Median : {(values.median()/ 60 /60):,.2f} h")
    print(f"Stddev : {(values.std(ddof=1)/ 60 /60):,.2f} h")