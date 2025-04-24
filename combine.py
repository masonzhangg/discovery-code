import pandas as pd

def combine_csv_files(*csv_files):
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)  
        df.reset_index(drop=True, inplace=True)      
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=1)
    return combined_df

if __name__ == "__main__":
    combined_df = combine_csv_files(
        "resultsMason1.csv",
        "resultsJenny1.csv",
        "resultsNathan1.csv",
    )
    combined_df.to_csv("combined_results.csv", index=False)
    print("Combined results saved to combined_results.csv")
