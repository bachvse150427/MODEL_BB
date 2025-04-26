import pandas as pd
import os
import glob
from datetime import datetime

def get_latest_predictions_folder():
    predictions_dir = 'predictions'
    folders = [f for f in os.listdir(predictions_dir) 
              if os.path.isdir(os.path.join(predictions_dir, f))]
    
    if not folders:
        raise Exception("No prediction folders found")
        
    folder_dates = []
    for folder in folders:
        try:
            date = datetime.strptime(folder, '%Y%m%d_%H%M%S')
            folder_dates.append((folder, date))
        except ValueError:
            continue
            
    if not folder_dates:
        raise Exception("No valid timestamp folders found")
        
    latest_folder = max(folder_dates, key=lambda x: x[1])[0]
    latest_path = os.path.join(predictions_dir, latest_folder)
    
    print(f"Found latest predictions folder: {latest_path}")
    return latest_path

def merge_prediction_files(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "predictions_*.csv"))
    dfs = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
            
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.sort_values(['Ticker', 'Model', 'Month-Year'])
        output_path = os.path.join(folder_path, "merged_predictions.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully merged {len(csv_files)} files into {output_path}")
        print("\nMerged file statistics:")
        print(f"Total rows: {len(merged_df)}")
        print(f"Unique tickers: {merged_df['Ticker'].nunique()}")
        print(f"Models used: {merged_df['Model'].unique()}")
        print(f"Date range: {merged_df['Month-Year'].min()} to {merged_df['Month-Year'].max()}")
        print(f"Overall prediction accuracy: {(merged_df['Correct'].sum() / len(merged_df)) * 100:.2f}%")
    else:
        print("No files were found to merge")

if __name__ == "__main__":
    try:
        latest_folder = get_latest_predictions_folder()
        merge_prediction_files(latest_folder)
    except Exception as e:
        print(f"Error: {str(e)}")
