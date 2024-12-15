import pandas as pd

def read_data_files(data,index_col):
    df = pd.read_csv(data,index_col=index_col)
    return df
    

def extract_first_two_letters(df):
    """
    Extracts the first two letters of each entry in the first column
    and creates a new DataFrame with those values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: New DataFrame with the first two letters
    """
    # Get the name of the first column
    first_column = df.columns[0]
    
    # Extract the first two letters
    new_data = df[first_column].astype(str).str[:2]
    
    # Create a new DataFrame with the extracted values
    new_df = pd.DataFrame({f"Country code": new_data})
    
    return new_df

