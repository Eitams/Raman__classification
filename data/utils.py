from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, params, normalization=True):
    start = params["start_wave"]
    end = params["end_wave"]
    test_size = params["test_size"]
    seed = params["seed"]
    
    ## preprocess data
    df.columns = df.iloc[0] # Set First row as column names
    df.drop(0, axis=0, inplace = True) # Drop first row
   # df.drop(df.columns[0], axis=1, inplace = True) # Drop patient id column
    df.columns = ["patient", "diabetic"] + list(df.columns[2:]) # Change names of first column

    ## Slice df to wanted wave boundaries
    tmp =df["diabetic"]
    df = df.iloc[:, start+1:end+2]
    df.insert(loc=0, column='diabetic', value=tmp)

    # Split the DataFrame into training and testing sets
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)

    ## Normalize data
    if normalization:
        scalar = MinMaxScaler()
        train_scaled = pd.DataFrame(scalar.fit_transform(train_df.iloc[:,1:]), columns = train_df.columns[1:])
        val_scaled = pd.DataFrame(scalar.transform(val_df.iloc[:,1:]), columns = val_df.columns[1:])
        
        train_scaled = pd.concat([train_df["diabetic"].reset_index(), train_scaled], axis=1).set_index("index")
        val_scaled = pd.concat([val_df["diabetic"].reset_index(), val_scaled], axis=1).set_index("index")


    return train_df, val_df