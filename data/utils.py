from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.signal as signal
from .MyDataset import CustomDataset
from torch.utils.data import DataLoader


def preprocess_raman_spectrum(df):
    """
    Preprocesses Raman spectra contained in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing Raman spectra, where rows represent samples,
                           and columns represent Raman shifts.

    Returns:
        pd.DataFrame: A new DataFrame containing preprocessed Raman spectra, where rows represent
                      samples, and columns represent Raman shifts.

    Preprocessing Steps:
        1. Baseline Correction: Using asymmetric least squares algorithm (Savitzky-Golay filter),
           the baseline noise in each Raman spectrum is removed to improve spectral clarity.

        2. Remove Baseline from Original Spectrum: The calculated baseline is subtracted from the
           original Raman spectrum to obtain the corrected spectrum.

        3. Normalization: The corrected spectrum is normalized to bring the intensities within the
           range [0, 1]. This scaling operation ensures comparability between different spectra.

        4. Smoothing: A Savitzky-Golay filter is applied to the normalized spectrum to smooth out
           noise and enhance important spectral features.

        5. Construct New DataFrame: The preprocessed spectra are added as columns to a new DataFrame.

        6. Transpose and Reset Column Names: The new DataFrame is transposed to have the original
           format (rows as samples, columns as Raman shifts). Column names are reset to match the
           original Raman shifts, and the index is also reset.

    Note:
        The input DataFrame 'df' is expected to have numeric values representing Raman intensities.
        Ensure that the DataFrame contains valid Raman spectra data before using this function.
    """
    
    preprocessed_df = pd.DataFrame()
    for i in range(len(df)):
        # Get Raman spectrum for the current sample
        spectrum = df.iloc[i, :].values

        # Baseline Correction (using asymmetric least squares algorithm)
        baseline = signal.savgol_filter(spectrum, 101, polyorder=3)

        # Remove baseline from the original spectrum
        corrected_spectrum = spectrum - baseline

        # Normalize the spectrum
        normalized_spectrum = (corrected_spectrum - np.min(corrected_spectrum)) / (np.max(corrected_spectrum) - np.min(corrected_spectrum))

        # Apply smoothing
        smoothed_spectrum = signal.savgol_filter(normalized_spectrum, 51, polyorder=3)

        # Add the preprocessed spectrum to the new DataFrame
        preprocessed_df = pd.concat([preprocessed_df, pd.Series(smoothed_spectrum)], axis=1)

    # Transpose the DataFrame to have the original format (rows as samples, columns as Raman shifts)
    preprocessed_df = preprocessed_df.T

    # Reset the column names to match the original Raman shifts
    preprocessed_df.columns = df.columns
    preprocessed_df.reset_index(drop=True, inplace=True)
    return preprocessed_df


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
        # assigning df without the target column to preprocess function
        preprocessed_train_df = preprocess_raman_spectrum(train_df.iloc[:,1:])
        preprocessed_val_df = preprocess_raman_spectrum(val_df.iloc[:,1:])

        # Mergin processed data frame with target column
        train_df = pd.concat([train_df["diabetic"].reset_index(), preprocessed_train_df], axis=1).set_index("index")
        val_df = pd.concat([val_df["diabetic"].reset_index(), preprocessed_val_df], axis=1).set_index("index")

        #scalar = MinMaxScaler()
        #train_scaled = pd.DataFrame(scalar.fit_transform(train_df.iloc[:,1:]), columns = train_df.columns[1:])
        #val_scaled = pd.DataFrame(scalar.transform(val_df.iloc[:,1:]), columns = val_df.columns[1:])
        
        #train_df = pd.concat([train_df["diabetic"].reset_index(), train_scaled], axis=1).set_index("index")
        #val_df = pd.concat([val_df["diabetic"].reset_index(), val_scaled], axis=1).set_index("index")


    return train_df, val_df

def preprocess_cv(df, params, normalization=True):
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
    df = df.reset_index().drop("index", axis = 1)
    # Perform K-Fold Cross-Validation
    num_folds = 7
    train_folds = {}
    val_folds = {}
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        
        if normalization:
            # Create data loaders for the current fold
            train_df, val_df = df.iloc[train_index,1:-1], df.iloc[test_index,1:-1]
            ## Normalize data
        
            # assigning df without the target column to preprocess function
            preprocessed_train_df = preprocess_raman_spectrum(train_df.iloc[:,1:])
            preprocessed_val_df = preprocess_raman_spectrum(val_df.iloc[:,1:])
            # print(train_index)
            # print(df.head())
            # print(df.loc[train_index,"diabetic"])
            # # print(preprocessed_train_df)
            # print(pd.concat([df.loc[train_index,"diabetic"].reset_index(), preprocessed_train_df], axis=1).set_index("index"))


            # Mergin processed data frame with target column
            train_df = pd.concat([df.loc[train_index,"diabetic"].reset_index(), preprocessed_train_df], axis=1).set_index("index")
            val_df = pd.concat([df.loc[test_index,"diabetic"].reset_index(), preprocessed_val_df], axis=1).set_index("index")
 
            #scalar = MinMaxScaler()
            #train_scaled = pd.DataFrame(scalar.fit_transform(train_df.iloc[:,1:]), columns = train_df.columns[1:])
            #val_scaled = pd.DataFrame(scalar.transform(val_df.iloc[:,1:]), columns = val_df.columns[1:])
            
            #train_df = pd.concat([train_df["diabetic"].reset_index(), train_scaled], axis=1).set_index("index")
            #val_df = pd.concat([val_df["diabetic"].reset_index(), val_scaled], axis=1).set_index("index")
        
        else:
            # Create data loaders for the current fold
            train_df, val_df = df.iloc[train_index,:-1], df.iloc[test_index,:-1]

        train_ds = CustomDataset(train_df)
        val_ds = CustomDataset(val_df)

        train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
        
        train_folds[fold+1] = train_dl
        val_folds[fold+1] = val_dl

    return train_folds, val_folds

