import pandas as pd
import pyreadstat
from collections import defaultdict
import os


def merge_datasets(df1, df2):
    # Get the set of columns for each DataFrame
    common_columns = df1.columns.intersection(df2.columns)
    
    if len(common_columns) != len(df1.columns) or len(common_columns) != len(df2.columns):
        # Output the different columns
        diff1 = df1.columns.difference(df2.columns)
        diff2 = df2.columns.difference(df1.columns)
        print(f"Columns in df1 but not in df2: {list(diff1)}")
        print(f"Columns in df2 but not in df1: {list(diff2)}")
    
    # Keep only the common columns and merge the datasets
    merged_df = pd.concat([df1[common_columns], df2[common_columns]], ignore_index=True)
    
    return merged_df

def summary_data(country_code, subject_char):
        # Read the .sav files
    directory = f'T19_G8_{country_code}_SPSS'
    file1 = f'bsa{country_code}z7.sav'
    file2 = f'bsg{country_code}z7.sav'
    
    df1, meta1 = pyreadstat.read_sav(os.path.join(directory, file1))
    df2, meta2 = pyreadstat.read_sav(os.path.join(directory, file2))

    # Function to get a tuple of columns with no NaN for a given row
    def columns_with_no_nan(row):
        return tuple(row.dropna().index)

    # Create a dictionary to hold groups of rows with the same non-NaN columns
    groups = defaultdict(list)

    # Group rows by columns with no NaN in the first dataset
    for index, row in df1.iterrows():
        key = columns_with_no_nan(row)
        groups[key].append(index)

    # Filter out groups that contain fewer than 100 samples
    filtered_groups = {key: indices for key, indices in groups.items() if len(indices) >= 100}

    # Assign numeric identifiers to each group
    group_ids = {key: idx for idx, key in enumerate(filtered_groups.keys())}

    print("Overview of the group information:")
    group_sizes = {group_ids[key]: (len(indices), [col for col in key if col.startswith(subject_char)][:5]) for key, indices in filtered_groups.items()}
    for group_id, (size, columns) in group_sizes.items():
        print(f"Group {group_id}: {size} samples; First few {subject_char} column names: {columns}")
    return len(group_ids)

def preprocess_data(country_code, label_char, subject_char, item_char, sheet_char, group_id_to_access):
    # Read the .sav files
    directory = f'T19_G8_{country_code}_SPSS'
    file1 = f'bsa{country_code}z7.sav'
    file2 = f'bsg{country_code}z7.sav'
    
    df1, meta1 = pyreadstat.read_sav(os.path.join(directory, file1))
    df2, meta2 = pyreadstat.read_sav(os.path.join(directory, file2))

    # Function to get a tuple of columns with no NaN for a given row
    def columns_with_no_nan(row):
        return tuple(row.dropna().index)

    # Create a dictionary to hold groups of rows with the same non-NaN columns
    groups = defaultdict(list)

    # Group rows by columns with no NaN in the first dataset
    for index, row in df1.iterrows():
        key = columns_with_no_nan(row)
        groups[key].append(index)

    # Filter out groups that contain fewer than 100 samples
    filtered_groups = {key: indices for key, indices in groups.items() if len(indices) >= 100}

    # Assign numeric identifiers to each group
    group_ids = {key: idx for idx, key in enumerate(filtered_groups.keys())}

    # print("Overview of the group information:")
    # group_sizes = {group_ids[key]: (len(indices), [col for col in key if col.startswith(subject_char)][:5]) for key, indices in filtered_groups.items()}
    # for group_id, (size, columns) in group_sizes.items():
    #     print(f"Group {group_id}: {size} samples; First few {subject_char} column names: {columns}")
        
    # Create a DataFrame for each remaining group, drop NaN columns, and store them in a dictionary with numeric IDs
    grouped_dfs = {}
    for key, indices in filtered_groups.items():
        group_df = df1.loc[indices]
        group_df = group_df.dropna(axis=1, how='any')  # Drop columns with any NaN values
        grouped_dfs[group_ids[key]] = group_df

    # Access group using its numeric ID
    chosen_group = grouped_dfs[group_id_to_access] if group_id_to_access in grouped_dfs else None

    if chosen_group is not None:
        # Filter the columns of the second dataset to keep only those starting with item_char and "IDSTUD"
        df2_filtered = df2[['IDSTUD'] + [col for col in df2.columns if col.startswith(item_char)]]

        # Merge the first group with the filtered second dataset on "IDSTUD"
        merged_df = pd.merge(chosen_group, df2_filtered, on='IDSTUD', how='inner')
        
    # columns_to_keep = [col for col in merged_df.columns if not col.startswith(item_char)]
    columns_to_keep = [col for col in merged_df.columns if not col.startswith(item_char)] + label_char
    
    # Filter the dataframe to keep only the selected columns
    df_filtered = merged_df[columns_to_keep]

    # Drop samples with NaN in label_char
    df_cleaned = df_filtered.dropna(subset=label_char)

    df_response = df_cleaned[[col for col in df_cleaned.columns if col.startswith(subject_char) and not col.endswith("_S") and not col.endswith("_F") and col not in label_char]]
    df_label = df_cleaned[label_char]
    df_label.to_csv(f'TIMSS/df_label_{country_code}_{label_char[0]}_group_{group_id_to_access}.csv', index=False)

    
    ############## store full response data #################
    # df_response.to_csv(f'TIMSS/df_response_{country_code}_{subject_char}_group_{group_id_to_access}.csv', index=False)
    ############## store the frequency and time data #################
    # df_S = df_cleaned[[col for col in df_cleaned.columns if col.startswith(subject_char) and col.endswith("_S")]]
    # df_F = df_cleaned[[col for col in df_cleaned.columns if col.startswith(subject_char) and col.endswith("_F")]]
    # df_S.to_csv(f'TIMSS/df_S_{country_code}_{subject_char}_group_{group_id_to_access}.csv', index=False)
    # df_F.to_csv(f'TIMSS/df_F_{country_code}_{subject_char}_group_{group_id_to_access}.csv', index=False) 

    # Load item information and response dataframes
    item_info_path = 'T19_G8_Item Information/eT19_G8_Item Information.xlsx'  # replace with actual path
    item_info_df = pd.read_excel(item_info_path, sheet_char)

    # Mapping from key letters to numbers
    key_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

    # Process the data
    # Create dictionaries for item types and keys
    item_types = item_info_df.set_index('Item ID')['Item Type'].to_dict()
    item_keys = item_info_df.set_index('Item ID')['Key'].to_dict()

    # Map the keys from letters to numbers
    item_keys = {k: key_mapping[v] for k, v in item_keys.items() if v in key_mapping}

    # Process each item in the response dataframe
    for column in df_response.columns:
        if column in item_types:
            if item_types[column] == 'MC':
                correct_key = item_keys[column]
                df_response.loc[:, column] = df_response[column].apply(lambda x: 1 if x == correct_key else 0)
            # CR items remain unchanged

    # Separate the dataframe into CR and MC questions
    cr_columns = [col for col in df_response.columns if item_types.get(col) == 'CR']
    mc_columns = [col for col in df_response.columns if item_types.get(col) == 'MC']

    df_cr = df_response[cr_columns].copy()
    df_mc = df_response[mc_columns].copy()

    # Save the resulting dataframes if needed
    df_cr.to_csv(f'TIMSS/df_cr_{country_code}_{subject_char}_group_{group_id_to_access}.csv', index=False)
    df_mc.to_csv(f'TIMSS/df_mc_{country_code}_{subject_char}_group_{group_id_to_access}.csv', index=False)


def ATC_TIMSS(country_code, label_char, group_id, B_bootstrap = 10):
    clustering_errs = np.ones(8)
    response_mat_ME = pd.read_csv(f'TIMSS/df_mc_{country_code}_ME_group_{group_id}.csv').to_numpy()
    response_mat_SE = pd.read_csv(f'TIMSS/df_mc_{country_code}_SE_group_{group_id}.csv').to_numpy()
    label = pd.read_csv(f'TIMSS/df_label_{country_code}_{label_char[0]}_group_{group_id}.csv').to_numpy()
    ############## Label setting: K = 2 #################
    K = 2
    set_1 = {1}
    new_labels = np.array([
        1 if all(val in set_1 for val in sample) else 0
        for sample in label
    ])
    warnings.filterwarnings('ignore')
    res_response_SE = fit_BMM_softEM(response_mat_SE, K)
    res_response_ME = fit_BMM_softEM(response_mat_ME, K)
    clustering_errs[1] = Hamming_aligned(res_response_SE['labels'],new_labels)/len(new_labels)
    clustering_errs[2] = Hamming_aligned(res_response_ME['labels'],new_labels)/len(new_labels)
    list_lbd = np.arange(0, 20, 0.2)
    [_, errs, _]= TL_demo(response_mat_SE, response_mat_ME, new_labels, K, ['Bernoulli', 'Bernoulli'], list_lbd, show = False, spectral = False)
    clustering_errs[3] = np.min(errs[:len(list_lbd)])/len(new_labels)
    # print(len(list_lbd),len(errs), len(errs[:len(list_lbd)]))
    clustering_errs[0] = errs[len(list_lbd)]/len(new_labels)
    
    # Adaptive clustering
    list_q = [0.8, 0.9, 0.95, 0.99]
    # Error estimation
    [errs_hat, _, _]= estimate_error(response_mat_SE, response_mat_ME, K = K, dist = ['Bernoulli', 'Bernoulli'], list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, spectral = False)
    # Adaptive selection
    list_selected_idx = np.argmin(errs_hat, axis = 1)
    for i in range(len(list_q)):
        clustering_errs[i+4] = errs[list_selected_idx[i]]/len(new_labels)

    return clustering_errs