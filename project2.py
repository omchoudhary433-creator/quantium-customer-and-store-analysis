import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\QVI_data (1).csv")
print(df.head(4))
print(df.info())
 #convert date column to datetime from which aggregation will be done #

df['DATE'] = pd.to_datetime(df['DATE'])
print(df['DATE'].dtype)
print(df['DATE'])

#create a new column for year and month for aggregation#

df['YEARMONTH'] = df['DATE'].dt.strftime('%Y%m')
print(df['YEARMONTH'].unique())
print(df['YEARMONTH'])


# MONTHLY METRICS AGGREGATION #

print("Calculating monthly metrics per store...")
monthly_metrics = df.groupby(['STORE_NBR', 'YEARMONTH']).agg(
    TOT_SALES=('TOT_SALES', 'sum'),
    N_CUSTOMERS=('LYLTY_CARD_NBR', 'nunique'),
    N_TXNS=('TXN_ID', 'nunique')
).reset_index()

monthly_metrics['TXN_PER_CUST'] = monthly_metrics['N_TXNS'] / monthly_metrics['N_CUSTOMERS']
update_cols = ['TOT_SALES', 'N_CUSTOMERS', 'N_TXNS', 'TXN_PER_CUST']
print("updated columns: ", update_cols)
print(monthly_metrics.head(4))

# STEP 3: SIMILARITY CALCULATOR FUNCTION

def calculate_similarity(trial_store, metrics_df, pre_trial_end="201902"):
    print(f"Calculating similarity for Trial Store {trial_store}...")
    print(f"Using pre-trial data up to {pre_trial_end} for matching.")
    print("matrics_df columns: ", metrics_df.columns)
    print("def calculate_similarity parameters: trial_store: ", trial_store, "pre_trial_end: ", pre_trial_end)
   
    # Sirf pre-trial period (July 2018 - Feb 2019) ka data use karenge matching ke liye

    pre_trial_data = metrics_df[metrics_df['YEARMONTH'] <= pre_trial_end]
    print("pre_trial_data columns: ", pre_trial_data.columns)
    print("pre_trial_data:")
    print(pre_trial_data)
    
    # extract trial store data#

    trial_store_data = pre_trial_data[pre_trial_data['STORE_NBR'] == trial_store].set_index('YEARMONTH')
    print("trial_store_data columns: ", trial_store_data.columns)
    similarity_results = []
    print(trial_store_data)
    
    # Iterate through all stores to find potential control stores

    for store in pre_trial_data['STORE_NBR'].unique():
        # Do not compare with itself or other trial stores
        if store in [77, 86, 88, trial_store]:
            continue
            
        control_store_data = pre_trial_data[pre_trial_data['STORE_NBR'] == store].set_index('YEARMONTH')
        
        # check if there are at least 8 common months of data for a valid comparison

        common_months = trial_store_data.index.intersection(control_store_data.index)
        if len(common_months) < 8:
            continue
            
        # Pearson Correlation (Sales & Customers)

        print(f"Calculating correlation for Control Store {store}...")
        corr_sales, _ = pearsonr(trial_store_data.loc[common_months, 'TOT_SALES'], control_store_data.loc[common_months, 'TOT_SALES'])
        corr_cust, _ = pearsonr(trial_store_data.loc[common_months, 'N_CUSTOMERS'], control_store_data.loc[common_months, 'N_CUSTOMERS'])
        print(f"Correlation for Control Store {store}: Sales={corr_sales}, Customers={corr_cust}")

        #  Absolute Magnitude Distance
        
        dist_sales = np.mean(np.abs(trial_store_data.loc[common_months, 'TOT_SALES'] - control_store_data.loc[common_months, 'TOT_SALES']))
        dist_cust = np.mean(np.abs(trial_store_data.loc[common_months, 'N_CUSTOMERS'] - control_store_data.loc[common_months, 'N_CUSTOMERS']))
        print(f"Distance for Control Store {store}: Sales={dist_sales}, Customers={dist_cust}")
        similarity_results.append({
            'CONTROL_STORE': store,
            'CORR_SALES': corr_sales,
            'CORR_CUST': corr_cust,
            'DIST_SALES': dist_sales,
            'DIST_CUST': dist_cust
        })
        
    results_df = pd.DataFrame(similarity_results)
    
    # Distance Metric Normalization (Min-Max Scaling: Score close to 1 means less distance)

    results_df['SCORE_DIST_SALES'] = 1 - (results_df['DIST_SALES'] - results_df['DIST_SALES'].min()) / (results_df['DIST_SALES'].max() - results_df['DIST_SALES'].min())
    results_df['SCORE_DIST_CUST'] = 1 - (results_df['DIST_CUST'] - results_df['DIST_CUST'].min()) / (results_df['DIST_CUST'].max() - results_df['DIST_CUST'].min())
    print("Similarity results before ranking:")
    print(results_df)
    # Final Similarity Score Calculation (Average of Correlation and Distance Scores)
    results_df['FINAL_SCORE'] = (results_df['CORR_SALES'] + results_df['CORR_CUST'] + results_df['SCORE_DIST_SALES'] + results_df['SCORE_DIST_CUST']) / 4
    print("Similarity results after final score calculation:")
    print(results_df)
    # Ranking Control Stores based on Final Similarity Score (Higher is better)
    results_df = results_df.sort_values(by='FINAL_SCORE', ascending=False).reset_index(drop=True)
    print("Similarity results after ranking:")
    print(results_df)
    return (results_df)

# calculate similarity for each trial store #

trial_stores = [77, 86, 88]
all_results = []

for trial_store in trial_stores:
    results_df = calculate_similarity(trial_store, monthly_metrics)
    all_results.append(results_df)

# Combine results from all trial stores
combined_results = pd.concat(all_results, ignore_index=True)


#best control stores for each trial store

best_control_stores = combined_results.groupby('CONTROL_STORE').first()
print(best_control_stores)



# BEST CONTROL STORE MAPPING

best_control_mapping = {}

for trial_store in trial_stores:

    results_df = calculate_similarity(
        trial_store,
        monthly_metrics
    )

    best_control = results_df.iloc[0]['CONTROL_STORE']

    best_control_mapping[trial_store] = best_control

    print(
        f"Trial Store {trial_store} "
        f"-> Best Control Store: {best_control}"
    )

trial_period = ['201903', '201904']

# PERFORMANCE ANALYSIS

for trial_store, control_store in best_control_mapping.items():

    print("\n")
    print("=" * 50)

    print(f"Trial Store: {trial_store}")
    print(f"Control Store: {control_store}")

    # Trial Store Data
    trial_data = monthly_metrics[
        (monthly_metrics['STORE_NBR'] == trial_store) &
        (monthly_metrics['YEARMONTH'].isin(trial_period))
    ]

    # Control Store Data
    control_data = monthly_metrics[
        (monthly_metrics['STORE_NBR'] == control_store) &
        (monthly_metrics['YEARMONTH'].isin(trial_period))
    ]

    # SALES COMPARISON

    trial_sales = trial_data['TOT_SALES'].sum()
    control_sales = control_data['TOT_SALES'].sum()

    print("Trial Sales:", trial_sales)
    print("Control Sales:", control_sales)

    # PERCENTAGE INCREASE

    increase = (
        (trial_sales - control_sales)
        / control_sales
    ) * 100

    print("Sales Increase %:", round(increase, 2))

    # T-TEST

    t_stat, p_value = ttest_ind(
        trial_data['TOT_SALES'],
        control_data['TOT_SALES']
    )

    print("P-value:", p_value)

    #sucess decision

    if p_value < 0.05 and increase > 0:

        print("RESULT: Trial was SUCCESSFUL")

    else:

        print("RESULT: Trial was NOT successful")

    # GRAPH

    plt.figure(figsize=(8,5))

    plt.plot(
        trial_data['YEARMONTH'],
        trial_data['TOT_SALES'],
        marker='o',
        label='Trial Store'
    )

    plt.plot(
        control_data['YEARMONTH'],
        control_data['TOT_SALES'],
        marker='o',
        label='Control Store'
    )

    plt.title(
        f"Trial Store {trial_store} vs Control Store {control_store}"
    )

    plt.xlabel("Month")
    plt.ylabel("Total Sales")

    plt.legend()

    plt.show()

#exporting the final combined results to a CSV file for further analysis

best_control_stores.to_csv("best_control_stores.csv", index=True)
print("Combined similarity results exported to CSV.")
