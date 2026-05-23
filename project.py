#IMPORT ALL LIABRARIES#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#LOAD DATASETS
CUSTOMER=pd.read_csv(r"C:\Users\user\OneDrive\Desktop\PURCHASE_BEHAVIOUR.csv")
print(CUSTOMER)
transaction=pd.read_excel(r"C:\Users\user\OneDrive\Desktop\QVI_transaction_data.xlsx")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#BASCIC DATA EXPLORE

print(transaction.head(2))
print(CUSTOMER.info())
print(transaction.info())
print(CUSTOMER.describe())
print(transaction.describe())
print(transaction.shape)
print(CUSTOMER.shape)
print(transaction.columns)
print(CUSTOMER.columns)

#CHECK MISSING VALUES
print(transaction.duplicated().sum())
print(CUSTOMER.duplicated().sum())
###REMOVE DUPLICATES###

transaction=transaction.drop_duplicates()
CUSTOMER=CUSTOMER.drop_duplicates()

#CONVERT DATE COLUMNS#
print(transaction["DATE"].dtype)
if np.issubdtype(transaction["DATE"].dtype, np.number):
    transaction["DATE"]=pd.to_datetime( transaction["DATE"])
    origin="1899-12-30",
unit="D"
print(transaction[["DATE"]])
print(transaction["PROD_QTY"].describe())

plt.boxplot(transaction["PROD_QTY"])
plt.title("Product's Quantity Outliers")
plt.show()

# REMOVING OPTIONAL#
transaction= transaction[
    transaction["PROD_QTY"] < 100
]
print(transaction["PROD_QTY"].describe())

## EXTRACTING BRAND NAME##
transaction["BRAND"] = (
    transaction["PROD_NAME"]
    .str.split()
    .str[0]
)
print(transaction[["PROD_NAME"]])

print(transaction[["PROD_NAME","BRAND"]])

 #EXTRACT PACK SIZE#

transaction["PACK_SIZE"] = (
    transaction["PROD_NAME"]
    .str.extract(r'(\d+)')
)

transaction["PACK_SIZE"] = pd.to_numeric(
    transaction["PACK_SIZE"]
)
print(transaction[["PROD_NAME","PACK_SIZE"]])
# MERGE DATASETS

data = pd.merge(
 transaction,
CUSTOMER,
on="LYLTY_CARD_NBR"
)
print(data.head(2))

# CHECK FINAL DATA
print(data.head())
print(data.info())
print(data.describe())

#TOTAL SALES ANALYSIS
total_sales = data["TOT_SALES"].sum()
print("Total Sales:", total_sales)

# SALES BY LIFESTAGE
sales_lifestage = (
 data.groupby("LIFESTAGE")["TOT_SALES"]
.sum()
.sort_values(ascending=False)
)

print(sales_lifestage)

# SALES BY PREMIUM CUSTOMER

premium_sales = (
    data.groupby("PREMIUM_CUSTOMER")["TOT_SALES"]
    .sum()
)
print(premium_sales)

#  MOST POPULAR BRAND
brand_count = (
    data["BRAND"]
    .value_counts()
)
print(brand_count)
#  MOST POPULAR PACK SIZE#
pack_size_count = (
    data["PACK_SIZE"]
    .value_counts()
)
print(pack_size_count)

#  AVERAGE SPENDING #

avg_spending = (
    data.groupby("PREMIUM_CUSTOMER")["TOT_SALES"]
    .mean()
)

print(avg_spending)
# AVERAGE PRODUCT QUANTITY #
avg_qty = (
    data.groupby("LIFESTAGE")["PROD_QTY"]
    .mean()
)
print(avg_qty)
#  BRAND SALES ANALYSIS #
brand_sales = (
    data.groupby("BRAND")["TOT_SALES"]
    .sum()
    .sort_values(ascending=False)
)
print(brand_sales)

#  VISUALIZATION - BRAND SALES
brand_sales.plot(
    kind="bar",
    figsize=(10,5)
)
plt.title("Brand Sales Analysis")
plt.xlabel("Brand")
plt.ylabel("Sales")
plt.show()

#  SALES BY LIFESTAGE GRAPH

sales_lifestage.plot(
    kind="bar",
    figsize=(10,5)
)

plt.title("Sales by Lifestage")
plt.xlabel("Lifestage")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()

# PACK SIZE ANALYSIS
pack_size_count.plot(
    kind="bar",
    figsize=(10,5)
)
plt.title("Pack Size Popularity")
plt.xlabel("Pack Size")
plt.ylabel("Count")
plt.show()

#  MONTHLY SALES TREND
data["MONTH"] = (
    data["DATE"]
    .dt.to_period("M")
)
monthly_sales = (
    data.groupby("MONTH")["TOT_SALES"]
    .sum()
)
monthly_sales.plot(
    kind="line",
    figsize=(10,5)
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
# STEP 24 : BUSINESS INSIGHTS
print("\nBUSINESS INSIGHTS\n")

print("1. Top-selling brands identified.")
print("2. High spending customer segments identified.")
print("3. Popular pack sizes identified.")
print("4. Premium customer behavior analyzed.")
print("5. Monthly sales trends analyzed.")


#  BUSINESS RECOMMENDATIONS #

print("\n BUSINESS RECOMMENDATIONS \n")

print("1. Increase marketing for top-selling brands.")
print("2. Focus on premium customers.")
print("3. Promote popular pack sizes.")
print("4. Provide family combo offers.")
print("5. Improve inventory for high-demand products.")

#  EXPORT RESULTS #
brand_sales.to_csv("brand_sales_output.csv")
sales_lifestage.to_csv("sales_lifestage_output.csv")
premium_sales.to_csv("premium_sales_output.csv")
brand_count.to_csv("brand_count_output.csv")
pack_size_count.to_csv("pack_size_count_output.csv")
print("Results exported csv files.")
print("Brand sales exported.")
print("Sales by lifestage exported.")
print("Premium sales exported.")
print("Brand count exported.")
print("Pack size count exported.")

print("analyse completed successfully")
