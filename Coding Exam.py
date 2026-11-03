#. Data validation and cleaning.
#DATA LOADING
import pandas as pd
account_info = pd.read_csv("da_fitly_account_info.csv")
customer_support = pd.read_csv("da_fitly_customer_support.csv")
user_activity = pd.read_csv("da_fitly_user_activity.csv")

#DATA INSPECTION
#Inspecting first few rows
print("\n--- HEAD ---")
print(account_info.head())
print(customer_support.head())
print(user_activity.head())


#check shape of the data
print("\n--- SHAPE ---")
print("Account information:", account_info.shape)
print("Customer support:", customer_support.shape)
print("User activity:", user_activity.shape)

#check structure of the data
print("\n--- STRUCTURE ---")
print("Account info:", account_info.info())
print("Customer support:", customer_support.info())
print("User activity: ", user_activity.info())

#check for missing values
print("\n--- MISSING VALUES ---")
print("\nAccount Info")
print(account_info.isna().sum())
print("\nCustomer Support")
print(customer_support.isna().sum())
print("\nUser Activity")
print(user_activity.isna().sum())

#check for duplicates
print("\n--- DUPLICATES ---")
print("\nAccount Info")
print(account_info.duplicated().sum())
print("\nCustomer Support")
print(customer_support.duplicated().sum())
print("\nUser Activity")
print(user_activity.duplicated().sum())

#checking the number of unique values in a categorical variable
print("\n--- UNIQUE VALUES OF CATEGORIES ---")
print("\nAccount Info")
print(account_info["state"].value_counts(dropna=False))
print(account_info["plan"].value_counts(dropna=False))
print(account_info["churn_status"].value_counts(dropna=False))

print("\nCustomer Support")
print(customer_support["channel"].value_counts(dropna=False))
print(customer_support["topic"].value_counts(dropna=False))
print(customer_support["state"].value_counts(dropna=False))

print("\nUser Activity")
print(user_activity["event_type"].value_counts(dropna=False))

#Getting a summary of numerical variables
print("\n--- NUMERICAL SUMMARY ---")

print("\nACCOUNT INFO > Plan list price:")
print(account_info["plan_list_price"].describe())

print("\nCUSTOMER SUPPORT> Resolution time:")
print(customer_support["resolution_time_hours"].describe())

print("\nThere is no categorical variable in USER ACTIVITY")

#checking the data type of the date time variables.
print("\n--- DATE COLUMNS ---")

print(customer_support["ticket_time"].dtype)
print(user_activity["event_time"].dtype)

#CROSS DATA VALIDATION
#checking that user IDs exist in account information

account_ids = account_info["customer_id"].str.replace("C", "").astype(int)

print("\n--- CROSS-DATASET ID VALIDATION ---")

print(
    "Customer support IDs found in account information:",
    customer_support["user_id"].isin(account_ids).sum(),
    "out of",
    len(customer_support)
)

print(
    "User activity IDs found in account information:",
    user_activity["user_id"].isin(account_ids).sum(),
    "out of",
    len(user_activity)
)

#checking date ranges
print("\n--- DATE RANGES ---")

print(
    "Customer support:",
    customer_support["ticket_time"].min(),
    "to",
    customer_support["ticket_time"].max()
)

print(
    "User activity:",
    user_activity["event_time"].min(),
    "to",
    user_activity["event_time"].max()
)

#Checking uniqueness of customer identifiers
print("\n--- UNIQUE IDENTIFIERS ---")

print(
    "Duplicate customer IDs:",
    account_info["customer_id"].duplicated().sum()
)

print(
    "Duplicate email addresses:",
    account_info["email"].duplicated().sum()
)


#DATA CLEANING
#replacing '-' with NA
customer_support["channel"] = customer_support["channel"].replace("-", pd.NA)
print(customer_support["channel"].value_counts(dropna=False))

#changing ticket time to pacific time data type
customer_support["ticket_time"] = pd.to_datetime(
    customer_support["ticket_time"]
).dt.tz_localize("US/Pacific")

#changing even time to pacific time data type
user_activity["event_time"] = pd.to_datetime(
    user_activity["event_time"]
).dt.tz_localize("US/Pacific")

#create a new column called churned based on the existing churn_status
account_info["churned"] = account_info["churn_status"].eq("Y")
print(account_info["churned"].value_counts())

#droping state column
customer_support = customer_support.drop(columns=["state"])
print(customer_support.columns)

#2. EDA
#Plan Distribution
import matplotlib.pyplot as plt

account_info["plan"].value_counts().plot(
    kind="bar",
    color=["steelblue", "orange", "seagreen", "purple"]
)

plt.title("Number of Customers by Plan")
plt.xlabel("Plan")
plt.ylabel("Number of Customers")
plt.yticks(range(0, 121, 10))
plt.xticks(rotation=0)
plt.show()

#Resolution Time.
customer_support["resolution_time_hours"].plot(kind="hist", bins=20)

plt.title("Distribution of Support Ticket Resolution Time")
plt.xlabel("Resolution Time (hours)")
plt.xticks(range(0, 35, 2))
plt.ylabel("Number of Tickets")
plt.show()

#Churn status by plan (stacked bar plot showing plan and churn status together)
#counting the customers on each plan by churn status
churn_status_by_plan = (
    account_info.groupby(["plan", "churned"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={True: "Marked churned (Y)", False: "No churn status recorded"})
)

#share of the customers on each plan that were marked as churned
churn_status_by_plan["total"] = churn_status_by_plan.sum(axis=1)
churn_status_by_plan["pct_churned"] = (
    churn_status_by_plan["Marked churned (Y)"] / churn_status_by_plan["total"] * 100
)
churn_status_by_plan = churn_status_by_plan.sort_values("pct_churned", ascending=False)
print(churn_status_by_plan.round(1))

ax = churn_status_by_plan[["Marked churned (Y)", "No churn status recorded"]].plot(
    kind="bar",
    stacked=True,
    color=["#D55E00", "#0072B2"],
    width=0.6,
    figsize=(8, 5.5)
)

#adding the number of customers inside each bar segment
for container in ax.containers:
    ax.bar_label(container, label_type="center", color="white", fontweight="bold")

#adding the percentage marked as churned above each bar
for i, (plan, row) in enumerate(churn_status_by_plan.iterrows()):
    ax.text(i, row["total"] + 2, f"{row['pct_churned']:.1f}% churned", ha="center")

plt.title("Customers Marked as Churned, by Plan")
plt.xlabel("Plan")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.ylim(0, churn_status_by_plan["total"].max() * 1.4)
plt.legend(title="Churn status", loc="upper right")
plt.show()

#3. Metric: share of customers marked as churned (overall and by plan)
overall_churn_rate = account_info["churned"].mean() * 100
print("Overall churn rate (marked churned / all customers):", round(overall_churn_rate, 1), "%")
print(churn_status_by_plan["pct_churned"].round(1))