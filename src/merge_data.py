import pandas as pd

inflation = pd.read_csv("/home/loreen-atenge/kenya-inflation-analysis/data/cleaned/kenya-final.csv")
debt = pd.read_csv("/home/loreen-atenge/kenya-debt-analysis/data/cleaned/kenya_debt.csv")

inflation = inflation[["year", "inflation", "exchange_rate", "broad_money"]]
debt = debt[["year", "external_debt_gni", "debt_service_exports"]]

merged = pd.merge(inflation, debt, on="year", how="inner")

print(merged.shape)
print(merged.head())
print(merged.isnull().sum())

merged.to_csv("data/cleaned/kenya_nexus.csv", index=False)