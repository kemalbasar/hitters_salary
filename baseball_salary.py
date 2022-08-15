import numpy as np
import pandas as pd
import seaborn as sns
import pre_methods as pt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

##################################################################################################
##################################################################################################
##################################################################################################

# 1. Data Pre-processing and Feature Engineeering.
#  a.Outliers ---> Detect outliers with quantiles ( box plot ) and apply supression
#                 ----> Detect with Local Outlier Facotr
#  b.Missing Values  --->  drop nas / simple assignment /  assignments at break points / assignment with KNN
#                       ----> missing vs targetn analysis
#  c.Encodings --->  label / one-hot / rare
#  d.Scaling ---> standart / robust / min-max
#  e.Feature Extraction / Feature Interactions

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv(r"/data/hitters.csv")
dff = df.copy()
df.describe().T

##################################################################################################
##################################################################################################
##################################################################################################
# Descriptive Analysis

cat_vars, num_vars, cat_but_car = pt.grab_col_name(df)

for col in num_vars:
    pt.num_summary(df, col, plot=False)

##################################################################################################
##################################################################################################
##################################################################################################
# Outlier Analysis
# cruns / chmrun and chits / catbat fuatures are seems like have outlier.

f = {col: pt.check_outlier(df, col, 0.1, 0.9) for col in num_vars}
outl_columns = [key for key in list(f.keys()) if f[key] == True]

# chmrun ve chits seems like have values which stands outside the limits.But its normal situation,
# outstanding perfomance can occur and effects salary , so we wont make any modifications here.

# before that we need to deal with None values for  because we will detect Multi-Var Outliers

# Missing Values Replacement
imputer = KNNImputer(n_neighbors=5)
df["Salary"] = pd.DataFrame(imputer.fit_transform(df[["Salary"]]), columns=[["Salary"]])

# multi-variable outlier analysis
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(df[[col for col in num_vars if col not in ['Hits', 'CHits']]])

df_scores = clf.negative_outlier_factor_
df_scores[df_scores < -2]
th = np.sort(df_scores)[5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# Dropping outliers.
df = df.loc[~df.index.isin(list(df[df_scores < th].index))]

# Label Encoding
df = pt.binary_encoder(df)

# Correlation Analysis
pt.get_core_triangle(df).applymap(lambda x: True if x > 0.8 else False)
sns.heatmap(df.corr())
plt.show()

# We wont gonna drop other correlated values , because they hold meaning that differs
# from each other, and we will use them for extract useful features from it.


##################################################################################################
##################################################################################################
##################################################################################################
# Feature Extraction
# Getting hit and homerun percentages
df["BA"] = df["Hits"] / df["AtBat"]
df["HR/H"] = df["HmRun"] / df["Hits"]

df["C_BA"] = df["CHits"] / df["CAtBat"]
df["C_HR/H"] = df["CHmRun"] / df["CHits"]

# Seasonal performance divergence from career.
df['NEW_Diff_Hits'] = df['Hits'] - (df['CHits'] / df['Years'])
df['NEW_Diff_HmRun'] = df['HmRun'] - (df['CHmRun'] / df['Years'])
df['NEW_Diff_Runs'] = df['Runs'] - (df['CRuns'] / df['Years'])
df['NEW_Diff_RBI'] = df['RBI'] - (df['CRBI'] / df['Years'])
df['NEW_Diff_Walks'] = df['Walks'] - (df['CWalks'] / df['Years'])

# Career Avgs
# df["NEW_Mean_Hits"] = df['CHits'] / df['Years']
df["NEW_Mean_HmRun"] = df['CHmRun'] / df['Years']
df["NEW_Mean_Runs"] = df['CRuns'] / df['Years']
# df["NEW_Mean_CRBI"] = df['CRBI'] / df['Years']
# df["NEW_Mean_CWalks"] = df['CWalks'] / df['Years']


df["New_CSLG"] = ((4 * df["CHmRun"]) + df["CRuns"]) / df["CAtBat"]
df["New_COBP"] = (df["CHits"] + df["CWalks"] + (df["CHits"] / df["CAtBat"])) \
                 / (df["CAtBat"] + df["CWalks"] + (df["CHits"] / df["CAtBat"]) +
                    (df["CRBI"] - df["CRuns"]))
df["New_COPS"] = df["New_CSLG"] + df["New_COBP"]

# we are dropping  careers sums because we will use career avg instead.
# In addition we are dropping batting statistics because percantages and hits together include that meaning.
df_dropped = df[["AtBat", "CAtBat", "CHits", "CHmRun", "CRuns", "CRBI", "CWalks"]]
# df = df[[col for col in df.columns if col not in ["Hits",
#                                                   "CHits"]]]

# grab new col names after feature extractions.
cat_vars, num_vars, cat_but_car = pt.grab_col_name(df)

# Feature Scaling; rescaling all numeric features between [0,1]
ss = StandardScaler()
for column in [col for col in num_vars if ((df[col].nunique() > 2) & (col != "Salary"))]:
    df[column] = ss.fit_transform(df[[column]])

df["New_Assister"] = [1 if df["Assists"][row] > 2.5 else 0 for row in list(df.index)]

# lof again
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(df[[col for col in num_vars if col not in ['Hits', 'CHits']]])

df_scores = clf.negative_outlier_factor_
df_scores[df_scores < -3]
th = np.sort(df_scores)[7]


##################################################################################################
##################################################################################################
##################################################################################################
# Now lets deploy model with final version of data.
LinR = LinearRegression()

X = df.drop('Salary', axis=1)
y = df[["Salary"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

reg_model.predict(pd.DataFrame(X_test.iloc[1]).T)

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
reg_model.score(X_test, y_test)

# cross validation score.
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))



# compare results with test values , check results that mse > 350.
y_test.reset_index(inplace=True)
y_test.drop("index", axis=1, inplace=True)
X_test.reset_index(inplace=True)
X_test.drop("index", axis=1, inplace=True)
compare_df = pd.merge(y_test, pd.DataFrame(y_pred), left_index=True, right_index=True)
compare_df["diff"] = abs(compare_df["Salary"] - compare_df[0])
X_test.iloc[list(compare_df[compare_df["diff"] > 350].index)]
compare_df.iloc[list(compare_df[compare_df["diff"].abs() > 350].index)].sort_values(by="diff", ascending=False)


#RMSE = 306