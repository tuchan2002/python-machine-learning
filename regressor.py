import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from ydata_profiling import ProfileReport

data = pd.read_csv("StudentScore.xls", delimiter=",")
# profile = ProfileReport(data, title="Score Report", explorative=True)
# profile.to_file("score.html")

target = "writing score"

x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# imputer = SimpleImputer(missing_values=-1, strategy="median")
# x_train[["math score", "reading score"]] = imputer.fit_transform(x_train[["math score", "reading score"]])
# scaler = StandardScaler()
# x_train[["math score", "reading score"]] = scaler.fit_transform(x_train[["math score", "reading score"]])

num_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
  ("scaler", StandardScaler()),
])
# result = num_transformer.fit_transform(x_train[["math score", "reading score"]])
# for i, j in zip(x_train[["math score", "reading score"]].values, result):
#   print("Before {}. After {}".format(i, j))

education_values = ['some high school', 'high school', 'some college',"associate's degree", "bachelor's degree", "master's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values])),
])
# result = ord_transformer.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])
# for i, j in zip(x_train[["parental level of education", "gender", "lunch", "test preparation course"]].values, result):
#   print("Before {}. After {}".format(i, j))

nom_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OneHotEncoder()),
])
# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
# for i, j in zip(x_train[["race/ethnicity"]].values, result):
#   print("Before {}. After {}".format(i, j))

preprocessor = ColumnTransformer(transformers=[
  ("num_feature", num_transformer, ["reading score", "math score"]),
  ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
  ("nom_feature", nom_transformer, ["race/ethnicity"]),
])

reg = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("model", LinearRegression()),
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

for i, j in zip(y_predict, y_test):
  print("Predicted Value {}. Actual Value {}".format(i, j))