import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn. preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

df = pd.read_csv('American_Housing_Data.csv')
null_values = df.isnull().sum()
print(null_values)
mode_income = df['Median Household Income'].dropna().mode()[0]
df['Median Household Income'] = df['Median Household Income'].fillna(mode_income)
cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
num_cols = [col for col in df.columns if df[col].dtypes != 'object']
print('Категориальные признаки:', cat_cols)
print('Числовые признаки:', num_cols)
df.describe()
print(df.info() )
for col in cat_cols:
     print(f'Признак : {col}.')
     print('Значения и их количество в наборе')
     print(df[col].value_counts())
corr_matrix = df[num_cols].corr()
fig = plt.figure(figsize=(18, 14))
g = sns.heatmap(corr_matrix, cmap=sns.cubehelix_palette(as_cmap=True), annot = True,fmt='.1g',cbar=False, annot_kws={'fontsize': 14})

sns.set(style='darkgrid')
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
sns.histplot(df['Price'], bins=30, ax=axs[0, 0], kde=True)
axs[0, 0].set_title('Price Distribution')
sns.countplot(x='Beds', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Beds Distribution')
sns.countplot(x='Baths', data=df, ax=axs[1, 0])
axs[1, 0].set_title('Baths Distribution')
sns.histplot(df['Living Space'], bins=30, ax=axs[1, 1], kde=True)
axs[1, 1].set_title('Living Space Distribution')
sns.histplot(df['Zip Code Population'], bins=30, ax=axs[2, 0], kde=True)
axs[2, 0].set_title('Zip Code Population Distribution')
sns.histplot(df['Median Household Income'], bins=30, ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Median Household Income Distribution')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.scatterplot(x='Living Space', y='Price', data=df, color='orange', alpha=1)
plt.title('Взаимосвязь между пространством и стомостью')
plt.xlabel('Жилое пространство')
plt.ylabel('Стоимость')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Median Household Income', y='Price', data=df, color='green', alpha=1)
plt.title('Взаимосвязь между доходом в районе и стоимостью')
plt.xlabel('Доход')
plt.ylabel('Стоимость')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Baths', y='Price', data=df, color='red', alpha=1)
plt.title('Взаимосвязь между количеством ванных комнат и стоимостью')
plt.xlabel('Количество ванных комнат')
plt.ylabel('Стоимость')
plt.show()

for col in num_cols:
     minLimit = df[col].quantile(0.025)
     maxLimit = df[col].quantile(0.975)
     df=df.loc[(df[col]>=minLimit)&(df[col]<=maxLimit)]
df.to_csv('American_Housing_Data_With_Quantile.csv', index=False)
sns.set(style='darkgrid')
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
sns.histplot(df['Price'], bins=30, ax=axs[0, 0], kde=True)
axs[0, 0].set_title('Price Distribution')
sns.countplot(x='Beds', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Beds Distribution')
sns.countplot(x='Baths', data=df, ax=axs[1, 0])
axs[1, 0].set_title('Baths Distribution')
sns.histplot(df['Living Space'], bins=30, ax=axs[1, 1], kde=True)
axs[1, 1].set_title('Living Space Distribution')
sns.histplot(df['Zip Code Population'], bins=30, ax=axs[2, 0], kde=True)
axs[2, 0].set_title('Zip Code Population Distribution')
sns.histplot(df['Median Household Income'], bins=30, ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Median Household Income Distribution')
plt.tight_layout()
plt.show()
df_with_quantile_without_objects = pd.read_csv('American_Housing_Data_With_Quantile.csv')
df_with_quantile_with_objects = pd.read_csv('American_Housing_Data_With_Quantile.csv')
df_with_quantile_without_objects.drop('Address', axis=1, inplace=True)
df_with_quantile_without_objects.drop('County', axis=1, inplace=True)
df_with_quantile_without_objects.drop('City', axis=1, inplace=True)
df_with_quantile_without_objects.drop('State', axis=1, inplace=True)
df_with_quantile_without_objects.to_csv('American_Housing_Data_With_Quantile2.csv', index=False)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Living Space', y='Price', data=df, color='orange', alpha=1)
plt.title('Взаимосвязь между пространством и стомостью')
plt.xlabel('Жилое пространство')
plt.ylabel('Стоимость')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Median Household Income', y='Price', data=df, color='green', alpha=1)
plt.title('Взаимосвязь между доходом в районе и стоимостью')
plt.xlabel('Доход')
plt.ylabel('Стоимость')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Baths', y='Price', data=df, color='red', alpha=1)
plt.title('Взаимосвязь между количеством ванных комнат и стоимостью')
plt.xlabel('Количество ванных комнат')
plt.ylabel('Стоимость')
plt.show()
cat_cols.remove('Address')
cat_cols.remove('County')
cat_cols.remove('City')
df_with_quantile_with_objects.drop('Address', axis=1, inplace=True)
df_with_quantile_with_objects.drop('County', axis=1, inplace=True)
df_with_quantile_with_objects.drop('City', axis=1, inplace=True)


df_with_binar = pd.get_dummies(df_with_quantile_with_objects, prefix=cat_cols)
cat_cols2 = [col for col in df_with_binar.columns]
print('cat_cols2',cat_cols2)
df_with_binar_replaced = df_with_binar.replace({True: 1, False: 0})
df_with_binar_replaced.to_csv('American_Housing_Data_With_Quantile_And_Binar.csv', index=False)

df.drop('Address', axis=1, inplace=True)
df.drop('County', axis=1, inplace=True)
df.drop('City', axis=1, inplace=True)

df_without_quantile_with_binar = pd.get_dummies(df, prefix=cat_cols)
cat_cols3 = [col for col in df_without_quantile_with_binar.columns]
print('cat_cols3',cat_cols3)
df_without_quantile_with_binar_replaced = df_without_quantile_with_binar.replace({True: 1, False: 0})
df_without_quantile_with_binar_replaced.to_csv('American_Housing_Data_without_quantile_with_binar.csv', index=False)

# def linear(data,scaler = None):
#      X = data.drop('Price', axis=1)
#      y = data['Price']
#      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#      model = LinearRegression()
#      params = {
#          'fit_intercept': [True, False],
#          'copy_X': [True, False],
#          'n_jobs': [-1, 1, 2, 4],
#          'positive': [False]
#      }
#      if scaler is not None:
#           scaler = scaler
#           X=scaler.fit_transform(X)
#      grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
#      grid_search.fit(X_train, y_train)
#      best_params = grid_search.best_params_
#      final_model = LinearRegression(**best_params)
#      final_model.fit(X_train, y_train)
#      y_pred = final_model.predict(X_test)
#      print("Лучшие параметры:", best_params)
#      print('linear_r2_score:', r2_score(y_test, y_pred))
#      print('linear_mean_squared_error',mean_squared_error(y_test, y_pred))
#      print('linear_mean_abs_error',mean_absolute_error(y_test, y_pred))

# linear(df_with_quantile_without_objects,StandardScaler())
# linear(df_with_quantile_without_objects,MinMaxScaler())
# linear(df_with_quantile_without_objects)
#
# linear(df_without_quantile_with_binar,StandardScaler())
# linear(df_without_quantile_with_binar,MinMaxScaler())
# linear(df_without_quantile_with_binar)
#
# linear(df_with_binar,StandardScaler())
# linear(df_with_binar,MinMaxScaler())
# linear(df_with_binar)

# def catboost(data,scaler = None):
#     X = data.drop('Price', axis=1)
#     y = data['Price']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = CatBoostRegressor()
#     params = {
#          'iterations': [300, 200, 100],
#          'learning_rate': [0.01, 0.1, 0.2],
#          'depth': [4, 6, 8]
#     }
#     if scaler is not None:
#         scaler = scaler
#         X = scaler.fit_transform(X)
#     grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     final_model = CatBoostRegressor(**best_params)
#     final_model.fit(X_train, y_train)
#     y_pred = final_model.predict(X_test)
#     print("Лучшие параметры:", best_params)
#     print('catboost_r2 score:', r2_score(y_test, y_pred))
#     print('catboost_Mean squared error:', mean_squared_error(y_test, y_pred))
#     print('catboost_Mean absolute error:', mean_absolute_error(y_test, y_pred))

# catboost(df_with_quantile_without_objects)
# catboost(df_without_quantile_with_binar)
# catboost(df_with_binar)

# catboost(df_with_quantile_without_objects,StandardScaler())
# catboost(df_with_quantile_without_objects,MinMaxScaler())
# catboost(df_with_quantile_without_objects)

# catboost(df_without_quantile_with_binar,StandardScaler())
# catboost(df_without_quantile_with_binar,MinMaxScaler())
# catboost(df_without_quantile_with_binar)
#
# catboost(df_with_binar,StandardScaler())
# catboost(df_with_binar,MinMaxScaler())
# catboost(df_with_binar)
#
# def decision_tree(data, scaler=None, max_depth=None, min_samples_split=2, min_samples_leaf=1):
#     X = data.drop('Price', axis=1)
#     y = data['Price']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = DecisionTreeRegressor()
#     params = {
#         'max_depth': [None, 5, 10],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2],
#         'max_leaf_nodes': [None, 10, 20],
#         'random_state': [0]
#     }
#     if scaler is not None:
#         scaler = scaler
#         X = scaler.fit_transform(X)
#     grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     final_model = DecisionTreeRegressor(**best_params)
#     final_model.fit(X_train, y_train)
#     y_pred = final_model.predict(X_test)
#     print("Лучшие параметры:", best_params)
#     print('decision_tree_r2 score:', r2_score(y_test, y_pred))
#     print('decision_tree_Mean squared error:', mean_squared_error(y_test, y_pred))
#     print('decision_tree_Mean absolute error:', mean_absolute_error(y_test, y_pred))
#
# def knn(data):
#     X = data.drop('Price', axis=1)
#     y = data['Price']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = KNeighborsRegressor()
#     params = {
#         'n_neighbors': [3, 5, 7],
#         'weights': ['uniform', 'distance'],
#         'algorithm': ['auto', 'brute'],
#         'leaf_size': [30, 50]
#     }
#     grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     final_model = KNeighborsRegressor(**best_params)
#     final_model.fit(X_train, y_train)
#     y_pred = final_model.predict(X_test)
#     print("Лучшие параметры:", best_params)
#     print('knn_r2 score:', r2_score(y_test, y_pred))
#     print('knn_Mean squared error:', mean_squared_error(y_test, y_pred))
#     print('knn_Mean absolute error:', mean_absolute_error(y_test, y_pred))
#
#
# def xgb(data):
#     X = data.drop('Price', axis=1)
#     y = data['Price']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = XGBRegressor()
#     params = {
#         'max_depth': [3, 6, 9],
#         'learning_rate': [0.1, 0.01],
#         'n_estimators': [100, 500],
#         'subsample': [0.8, 0.9],
#         'colsample_bytree': [0.6, 0.8],
#         'min_child_weight': [1, 3]
#     }
#     grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     final_model = XGBRegressor(**best_params)
#     final_model.fit(X_train, y_train)
#     y_pred = final_model.predict(X_test)
#     print("Лучшие параметры:", best_params)
#     print('xgb_r2 score:', r2_score(y_test, y_pred))
#     print('xgb_Mean squared error:', mean_squared_error(y_test, y_pred))
#     print('xgb_Mean absolute error:', mean_absolute_error(y_test, y_pred))
#
# def adaboost(data):
#     X = data.drop('Price', axis=1)
#     y = data['Price']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = AdaBoostRegressor()
#     params = {
#         'n_estimators': [50, 100],
#         'learning_rate': [0.1, 0.2],
#     }
#     grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     final_model = AdaBoostRegressor(**best_params)
#     final_model.fit(X_train, y_train)
#     y_pred = final_model.predict(X_test)
#     print("Лучшие параметры:", best_params)
#     print('adaboost_r2 score:', r2_score(y_test, y_pred))
#     print('adaboost_Mean squared error:', mean_squared_error(y_test, y_pred))
#     print('adaboost_Mean absolute error:', mean_absolute_error(y_test, y_pred))



# def random_forest(data):
#     X = data.drop('Price', axis=1)
#     y = data['Price']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestRegressor()
#     params = {
#         'n_estimators': [50, 100],
#         'max_features': ['sqrt'],
#         'max_depth': [2,7,15],
#         'min_samples_split': [3,23],
#         'min_samples_leaf': [3,6],
#         'bootstrap': [False]
#     }
#     grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     final_model = RandomForestRegressor(**best_params)
#     final_model.fit(X_train, y_train)
#     y_pred = final_model.predict(X_test)
#     print("Лучшие параметры:", best_params)
#     print('random_forest_r2 score:', r2_score(y_test, y_pred))
#     print('random_forest_Mean squared error:', mean_squared_error(y_test, y_pred))
#     print('random_forest_Mean absolute error:', mean_absolute_error(y_test, y_pred))
#
#
#
#
#
# decision_tree(df_with_quantile_without_objects,StandardScaler())
# decision_tree(df_with_quantile_without_objects,MinMaxScaler())
# decision_tree(df_with_quantile_without_objects)
#
# decision_tree(df_without_quantile_with_binar,StandardScaler())
# decision_tree(df_without_quantile_with_binar,MinMaxScaler())
# decision_tree(df_without_quantile_with_binar)
#
# decision_tree(df_with_binar,StandardScaler())
# decision_tree(df_with_binar,MinMaxScaler())
# decision_tree(df_with_binar)
#
#
# knn(df_with_quantile_without_objects)
# knn(df_without_quantile_with_binar)
# knn(df_with_binar)
#
#
# xgb(df_with_quantile_without_objects)
# xgb(df_without_quantile_with_binar)
# xgb(df_with_binar)
#
# adaboost(df_with_quantile_without_objects)
# adaboost(df_without_quantile_with_binar)
# adaboost(df_with_binar)
#
# random_forest(df_with_quantile_without_objects)
# random_forest(df_without_quantile_with_binar)
# random_forest(df_with_binar)
#



