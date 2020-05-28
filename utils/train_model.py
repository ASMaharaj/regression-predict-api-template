"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
riders = pd.read_csv('data/riders.csv')
train = train.merge(riders, how='left', on='Rider Id')

def to_datetime(df):
    times = []
    for col in df.columns.values:
        if col.endswith('Time'):
            times.append(col)
        else:
            pass
    df[times] = df[times].apply(lambda x: pd.to_datetime(x, format='%I:%M:%S %p') )
    for i in times:
        df[i] = df[i].dt.time
        df[i] = df[i].apply(lambda x: 3600 * int(str(x)[0:2]) + 60 * int(str(x)[3:5]) + int(str(x)[7:]))
        
    return df

def impute_nan(input_df):
    
    from pandas.api.types import is_numeric_dtype
    
    def imp_mean(d):
        return d.fillna(round(d.mean(),1))
    
    def imp_mode(d):
        A = [x for x in d if pd.notnull(x) == True]
        most = max(list(map(A.count, A)))
        m = sorted(list(set(filter(lambda x: A.count(x) == most, A))))
        return d.fillna(m[0])
    
    for col in input_df.columns.values:
        if is_numeric_dtype(input_df[col]) == True:
            input_df[col] = input_df[col].transform(imp_mean)
        else:
            input_df[col] = input_df[col].transform(imp_mode)
            
    return input_df

train = to_datetime(train)

train.drop(columns = ['Vehicle Type', 'User Id', 'Precipitation in millimeters','Arrival at Destination - Day of Month','Arrival at Destination - Weekday (Mo = 1)', 'Arrival at Destination - Time', 'Rider Id'], inplace = True)

train = impute_nan(train)

x_cols = list(train.drop('Time from Pickup to Arrival', axis = 1).columns)
X = train[x_cols]
y = train['Time from Pickup to Arrival']

X['Personal or Business'] = pd.get_dummies(X['Personal or Business'], drop_first=True)
df1dum = pd.get_dummies(X['Platform Type'], prefix = 'Plat', drop_first = True)
X = pd.concat([X, df1dum], axis=1)

X.reset_index(drop=True, inplace=True)
X.set_index('Order No', inplace = True)
X.drop(columns=['Platform Type'], inplace = True)

from sklearn.model_selection import train_test_split
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=1)

# Fit model

gbr = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1592, loss='ls', max_depth=3,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=275,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=60, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
print ("Training Model...")
gbr.fit(X_train, y_train)

# Pickle model for use within our API
save_path = 'D:/kopan/Documents/EDSA/Machine Learning Sprint/Zindi_competition/regression-predict-api-template/assets/trained-models/gbr_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(gbr, open(save_path,'wb'))
