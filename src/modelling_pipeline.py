import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_and_split_data(file_path, target_col, is_credit=False):
    df = pd.read_csv(file_path)
    
    if is_credit:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        scaler = StandardScaler()
        X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
        X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
        return X_train, X_test, y_train, y_test, None
    else:
        drop_cols = [target_col, 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df[target_col]
        
        top_countries = X['country'].value_counts().index[:15]
        X['country'] = X['country'].apply(lambda x: x if x in top_countries else 'Other')
        
        num_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'device_velocity_30m']
        cat_cols = ['source', 'browser', 'sex', 'country']
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        return X_train_trans, X_test_trans, y_train, y_test, preprocessor
