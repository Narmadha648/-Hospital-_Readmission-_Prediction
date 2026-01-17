
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def train_model():
   
    df = pd.read_csv("Train.csv")
    
    target = "readmitted" 
    X = df.drop(target, axis=1)
    y = df[target]

   
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

   
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        pickle.dump(le_target, open("target_encoder.pkl", "wb"))

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    
    return model, X.columns, label_encoders
