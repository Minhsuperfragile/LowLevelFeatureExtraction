import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.constant import param_list
import xgboost as xgb

result_df = pd.read_csv("./ckpts/multiple_result.csv", index_col="features_name")
md = "md"
features_mark = 2 if md == "md" else 8

def process_confusion_matrix(label, pred):
    cm = confusion_matrix(label, pred)
    l = []

    for i in range(cm.shape[0]):
        class_accuracy = cm[i, i] / cm[i].sum() * 100
        l.append(class_accuracy)
        print(f"Class {i} Accuracy: {class_accuracy:.2f}%")

    return l

for param in param_list[:1]:

    # features_name = param['function'].__name__
    features_name = "vip"

    # Load training data from CSV
    train_file_path = f"./data/{features_name}/vaynen_train_{features_name}_new.csv"  # Change this to your actual file path
    train_data = pd.read_csv(train_file_path)

    # Assuming the first column is the label
    y_train = train_data.iloc[:, 1]  # Labels
    X_train = train_data.iloc[:, features_mark:]  # Features

    # Load test data from another CSV
    test_file_path = f"./data/{features_name}/vaynen_test_{features_name}_new.csv"  # Change this to your actual file path
    test_data = pd.read_csv(test_file_path)

    y_test = test_data.iloc[:, 1]  # Labels
    X_test = test_data.iloc[:, features_mark:]  # Features

    GBC = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    GBC.fit(X_train, y_train)
    GBC_y_pred = GBC.predict(X_test)
    GBC_accuracy = accuracy_score(y_test, GBC_y_pred)
    GBC_class_acc = process_confusion_matrix(y_test, GBC_y_pred)
    
    RFC = RandomForestClassifier(n_estimators=200, random_state=42)
    RFC.fit(X_train, y_train)
    RFC_y_pred = RFC.predict(X_test)
    RFC_accuracy = accuracy_score(y_test, RFC_y_pred)
    RFC_class_acc = process_confusion_matrix(y_test, RFC_y_pred)

    DTC = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)
    DTC.fit(X_train, y_train)
    DTC_y_pred = DTC.predict(X_test)
    DTC_accuracy = accuracy_score(y_test, DTC_y_pred)
    DTC_class_acc = process_confusion_matrix(y_test, DTC_y_pred)

    XGB = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=20,
    objective="multi:softmax",  # Use softmax for multi-class
    num_class=2,  # Dynamically set num_class
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    # verbose=True
)
    XGB.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],   # Validation data
    # early_stopping_rounds=50,        # Stop if no improvement in 50 rounds
    verbose=True                      # Show training progress
)
    XGB_y_pred = XGB.predict(X_test)
    XGB_accuracy = accuracy_score(y_test, XGB_y_pred)
    XGB_class_acc = process_confusion_matrix(y_test, XGB_y_pred)

    result_df.at[features_name, f'gradient_boost_{md}'] = GBC_accuracy*100
    result_df.at[features_name, f'random_forest_{md}'] = RFC_accuracy*100
    result_df.at[features_name, f'decision_tree_{md}'] = DTC_accuracy*100
    result_df.at[features_name, f'xgboost_{md}'] = XGB_accuracy*100

    result_df.at[f'{features_name}_0', f'gradient_boost_{md}'] = GBC_class_acc[0]
    result_df.at[f'{features_name}_0', f'random_forest_{md}'] = RFC_class_acc[0]
    result_df.at[f'{features_name}_0', f'decision_tree_{md}'] = DTC_class_acc[0]
    result_df.at[f'{features_name}_0', f'xgboost_{md}'] = XGB_class_acc[0]

    result_df.at[f'{features_name}_1', f'gradient_boost_{md}'] = GBC_class_acc[1]
    result_df.at[f'{features_name}_1', f'random_forest_{md}'] = RFC_class_acc[1]
    result_df.at[f'{features_name}_1', f'decision_tree_{md}'] = DTC_class_acc[1]
    result_df.at[f'{features_name}_1', f'xgboost_{md}'] = XGB_class_acc[1]

result_df.to_csv("./ckpts/multiple_result.csv")