
import time

# Third-party
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline

# scikit-learn
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Encoders
from category_encoders import TargetEncoder



id_col = "userId"
target_col = "clickedCTA"  # as specified in the case
numeric_features = ["estimatedAnnualIncome", "visitCount", "scrollDepth"]
categorical_features = ["sessionReferrer", "browser", "deviceType", "estimatedPropertyType", "ctaCopy", "ctaPlacement"]
date_features = ['date']
text_features = ['pageURL','editorialSnippet']


def _date_to_dow_2d(X):
    s = pd.Series(np.asarray(X).ravel(), dtype="string").str.strip()
    s = s.replace("", pd.NA)  # treat empty as missing
    dt = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    dow = dt.dt.day_name().fillna("(unknown)")
    return dow.to_numpy().reshape(-1, 1)


def preprocessing(df):# Identify columns
    all_cols = df.columns.tolist()
    post_click_cols = ["submittedform","scheduledappointment","mortgagevariation","revenue"]

    # Target & identifier

    # Candidate features (exclude target, post-click, and id)
    feature_cols = [c for c in all_cols if c not in set([target_col]) | set(post_click_cols) | set([id_col])]

    # Preprocessors
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # for date convert to day of the week and one hot encode

    date_transformer = Pipeline(steps=[
        ("to_dow", FunctionTransformer(_date_to_dow_2d, validate=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    #for larger text use target encoding
    text_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("targetencode", TargetEncoder())
    ])

    #building preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("dow", date_transformer, date_features),
            ("text", text_transformer, text_features)
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return preprocessor, feature_cols

def train_model(df):
    #models to compare and their hyperparameters to tune
    model_name = "neural net (MLP)"
    model_params={"model": Pipeline([
                ("nn", MLPClassifier(
                    activation="relu",
                    solver="adam",
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=5,
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "clf__nn__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "clf__nn__alpha": [1e-4, 1e-3],
                "clf__nn__learning_rate_init": [1e-3, 5e-4]
            }
        }
    preprocessor, feature_cols = preprocessing(df)
    X = df[feature_cols]
    y = df[target_col].values
    #seperate validation set for probability calibrations
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)



    print(f"\nRunning GridSearchCV for {model_name} ...")

    steps = [
        ("preprocessor", preprocessor),
        ("sampler", "passthrough"), #random over sampling to treat imbalanced data
        ("clf", model_params["model"]),
    ]
    pipeline = ImbPipeline(steps)

    grid = GridSearchCV(
        pipeline,
        param_grid=model_params["params"],
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1
    )

    # Training latency
    start_train = time.time()

    grid.fit(X_train, y_train)
    end_train = time.time()
    training_latency = end_train - start_train
    print('\ntraining time:',training_latency)

    best_model = grid.best_estimator_
    calibratedclf=CalibratedClassifierCV(FrozenEstimator(best_model), method='isotonic')
    calibratedclf.fit(X_val, y_val)
    best_model=calibratedclf

    print("\nFinished Training.")
    return best_model