from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_once():
    csv_path = ROOT / "titanic.csv"
    try:
        df = sns.load_dataset("titanic")
        df.to_csv(csv_path, index=False)
        print("Loaded Titanic through Seaborn once and saved offline fallback.")
    except Exception:
        if not csv_path.exists():
            raise
        df = pd.read_csv(csv_path)
        print("Used existing offline titanic.csv fallback.")
    return df


def missing_report(df):
    miss = (df.isna().mean() * 100).loc[lambda x: x > 0]
    print("\nMissing percentages:\n", miss)
    return miss


def clean_eda(df):
    df = df.copy()
    miss = missing_report(df)
    # Under 5%: drop rows. 5%-30%: median/mode impute.
    for col, pct in miss.items():
        if pct < 5:
            df = df.dropna(subset=[col])
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
        else:
            # Retain high-missing columns as an explicit "missing" category.
            df[col] = df[col].fillna("missing") if not pd.api.types.is_numeric_dtype(df[col]) else df[col].fillna(df[col].median())
    return df


def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())


def eda(df):
    print("\nINFO"); df.info()
    print("\nDESCRIBE\n", df.describe(include="all"))
    print("\nSHAPE", df.shape)
    cleaned = clean_eda(df)

    for col in ["age", "fare"]:
        plt.figure()
        plt.hist(cleaned[col].dropna(), bins=30)
        plt.title(f"{col} histogram")
        plt.xlabel(col); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(OUT / f"{col}_hist.png"); plt.close()

        plt.figure()
        plt.boxplot(cleaned[col].dropna())
        plt.title(f"{col} box plot")
        plt.ylabel(col)
        plt.tight_layout(); plt.savefig(OUT / f"{col}_box.png"); plt.close()

    print("Age IQR outliers:", iqr_outliers(cleaned["age"]))
    print("Fare IQR outliers:", iqr_outliers(cleaned["fare"]))
    fare = cleaned["fare"]
    print("Fare mean:", fare.mean(), "median:", fare.median(), "mode:", fare.mode().iloc[0])

    for label, group in cleaned.groupby("sex"):
        print("Survival by sex:", label, group["survived"].mean())
    for label, group in cleaned.groupby("pclass"):
        print("Survival by pclass:", label, group["survived"].mean())
    print("\nSurvival by sex+pclass:\n", cleaned.groupby(["sex", "pclass"])["survived"].mean())

    cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = cleaned[cols].corr()
    print("\nCorrelation:\n", corr)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Titanic numeric correlation heatmap")
    plt.tight_layout(); plt.savefig(OUT / "correlation_heatmap.png"); plt.close()

    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            pairs.append((abs(corr.loc[a,b]), a, b, corr.loc[a,b]))
    print("Two strongest correlations:", sorted(pairs, reverse=True)[:2])

    charts = [
        ("survival_by_sex.png", cleaned.groupby("sex")["survived"].mean(), "Survival rate by sex"),
        ("survival_by_class.png", cleaned.groupby("pclass")["survived"].mean(), "Survival rate by class"),
    ]
    for filename, series, title in charts:
        plt.figure()
        series.plot(kind="bar")
        plt.title(title); plt.ylabel("survival rate")
        plt.tight_layout(); plt.savefig(OUT / filename); plt.close()

    plt.figure()
    sns.boxplot(data=cleaned, x="pclass", y="fare")
    plt.title("Fare by passenger class")
    plt.tight_layout(); plt.savefig(OUT / "fare_by_class.png"); plt.close()

    plt.figure()
    sns.barplot(data=cleaned, x="sex", y="survived", hue="pclass")
    plt.title("Survival by sex and class")
    plt.tight_layout(); plt.savefig(OUT / "survival_sex_class.png"); plt.close()

    for col in ["age", "fare"]:
        z = (cleaned[col] - cleaned[col].mean()) / cleaned[col].std()
        print(f"\n{col} before mean/std:", cleaned[col].mean(), cleaned[col].std())
        print(f"{col} after mean/std:", z.mean(), z.std())

    return cleaned


def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])


def classify(df):
    target = "survived"
    features = ["pclass","sex","age","sibsp","parch","fare","embarked"]
    X, y = df[features], df[target]
    print("\nClass balance:\n", y.value_counts(normalize=True))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200)
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:,1]
        results[name] = [
            accuracy_score(yte,pred), precision_score(yte,pred),
            recall_score(yte,pred), f1_score(yte,pred), roc_auc_score(yte,proba)
        ]
        print(name, confusion_matrix(yte,pred))
        fpr,tpr,_ = roc_curve(yte,proba)
        plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--")
        plt.title(f"ROC - {name}"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.tight_layout(); plt.savefig(OUT / (name.lower().replace(" ","_")+"_roc.png")); plt.close()

        if name == "Decision Tree":
            prep = pipe.named_steps["prep"]
            names = prep.get_feature_names_out()
            plt.figure(figsize=(16,9))
            plot_tree(pipe.named_steps["model"], feature_names=names,
                      class_names=["not survived","survived"], filled=False, max_depth=4)
            plt.tight_layout(); plt.savefig(OUT / "decision_tree.png"); plt.close()

    table = pd.DataFrame(results, index=["accuracy","precision","recall","f1","auc"]).T
    print("\nClassifier comparison:\n", table)
    table.to_csv(OUT / "classifier_comparison.csv")

    # Imbalance variants using the same split.
    variants = {}
    for label, estimator in [
        ("baseline", RandomForestClassifier(random_state=42, n_estimators=200)),
        ("class_weight_balanced", RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")),
    ]:
        p = Pipeline([("prep", build_preprocessor(Xtr)), ("model", estimator)])
        p.fit(Xtr,ytr); pred=p.predict(Xte)
        variants[label]=[precision_score(yte,pred), recall_score(yte,pred), f1_score(yte,pred)]
    smote_pipe = ImbPipeline([
        ("prep", build_preprocessor(Xtr)),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(random_state=42,n_estimators=200))
    ])
    smote_pipe.fit(Xtr,ytr); pred=smote_pipe.predict(Xte)
    variants["SMOTE"]=[precision_score(yte,pred),recall_score(yte,pred),f1_score(yte,pred)]
    print("\nImbalance comparison:\n", pd.DataFrame(variants,index=["precision","recall","f1"]).T)

    # Grid search with OOB enabled in final estimator.
    rf = RandomForestClassifier(random_state=42, oob_score=True)
    pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", rf)])
    grid = GridSearchCV(pipe, {
        "model__n_estimators":[100,200],
        "model__max_depth":[None,5,10],
        "model__max_features":["sqrt","log2"]
    }, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(Xtr,ytr)
    best = grid.best_estimator_
    print("Best RF params:", grid.best_params_)
    print("Best RF OOB score:", best.named_steps["model"].oob_score_)

    joblib.dump(best, OUT / "best_pipeline.joblib")
    reloaded = joblib.load(OUT / "best_pipeline.joblib")
    print("Reloaded raw-input prediction:", reloaded.predict(Xte.head(2)).tolist())

    return table, (Xtr,Xte,ytr,yte), best


def regression(df):
    # Fare is target; use other available features excluding fare and redundant target survived.
    features = ["pclass","sex","age","sibsp","parch","embarked","adult_male","alone"]
    data = df[features + ["fare"]].copy()
    X = data.drop(columns="fare")
    y = data["fare"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)
    pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("model", LinearRegression())])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    mae=mean_absolute_error(yte,pred)
    rmse=mean_squared_error(yte,pred,squared=False)
    r2=r2_score(yte,pred)
    n,p=Xte.shape[0], pipe.named_steps["prep"].transform(Xte).shape[1]
    adj=1-(1-r2)*(n-1)/(n-p-1) if n>p+1 else np.nan
    residuals=yte-pred
    plt.figure(); plt.scatter(pred,residuals); plt.axhline(0,linestyle="--")
    plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.title("Fare regression residuals")
    plt.tight_layout(); plt.savefig(OUT/"fare_residuals.png"); plt.close()
    print("\nRegression metrics:", {"MAE":mae,"RMSE":rmse,"R2":r2,"Adjusted_R2":adj})
    return {"MAE":mae,"RMSE":rmse,"R2":r2,"Adjusted_R2":adj}


def main():
    raw = load_once()
    cleaned = eda(raw)
    clf_table, split, best = classify(cleaned)
    reg = regression(cleaned)
    print("\nRegression metrics are a separate metric group from classifier metrics.")
    print("A written final recommendation should reference the measured classifier metrics rather than treating regression and classification metrics as directly comparable.")


if __name__ == "__main__":
    main()
