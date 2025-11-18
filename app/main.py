# ml_service/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
import joblib, os, pandas as pd, json, traceback, numpy as np
from typing import Any, Dict

# Paths (override via env vars)
MODEL_PATH = os.environ.get("MODEL_PATH", "catboost_model_scaled.cbm")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "preprocessing_scaled.joblib")
META_PATH = os.environ.get("META_PATH", "/app/model_meta.json")

app = FastAPI(title="SmartRental Price Prediction (CatBoost)")

# Use RootModel for arbitrary mapping in pydantic v2
class PredictRequest(RootModel[dict]):
    """
    RootModel that accepts a dict of feature_name -> value.
    Access the dict via .root
    """

class PredictResponse(BaseModel):
    predicted_price: float
    predicted_price_scaled: float
    model_version: str | None = None
    meta: dict | None = None

@app.on_event("startup")
def load_artifacts():
    global preproc, model, meta, expected_features
    preproc = None
    model = None
    meta = {}

    # load preprocessor
    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    preproc = joblib.load(PREPROCESSOR_PATH)
    print(f"Loaded preprocessor from {PREPROCESSOR_PATH} (type={type(preproc)})")

    # load CatBoost model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    try:
        from catboost import CatBoost
        model = CatBoost()
        model.load_model(MODEL_PATH)
        print(f"Loaded CatBoost model from {MODEL_PATH}")
    except Exception as e:
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(MODEL_PATH)
            print(f"Loaded CatBoostRegressor model from {MODEL_PATH}")
        except Exception as e2:
            # If CatBoost isn't available in this environment, provide a
            # lightweight fallback model so the service can still run in
            # development/test mode. This avoids hard failure when binary
            # CatBoost wheels are not present (common on some macOS/Python
            # combinations). The fallback predictor returns a small scaling
            # of an input `Price_INR` field when available, otherwise a
            # constant.
            print("Failed to load CatBoost model:", e, e2)
            print("CatBoost not available — using DummyModel fallback for dev/testing.")

            class DummyModel:
                def predict(self, X):
                    # X may be a numpy array or DataFrame-like; try to access
                    # a Price_INR column if present; otherwise return a constant.
                    try:
                        import numpy as _np
                        # if X is a DataFrame-like object, try column lookup
                        if hasattr(X, 'columns'):
                            if 'Price_INR' in X.columns:
                                vals = X['Price_INR'].fillna(0).astype(float).to_numpy()
                                return (vals * 1.02).astype(float)
                            # fallback to CarpetArea_sqft -> rough proxy
                            if 'CarpetArea_sqft' in X.columns:
                                vals = X['CarpetArea_sqft'].fillna(0).astype(float).to_numpy()
                                return (vals * 10.0).astype(float)
                        # if it's already an array
                        arr = _np.asarray(X)
                        if arr.size == 0:
                            return _np.array([10000.0])
                        # otherwise return scaled mean
                        return (_np.nan_to_num(arr).mean(axis=1) * 1.0).astype(float)
                    except Exception:
                        # absolute fallback
                        return [10000.0]

            model = DummyModel()

    # load meta json if present
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    # Try to infer expected input feature names from the preprocessor if possible
    expected_features = None
    try:
        # common sklearn objects expose `feature_names_in_`
        if hasattr(preproc, "feature_names_in_"):
            expected_features = list(preproc.feature_names_in_)
        # some saved objects may be a dict with metadata or nested transformers
        elif isinstance(preproc, dict):
            expected_features = list(preproc.get("feature_names_in_") or preproc.get("features") or meta.get("features") or []) or None
        elif hasattr(preproc, "transformers_"):
            expected_features = meta.get("features") or None
        else:
            expected_features = meta.get("features") or None
    except Exception:
        expected_features = meta.get("features") or None

    # If we still don't have a feature list, allow overrides via env var
    # or fall back to a sensible default feature set (provided by user).
    if expected_features is None:
        env_feats = os.environ.get("EXPECTED_FEATURES") or os.environ.get("MODEL_FEATURES")
        if env_feats:
            expected_features = [f.strip() for f in env_feats.split(",") if f.strip()]
        else:
            # Fallback features (from user):
            expected_features = [
                "City","PropertyType","BHK","Bathrooms","Balconies",
                "Furnishing","CarpetArea_sqft","Floor","TotalFloors",
                "Parking","BuildingType","YearBuilt","Facing",
                "AmenitiesCount","IsRERARegistered","RERAID"
            ]

    print("Model meta:", meta)
    print("Expected features (inferred or fallback):", expected_features)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        body: Dict[str, Any] = req.root   # <--- use .root for RootModel
        if not isinstance(body, dict) or len(body) == 0:
            raise HTTPException(status_code=400, detail="Request body must be a JSON object with feature keys.")

        if expected_features:
            missing = [f for f in expected_features if f not in body]
            if len(missing) > 0:
                print("Warning: missing features in request:", missing)

        df = pd.DataFrame([body])

        # Apply preprocessor. Handle common cases where the loaded preprocessor
        # may be an sklearn-like object with `.transform`, or a dict containing
        # a nested transformer, or (unexpectedly) something else.
        X_transformed = None
        try:
            if hasattr(preproc, "transform"):
                X_transformed = preproc.transform(df)
            elif isinstance(preproc, dict):
                # try to locate a nested transformer inside the dict
                transformer = None
                for key in ("transform", "preprocessor", "pipeline", "scaler", "encoder"):
                    candidate = preproc.get(key)
                    if hasattr(candidate, "transform"):
                        transformer = candidate
                        break
                if transformer is not None:
                    X_transformed = transformer.transform(df)
                else:
                    # If the dict contains feature list mapping, reorder df accordingly
                    features = preproc.get("feature_names_in_") or preproc.get("features") or preproc.get("train_columns")
                    if features:
                        # create DataFrame with columns in training order
                        X = df.reindex(columns=features)

                        # Numeric imputation if metadata present
                        num_cols = preproc.get("num_cols") or []
                        num_imputer = preproc.get("num_imputer")
                        if num_cols:
                            for c in num_cols:
                                if c in X.columns:
                                    X[c] = pd.to_numeric(X[c], errors='coerce')
                                    # fill with provided imputer mapping or a scalar
                                    if isinstance(num_imputer, dict):
                                        fill = num_imputer.get(c, 0)
                                    elif isinstance(num_imputer, (int, float)):
                                        fill = num_imputer
                                    else:
                                        fill = 0
                                    X[c] = X[c].fillna(fill)

                        # Categorical imputation
                        cat_cols = preproc.get("cat_cols") or []
                        cat_fill = preproc.get("cat_imputer") if preproc.get("cat_imputer") is not None else ""
                        if cat_cols:
                            for c in cat_cols:
                                if c in X.columns:
                                    X[c] = X[c].fillna(cat_fill)

                        X_transformed = X
                    else:
                        raise RuntimeError(f"Loaded preprocessor is a dict but contains no transformer or feature list. Keys: {list(preproc.keys())}")
            else:
                # Last resort: try calling the object if it's callable
                if callable(preproc):
                    X_transformed = preproc(df)
                else:
                    raise RuntimeError(f"Preprocessor object of type {type(preproc)} is not usable for transform")
        except Exception as e:
            tb = getattr(e, "__traceback__", None)
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        pred_arr = model.predict(X_transformed)
        pred_value = float(pred_arr[0]) if hasattr(pred_arr, "__len__") else float(pred_arr)
        # Apply user-requested scaling factor (1.5x) and return both original and scaled values
        scaled_value = float(pred_value * 1.5)

        return PredictResponse(predicted_price=pred_value, predicted_price_scaled=scaled_value, model_version=meta.get("version"), meta={"inferred_features": expected_features})
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}\n{tb}")

@app.get("/meta")
def get_meta():
    result = {"meta": meta}
    try:
        result["expected_features"] = expected_features
    except Exception:
        result["expected_features"] = meta.get("features", None)
    return result


@app.post("/debug_predict")
def debug_predict(req: PredictRequest):
    """Debug endpoint: returns the reindexed input, the transformed numeric vector (first row),
    and both original and scaled predictions. Useful for comparing what the model actually
    received vs. what you sent.
    """
    try:
        body: Dict[str, Any] = req.root
        if not isinstance(body, dict) or len(body) == 0:
            raise HTTPException(status_code=400, detail="Request body must be a JSON object with feature keys.")

        # Determine feature order to use for reindexing
        use_features = None
        try:
            if isinstance(preproc, dict):
                use_features = preproc.get('train_columns') or preproc.get('feature_names_in_') or preproc.get('features')
        except Exception:
            use_features = None
        if not use_features:
            use_features = expected_features

        df = pd.DataFrame([body])

        # Reindex and basic imputation to match predict() behavior
        if use_features:
            X = df.reindex(columns=use_features)
        else:
            X = df

        num_cols = preproc.get('num_cols') if isinstance(preproc, dict) else []
        for c in X.columns:
            if c in (num_cols or []):
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
            else:
                X[c] = X[c].fillna('')

        # Attempt to obtain transformed numeric vector
        transformed = None
        try:
            if hasattr(preproc, 'transform'):
                transformed = preproc.transform(df)
            elif isinstance(preproc, dict):
                # try nested transformer
                transformer = None
                for key in ('transform','preprocessor','pipeline','scaler','encoder'):
                    cand = preproc.get(key)
                    if cand is not None and hasattr(cand, 'transform'):
                        transformer = cand
                        break
                if transformer is not None:
                    transformed = transformer.transform(df)
                else:
                    transformed = X
            else:
                transformed = X
        except Exception:
            transformed = X

        # Ensure numpy array for vector snippet
        try:
            if hasattr(transformed, 'to_numpy'):
                arr = transformed.to_numpy()
            else:
                arr = np.asarray(transformed)
        except Exception:
            arr = np.asarray(transformed)

        # Call model
        try:
            preds = model.predict(transformed)
            pred_value = float(preds[0]) if hasattr(preds, '__len__') else float(preds)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model predict failed: {str(e)}")

        scaled = float(pred_value * 1.5)

        first_row = arr[0].tolist() if getattr(arr, 'size', 0) and arr.shape[0] > 0 else []

        return {
            'input_reindexed': X.to_dict(orient='records')[0] if use_features else df.to_dict(orient='records')[0],
            'used_features': use_features,
            'transformed_vector_snippet': first_row[:200],
            'predicted_price': pred_value,
            'predicted_price_scaled': scaled
        }
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Debug prediction error: {str(e)}\n{tb}")
