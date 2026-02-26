import copy

# For example: create a slightly different calibration for shadow evaluation
shadow_model = copy.deepcopy(model)

# Simulate hyperparameter variation or threshold experimentation
shadow_threshold = np.clip(THRESHOLD * 0.9, 0.05, 0.95)

def score_shadow_batch(df: pd.DataFrame) -> pd.DataFrame:
    X = validate_features(df)
    assert_feature_health(X)
    
    shadow_probs = shadow_model.predict_proba(X)[:, 1]
    shadow_preds = (shadow_probs >= shadow_threshold).astype(int)
    
    result = pd.DataFrame({
        "utc_timestamp": df["utc_timestamp"].values,
        "shadow_proba": shadow_probs,
        "shadow_pred": shadow_preds,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_hash": MODEL_HASH,
        "feature_version": FEATURE_VERSION,
        "shadow_threshold": shadow_threshold
    })
    
    # Compare with production predictions
    result["prod_pred"] = prod_preds["price_spike_pred"].values
    result["disagreement"] = (result["shadow_pred"] != result["prod_pred"]).astype(int)
    disagreement_rate = result["disagreement"].mean()
    
    print(f"Shadow disagreement rate vs prod: {disagreement_rate:.2%}")
    
    log_file = LOG_DIR / f"shadow_predictions_{datetime.now(timezone.utc).date()}.csv"
    log_predictions(result, log_file)
    
    return result


# Run shadow inference

shadow_preds = score_shadow_batch(feature_df)
display(shadow_preds.head())
print("Shadow deployment inference completed successfully")