import numpy as np

# threshold is a parameter to mark the minimum required score difference to continue iterating.
def SFS(X_train, y_train, X_test, y_test, base_model, threshold=0.000001):
    best_features = set([])
    features = set(X_train.keys())

    max_score = -np.inf
    diff = abs(max_score)
    while len(features) > 0 and diff > threshold:
        cur_score = max_score

        # Indicator for a change in the final set.
        changed = False
        best_feature = None

        for feature in features:

            best_features.add(feature)
            # Train the model on the train set.
            base_model.fit((X_train[best_features]).astype(np.float), y_train)
            # Evaluate based on the test set.
            tmp_score = base_model.score(X_test[best_features], y_test)
            best_features.remove(feature)

            if max_score < tmp_score:
                max_score = tmp_score
                best_feature = feature
                changed = True

        # After evaluating the model with all the features, pick the best one.
        if changed:
            best_features.add(best_feature)
            features.discard(best_feature)

        diff = abs(max_score - cur_score)

    return best_features
