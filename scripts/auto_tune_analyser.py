import json, os, itertools
from analyser_relevancy import predict_relevancy
from evaluate_analyser_model import load_eval_set

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_FILE = os.path.join(ROOT, "data/best_params.json")

# Parameter grid to search
TYPE_WEIGHTS = [0.3, 0.5, 0.6, 0.75]
TOKEN_WEIGHTS = [0.2, 0.3, 0.4]
CENTERS = [0.85, 0.95, 1.05]
SCALES = [0.08, 0.12, 0.18]
REL_THRESHOLDS = [0.6, 0.7, 0.8]

def auto_tune():
    eval_set = load_eval_set()
    best_score = -1
    best_params = None
    results = []

    for tw, to, cen, sc, thr in itertools.product(
        TYPE_WEIGHTS, TOKEN_WEIGHTS, CENTERS, SCALES, REL_THRESHOLDS
    ):
        # temporarily override global params inside predict function
        import analyser_relevancy as AR
        AR.TYPE_WEIGHT = tw
        AR.TOKEN_WEIGHT = to
        AR.LOGISTIC_CENTER = cen
        AR.LOGISTIC_SCALE = sc
        AR.RELEVANCY_SCORE_THRESHOLD = thr

        # Evaluate
        total = len(eval_set)
        correct = 0

        for item in eval_set:
            result = AR.predict_relevancy(item["query"], top_k=3)
            if result["best_match"] and result["best_match"]["product_code"] == item["expected_product_code"]:
                correct += 1

        accuracy = correct / total
        results.append({
            "TYPE_WEIGHT": tw,
            "TOKEN_WEIGHT": to,
            "LOGISTIC_CENTER": cen,
            "LOGISTIC_SCALE": sc,
            "RELEVANCY_THRESHOLD": thr,
            "accuracy": accuracy
        })

        if accuracy > best_score:
            best_score = accuracy
            best_params = results[-1]

    with open(BEST_FILE, "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score, "best_params": best_params}, f, indent=2)

    print("Best Params:")
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    auto_tune()
