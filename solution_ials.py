import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def main(input_path: str, output_path: str):
    train = pd.read_parquet(input_path)

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(train['user_id'].unique())}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(train['item_id'].unique())}

    user_idx = train['user_id'].map(user_id_to_idx).to_numpy()
    item_idx = train['item_id'].map(item_id_to_idx).to_numpy()

    alpha = 10

    weights = {
        'view': 1,
        'time': 1,
        'like': 3,
        'share': 4,
        'bookmark': 2,
        'click_on_author': 2,
        'open_comments': 1
    }

    r_ui = (
        weights['view']
        + weights['time'] * np.log(1 + train['timespent']) / np.log(256)
        + weights['like'] * train['like'].astype(int)
        + weights['share'] * train['share'].astype(int)
        + weights['bookmark'] * train['bookmark'].astype(int)
        + weights['click_on_author'] * train['click_on_author'].astype(int)
        + weights['open_comments'] * train['open_comments'].astype(int)
    )

    s_ui = np.where(train['dislike'], -(1 + alpha * r_ui), 1 + alpha * r_ui)

    R = csr_matrix(
        (s_ui, (user_idx, item_idx)),
        shape=(len(user_id_to_idx), len(item_id_to_idx))
    )

    R.sum_duplicates()
    R.eliminate_zeros()

    ials = AlternatingLeastSquares(
        factors=64,
        regularization=0.1,
        iterations=20,
        calculate_training_loss=True,
        random_state=42
    )

    print("Training iALS..")
    ials.fit(R)
    print("iAlS trained.")

    print("Compute recommendations..")

    idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}

    recommendations = []

    for user_id, user_idx in user_id_to_idx.items():
        item_indices, scores = ials.recommend(
            userid=user_idx,
            user_items=R[user_idx],
            N=10,
            filter_already_liked_items=True
        )

        for item_idx, score in zip(item_indices, scores):
            recommendations.append({
                'user_id': user_id,
                'recs': idx_to_item_id[item_idx]
            })

    result = pd.DataFrame(recommendations, columns=["user_id", "recs"])

    print(f"Save result to file {output_path}.")
    result.to_csv(output_path, index=False)
    print("File saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender arguments.")
    parser.add_argument("--input_path", type=str, required=True, help="Input path to train parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to csv with recommendations")

    args = parser.parse_args()
    main(args.input_path, args.output_path)