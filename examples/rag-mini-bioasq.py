import argparse
import ast
import concurrent.futures
from collections import Counter

import pandas as pd
import weave
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

weave.init("bioasq-rag-data")


@weave.op()
def load_data(dataset_name: str = "rag-datasets/rag-mini-bioasq"):
    qap = load_dataset(dataset_name, "question-answer-passages")["test"]
    tc = load_dataset(dataset_name, "text-corpus")["passages"]
    return qap, tc


@weave.op()
def preprocess_data(qap, tc):
    qap_df = qap.to_pandas()
    qap_df["relevant_passage_ids"] = qap_df["relevant_passage_ids"].apply(
        ast.literal_eval
    )
    qap_df.rename(
        columns={"answer": "ground_truth"}, inplace=True
    )  # Rename column here
    tc_df = tc.to_pandas()

    tc_df["passage"] = tc_df["passage"].str.strip()
    nan_passage_ids = tc_df[tc_df["passage"].isin(["nan", "passage", "1."])][
        "id"
    ].tolist()

    qap_df["relevant_passage_ids"] = qap_df["relevant_passage_ids"].apply(
        lambda ids: [id for id in ids if id not in nan_passage_ids]
    )
    qap_df = qap_df[qap_df["relevant_passage_ids"].apply(bool)]
    tc_df = tc_df[~tc_df["passage"].isin(["nan", "passage", "1."])]

    return qap_df, tc_df


@weave.op()
def analyze_distribution(qap_df):
    passage_id_counts = Counter()
    for ids in qap_df["relevant_passage_ids"]:
        passage_id_counts.update(ids)
    return passage_id_counts


@weave.op()
def get_freq_bin(count, bins=[1, 2, 3, 4, 5, float("inf")]):
    for i in range(len(bins) - 1):
        if bins[i] <= count < bins[i + 1]:
            return f"{bins[i]}-{bins[i+1]}"
    return f"{bins[-1]}+"


@weave.op()
def select_passage_ids(passage_id_counts, sample_frac=0.015, random_state=42):
    passage_id_df = pd.DataFrame(
        {
            "id": list(passage_id_counts.keys()),
            "count": list(passage_id_counts.values()),
        }
    )
    passage_id_df["freq_bin"] = passage_id_df["count"].apply(get_freq_bin)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=sample_frac, random_state=random_state
    )
    _, selected_indices = next(splitter.split(
        passage_id_df, passage_id_df["freq_bin"]))
    return passage_id_df.iloc[selected_indices]["id"].tolist()


@weave.op()
def filter_passage_ids(ids, selected_passage_ids):
    return [id for id in ids if id in selected_passage_ids]


@weave.op()
def filter_qap_df(qap_df, selected_passage_ids):
    qap_df_filtered = qap_df.copy(deep=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        qap_df_filtered["relevant_passage_ids"] = list(
            executor.map(
                lambda ids: filter_passage_ids(ids, selected_passage_ids),
                qap_df_filtered["relevant_passage_ids"],
            )
        )
    qap_df_filtered = qap_df_filtered[
        qap_df_filtered["relevant_passage_ids"].apply(bool)
    ]
    return qap_df_filtered


@weave.op()
def create_weave_dataset(df, name):
    return weave.Dataset(name=name, rows=df.to_dict(orient="records"))


@weave.op()
def create_filtered_bioasq_datasets(args):

    qap, tc = load_data()
    qap_df, tc_df = preprocess_data(qap, tc)
    passage_id_counts = analyze_distribution(qap_df)
    selected_passage_ids = select_passage_ids(
        passage_id_counts, sample_frac=args.sample_frac, random_state=args.random_state
    )

    tc_df_filtered = tc_df[tc_df["id"].isin(selected_passage_ids)]
    qap_df_filtered = filter_qap_df(qap_df, selected_passage_ids)

    qap_train_df, qap_test_df = train_test_split(
        qap_df_filtered, test_size=0.2)

    tc_dataset_filtered = create_weave_dataset(
        tc_df_filtered, name="TextCorpusFiltered"
    )
    qap_train_dataset_filtered = create_weave_dataset(
        qap_train_df, name="QuestionAnswerPairsTrainFiltered"
    )
    qap_test_dataset_filtered = create_weave_dataset(
        qap_test_df, name="QuestionAnswerPairsTestFiltered"
    )

    weave.publish(tc_dataset_filtered)
    weave.publish(qap_train_dataset_filtered)
    weave.publish(qap_test_dataset_filtered)

    return {
        "text_corpus": tc_dataset_filtered,
        "qap_train": qap_train_dataset_filtered,
        "qap_test": qap_test_dataset_filtered,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process RAG mini BioASQ dataset")
    parser.add_argument(
        "--sample_frac", type=float, default=0.015, help="Fraction of samples to select"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility"
    )
    args = parser.parse_args()
    create_filtered_bioasq_datasets(args)
