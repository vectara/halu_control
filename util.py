def is_summary_valid(summary: str) -> bool:
    """
    Checks if the summary is valid.

    A summary is valid if it is not empty and contains at least five words. 
    
    Args:
        summary (str): The summary to check.

    Returns:
        bool: True if the summary is valid, False otherwise.
    """
    if isinstance(summary, str):
        words = summary.split()
        if len(words) >= 5:
            return True
    elif isinstance(summary, list):
        for item in summary:
            words = item.split()
            if len(words) >= 5:
                return True
    return False


def create_pairs(df):
    """
    Creates pairs of source and summary from the dataframe.

    Args:
        df (DataFrame): The dataframe containing source and summary columns.

    Returns:
        list: A list of pairs [source, summary].
    """
    pairs = []
    for _, row in df.iterrows():
        pairs.append([row['source'], row['summary']])

    return pairs


def format_results(model_name: str, revision: str, precision: str,
                factual_consistency_rate: float, hallucination_rate: float,
                answer_rate: float, avg_summary_len: float) -> dict:
    """
    Formats the evaluation results into a structured dictionary.

    Args:
        model_name (str): The name of the evaluated model.
        revision (str): The revision hash of the model.
        precision (str): The precision with which the evaluation was run.
        factual_consistency_rate (float): The factual consistency rate.
        hallucination_rate (float): The hallucination rate.
        answer_rate (float): The answer rate.
        avg_summary_len (float): The average summary length.

    Returns:
        dict: A dictionary containing the structured evaluation results.
    """
    results = {
        "config": {
            "model_dtype": precision, # Precision with which you ran the evaluation
            "model_name": model_name, # Name of the model
            "model_sha": revision # Hash of the model 
        },
        "results": {
            "hallucination_rate": {
                "hallucination_rate": hallucination_rate
            },
            "factual_consistency_rate": {
                "factual_consistency_rate": factual_consistency_rate
            },
            "answer_rate": {
                "answer_rate": answer_rate
            },
            "average_summary_length": {
                "average_summary_length": avg_summary_len
            },
        }
    }

    return results
