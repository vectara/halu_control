import os
import time
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import util

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load spacy model for word tokenization
nlp = spacy.load("en_core_web_sm")


def load_evaluation_model(model_path):
    """Load the evaluation model from the given path

    Args:
        model_path (str): Path to the evaluation model

    Returns:
        CrossEncoder: The evaluation model
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)
    return model

class ModelLoadingException(Exception):
    """Exception raised for errors in loading a model.

    Attributes:
        model_id (str): The model identifier.
        revision (str): The model revision.
    """

    def __init__(self, model_id, revision, messages="Error initializing model"):
        self.model_id = model_id
        self.revision = revision
        super().__init__(f"{messages} id={model_id} revision={revision}")


class SummaryGenerator:
    def __init__(self):
        self.summaries_df = pd.DataFrame()
        self.avg_length = None
        self.answer_rate = None
        self.exceptions = None
    
    def generate_summaries(self, df, gen_func):
        source, summary, dataset = [], [], []
        exceptions = []
        # df = df.head(10)
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            _source = row['text']
            _dataset = row['dataset']
            while True:
                try:
                    _summary = gen_func(_source)
                    break
                except Exception as e:
                    if 'Rate limit reached' in str(e):
                        wait_time = 3660
                        current_time = datetime.now().strftime('%H:%M:%S')
                        print(f"Rate limit hit at {current_time}. Waiting for 1 hour before retrying...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error at index {index}: {e}")
                        _summary = ""
                        exceptions.append(index)
                        break

            summary.append(_summary)
            source.append(_source)
            dataset.append(_dataset)

            # Sleep to prevent hitting rate limits too frequently
            # time.sleep(1)

        self.summaries_df = pd.DataFrame(list(zip(source, summary, dataset)),
                                        columns=["source", "summary", "dataset"])
        self.exceptions = exceptions
        self._compute_avg_length()
        self._compute_answer_rate()

        return self.summaries_df

    def _compute_avg_length(self):
        """
        Compute the average length of non-empty summaries using SpaCy.
        """
        total_word_count = 0
        total_count = 0

        for summary in self.summaries_df['summary']:
            if isinstance(summary, list):
                for item in summary:
                    if util.is_summary_valid(item):
                        doc = nlp(item)
                        words = [token.text for token in doc if token.is_alpha]
                        total_word_count += len(words)
                        total_count += 1
            elif util.is_summary_valid(summary):
                doc = nlp(summary)
                words = [token.text for token in doc if token.is_alpha]
                total_word_count += len(words)
                total_count += 1

        self.avg_length = 0 if total_count == 0 else total_word_count / total_count

    def _compute_answer_rate(self):
        """
        Compute the rate of non-empty summaries.
        """
        valid_count = sum(1 for summary in self.summaries_df['summary']
                            if util.is_summary_valid(summary))

        total_count = len(self.summaries_df)

        self.answer_rate = 0 if total_count == 0 else valid_count / total_count


class EvaluationModel:
    """A class to evaluate generated summaries.

    Attributes:
        model (CrossEncoder): The evaluation model.
        scores (list): List of evaluation scores.
        accuracy (float): Accuracy of the summaries.
        hallucination_rate (float): Rate of hallucination in summaries.
    """

    def __init__(self, model_path):
        """
        Initializes the EvaluationModel with a CrossEncoder model.

        Args:
            model_path (str): Path to the CrossEncoder model.
        """
        self.model = load_evaluation_model(model_path)
        self.scores = []
        self.factual_consistency_rate = None
        self.hallucination_rate = None

    def evaluate_hallucination(self, summaries_df):
        """
        Evaluate the hallucination rate in summaries. Updates the 'scores' attribute 
        of the instance with the computed scores.

        Args:
            summaries_df (DataFrame): DataFrame containing source docs and summaries.

        Returns:
            list: List of hallucination scores. Also updates the 'scores' attribute of the instance.
        """
        hem_scores = []
        full_scores = []
        source_summary_pairs = util.create_pairs(summaries_df)

        for doc, summary in tqdm(source_summary_pairs, desc="Evaluating hallucinations"):
            if util.is_summary_valid(summary):
                try:
                    if isinstance(summary, str):
                        score = float(self.model.predict([doc, summary], show_progress_bar=False))
                    elif isinstance(summary, list):
                        scores = self.model.predict([[doc, _sum] for _sum in summary], show_progress_bar=False)    
                        score = float(max(scores))
                    if not isinstance(score, float):
                        logging.warning(f"Score type mismatch: Expected float, got {type(score)}.")
                        continue
                    hem_scores.append(score)
                    full_scores.append(score)
                except Exception as e:
                    logging.error(f"Error while running HEM: {e}")
                    raise
            else:
                full_scores.append(float('nan'))

        self.scores = hem_scores
        return full_scores


    def compute_factual_consistency_rate(self, threshold=0.5):
        """
        Compute the factual consistency rate of the evaluated summaries based on
        the previously calculated scores. This method relies on the 'scores'
        attribute being populated, typically via the 'evaluate_hallucination' method.

        Returns:
            float: Factual Consistency Rate. Also updates the 'factual_consistency_rate'
            and 'hallucination_rate' attributes of the instance.

        Raises:
            ValueError: If scores have not been calculated prior to calling this method.
        """
        if not self.scores:
            error_msg = "Scores not calculated. Call evaluate_hallucination() first."
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Use threshold of 0.5 to compute factual_consistency_rate
        num_above_threshold = sum(score >= threshold for score in self.scores)
        num_total = len(self.scores)

        if not num_total:
            raise ValueError("No scores available to compute factual consistency rate.")

        self.factual_consistency_rate = (num_above_threshold / num_total) * 100
        self.hallucination_rate = 100 - self.factual_consistency_rate

        return self.factual_consistency_rate

def run_eval(input_csv, output_csv="hhem_eval.csv"):
    summ = SummaryGenerator()
    summ.summaries_df = pd.read_csv(input_csv)
    summ._compute_avg_length()
    summ._compute_answer_rate()
    hem = EvaluationModel("vectara/hallucination_evaluation_model")
    hscore = hem.evaluate_hallucination(summ.summaries_df)
    hrate = hem.compute_factual_consistency_rate()
    print("Average Length", summ.avg_length)
    print("Answer Rate", summ.answer_rate)
    print("Consistent Rate", hrate)
    summ.summaries_df.insert(0, "Score", hscore, allow_duplicates=True)
    summ.summaries_df.to_csv(output_csv, index=False)

def run_eval_TT(input_csv, output_csv="hhem_eval.csv"):
    summ = SummaryGenerator()
    summ.summaries_df = pd.read_csv(input_csv)
    summ._compute_avg_length()
    summ._compute_answer_rate()

    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model_path = 'google/t5_11b_trueteacher_and_anli'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda")

    hscore = []
    for premise, hypothesis in tqdm(zip(summ.summaries_df["source"],
                                   summ.summaries_df["summary"])):
        input_ids = tokenizer(
            f'premise: {premise} hypothesis: {hypothesis}',
            return_tensors='pt',
            truncation=True,
            max_length=2048).input_ids.to("cuda")
        outputs = model.generate(input_ids)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        judgement = result[0] == "1"
        hscore.append(judgement)

    print("Average Length", summ.avg_length)
    print("Answer Rate", summ.answer_rate)
    print("Consistent Rate", sum(hscore) / len(hscore) )
    summ.summaries_df.insert(0, "Score", hscore, allow_duplicates=True)
    summ.summaries_df.to_csv(output_csv, index=False)