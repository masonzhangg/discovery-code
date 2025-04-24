import pandas as pd
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
import re
import time
import sklearn.metrics as metrics

# accuracy score for sklearn metrics.
# pandas df comparison, need to drop the nulls.

load_dotenv()
client = OpenAI()

aging_types = [
    "successful aging",
    "productive aging",
    "healthy aging",
    "active aging",
    "vital aging",
    "conscious aging",
    "sustainable aging",
    "optimal aging",
    "effective aging",
    "independent aging",
    "joyful aging",
    "harmonious aging",
]

article_types = ["Emperical", "Review", "Theoretical/Conceptual", "Other"]


def clean_json_response(response_text):
    json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return f"Error: Unable to parse JSON. Raw response: {response_text}"


def articleManipulation(article_text):
    try:
        input_prompt = (
            f"Analyze the following journal paper abstract and extract the information in JSON format:\n\n"
            f"1. Determine if the abstract is relevant to gerontology in general (such as healthy aging), (answer with 1 FOR 'YES or 0 FOR 'NO').\n"
            f"2. Identify if the abstract is purely theoretical or conceptual (answer with 1 FOR 'YES or 0 FOR 'NO').\n"
            f"3. Identify if the abstract is a scoping review, systematic review, meta-analysis review, narrative review, or other type of review (answer with 1 FOR 'YES or 0 FOR 'NO')\n"
            f"4. Identify the country the abstract is based in (answer with 'COUNTRY' IN ALL CAPS). If there is no country mentioned, answer with 'NA'.\n"
            f"Abstract:\n{article_text}\n\n"
            f"Please return the response as a **valid JSON object** with the keys 'gerontology_related', 't_or_c', 'review', and 'origin_country' only. Do not include any other text, explanations, or formatting. The response should be JSON only.\n\n"
            f"Ensure the response is ONLY valid JSON without explanations, notes, or formatting outside of JSON.\n"
            f"Example JSON format:\n"
            f'{{\n  "aging_related": 1,\n  "theoretical_conceptual": 1,\n  "review_type": 0,\n  "country": "CHINA"\n}}\n\n'
        )

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "The following is a conversation with an AI assistant.",
                },
                {"role": "user", "content": input_prompt},
            ],
        )

        response_text = completion.choices[0].message.content.strip()
        return clean_json_response(response_text)

    except json.JSONDecodeError:
        return "An error occurred while parsing the JSON response"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def run_batch(articles, batch_size=10):
    results = []

    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        batch_results = []

        for article in batch:
            try:
                input_prompt = (
                    f"Analyze the following journal paper abstract and extract the information in JSON format:\n\n"
                    f"1. Determine if the abstract is relevant to gerontology in general (such as healthy aging), (answer with 1 FOR 'YES' or 0 FOR 'NO').\n"
                    f"2. Identify if the abstract is purely theoretical or conceptual (answer with 1 FOR 'YES' or 0 FOR 'NO').\n"
                    f"3. Identify if the abstract is a scoping review, systematic review, meta-analysis review, narrative review, or other type of review (answer with 1 FOR 'YES' or 0 FOR 'NO').\n"
                    f"4. Identify the country the abstract is based in (answer with 'COUNTRY' IN ALL CAPS). If there is no country mentioned, answer with 'NA'.\n\n"
                    f"Abstract:\n{article}\n\n"
                    f"Please return the response as a **valid JSON object** with the keys 'aging_related', 'theoretical_conceptual', 'review_type', and 'country' only. "
                    f"Ensure the response is ONLY valid JSON without explanations, notes, or formatting outside of JSON."
                )

                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "The following is a conversation with an AI assistant.",
                        },
                        {"role": "user", "content": input_prompt},
                    ],
                )

                response_text = completion.choices[0].message.content.strip()
                result = clean_json_response(response_text)
                batch_results.append(result)

            except Exception as e:
                batch_results.append({"error": str(e)})

            time.sleep(0.5)

        results.extend(batch_results)
        print(
            f"Processed batch {i//batch_size + 1}, {len(results)}/{len(articles)} abstracts completed"
        )

        time.sleep(2)

    return results

def calculate_multi_accuracy(results_file, validation_file):
    results_df = pd.read_csv(results_file)
    validation_df = pd.read_csv(validation_file, encoding="ISO-8859-1")

    num_rows = min(len(results_df), len(validation_df))
    results_df = results_df.iloc[:num_rows].copy()
    validation_df = validation_df.iloc[:num_rows].copy()

    model_columns = [
        "aging_related",
        "theoretical_conceptual",
        "review_type",
        "country",
        "draws_from_education_aghe",
        "draws_from_bss",
        "draws_from_biosci",
        "draws_from_hs",
        "draws_from_srpp",
        "draws_from_humanities",
        "empirical",
        "qualitative",
        "uses_interviews",
        "uses_observation",
        "uses_focus_groups",
        "uses_content_review",
        "quantitative",
        "uses_secondary_data",
        "uses_primary_data",
        "other_quant_method",
        "mixed_methods",
        "other_method",
    ]

    available_columns = []
    for column in model_columns:
        if column in results_df.columns and column in validation_df.columns:
            available_columns.append(column)
        else:
            print(f"Warning: Column '{column}' not found in both datasets, skipping.")

    accuracy_results = {}

    if "aging_related" in available_columns:
        correct_count = 0
        total_count = 0
        
        for i in range(num_rows):
            res_value = results_df["aging_related"].iloc[i]
            val_value = validation_df["aging_related"].iloc[i]
            
            if pd.isna(res_value) or pd.isna(val_value):
                continue
                
            res_value_str = str(res_value).strip()
            val_value_str = str(val_value).strip()
            
            if res_value_str == val_value_str:
                correct_count += 1
            total_count += 1
        
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        accuracy_results["aging_related"] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
        }
    
    for column in available_columns:
        if column == "aging_related":
            continue
            
        correct_count = 0
        total_count = 0

        for i in range(num_rows):
            if "aging_related" in validation_df.columns:
                val_aging = validation_df["aging_related"].iloc[i]
                if pd.isna(val_aging) or int(val_aging) != 1:
                    continue
            
            res_value = results_df[column].iloc[i]
            val_value = validation_df[column].iloc[i]

            if pd.isna(res_value) or pd.isna(val_value):
                continue

            if column == "country" and isinstance(res_value, str) and isinstance(val_value, str):
                res_value = res_value.strip().upper()
                val_value = str(val_value).strip().upper()

                if res_value in ["NA", "N/A"] and val_value in ["", "0", "NAN", "NA", "N/A"]:
                    correct_count += 1
                elif res_value == val_value:
                    correct_count += 1
                total_count += 1
            else:
                res_value_str = str(res_value).strip()
                val_value_str = str(val_value).strip()

                if res_value_str == val_value_str:
                    correct_count += 1
                total_count += 1

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        accuracy_results[column] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
        }

    total_correct = sum(
        metrics["correct_count"]
        for metrics in accuracy_results.values()
        if isinstance(metrics.get("correct_count"), (int, float))
    )
    total_comparisons = sum(
        metrics["total_count"]
        for metrics in accuracy_results.values()
        if isinstance(metrics.get("total_count"), (int, float))
    )
    overall_accuracy = (total_correct / total_comparisons * 100) if total_comparisons > 0 else 0

    accuracy_results["overall"] = {
        "accuracy": overall_accuracy,
        "correct_count": total_correct,
        "total_count": total_comparisons,
    }

    return accuracy_results

if __name__ == "__main__":
    try:
        print("\n=== EVALUATING EXISTING RESULTS ===")
        all_accuracies = calculate_multi_accuracy(
            "combined_results.csv", "validation_2025_binaryJoy.csv"
        )

        print("\n=== ACCURACY RESULTS ===")
        for column, metrics in all_accuracies.items():
            if column == "overall":
                continue
            if isinstance(metrics.get("accuracy"), (int, float)):
                print(
                    f"{column:<25}: {metrics['accuracy']:.2f}% ({metrics['correct_count']}/{metrics['total_count']})"
                )
            else:
                print(
                    f"{column:<25}: {metrics.get('accuracy')} {metrics.get('status', '')}"
                )

        if "overall" in all_accuracies:
            print("\n=== OVERALL ACCURACY ===")
            metrics = all_accuracies["overall"]
            print(
                f"Overall Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct_count']}/{metrics['total_count']})"
            )
    except Exception as e:
        print(f"Error processing existing results: {str(e)}")

    # ---------- Can Uncomment This Later ------------#
    #     try:
    #         validation_df = pd.read_csv(
    #             "validation_2025_binaryJoy.csv", encoding="ISO-8859-1"
    #         )

    #         if "Abstract" in validation_df.columns:
    #             abstracts = validation_df["Abstract"].dropna().tolist()

    #             print("\n=== RUNNING BATCH PROCESSING ===")
    #             batch_results = run_batch(abstracts, batch_size=20)
    #             pd.DataFrame(batch_results).to_csv("resultsbatch.csv", index=False)
    #             print("Batch results saved to resultsbatch.csv")

    #             print("\n=== EVALUATING BATCH RESULTS ===")
    #             batch_accuracies = calculate_multi_accuracy(
    #                 "resultsbatch.csv", "validation_2025_binaryJoy.csv"
    #             )

    #             print("\n=== BATCH ACCURACY RESULTS ===")
    #             for column, metrics in batch_accuracies.items():
    #                 if column == "overall":
    #                     continue
    #                 if isinstance(metrics.get("accuracy"), (int, float)):
    #                     print(
    #                         f"{column:<25}: {metrics['accuracy']:.2f}% ({metrics['correct_count']}/{metrics['total_count']})"
    #                     )
    #                 else:
    #                     print(
    #                         f"{column:<25}: {metrics.get('accuracy')} {metrics.get('status', '')}"
    #                     )

    #             if "overall" in batch_accuracies:
    #                 print("\n=== BATCH OVERALL ACCURACY ===")
    #                 metrics = batch_accuracies["overall"]
    #                 print(
    #                     f"Overall Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct_count']}/{metrics['total_count']})"
    #                 )

    #             print("\n=== RUNNING RANDOM SAMPLE PROCESSING ===")
    #             random_sample = random.sample(abstracts, k=min(10, len(abstracts)))
    #             individual_results = []

    #             for i, abstract in enumerate(random_sample, 1):
    #                 print(f"Processing abstract {i}/{len(random_sample)}")
    #                 result = articleManipulation(abstract)
    #                 individual_results.append(result)
    #                 print(json.dumps(result, indent=2))
    #                 time.sleep(2)

    #             pd.DataFrame(individual_results).to_csv("results.csv", index=False)
    #             print("Sampled results saved to results.csv")
    #         else:
    #             print(
    #                 "The required column 'Abstract' was not found in the validation CSV."
    #             )
    #     except Exception as e:
    #         print(f"Error processing validation file: {str(e)}")

    # except Exception as e:
    #     print(f"An error occurred in the main execution: {str(e)}")
