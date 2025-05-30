import argparse
import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv

from utils.evaluator import GemniEvaluator, InternVLEvaluator
from utils.utils import load_jsonl
from utils.prompt import *
import statistics

load_dotenv()


def find_deepest_step_folders(base_folder):
    final_step_folders = []
    subdirs = [
        d for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d)) and d.isdigit()
    ]
    subdirs = sorted(subdirs, key=lambda x: int(x))

    for subdir in subdirs:
        subdir_path = os.path.join(base_folder, subdir)
        step_dirs = [
            d for d in os.listdir(subdir_path)
            if re.match(r'step\d+', d.lower()) and os.path.isdir(os.path.join(subdir_path, d))
        ]
        print(f"\n📁 Processing {subdir_path}")
        print(f"Found step dirs: {step_dirs}")
        if step_dirs:
            sorted_steps = sorted(
                step_dirs, key=lambda name: int(re.search(r'\d+', name).group())
            )
            final_step = sorted_steps[-1]
            final_step_folders.append(os.path.join(subdir_path, final_step))
    return final_step_folders


def get_latest_edited_image_file(folder):
    edited_images = []
    for file in Path(folder).glob("generated_image_*.jpg"):
        match = re.search(r'generated_image_(\d+)\.jpg', file.name)
        if match:
            edited_images.append((int(match.group(1)), file))
    if edited_images:
        return sorted(edited_images, key=lambda x: x[0], reverse=True)[0][1]
    return None


def extract_summary(text):
    match = re.search(r"\*\*Evaluation Summary\*\*:\s*(.+?)\n\*\*Score\*\*", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def save_results_to_json(index, image_path, object_results, avg_score, output_dir):
    result = {
        "idx": index,
        "image": str(image_path),
        "objects": object_results,
        "average_score": avg_score
    }
    with open(output_dir, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate final images.")
    parser.add_argument('--method', type=str, default="plan4gen_3", help="Method")
    parser.add_argument('--eval_folder', type=str, default="./eval", help="Path to the eval folder")
    parser.add_argument(
        '--object_file',
        type=str,
        default="data/instruction.jsonl",
        help="Path to the object .jsonl file"
    )

    parser.add_argument(
        '--evaluator',
        type=str,
        default="gemini-2.0-flash",
        choices=["gemini-2.0-flash", "OpenGVLab/InternVL3-78B"],
        help="Evaluator model name. Choose from: 'gemini-2.0-flash', 'OpenGVLab/InternVL3-78B'"
    )

    parser.add_argument(
        "--Gemni_API_Key",
        # required=True,
        type=str,
        nargs='+',
        help="Your Gemini API key for authentication. Keep it secure."
    )

    args = parser.parse_args()
    folder = f"./outputs/{args.method}"
    folders = find_deepest_step_folders(folder)
    data = load_jsonl(args.object_file)
    evaluator = args.evaluator

    os.makedirs(args.eval_folder, exist_ok=True)
    eval_file = os.path.join(args.eval_folder,
                             f"eval_{os.path.splitext(os.path.basename(args.object_file))[0]}_{args.method}.jsonl")

    if not folders or not data:
        print("❌ No folders or object descriptions found.")
        return

    if len(folders) != len(data):
        print(f"⚠️ Mismatch: {len(folders)} folders vs {len(data)} objects. Using minimum of the two.")
    if evaluator == "gemini-2.0-flash":
        evaluator = GemniEvaluator(model=args.Gemni_Model, api_keys=args.Gemni_API_Key)
    elif evaluator == "OpenGVLab/InternVL3-78B":
        evaluator = InternVLEvaluator(evaluator)

    all_avg_scores = []
    dimension_scores = {}

    for i, (folder, obj) in enumerate(zip(folders, data)):
        image_file = get_latest_edited_image_file(folder)
        if not image_file:
            print(f"⚠️ No image found in {folder}")
            continue
        label_entry = {}
        if isinstance(obj.get("label"), list) and len(obj["label"]) > 0:
            label_entry = obj["label"][0]
        else:
            print("⚠️ No valid 'label' entry found. Skipping.")
            continue

        if not isinstance(label_entry, dict):
            print("⚠️ Label entry is not a dictionary. Skipping.")
            continue

        total_score = 0
        object_results = []

        for category_name, category_desc in label_entry.items():
            try:
                prompt = evaluator_category_prompt.format(attribute_name=category_name,
                                                          attribute_description=category_desc)
                result_text, score = evaluator.evaluate(str(image_file), prompt)
                result = {
                    "category_name": category_name,
                    "description": category_desc,
                    "score": int(score),
                    "evaluation": result_text
                }
                object_results.append(result)
                print(result)
                if score != -1:
                    total_score += int(score)
                    if category_name not in dimension_scores:
                        dimension_scores[category_name] = []
                    dimension_scores[category_name].append(int(score))

                print(f"🧠 {category_name} — Score: {score}, Evaluation: {result_text}")
            except Exception as e:
                print(f"❌ Error evaluating '{category_name}': {e}")

        if object_results:
            avg_score = round(total_score / len(object_results), 2)
            all_avg_scores.append(avg_score)
            print(f"📊 Average Score: {avg_score}/5")
            save_results_to_json(i, image_file, object_results, avg_score, output_dir=eval_file)

    if all_avg_scores:
        overall_avg = round(sum(all_avg_scores) / len(all_avg_scores), 2)
        max_score = max(all_avg_scores)
        min_score = min(all_avg_scores)

        print("\n📈 Evaluation Summary Across All Samples:")
        print(f" - Samples Evaluated: {len(all_avg_scores)}")
        print(f" - Overall Average Score: {overall_avg}/5")
        print(f" - Highest Sample Average: {max_score}/5")
        print(f" - Lowest Sample Average: {min_score}/5")

        summary_data = {
            "samples_evaluated": len(all_avg_scores),
            "overall_average": overall_avg,
            "highest_average": max_score,
            "lowest_average": min_score
        }
        with open(eval_file, "a", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False)
            f.write("\n")
        print("\n📊 Per-Dimension Statistics:")
        dimension_summary = {}
        for category_name, scores in dimension_scores.items():
            avg_dim_score = round(sum(scores) / len(scores), 2)
            max_dim_score = max(scores)
            min_dim_score = min(scores)
            if len(scores) > 1:
                variance = round(statistics.variance(scores), 2)
                std_dev = round(statistics.stdev(scores), 2)
            else:
                variance = 0.0
                std_dev = 0.0

            dimension_summary[category_name] = {
                "samples_evaluated": len(scores),
                "average_score": avg_dim_score,
                "highest_score": max_dim_score,
                "lowest_score": min_dim_score,
                "variance": variance,
                "std_dev": std_dev
            }
            print(
                f" - {category_name}: {avg_dim_score}/5 (Evaluated: {len(scores)}, Max: {max_dim_score}, Min: {min_dim_score})")

        with open(eval_file, "a", encoding="utf-8") as f:
            json.dump({"per_dimension_summary": dimension_summary}, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
