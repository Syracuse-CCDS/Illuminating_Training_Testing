'''
Train and test a model against "gold data'

Requires:
    python3 -m pip install torch pytorch_pretrained_bert scikit-learn

To install locally you might need to use:
    python3 -m pip install torch pytorch_pretrained_bert scikit-learn --user
'''

import csv
import json
import logging
import math
import os
import pathlib
import random
import warnings

import numpy
import pytorch_pretrained_bert
import sklearn.metrics
import torch
import torch.utils.data

import illuminating.insomnia
import illuminating.utility


def ellipsis(text, length):
    return f"{text[:length]}..." if len(text) > length else text

def set_seed(seed):
    if not seed:
        print("** WARNING** Training is non-deterministic")
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_datasets(dataset_configs, training_pct, output_folder):
    template = "Dataset: \"{name}\" Rows: {row_count} True: {row_count_true} False: {row_count_false}"
    combined_gold = []
    combined_training = []
    combined_testing = []
    results = {
        "combined_gold": f"{output_folder}/Combined_gold.csv",
        "combined_training": f"{output_folder}/Combined_training.csv",
        "combined_testing": f"{output_folder}/Combined_testing.csv",
        "individual_datasets": []
    }

    for config in dataset_configs:
        name = config["name"]
        path = config["path"]
        text_key = config["text_key"]
        scoring_key = config["scoring_key"]
        scoring_map = config["scoring_map"]

        print_and_log(f"\tLoading \"gold\" data from \"{path}\"")
        print_and_log(f"\t\tTagging Key: \"{scoring_key}\" Text Key: \"{text_key}\"")
        with open(path, "r", encoding="utf-8") as file_in:
            reader = csv.DictReader(file_in)
            gold = standardize_gold_dataset(reader, scoring_key, text_key)
        print_and_log("\t\t" + template.format(**dataset_report("gold", gold)))

        print_and_log("\t\tExamples:")
        for row in gold[:5]:
            print_and_log(f"\t\t\t{row[1]} :: {ellipsis(row[2], 50)}")

        print_and_log(f"\t\tCreating training/testing splits in \"{output_folder}\"...")
        training, testing = create_train_test_split(gold, training_pct)
        training_filename = f"{name}_training.csv"
        testing_filename = f"{name}_testing.csv"
        training_filename_path = f"{output_folder}/{training_filename}"
        testing_filename_path = f"{output_folder}/{testing_filename}"

        print_and_log("\t\t" + template.format(**dataset_report(training_filename, training)))
        print_and_log("\t\t" + template.format(**dataset_report(testing_filename, testing)))

        with open(training_filename_path, "w", encoding="utf-8", newline="") as file_out:
            writer = csv.writer(file_out)
            writer.writerows(training)

        with open(testing_filename_path, "w", encoding="utf-8", newline="") as file_out:
            writer = csv.writer(file_out)
            writer.writerows(testing)

        results["individual_datasets"].append({
            "training": training_filename_path,
            "testing": testing_filename_path
        })
        combined_gold.extend(gold)
        combined_training.extend(training)
        combined_testing.extend(testing)

    random.shuffle(combined_gold)
    random.shuffle(combined_training)
    random.shuffle(combined_testing)

    with open(results["combined_gold"], "w", encoding="utf-8", newline="") as file_out:
        writer = csv.writer(file_out)
        writer.writerows(combined_gold)

    with open(results["combined_training"], "w", encoding="utf-8", newline="") as file_out:
        writer = csv.writer(file_out)
        writer.writerows(combined_training)

    with open(results["combined_testing"], "w", encoding="utf-8", newline="") as file_out:
        writer = csv.writer(file_out)
        writer.writerows(combined_testing)

    return results

def standardize_gold_dataset(gold_data, gold_scoring_column_key, gold_text_column_key):
    return [
        [
            index,
            str(row[gold_scoring_column_key]),
            row[gold_text_column_key]
        ]
        for index, row in enumerate(gold_data)
    ]

def create_train_test_split(
        gold_data,
        training_percent
    ):
    ## ----------------------
    # Cast gold data into essentials
    ## ----------------------
    refined_gold = {}
    for row in gold_data:
        gold_tag = row[1]
        gold_text = illuminating.utility.clean_text(row[2])
        refined_gold.setdefault(gold_tag, []).append(gold_text)
    ## ----------------------

    ## ----------------------
    # Create splits with the same proportions as the original gold data
    ## ----------------------
    train_data = []
    test_data = []
    for gold_tag, gold_texts in refined_gold.items():
        breakpoint = int((len(gold_texts) + 1) * training_percent)
        random.shuffle(gold_texts)
        train_data.extend([-1, gold_tag, ["a"], text] for text in gold_texts[:breakpoint])
        test_data.extend([-1, gold_tag, ["a"], text] for text in gold_texts[breakpoint:])
    ## ----------------------

    ## ----------------------
    ## reset the indexes
    ## ----------------------
    random.shuffle(train_data)
    random.shuffle(test_data)
    for index, row in enumerate(train_data):
        row[0] = index

    for index, row in enumerate(test_data):
        row[0] = index
    ## ----------------------

    return (train_data, test_data)

def dataset_report(name, data):
    return {
        "name": name,
        "row_count": len(data),
        "row_count_true": len(list(filter(lambda x: x[1] == "1", data))),
        "row_count_false": len(list(filter(lambda x: x[1] == "0", data)))
    }

def get_dataloader(
        data,
        guid_key,
        gold_scoring_map,
        max_seq_len,
        tokenizer,
        batch_size
    ):

    features = []
    for index, row in enumerate(data):
        example = illuminating.utility.InputExample(
            guid=f"{guid_key}-{ index }",
            text_a=row[3],
            text_b=None,
            label=row[1]
        )

        feature = illuminating.utility.convert_example_to_feature((
            example,
            gold_scoring_map,
            max_seq_len,
            tokenizer,
            "classification"
        ))

        features.append(feature)

    data = torch.utils.data.TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.input_mask for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        torch.tensor([f.label_id for f in features], dtype=torch.long)
    )
    sampler = torch.utils.data.SequentialSampler(data)
    return torch.utils.data.DataLoader(data, sampler=sampler, batch_size=batch_size, shuffle=False)

def get_optimizer(
        row_count,
        batch_size,
        gradient_accumulation_steps,
        epochs,
        model_parameters,
        learning_rate,
        warmup_proportion
    ):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    num_train_optimization_steps = int(math.ceil(row_count / batch_size) / gradient_accumulation_steps) * epochs
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model_parameters if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    return pytorch_pretrained_bert.optimization.BertAdam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        warmup=warmup_proportion,
        t_total=num_train_optimization_steps
    )

def get_eval_report(gold_labels, all_predictions):
    mcc = sklearn.metrics.matthews_corrcoef(gold_labels, all_predictions)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gold_labels, all_predictions).ravel()
    recall = tp / (tp + fn) if int(tp + fn) !=0 else None
    precision = tp / (tp + fp) if int(tp + fp) !=0 else None
    f1 = 2 / ((1 / recall) + (1 / precision)) if recall and precision else None
    return {
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "recall": round(recall, 4) if recall else None,
        "precision": round(precision, 4) if precision else None,
        "f1" : round(f1, 4) if f1 else None,
        "matthews_cc": round(mcc, 4) if mcc else None
    }

def init_logger():
    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging_file_handler = logging.FileHandler("./empty.log", delay=True)
    logging_file_handler.setFormatter(log_formatter)

    info_logger = logging.getLogger("print")
    info_logger.setLevel(logging.INFO)
    info_logger.addHandler(logging_file_handler)
    return info_logger

def print_and_log(*args, **kwargs):
    print(*args, **kwargs)
    logging.getLogger("print").info(args[0])

## ----------------------------
## Global Parameters
## https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
## ----------------------------
CACHE_DIR = "./Cache"
TORCH_DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN_EPOCHS = 4
BASE_MODEL = "bert-base-cased"
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
TRAINING_ROWS_PERCENT = 0.8
FN_LOSS = torch.nn.CrossEntropyLoss()
DATASET_REPORT_TEMPLATE = "Dataset: {name} Rows: {row_count} True: {row_count_true} False: {row_count_false}"
## ----------------------------

gold_dataset_configs = [
    {
        "name": "Gold_FB_Ads",
        "path" : "./Data/CTA_Subwork/2020_FB_Ads_CTA_GOLD.csv",
        "text_key" : "ad_creative_body",
        "scoring_key" : "Fundraising",
        "scoring_map": {"0": 0, "1": 1}
    },
    {
        "name": "Gold_FB_Posts",
        "path" : "./Data/CTA_Subwork/2020_FB_Posts_CTA_Sub_GOLD.csv",
        "text_key" : "body",
        "scoring_key" : "Fundraising",
        "scoring_map": {"0": 0, "1": 1}
    },
    {
        "name": "Gold_TW_Posts",
        "path" : "./Data/CTA_Subwork/2020_TW_Posts_CTA_Sub_GOLD.csv",
        "text_key" : "doc.text",
        "scoring_key" : "Fundraising",
        "scoring_map": {"0": 0, "1": 1}
    }
]
gold_scoring_map = {"0": 0, "1": 1}

fold_results_parent_folder = "./Data/CTA_Subwork"
fold_seeds = [2020, 2021, 2022, 2023, 2024]

if __name__ == "__main__":
    info_logger = init_logger()

    for fold_number, fold_seed in enumerate(fold_seeds, start=1):
        set_seed(fold_seed)
        fold_path = f"{fold_results_parent_folder}/fold_{fold_seed}"

        print("--------------------------------")
        print(f"## Creating Fold {fold_number} of {len(fold_seeds)} at \"{fold_path}\"...")
        print("--------------------------------")
        pathlib.Path(fold_path).mkdir(parents=True, exist_ok=True)

        ## ---------------------
        ## if the logger has a file open close it before switching
        ## ---------------------
        info_logger.handlers[0].close()
        info_logger.handlers[0].setStream(open(f"{fold_path}/report.log", mode="w"))
        ## ---------------------

        print_and_log("\nCREATING SPLITS...")
        datasets = build_datasets(gold_dataset_configs, TRAINING_ROWS_PERCENT, fold_path)

        print_and_log(f"\nTRAINING...")

        ## ----------------------
        ## reload the training data
        ## ----------------------
        print_and_log(f"\tLoading training Data: \"{datasets['combined_training']}\"")
        with open(datasets["combined_training"], "r", encoding="utf-8") as file_in:
            train_data = list(csv.reader(file_in))
        ## ----------------------

        ## ----------------------
        # Configure the Model
        ## ----------------------
        print_and_log(f"\tLoading the base model: \"{ BASE_MODEL }\"")
        tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR, do_lower_case=False)
        model = pytorch_pretrained_bert.BertForSequenceClassification.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR, num_labels=len(gold_scoring_map))
        model.to(TORCH_DEVICE_NAME)
        model.train()
        ## ----------------------

        ## ----------------------
        # Configure the training optimizer
        ## ----------------------
        print_and_log(f"\tConfiguring the optimizer")
        optimizer = get_optimizer(
            row_count=len(train_data),
            batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            epochs=NUM_TRAIN_EPOCHS,
            model_parameters=list(model.named_parameters()),
            learning_rate=LEARNING_RATE,
            warmup_proportion=WARMUP_PROPORTION
        )
        ## ----------------------

        ## ----------------------
        # Reshape training data into tensors for batch processing
        ## ----------------------
        print_and_log(f"\tConstructing tensors")
        train_dataloader = get_dataloader(
            data=train_data,
            guid_key="train",
            gold_scoring_map=gold_scoring_map,
            max_seq_len=MAX_SEQ_LENGTH,
            tokenizer=tokenizer,
            batch_size=TRAIN_BATCH_SIZE
        )
        ## ----------------------

        ## ----------------------
        ## Training
        ## ----------------------
        for epoc in range(NUM_TRAIN_EPOCHS):
            print_and_log(f"\tTraining epoc {epoc + 1} of {NUM_TRAIN_EPOCHS}")
            for index, (input_ids, input_mask, segment_ids, label_ids) in enumerate(train_dataloader):
                ## -------------------
                ## The model and data must be on the same device
                ## -------------------
                input_ids = input_ids.to(TORCH_DEVICE_NAME)
                input_mask = input_mask.to(TORCH_DEVICE_NAME)
                segment_ids = segment_ids.to(TORCH_DEVICE_NAME)
                label_ids = label_ids.to(TORCH_DEVICE_NAME)
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                ## -------------------

                loss = FN_LOSS(
                    logits.view(-1, 2),
                    label_ids.view(-1)
                )

                if GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                if (index + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    ## -------------------------------
                    ## May produce a depreciation warning:
                    ## "This overload of add_ is deprecated:""
                    ## -------------------------------
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        optimizer.step()
                        optimizer.zero_grad()
                    ## -------------------------------

                pct_complete = (index + 1) * TRAIN_BATCH_SIZE / len(train_data)
                pct_complete = (index + 1) / math.ceil(len(train_data) / TRAIN_BATCH_SIZE)
                pct_complete = min(100.0, 100.0 * pct_complete)
                print(f"\tCompleted: {round(pct_complete, 2)}% -- loss: {round(loss.item(), 4)}", end="\r", flush=True)
            print_and_log(f"\tCompleted: {round(pct_complete, 2)}% -- loss: {round(loss.item(), 4)}")
        ## ----------------------

        ### ---------------------
        ### Persist the trained model
        ### ---------------------
        print(f"\tSaving training artifacts")
        tokenizer.save_vocabulary(f"{ fold_path }")
        model_to_save = model.module if hasattr(model, "module") else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), f"{ fold_path }/pytorch_model.bin")
        model_to_save.config.to_json_file(f"{ fold_path }/config.json")
        ### ---------------------

        print_and_log(f"\nTESTING...")

        ## ----------------------
        # Configure the Model
        ## ----------------------
        print_and_log(f"\tLoading the trained model: \"{ fold_path }\"")
        tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(fold_path, do_lower_case=False)
        model = pytorch_pretrained_bert.BertForSequenceClassification.from_pretrained(fold_path, num_labels=len(gold_scoring_map))
        model.to(TORCH_DEVICE_NAME)
        model.eval()
        ## ----------------------

        ## ----------------------
        ## ----------------------
        ## ----------------------
        test_cases = datasets["individual_datasets"].copy()
        if len(datasets["individual_datasets"]) > 1:
            test_cases.append({"testing": datasets["combined_testing"]})

        for test_case in test_cases:
            test_data_path = test_case["testing"]
            test_data_path_base = pathlib.Path(test_data_path).with_suffix("")

            ## ----------------------
            ## ----------------------
            ## ----------------------
            print_and_log(f"\tLoading testing data: \"{test_data_path}\"")
            with open(test_data_path, "r", encoding="utf-8") as file_in:
                test_data = list(csv.reader(file_in))
            ## ----------------------

            ### ---------------------
            ### ---------------------
            ### ---------------------
            print_and_log(f"\t\tConstructing tensors")
            test_dataloader = get_dataloader(
                data=test_data,
                guid_key="test",
                gold_scoring_map=gold_scoring_map,
                max_seq_len=MAX_SEQ_LENGTH,
                tokenizer=tokenizer,
                batch_size=TEST_BATCH_SIZE
            )
            ### ---------------------

            ## ----------------------
            ## ----------------------
            ## ----------------------
            print_and_log(f"\t\tPerforming testing...")
            eval_loss = 0
            nb_eval_steps = len(test_dataloader)
            all_predictions = numpy.empty(shape=(0,2))
            for index, (input_ids, input_mask, segment_ids, label_ids) in enumerate(test_dataloader):
                ## -------------------
                ## The model and the data it opperates on must be on the same device
                ## -------------------
                input_ids = input_ids.to(TORCH_DEVICE_NAME)
                input_mask = input_mask.to(TORCH_DEVICE_NAME)
                segment_ids = segment_ids.to(TORCH_DEVICE_NAME)
                label_ids = label_ids.to(TORCH_DEVICE_NAME)
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)
                ## -------------------

                eval_loss += FN_LOSS(
                    logits.view(-1, len(gold_scoring_map)),
                    label_ids.view(-1)
                ).mean().item()

                batch_predictions = logits.detach().cpu().numpy()
                all_predictions = numpy.append(all_predictions, batch_predictions, axis=0)

                pct_complete = 100.0 * (index + 1) * TEST_BATCH_SIZE / len(test_data)
                pct_complete = round(min(100.0, pct_complete))
                print(f"\t\tCompleted: {pct_complete}%", end="\r", flush=True)
            print_and_log(f"\t\tCompleted: {pct_complete}%")
            final_eval_loss = eval_loss / nb_eval_steps
            ## ----------------------

            ## ----------------------
            ## ----------------------
            ## ----------------------
            gold_labels = []
            predictions = []
            evaluated_data = []
            for t, p in zip(test_data, all_predictions):
                index = t[0]
                gold_label = t[1]
                gold_text = t[3]
                prediction = str(numpy.argmax(p, axis=0))
                prediction_values = p.tolist()
                prediction_delta = round(prediction_values[1] - prediction_values[0], 2)

                evaluated_data.append([index, gold_label, prediction, prediction_delta, prediction_values, gold_text])
                gold_labels.append(gold_label)
                predictions.append(prediction)

            result = get_eval_report(gold_labels, predictions)
            result["final_eval_loss"] = round(final_eval_loss, 4)

            print_and_log(f"\t\tResults:")
            for key, value in result.items():
                print_and_log(f"\t\t\t{key:<20} = {value}")

            result_file_name = f"{test_data_path_base}_results.csv"
            print_and_log(f"\t\tSaving testing results to \"{result_file_name}\"")
            with open(result_file_name, "w", encoding="utf-8", newline="") as file_out:
                writer = csv.writer(file_out)
                writer.writerow(["index", "gold", "prediction", "prediction_delta", "prediction_details", "text"])
                writer.writerows(evaluated_data)

            result_file_name = f"{test_data_path_base}_results.txt"
            print_and_log(f"\t\tSaving testing results to \"{result_file_name}\"")
            with open(result_file_name, "w") as file_out:
                file_out.write(f"{ json.dumps(result, indent=4, default=str) }\n")
            ## ----------------------
