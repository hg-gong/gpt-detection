import json

from sklearn.model_selection import train_test_split


def split_data(
    input_file,
    train_file,
    val_file,
    test_file,
    random_state=42,
    test_ratio=0.2,
    val_ratio=0.1,
):
    # Load data from input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Split data into training and test sets
    train_data, test_data = train_test_split(
        data, test_size=test_ratio, random_state=random_state
    )

    # Split training set into training and validation sets
    train_set, val_set = train_test_split(
        train_data, test_size=val_ratio, random_state=random_state
    )

    # Save data sets to JSON files
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=4)

    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=4)

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


def merge_data(input_file, output_file):
    # Load data from input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Create samples from data
    samples = []
    for d in data:
        for answer in d["human_answers"]:
            sample = {"question": d["question"], "text": answer, "fake": 0}
            samples.append(sample)

        for answer in d["chatgpt_answers"]:
            sample = {"question": d["question"], "text": answer, "fake": 1}
            samples.append(sample)

    # Save samples to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    input_file = "./dataset/all.jsonl"
    merged_file = "./dataset/merged.jsonl"
    train_file = "./dataset/train.json"
    val_file = "./dataset/val.json"
    test_file = "./dataset/test.json"

    # Merge data from input file and save to output file
    merge_data(input_file, merged_file)

    # Split merged data into training, validation, and test sets and save to files
    split_data(merged_file, train_file, val_file, test_file)


if __name__ == "__main__":
    main()
