from datasets import load_dataset

# Replace 'your_dataset_name' and 'split' with the actual values
train_dataset = load_dataset("taylor-joren/uniref50", split='train')
val_dataset = load_dataset("taylor-joren/uniref50", split='validation')

def write_fasta(dataset, output_path, seq_key="sequence"):
    id_num = 0
    with open(output_path, "w") as f:
        for i, item in enumerate(dataset):
            seq = item[seq_key]
            seq_id = f"seq{id_num}"
            id_num += 1
            f.write(f">{seq_id}\n{seq}\n")

write_fasta(train_dataset, "train.fasta")
write_fasta(val_dataset, "val.fasta")
