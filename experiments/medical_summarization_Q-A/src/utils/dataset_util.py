from datasets import load_dataset, DatasetDict

class TargetDataset:
  def __init__(self, dataset_name):
    self.data_qa = load_dataset(dataset_name)    
    
    def split_dataset(self, data):
        # Perform a train-test split
        data = data["train"].train_test_split(test_size=0.1)

        # Further split the training data into training and validation sets
        test_data = data["test"]
        train_data = data["train"].train_test_split(test_size=0.1)
        valid_data = train_data["test"]
        train_data = train_data["train"]
        # Print dataset splits
        print("Train data:", len(train_data))
        print("Validation data:", len(valid_data))
        print("Test data:", len(test_data))
 
        data = DatasetDict({
            "train": train_data,
            "validation": valid_data,
            "test": test_data
        })
        return data
