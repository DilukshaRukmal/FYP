import logging
import os
import re
from string import punctuation

import model
import pandas as pd
import torch
import torchvision.transforms as transforms
from model import DOGS, ContrastiveLoss
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import BertTokenizer

captions = pd.read_csv("/home/diluksha/Dataset/flickr30k_images/results.csv", sep="|")

IMG_DIR = "/home/diluksha/Dataset/flickr30k_images/flickr30k_images"
IMG_SIZE = 224


def clean_text(row):
    regex = re.compile("[%s]" % re.escape(punctuation))
    row = str(row).strip()
    row = row.lower()
    return regex.sub("", row)


captions.columns = [col.strip() for col in captions.columns]
captions["comment"] = captions["comment"].apply(clean_text)

train_captions, test_captions = train_test_split(
    captions, train_size=0.8, random_state=42
)

statements = train_captions["comment"]


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_statements = [statement for statement in statements]

TRAIN_PATHS = list(train_captions["image_name"])
TEST_PATHS = list(test_captions["image_name"])


class InvalidDatasetException(Exception):
    def __init__(self, len_of_paths, len_of_labels):
        super().__init__(
            f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})"
        )


class CustomData(Dataset):
    def __init__(self, img_paths, caps):
        self.img_paths = [os.path.join(IMG_DIR, path) for path in img_paths]
        self.caps = caps
        self.transforms = transforms.Compose(
            [transforms.Resize(size=(IMG_SIZE, IMG_SIZE)), ToTensor()]
        )
        if len(self.img_paths) != len(self.caps):
            raise InvalidDatasetException(self.img_paths, self.caps)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        tensor_img = self.transforms(img)

        label = self.caps[idx]
        # label = torch.tensor(label)

        return tensor_img, label


train_set = CustomData(TRAIN_PATHS, tokenized_statements)
print(f"The number of images in the train set is : {train_set.__len__()}")

test_set = CustomData(TEST_PATHS, test_captions["comment"])
print(f"The number of images in the test set is : {test_set.__len__()}")

BATCH_SIZE = 4

torch.manual_seed(42)
train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

print(
    f"the size of the train dataloader {len(train_dataloader)} batches of {BATCH_SIZE}"
)

torch.manual_seed(42)
test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"the size of the test dataloader {len(test_dataloader)} batches of {BATCH_SIZE}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Check if GPU is available
if torch.cuda.is_available():
    # Create your model
    model = DOGS()
    # Move the model to the GPU
    model = model.to(
        "cuda:0"
    )  # You can also use model.to('cuda:0') for a specific GPU index

    for param in model.parameters():
        param.to("cuda:0")
    # Alternatively, you can use the .cuda() method
    # model = model.cuda()
else:
    model = DOGS()
    print("GPU is not available. Using CPU.")

# Count the learnable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total learnable parameters: {total_params}")

criterion = ContrastiveLoss(margin=0.8, max_violation=False)
optimizer = Adam(model.parameters(), lr=0.001)


# Define a directory to save your model and checkpoints
save_dir = "/data/ood/teacher_model_06_12/checkpoints"

# Make sure the directory exists, create it if it doesn't
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Configure logging
log_file = "training.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

Epochs = 10
training_loss = []
for i in tqdm(range(Epochs)):
    epoch_loss = 0
    count = 0
    for batch, (image, caption) in enumerate(train_dataloader):
        image = image.to("cuda:0")
        # caption = caption.to("cuda:0")
        score, attn, text_embeddings = model(image, caption)

        # print("Score  :", score)
        # print("Attan : ", attn)

        loss = criterion(score)

        if batch % 5000 == 0:
            # break
            print("Score  :", score)
            print("Attan : ", attn)
            print("Loss : ", loss)
            print(
                f"Looked at {batch * len(image)}/{len(train_dataloader.dataset)} samples."
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss

    training_loss.append((epoch_loss / len(train_dataloader)).cpu().detach().numpy())
    print(f"Epoch {i+1}: Loss: {training_loss[-1]}\n\n")

    # Save the model and training checkpoint
    if (i + 1) % 1 == 0:  # Save every 2 epochs, adjust as needed
        # Save the model's state dictionary
        model_checkpoint_path = os.path.join(save_dir, f"model_epoch_{i+1}.pt")
        torch.save(model.state_dict(), model_checkpoint_path)

        # Save training checkpoint information
        checkpoint = {
            "epoch": i + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": training_loss[-1],
        }
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{i+1}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Log the saved paths and other details
        logging.info(f"Epoch {i+1} - Model saved to: {model_checkpoint_path}")
        logging.info(f"Epoch {i+1} - Checkpoint saved to: {checkpoint_path}")

# Close the logging file handler
logging.shutdown()

