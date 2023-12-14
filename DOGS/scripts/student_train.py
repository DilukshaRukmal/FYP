import logging
import os
import warnings

warnings.filterwarnings("ignore")

import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from student_model_original import Knowledge_distiller, SimilarityLoss
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm
from transformers import BertModel, BertTokenizer


def data_loader(
    train_data_dir,
    valid_data_dir,
    test_data_dir,
    batch_size,
    random_seed=38,
    valid_size=0.1,
    shuffle=True,
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((Image_size, Image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = ImageFolder(root=train_data_dir, transform=transform)
    valid_dataset = ImageFolder(root=valid_data_dir, transform=transform)
    test_dataset = ImageFolder(root=test_data_dir, transform=transform)

    train_indices, _ = train_test_split(
        list(range(len(train_dataset))),
        train_size=trainset_size,
        random_state=random_seed,
    )
    valid_indices, _ = train_test_split(
        list(range(len(valid_dataset))), train_size=0.99, random_state=random_seed
    )

    # Create DataLoader for train and test sets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    DATASET_DIR = "/data/ood/PACS/pacs_data/pacs_data/art_painting/dog"
    Image_size = 224
    trainset_size = 0.5
    # Define batch_size
    batch_size = 16

    DIR = {
        "P": "/data/ood/PACS/pacs_data/pacs_data/photo",
        "A": "/data/ood/PACS/pacs_data/pacs_data/art_painting",
        "C": "/data/ood/PACS/pacs_data/pacs_data/cartoon",
        "S": "/data/ood/PACS/pacs_data/pacs_data/sketch",
    }

    # set the relevant domain as train, valid, test sets
    train_dataloader, validation_dataloader, test_dataloader = data_loader(
        DIR["P"], DIR["A"], DIR["S"], batch_size
    )

    # lists to save embeddings to do PCA
    text_embedding_global = []
    S_img_embedding_global = []
    T_img_embedding_global = []
    label_set = []
    epoch_flag = False  # flag to identify last epoch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if torch.cuda.is_available():
        model = Knowledge_distiller(num_classes=7)
        model = model.to("cuda:0")  # model moved to specified GPU
        print("Model loaded to GPU.")
    else:
        model = Knowledge_distiller(num_classes=7)
        print("GPU is unavailable. model loaded to CPU.")

    criterion_1 = SimilarityLoss()  # Loss function for knowledge distillation
    criterion_2 = nn.CrossEntropyLoss()  # Loss function for classifier
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # PACS classes
    labels = {
        "0": "Dog",
        "1": "Elephant",
        "2": "Giraffe",
        "3": "Guitar",
        "4": "Horse",
        "5": "House",
        "6": "Person",
    }

    # Define a directory to save your model and checkpoints
    save_dir = "Model_checkpoints_for_distiller"

    # Make sure the directory exists, create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # training only student network (classifier-> freeze)
    for name, param in model.named_parameters():
        if name in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    Epochs = 5
    training_loss = []
    threshold = 0.2  # threshold value for classify invariant and variant features

    logging.basicConfig(
        filename="Distillation.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    for i in tqdm(range(Epochs)):
        epoch_loss = 0
        count = 0
        for batch, (image, label) in enumerate(tqdm(train_dataloader)):
            if epoch_flag:
                label_set.append(label.detach().numpy())
            label_tensor = []
            for idx, val in enumerate(label):
                label_tensor.append(labels[str(label[idx].item())])

            # Moving Img_data to GPU
            if torch.cuda.is_available():
                image = image.to("cuda:0")
            
            print("Label {}".format(label_tensor))
            x, teacher_img_emb, student_img_emb = model(image, label_tensor, threshold)

            # Calculating loss
            loss = criterion_1(teacher_img_emb, student_img_emb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss

        if i == Epochs - 2:
            epoch_flag = True

        training_loss.append(
            (epoch_loss / len(train_dataloader)).cpu().detach().numpy()
        )
        print(f"Epoch {i+1}: Loss: {training_loss[-1]}\n\n")

        # Save the model and training checkpoint
        if (i + 1) % 2 == 0:  # Save every 2 epochs, adjust as needed
            # Save the model's state dictionary
            model_checkpoint_path = os.path.join(save_dir, f"model_epoch_{i+1}.pt")
            # torch.save(model.state_dict(), model_checkpoint_path)

            # Save training checkpoint information
            checkpoint = {
                "epoch": i + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": training_loss[-1],
            }
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{i+1}.pt")
            # torch.save(checkpoint, checkpoint_path)

            # Log the saved paths and other details
            logging.info(f"Epoch {i+1} - Model saved to: {model_checkpoint_path}")
            logging.info(f"Epoch {i+1} - Checkpoint saved to: {checkpoint_path}")

    logging.shutdown()

    final_model = model.student_net
    # freezing image encoder
    for name, param in model.named_parameters():
        if "cropped_resnet152" in name:
            param.requires_grad = False

    # Define a directory to save your model and checkpoints
    save_dir = "Model_checkpoints_for_classifier"

    # Make sure the directory exists, create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## Training for the classifier by freezing feature extractor
    Epochs = 5
    training_loss = []

    logging.basicConfig(
        filename="Classification.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    for i in tqdm(range(Epochs)):
        epoch_loss = 0
        count = 0
        for batch, (image, label) in enumerate(tqdm(train_dataloader)):
            if torch.cuda.is_available():
                image = image.to("cuda:0")

            y = final_model(image)
            if torch.cuda.is_available():
                label = label.to("cuda:0")

            loss = criterion_2(y[1], label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss

        training_loss.append(
            (epoch_loss / len(train_dataloader)).cpu().detach().numpy()
        )
        print(f"Epoch {i+1}: Loss: {training_loss[-1]}\n\n")

        # Save the model's state dictionary
        model_checkpoint_path = os.path.join(save_dir, f"model_epoch_{i+1}.pt")
        # torch.save(model.state_dict(), model_checkpoint_path)

        # Save training checkpoint information
        checkpoint = {
            "epoch": i + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": training_loss[-1],
        }
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{i+1}.pt")
        # torch.save(checkpoint, checkpoint_path)

        # Log the saved paths and other details
        logging.info(f"Epoch {i+1} - Model saved to: {model_checkpoint_path}")
        logging.info(f"Epoch {i+1} - Checkpoint saved to: {checkpoint_path}")

    logging.shutdown()

    ## Validation
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((Image_size, Image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    epoch_flag = False
    total_loss = 0
    correct = 0
    total = 0
    for batch, (image, label) in enumerate(tqdm(validation_dataloader)):
        label_tensor = []
        for idx, val in enumerate(label):
            label_tensor.append(labels[str(label[idx].item())])

        # Moving Img_data to GPU
        if torch.cuda.is_available():
            image = image.to("cuda:0")

        prediction = final_model(image)
        prediction = prediction[1].cpu()
        # Calculating loss
        loss = criterion_2(prediction, label)
        total += label.size(0)
        predict_labels = torch.argmax(prediction, dim=1)
        predict_eval_list = [a == b for a, b in zip(predict_labels, label)]
        correct += sum(predict_eval_list)
        total_loss += loss
    # Calculate and print validation loss and accuracy
    validation_loss = total_loss / len(validation_dataloader)
    validation_accuracy = 100 * correct / total
    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")

