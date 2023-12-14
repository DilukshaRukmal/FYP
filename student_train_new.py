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
from student_model_new import Knowledge_distiller, SimilarityLoss
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm
from transformers import BertModel, BertTokenizer



def data_loader(
    train_data_dir_1,
    train_data_dir_2,
    train_data_dir_3,
    valid_data_dir,
    batch_size,
    random_seed=38,
    valid_size=0.9,#train_size is not 0.1 it comes from totaly seperate domain
    shuffle=True,
):
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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

    train_dataset_1 = ImageFolder(root=train_data_dir_1, transform=transform)
    train_dataset_2 = ImageFolder(root=train_data_dir_2, transform=transform)
    train_dataset_3 = ImageFolder(root=train_data_dir_3, transform=transform)
    valid_dataset = ImageFolder(root=valid_data_dir, transform=transform)

    #train_dataset = train_dataset_1 + train_dataset_2 + train_dataset_3
    train_dataset = train_dataset_1

    train_indices, _ = train_test_split(
        list(range(len(train_dataset))),
        train_size=trainset_size,
        random_state=random_seed,
    )
    valid_indices, _ = train_test_split(
        list(range(len(valid_dataset))), train_size=valid_size, random_state=random_seed
    )

    # Create DataLoader for train and test sets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


if __name__ == "__main__":

    Image_size = 224
    trainset_size = 0.99
    Epochs = 10
    training_loss = []
    validation_enabled = True
    saved_teacher_net_path = "/data/ood/teacher_model_13_11/checkpoints/model_epoch_10.pt"

    # Define batch_size
    batch_size = 16

    DIR = {
        "P": "/data/ood/PACS/pacs_data/pacs_data/photo",
        "A": "/data/ood/PACS/pacs_data/pacs_data/art_painting",
        "C": "/data/ood/PACS/pacs_data/pacs_data/cartoon",
        "S": "/data/ood/PACS/pacs_data/pacs_data/sketch",
    }

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

    # set the relevant domain as train, valid sets
    train_dataloader, validation_dataloader= data_loader(
        DIR["P"],   #train
        DIR["A"],   #train
        DIR["C"],   #train
        DIR["P"],   #validation
        batch_size)

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
        if torch.backends.mps.is_available() #check whether multi process service is enabled
        else "cpu"
    )

    if torch.cuda.is_available():
        model = Knowledge_distiller(num_classes=7,model_path=saved_teacher_net_path)
        model = model.to("cuda:0")  # model moved to specified GPU
        print("Model loaded to GPU.")
    else:
        model = Knowledge_distiller(num_classes=7,model_path=saved_teacher_net_path)
        print("GPU is unavailable. model loaded to CPU.")

    criterion_1 = SimilarityLoss()  # Loss function for knowledge distillation
    criterion_2 = nn.CrossEntropyLoss()  # Loss function for classifier
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define a directory to save your model and checkpoints
    save_dir = "Model_checkpoints_for_distiller"

    # Make sure the directory exists, create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # training only student network (classifier-> frozen)
    for name, param in model.named_parameters():
        if name in ["student_net.fc.weight", "student_net.fc.bias"]:
            param.requires_grad = False

    logging.basicConfig(
        filename="Distillation.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("-------------------Training feature extractor of student---------------------")

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
                
            #print("Label {}".format(label))
            teacher_img_emb, student_img_emb,_ = model(image, label)

            # Calculating loss
            loss = criterion_1(teacher_img_emb, student_img_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss


        if i == Epochs - 2:
            epoch_flag = True

        training_loss.append(
            (epoch_loss / len(train_dataloader)).cpu().detach().numpy()
        )
        # Print average loss for a batch in each epoch
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

    # freezing feature extractor
    for name, param in model.named_parameters():
        if name in ["student_net.fc.weight", "student_net.fc.bias"]:
            param.requires_grad = True


    # Define a directory to save your model and checkpoints
    save_dir = "Model_checkpoints_for_classifier"

    # Make sure the directory exists, create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## Training for the classifier by freezing feature extractor
    training_loss = []

    logging.basicConfig(
        filename="Classification.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    Epochs  =20
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("-------------------Training classifier of student---------------------")

    for i in tqdm(range(Epochs)):
        epoch_loss = 0
        count = 0
        for batch, (image, label) in enumerate(tqdm(train_dataloader)):
            if torch.cuda.is_available():
                image = image.to("cuda:0")

            imgt,imgs, logits = model(image,label)
            logits = torch.squeeze(logits,0)
            if torch.cuda.is_available():
                label = label.to("cuda:0")
            loss = criterion_2(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss

        print(torch.argmax(logits,dim=1))
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
    if (validation_enabled):

        print("-------------------Validating the model---------------------")
        epoch_flag = False
        total_loss = 0
        correct = 0
        total = 0
        i=0
        for batch, (image, label) in enumerate(tqdm(validation_dataloader)):     
                # Moving Img_data to GPU
                if torch.cuda.is_available():
                    image = image.to("cuda:0")

                imgt,imgs,prediction = model(image,label)
                prediction =torch.squeeze(prediction,0)
                loss = criterion_2(prediction.cpu(), label)
                total += label.size(0)
                predict_labels = torch.argmax(prediction, dim=1)
                predict_labels = torch.tensor(predict_labels,dtype = int)
                predict_eval_list = [a == b for a, b in zip(predict_labels, label)]
                correct += sum(predict_eval_list)
                total_loss += loss.item()
        validation_loss = total_loss / len(validation_dataloader)
        validation_accuracy = 100 * correct / total
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.2f}%")

