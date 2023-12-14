import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model import DOGS
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer

#Path to saved checkpoints of teacher model
model_path = "/data/ood/teacher_model_13_11/checkpoints/model_epoch_10.pt"

# Define the ProjectionHead class
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=512,  # Update the projection dimension
        dropout=0.1,
    ):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected  # skip connection
        x = self.layer_norm(x)  # Layer Normalization
        return x


# Student feature extractor + Classifier
class StudentNet(nn.Module):
    def __init__(self,num_classes, hidden_layers=512):
        super(StudentNet, self).__init__()

        # download resnet152 model
        model = models.resnet152(pretrained=True)
        self.cropped_resnet152 = torch.nn.Sequential(*list(model.children())[:-2])
        self.projection_head = ProjectionHead(2048, projection_dim=512, dropout=0.1)

        # for classifier
        self.fc = nn.Linear(hidden_layers, num_classes)

    def forward(self, image):
        img_embedding = self.cropped_resnet152(image)
        image_embedding = img_embedding.view(len(img_embedding), 2048, 49)
        image_embedding = image_embedding.permute(0, 2, 1)
        image_embedding = self.projection_head(image_embedding)
        image_embedding = image_embedding.permute(1, 0, 2)

        x = torch.mean(image_embedding, dim=0)
        x = self.fc(x)

        return image_embedding, x


# Cosine similarity between embeddings
def embedding_distance(text_emb, img_emb, thresh: float = 0.5):
    # Compute the Cosine similarity -> check for any better distance function

    # Expand dimensions to make the computation broadcastable-------------------------------------///////////
    text_emb_expanded = text_emb.unsqueeze(0)  # Shape: (1, 16, 512)
    img_emb_expanded = img_emb.unsqueeze(1)    # Shape: (49, 1, 512)

    # Compute cosine similarity
    cos_distance = F.cosine_similarity(text_emb_expanded, img_emb_expanded, dim=2)
    thresh_tens = torch.full(cos_distance.shape, thresh)
    if torch.cuda.is_available():
        thresh_tens = thresh_tens.to("cuda:0")
    #Need to check the comparison vector-----------------------------------------------------------///////////
    comparison = torch.ge(cos_distance, thresh_tens)
    return comparison


def Ideal_teacher_nn_output_matrix(
    text_embeddings, Image_embeddings, student_img_emb, threshold, epoch_flag=False
):
    # Check if epoch_flag is True
    if epoch_flag:
        # Assuming you have these lists defined globally somewhere in your code
        # text_embedding_global = []
        S_img_embedding_global = []
        # T_img_embedding_global = []
        # Append tensors to the global lists
        S_img_embedding_global.append(student_img_emb.cpu().detach().tolist())

    # Create a zero tensor with the same shape as Image_embeddings
    zero_tensor = torch.zeros_like(Image_embeddings)

    for i in range(Image_embeddings.shape[2]):
        region_em = Image_embeddings[:, :, i]

        # Assuming embedding_distance is a function that compares text_embeddings and region_em
        is_inside = embedding_distance(text_embeddings, region_em, threshold)

        # Use boolean indexing to copy values from region_em to zero_tensor
        # zero_tensor[:, :, i] = torch.where(
        #     is_inside.unsqueeze(1), region_em.unsqueeze(0), zero_tensor[:, :, i]
        # )

    return zero_tensor


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, teacher_em, student_em):
        batch_cs = torch.zeros(
            teacher_em.shape[0]
        )  # tensor to hold cs for each feature

        for i in range(teacher_em.shape[0]):
            feature_cs = F.cosine_similarity(
                teacher_em[i, :, :], student_em[i, :, :], dim=0
            )
            mean_cs = torch.mean(feature_cs)
            batch_cs[i] = mean_cs

        mean_batch_cs = torch.mean(batch_cs)
        final_loss = 1 - mean_batch_cs
        return final_loss


class Knowledge_distiller(nn.Module):
    def __init__(self, model_path="path_to_teacher", p_size=2, num_classes=2):
        super(Knowledge_distiller, self).__init__()

        self.student_net = StudentNet(num_classes=num_classes)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        #original large bert tokenizor
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # loading the teacher
        self.teacher_img_net = DOGS()       
        self.teacher_img_net.load_state_dict(
            torch.load(model_path, map_location="cuda:0")
        )
        #freezing the teacher
        for param in self.teacher_img_net.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        #Moving networks to GPU if available
        if torch.cuda.is_available():
            self.student_net = self.student_net.to("cuda:0")
            self.teacher_img_net = self.teacher_img_net.to("cuda:0")
        ##  self.text_encoder = self.text_encoder.to("cuda:0")

    def forward(self, image, label, threshold):
        # label = [
        #     self.tokenizer.encode(
        #         statement,
        #         add_special_tokens=True,
        #         padding="max_length",
        #         max_length=30,
        #         truncation=True,
        #     )
        #     for statement in label
        # ]

        # label = torch.tensor(label)

        # Move the token tensor to GPU if cuda available
        # if torch.cuda.is_available():
        #     label = label.to("cuda:0")

        score, teacher_img_emb, txt_emb = self.teacher_img_net(image, label)
        teacher_img_emb = teacher_img_emb.permute(1, 2, 0)

        student_img_emb, prediction = self.student_net(image)
        student_img_emb = student_img_emb.permute(1, 2, 0)

        txt_emb = torch.mean(txt_emb, dim=1)

        ideal_teacher_out = Ideal_teacher_nn_output_matrix(
            txt_emb, teacher_img_emb, student_img_emb, threshold
        )

        return prediction, teacher_img_emb, student_img_emb

