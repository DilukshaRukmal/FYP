import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Sajeepan.DOGS.scripts.model import DOGS


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
        x = self.layer_norm(x)  # Layer Normalization-
        return x


# Student feature extractor + Classifier
class StudentNet(nn.Module):
    def __init__(self,num_classes, hidden_layers=512):
        super(StudentNet, self).__init__()

        # download resnet152 model
        model = models.resnet152(pretrained=False)
        self.cropped_resnet152 = torch.nn.Sequential(*list(model.children())[:-2])
        self.projection_head = ProjectionHead(2048, projection_dim=512, dropout=0.1)

        # for classifier
        self.fc = nn.Linear(hidden_layers, num_classes)

    def forward(self, image):
        #feature extractor
        img_embedding = self.cropped_resnet152(image)
        image_embedding = img_embedding.view(len(img_embedding), 2048, 49)#(b,2048,49)
        image_embedding = image_embedding.permute(0, 2, 1)#(b,49,2048)
        image_embedding = self.projection_head(image_embedding)#(b,49,512)
        image_embedding = image_embedding.permute(1, 0, 2)#(49,b,512)
        #classifier
        x = torch.mean(image_embedding, dim=0)#(1,b,512)
        x = torch.unsqueeze(x,0)
        logit = self.fc(x)

        return image_embedding, logit



#Loss function for knowledge distillation
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, teacher_em, student_em):#(b,512,49)
        batch_cs = torch.zeros(teacher_em.shape[0])  # tensor to hold cs for each feature
        for i in range(teacher_em.shape[0]):
            feature_cs = F.cosine_similarity(teacher_em[i, :, :], student_em[i, :, :], dim=1)
            mean_cs = torch.mean(feature_cs)
            batch_cs[i] = mean_cs

        mean_batch_cs = torch.mean(batch_cs)
        final_loss = 1 - mean_batch_cs
        return final_loss ##a value between 0 & 2 


#Main model for knowledge distillation
class Knowledge_distiller(nn.Module):
    def __init__(self, model_path="path_to_teacher", num_classes=2):
        super(Knowledge_distiller, self).__init__()

        self.student_net = StudentNet(num_classes=num_classes)

        #loading the teacher
        self.teacher_img_net = DOGS()       
        self.teacher_img_net.load_state_dict(
            torch.load(model_path, map_location="cuda:0")
        )
        
        #freezing the teacher
        for param in self.teacher_img_net.parameters():
            param.requires_grad = False

        #Moving networks to GPU if available
        if torch.cuda.is_available():
            self.student_net = self.student_net.to("cuda:0")
            self.teacher_img_net = self.teacher_img_net.to("cuda:0")

    def forward(self, image, label):
        _, teacher_img_emb, __ = self.teacher_img_net(image, label)#(b,49,512)
        teacher_img_emb = teacher_img_emb.permute(0, 2, 1)#(b,512,49)

        student_img_emb, prediction = self.student_net(image)#(49,b,512)
        student_img_emb = student_img_emb.permute(1, 2, 0)#(b,512,49)

        return  teacher_img_emb, student_img_emb, prediction