import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable

class ImageEncoder(nn.Module):
    def __init__(self, model_name):
        super(ImageEncoder, self).__init__()
        self.model = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

    def forward(self, image_batch):
        # Reshape the input to (batch_size, 3, 224, 224)
        image_batch = image_batch.view(-1, 3, 224, 224)
        # Encode the images using ResNet50
        image_features = self.model(image_batch)
        # Reshape to (batch_size, 2048, 49)
        image_features = image_features.view(image_features.size(0), 2048, -1)
        return image_features

class TextEncoder(nn.Module):
    def __init__(self, model_name, max_sequence_length):
        super(TextEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.text_encoder = DistilBertModel.from_pretrained(model_name)
        self.max_sequence_length = max_sequence_length

    def forward(self, text_batch):
        # Tokenize and ensure the text has a fixed length of max_sequence_length
        text_encoded = self.tokenizer(text_batch, padding='max_length', truncation=True, max_length=self.max_sequence_length, return_tensors="pt")
        text_encoded = text_encoded.to("cuda:0")
        # Encode the text using DistilBERT
        text_output = self.text_encoder(**text_encoded)

        # Extract embeddings for all tokens (last_hidden_state)
        text_embeddings = text_output.last_hidden_state  # This will have shape (batch_size, max_sequence_length, hidden_size)

        return text_embeddings

# Define the ProjectionHead class
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,  # Update the projection dimension
        dropout=0.1
    ):
        super(ProjectionHead, self).__init__()  # Correct usage of super()
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
        x = x + projected #skip connection
        x = self.layer_norm(x) #Layer Normalization
        return x

# Cross-Attention module
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_atten = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, image_features, text_features):
        key = text_features
        value = text_features
        query = image_features
        attn_output, attn_output_weights = self.multihead_atten(query, key, value)
        return attn_output, attn_output_weights

#cosine similarity
def cosine_similarity(avg_attn_out, avg_text_embed):
    dim = len(avg_attn_out)
    similarity_matrix = torch.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            similarity = F.cosine_similarity(avg_attn_out[i], avg_text_embed[j], dim=0)
            similarity_matrix[i, j] = similarity
    return similarity_matrix

#model
class ImageTextMatchingModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, image_projection_head, text_projection_head, cross_attention_layer):
        super(ImageTextMatchingModel, self).__init__()
                # Freeze the parameters of the pretrained image encoder
        for param in image_encoder.parameters():
            param.requires_grad = False

        # Freeze the parameters of the pretrained text encoder
        for param in text_encoder.parameters():
            param.requires_grad = False

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection_head = image_projection_head
        self.text_projection_head = text_projection_head
        self.cross_attention_layer = cross_attention_layer

    def forward(self, image_batch, text_batch):
        # Encode images and texts
        image_features = self.image_encoder(image_batch)
        text_embeddings = self.text_encoder(text_batch)
        image_features = image_features.permute(0, 2, 1)
        # Project images and texts
        projected_images = self.image_projection_head(image_features)
        projected_texts = self.text_projection_head(text_embeddings)
        # Cross-attention
        attn_output, attn_output_weights = self.cross_attention_layer(projected_images.permute(1, 0, 2), projected_texts.permute(1, 0, 2))
        attn_output = attn_output.permute(1, 0, 2)
        '''
        # Average pooling for image output
        avg_pool = nn.AdaptiveAvgPool1d(1)
        avg_attn_output = avg_pool(attn_output.permute(0, 2, 1))
        avg_attn_output = avg_attn_output.squeeze(dim=2)
        # CLS pooling for text output
        cls_embeddings = projected_texts[:, 0, :]
        '''

        avg_attn_output = torch.mean(attn_output, dim=1)
        cls_embeddings = torch.mean(projected_texts, dim=1)


        # Compute the cosine similarity score
        score = cosine_similarity(avg_attn_output, cls_embeddings)
        #print("Score",score)
        return score, attn_output, attn_output_weights, avg_attn_output, cls_embeddings, projected_texts#------------------editted----------attn_output, attn_output_weights

# model
class DOGS(nn.Module):
    def __init__(
        self,
        max_sequence_length=50,
        embedding_dim=512,
    ):
        super(DOGS, self).__init__()
        # Create an instance of the ImageTextMatchingModel
        self.image_encoder = ImageEncoder("resnet50")
        self.max_sequence_length = 30
        self.text_encoder = TextEncoder("distilbert-base-uncased", max_sequence_length)
        self.image_projection_head = ProjectionHead(2048)
        self.text_projection_head = ProjectionHead(768)
        self.embedding_dim = 256
        #self.cross_aatn = CrossAttentionLayer(embedding_dim, num_heads=4)
        self.cross_attention_layer = CrossAttentionLayer(embedding_dim, num_heads=4)


    

    def forward(self, image_batch, text_batch):

        

        model = ImageTextMatchingModel(
                    image_encoder=self.image_encoder,
                    text_encoder=self.text_encoder,
                    image_projection_head=self.image_projection_head,
                    text_projection_head=self.text_projection_head,
                    cross_attention_layer=self.cross_attention_layer
                )
        score = model(image_batch, text_batch)

        return score 
        
#new loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, scores):
        diagonal = scores.diag()
        d1 = diagonal.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d1.t()).clamp(min=0)
        #print("cost", cost_s.sum() + cost_im.sum())
        return cost_s.sum() + cost_im.sum()