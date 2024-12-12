import torch
import torch.nn as nn
from torchvision import models

class DimensionExtractionModel(nn.Module):
    def __init__(self, num_units, num_entity_types, num_groups):
        super(DimensionExtractionModel, self).__init__()

        # CNN for image (edge image)
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 256)
        
        # Text embedding for OCR text and coordinates
        self.text_embedding = nn.EmbeddingBag(num_units, 128, sparse=True)  # OCR text embedding
        self.coord_embedding = nn.Linear(4, 64)  # OCR coordinates embedding (x, y, width, height)

        # Line coordinates embedding (optional line coordinates)
        self.line_embed = nn.Linear(4, 64)  # x1, y1, x2, y2
        
        # Metadata embedding
        self.unit_embed = nn.Embedding(num_units, 32)
        self.entity_type_embed = nn.Embedding(num_entity_types, 32)
        self.group_embed = nn.Embedding(num_groups, 32)
        
        # Fully connected layer to combine features
        self.fc1 = nn.Linear(256 + 128 + 64 + 64 + 32 + 32 + 32, 128)
        
        # Separate heads for value prediction and unit classification
        self.value_output = nn.Linear(128, 1)  # Numeric output
        self.unit_output = nn.Linear(128, num_units)  # Unit classification
        
    def forward(self, edge_image, ocr_text, ocr_coords, line_coords, entity_unit, entity_type, group_id):
        # Image feature extraction
        image_features = self.image_model(edge_image)

        # OCR text embedding and OCR coordinates embedding
        text_features = self.text_embedding(ocr_text)
        coord_features = self.coord_embedding(ocr_coords)

        # Combine OCR text and coordinate features
        ocr_features = text_features + coord_features  # or torch.cat((text_features, coord_features), dim=1) if using separate embeddings

        # Line coordinates embedding
        line_features = self.line_embed(line_coords)

        # Metadata embeddings
        unit_features = self.unit_embed(entity_unit)
        entity_type_features = self.entity_type_embed(entity_type)
        group_features = self.group_embed(group_id)
        
        # Concatenate all features
        combined_features = torch.cat((image_features, ocr_features, line_features,
                                       unit_features, entity_type_features, group_features), dim=1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(combined_features))
        
        # Separate predictions
        predicted_value = self.value_output(x)  # Numeric part
        predicted_unit_logits = self.unit_output(x)  # Unit classification logits
        predicted_unit = torch.argmax(predicted_unit_logits, dim=1)  # Get unit as a class
        
        return predicted_value, predicted_unit  # Returning as two parts for structured output
