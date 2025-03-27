import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadResNet50(nn.Module):
    """
    A multi-head ResNet-50 model that outputs:
      - fruit_logits for fruit classification
      - rotten_logits for fresh vs. rotten
    """
    def __init__(self, num_fruits=5, freeze_layers=True):
        super(MultiHeadResNet50, self).__init__()

        # 1) Load a pretrained ResNet-50
        self.base_model = models.resnet50(pretrained=True)
        
        # 2) Remove the default final classification layer
        self.base_model.fc = nn.Identity()  # outputs 2048-dim features
        
        # 3) Optionally freeze layers (besides layer4 & new heads) to avoid overfitting
        if freeze_layers:
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze only layer4 (the last block)
            for name, param in self.base_model.layer4.named_parameters():
                param.requires_grad = True

        # 4) Define two heads: fruit & rotten
        feature_dim = 2048
        # Fruit classification head
        self.fruit_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_fruits)  # e.g., 5 fruit classes
        )
        # Fresh vs. rotten head (2 classes)
        self.rotten_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # (0=fresh, 1=rotten)
        )

    def forward(self, x):
        """
        Returns:
          fruit_logits  - shape [batch_size, num_fruits]
          rotten_logits - shape [batch_size, 2]
        """
        # Extract features using the shared ResNet-50 backbone
        features = self.base_model(x)  # [batch_size, 2048]
        
        # Pass through heads
        fruit_logits = self.fruit_head(features)
        rotten_logits = self.rotten_head(features)
        
        return fruit_logits, rotten_logits
