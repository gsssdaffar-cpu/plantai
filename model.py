import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_model(num_classes, pretrained=True):

    # ========================================================
    # LOAD EFFICIENTNET-B0
    # ========================================================

    if pretrained:

        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

    else:

        model = efficientnet_b0(
            weights=None
        )


    # ========================================================
    # REPLACE CLASSIFIER
    # ========================================================

    input_features = (
        model.classifier[1].in_features
    )


    model.classifier[1] = nn.Linear(
        input_features,
        num_classes
    )


    return model