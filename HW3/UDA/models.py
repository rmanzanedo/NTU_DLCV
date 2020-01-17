# from torch import nn


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(3, 10, kernel_size=5),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Conv2d(10, 20, kernel_size=5),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(),
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(320, 50),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(50, 10),
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         features = features.view(x.shape[0], -1)
#         logits = self.classifier(features)
#         return logits


# import torch.nn.functional as F
# from torch import nn


# class LeNetEncoder(nn.Module):
#     """LeNet encoder model for ADDA."""

#     def __init__(self):
#         """Init LeNet encoder."""
#         super(LeNetEncoder, self).__init__()

#         self.restored = False

#         self.encoder = nn.Sequential(
#             # 1st conv layer
#             # input [1 x 28 x 28]
#             # output [20 x 12 x 12]
#             nn.Conv2d(1, 20, kernel_size=5),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU(),
#             # 2nd conv layer
#             # input [20 x 12 x 12]
#             # output [50 x 4 x 4]
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.Dropout2d(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU()
#         )
#         self.fc1 = nn.Linear(50 * 4 * 4, 500)

#     def forward(self, input):
#         """Forward the LeNet."""
#         conv_out = self.encoder(input)
#         feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
#         return feat


# class LeNetClassifier(nn.Module):
#     """LeNet classifier model for ADDA."""

#     def __init__(self):
#         """Init LeNet encoder."""
#         super(LeNetClassifier, self).__init__()
#         self.fc2 = nn.Linear(500, 10)

#     def forward(self, feat):
#         """Forward the LeNet classifier."""
#         out = F.dropout(F.relu(feat), training=self.training)
#         out = self.fc2(out)
#         return out


# class Discriminator(nn.Module):
#     """Discriminator model for source domain."""

#     def __init__(self, input_dims, hidden_dims, output_dims):
#         """Init discriminator."""
#         super(Discriminator, self).__init__()

#         self.restored = False

#         self.layer = nn.Sequential(
#             nn.Linear(input_dims, hidden_dims),
#             nn.ReLU(),
#             nn.Linear(hidden_dims, hidden_dims),
#             nn.ReLU(),
#             nn.Linear(hidden_dims, output_dims),
#             nn.LogSoftmax()
#         )

#     def forward(self, input):
#         """Forward the discriminator."""
#         out = self.layer(input)
#         return out

import torch.nn as nn
from functions import ReverseLayerF


class DSN(nn.Module):
    def __init__(self, code_size=100, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder_conv = nn.Sequential()
        self.source_encoder_conv.add_module('conv_pse1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                padding=2))
        self.source_encoder_conv.add_module('ac_pse1', nn.ReLU(True))
        self.source_encoder_conv.add_module('pool_pse1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.source_encoder_conv.add_module('conv_pse2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                                padding=2))
        self.source_encoder_conv.add_module('ac_pse2', nn.ReLU(True))
        self.source_encoder_conv.add_module('pool_pse2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.source_encoder_fc = nn.Sequential()
        self.source_encoder_fc.add_module('fc_pse3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.source_encoder_fc.add_module('ac_pse3', nn.ReLU(True))

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential()
        self.target_encoder_conv.add_module('conv_pte1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                padding=2))
        self.target_encoder_conv.add_module('ac_pte1', nn.ReLU(True))
        self.target_encoder_conv.add_module('pool_pte1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.target_encoder_conv.add_module('conv_pte2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                                padding=2))
        self.target_encoder_conv.add_module('ac_pte2', nn.ReLU(True))
        self.target_encoder_conv.add_module('pool_pte2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.target_encoder_fc = nn.Sequential()
        self.target_encoder_fc.add_module('fc_pte3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.target_encoder_fc.add_module('ac_pte3', nn.ReLU(True))

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.shared_encoder_conv = nn.Sequential()
        self.shared_encoder_conv.add_module('conv_se1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                  padding=2))
        self.shared_encoder_conv.add_module('ac_se1', nn.ReLU(True))
        self.shared_encoder_conv.add_module('pool_se1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.shared_encoder_conv.add_module('conv_se2', nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,
                                                                  padding=2))
        self.shared_encoder_conv.add_module('ac_se2', nn.ReLU(True))
        self.shared_encoder_conv.add_module('pool_se2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.shared_encoder_fc = nn.Sequential()
        self.shared_encoder_fc.add_module('fc_se3', nn.Linear(in_features=7 * 7 * 48, out_features=code_size))
        self.shared_encoder_fc.add_module('ac_se3', nn.ReLU(True))

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=100, out_features=n_class))

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=100, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=100, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=code_size, out_features=588))
        self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))

        self.shared_decoder_conv = nn.Sequential()
        self.shared_decoder_conv.add_module('conv_sd2', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.shared_decoder_conv.add_module('relu_sd2', nn.ReLU())

        self.shared_decoder_conv.add_module('conv_sd3', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.shared_decoder_conv.add_module('relu_sd3', nn.ReLU())

        self.shared_decoder_conv.add_module('us_sd4', nn.Upsample(scale_factor=2))

        self.shared_decoder_conv.add_module('conv_sd5', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd5', nn.ReLU(True))

        self.shared_decoder_conv.add_module('conv_sd6', nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                                                                  padding=1))

    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':

            # target private encoder
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.target_encoder_fc(private_feat)

        result.append(private_code)

        # shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 48 * 7 * 7)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 3, 14, 14)

        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result