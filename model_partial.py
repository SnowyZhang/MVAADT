import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
    
class Encoder_overall(Module):
     
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics2)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics1)
        self.discriminator_omics1 = Discriminator(self.dim_out_feat_omics1)
        self.discriminator_omics2 = Discriminator(self.dim_out_feat_omics2)


        self.translation_12 = translationModel(self.dim_out_feat_omics1, self.dim_out_feat_omics2)  # translation from omics1 to omics2
        self.translation_21 = translationModel(self.dim_out_feat_omics2, self.dim_out_feat_omics1)  # translation from omics2 to omics1
        # 加载预训练权重
        # self.translation_12.load_state_dict(torch.load('HumanBrain_translation12.pth'))
        # self.translation_21.load_state_dict(torch.load('HumanBrain_translation21.pth'))
        
        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)
        
    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2, mask_index, remain_index):
        
        # graph1
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)  
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
        
        # graph2
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)
        
        # within-modality attention aggregation layer
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)

        #translation between omics1 and omics2
        emb_latent_omics1_to_omics2 = self.translation_12(emb_latent_omics1)
        emb_latent_omics2_to_omics1 = self.translation_21(emb_latent_omics2)

        emb_latent_omics1_new = self.fuse_representations(emb_latent_omics1, emb_latent_omics2_to_omics1, remain_index, mask_index)
        
        # discriminator for each modality
        pred_omics1 = self.discriminator_omics1(emb_latent_omics1_new)
        pred_omics2 = self.discriminator_omics2(emb_latent_omics2)
        # pred_omics2 = self.discriminator_omics2(emb_latent_omics1_to_omics2)


        # between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1_new, emb_latent_omics2)
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics1_to_omics2)
        
        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        
        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics1_new':emb_latent_omics1_new,
                   'emb_latent_omics2':emb_latent_omics2,
                    # 'emb_latent_omics2':emb_latent_omics1_to_omics2,
                #    'emb_latent_translated_omics1_to_omics2':emb_latent_omics1_to_omics2,
                    'emb_latent_translated_omics2_to_omics1':emb_latent_omics2_to_omics1,
                   'emb_latent_omics1_discriminator':pred_omics1,
                   'emb_latent_omics2_discriminator':pred_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'alpha_omics1':alpha_omics1,
                   'alpha_omics2':alpha_omics2,
                   'alpha':alpha_omics_1_2
                   }
        
        return results  
    
    def fuse_representations(self,emb_latent_omics1, emb_latent_omics2_to_omics1, remain_index, mask_index):
        """
        Select the corresponding samples from emb_latent_omics2_to_omics1 based on mask_index 
        and add them to emb_latent_omics1 in the order specified by the index.

        Parameters:
        emb_latent_omics1 (torch.Tensor): The original emb_latent_omics1 representation.
        emb_latent_omics2_to_omics1 (torch.Tensor): The transformed emb_latent_omics2_to_omics1 representation.
        remain_index (torch.Tensor): The actual index of each sample in emb_latent_omics1.
        mask_index (torch.Tensor): The indices of the masked samples.

        Returns:
        torch.Tensor: The merged representation.
        """

        
        emb_latent_omics1_new = torch.zeros((mask_index.shape[0] + remain_index.shape[0], emb_latent_omics1.shape[1]), device=emb_latent_omics1.device)

        
        emb_latent_omics1_new[remain_index] = emb_latent_omics1

        
        emb_latent_omics1_new[mask_index] = emb_latent_omics2_to_omics1[mask_index]

        return emb_latent_omics1_new

    # def compute_loss(self, results, features_omics1, features_omics2, device):

    #     features_omics1 = features_omics1.to(device)
    #     features_omics2 = features_omics2.to(device)
    #     results = {k: v.to(device) for k, v in results.items()}
        
    #     # reconstruction loss
    #     loss_recon_omics1 = F.mse_loss(features_omics1, results['emb_recon_omics1'])
    #     loss_recon_omics2 = F.mse_loss(features_omics2, results['emb_recon_omics2'])
        
        
    #     # alignment loss
    #     real_labels1 = torch.ones(results['emb_latent_omics1'].size(0), 1)
    #     real_labels2 = torch.ones(results['emb_latent_omics2'].size(0), 1)
    #     fake_labels1 = torch.zeros(results['emb_latent_omics2'].size(0), 1)
    #     fake_labels2 = torch.zeros(results['emb_latent_omics1'].size(0), 1)

    #     # loss_gen_omics1 = F.binary_cross_entropy(results['emb_latent_omics1_discriminator'], real_labels1)
    #     # loss_gen_omics2 = F.binary_cross_entropy(results['emb_latent_omics2_discriminator'], real_labels2)
    #     loss_omics1_ref = F.binary_cross_entropy(results['emb_latent_omics1_discriminator'], real_labels1) + F.binary_cross_entropy(results['emb_latent_omics2_discriminator'], fake_labels1)
    #     loss_omics2_ref = F.binary_cross_entropy(results['emb_latent_omics2_discriminator'], real_labels2) + F.binary_cross_entropy(results['emb_latent_omics1_discriminator'], fake_labels2)

    #     return loss_recon_omics1, loss_recon_omics2, loss_omics1_ref, loss_omics2_ref  

class translationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(translationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class translationModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(translationModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 64)
#         self.fc5 = nn.Linear(64, 32)
#         self.fc6 = nn.Linear(32, output_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.relu(self.fc4(x))
#         x = self.relu(self.fc5(x))
#         x = self.fc6(x)
#         return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class Encoder(Module): 
    
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Latent representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x
    
class Decoder(Module):
    
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Reconstructed representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x                  

class AttentionLayer(Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha      
