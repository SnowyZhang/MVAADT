import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
import os
from datetime import datetime
from .preprocess import adjacent_matrix_preprocessing

class Train_MVAADT:
    def __init__(self, 
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        random_seed = 0,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 5, 1, 1]
        ):
        """
        data: dict, including adata_omics1 and adata_omics2
        datatype: str, the type of data, default is 'SPOTS'
        device: torch.device, default is torch.device('cpu')
        random_seed: int, default is 2024
        learning_rate: float, default is 0.0001
        weight_decay: float, default is 0.00
        epochs: int, default is 600
        dim_input: int, default is 3000
        dim_output: int, default is 64

        """
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
          
        if self.datatype == '10x':
           self.epochs = 800 
           self.weight_factors = [1,5,1,10,1,1]
            
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
           self.epochs = 1600
           self.weight_factors = [1,5,1,1,1,1]
    
    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer_G = torch.optim.Adam(
            list(self.model.encoder_omics1.parameters()) +
            list(self.model.encoder_omics2.parameters()) +
            list(self.model.decoder_omics1.parameters()) +
            list(self.model.decoder_omics2.parameters()) +
            list(self.model.translation_12.parameters()) +
            list(self.model.translation_21.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.model.discriminator_omics1.parameters()) +
            list(self.model.discriminator_omics2.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.model.train()
        os.makedirs("./logs", exist_ok=True)
        log_path = f"./logs/train_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_path, "w") as f:
            f.write("Epoch\tRecon1\tRecon2\tAlignD1\tAlignD2\tAlignG1\tAlignG2\tTrans1to2\tTrans2to1\tG_total\tD_total\n")


        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            # 1) discriminator forward
            results_D = self.model(self.features_omics1, self.features_omics2,
                                self.adj_spatial_omics1, self.adj_feature_omics1,
                                self.adj_spatial_omics2, self.adj_feature_omics2)

            # compute_loss
            (loss_recon_omics1, loss_recon_omics2,
            loss_alignD1, loss_alignD2,
            _, _,  
            loss_trans_1to2, loss_trans_2to1) = self.compute_loss(results_D)

            # update D
            for p in self.model.discriminator_omics1.parameters():
                p.requires_grad = True
            for p in self.model.discriminator_omics2.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            loss_D = loss_alignD1 + loss_alignD2
            loss_D.backward()
            self.optimizer_D.step()

            # 2) generator fresh forward 
            for p in self.model.discriminator_omics1.parameters():
                p.requires_grad = False
            for p in self.model.discriminator_omics2.parameters():
                p.requires_grad = False

            results_G = self.model(self.features_omics1, self.features_omics2,
                                self.adj_spatial_omics1, self.adj_feature_omics1,
                                self.adj_spatial_omics2, self.adj_feature_omics2)

            (loss_recon_omics1, loss_recon_omics2,
            _, _,
            loss_alignG1, loss_alignG2,
            loss_trans_1to2, loss_trans_2to1) = self.compute_loss(results_G)


            loss_G = (
                self.weight_factors[0] * loss_recon_omics1 +
                self.weight_factors[1] * loss_recon_omics2 +
                self.weight_factors[2] * loss_alignG1 +   
                self.weight_factors[3] * loss_alignG2 +
                self.weight_factors[4] * loss_trans_1to2 +
                self.weight_factors[5] * loss_trans_2to1
            )

            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            # -------- logging --------
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | D_loss {loss_D.item():.6f} | G_loss {loss_G.item():.6f}")
            with open(log_path, "a") as f:
                f.write(
                    f"{epoch+1}\t"
                    f"{loss_recon_omics1.item():.6f}\t{loss_recon_omics2.item():.6f}\t"
                    f"{loss_alignD1.item():.6f}\t{loss_alignD2.item():.6f}\t"
                    f"{loss_alignG1.item():.6f}\t{loss_alignG2.item():.6f}\t"
                    f"{loss_trans_1to2.item():.6f}\t{loss_trans_2to1.item():.6f}\t"
                    f"{loss_G.item():.6f}\t{loss_D.item():.6f}\n"
                )
        print("Model training finished!\n")

    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'GAN_Align': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()
                }
        
        #  save translationModel
        torch.save(self.model.translation_12.state_dict(), './translationSet/simulated_data_translation12.pth')
        torch.save(self.model.translation_21.state_dict(), './translationSet/simulated_data_translation21.pth')


        return output
    
    def compute_loss(self, results):

        
        # reconstruction loss
        loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
        loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

        #translation loss
        loss_translated_omics1_to_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_translated_omics1_to_omics2'])
        loss_translated_omics2_to_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_translated_omics2_to_omics1'])
        
        
        # alignment loss
        results = {k: v.to(self.device) for k, v in results.items()}
        real_labels1 = torch.ones(results['emb_latent_omics1'].size(0), 1).to(self.device)
        real_labels2 = torch.ones(results['emb_latent_omics2'].size(0), 1).to(self.device)
        fake_labels1 = torch.zeros(results['emb_latent_omics2'].size(0), 1).to(self.device)
        fake_labels2 = torch.zeros(results['emb_latent_omics1'].size(0), 1).to(self.device)

        loss_omics1_ref = F.binary_cross_entropy(results['emb_latent_omics1_discriminator'], real_labels1).to(self.device) + F.binary_cross_entropy(results['emb_latent_omics2_discriminator'], fake_labels1).to(self.device)
        loss_omics2_ref = F.binary_cross_entropy(results['emb_latent_omics2_discriminator'], real_labels2).to(self.device) + F.binary_cross_entropy(results['emb_latent_omics1_discriminator'], fake_labels2).to(self.device)

        loss_G_omics1 = F.binary_cross_entropy(results['emb_latent_omics1_discriminator'], fake_labels1)  
        loss_G_omics2 = F.binary_cross_entropy(results['emb_latent_omics2_discriminator'], fake_labels2)

            
        return loss_recon_omics1, loss_recon_omics2, loss_omics1_ref, loss_omics2_ref,loss_G_omics1,loss_G_omics2,loss_translated_omics1_to_omics2,loss_translated_omics2_to_omics1
    