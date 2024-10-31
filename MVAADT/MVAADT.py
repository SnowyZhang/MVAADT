import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing

class Train_MVAADT:
    def __init__(self, 
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        random_seed = 2022,
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
           self.weight_factors = [1,5,1,10,1,1] #1,5,1,10
            
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
           self.epochs = 1600
           self.weight_factors = [1,5,1,1,1,1]#1,5,1,1
    
    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
            
            self.loss_recon_omics1, self.loss_recon_omics2, self.loss_omics1_ref, self.loss_omics2_ref,self.loss_translated_omics1_to_omics2,self.loss_translated_omics2_to_omics1 = self.compute_loss(results)
            loss = self.weight_factors[0]*self.loss_recon_omics1 + self.weight_factors[1]*self.loss_recon_omics2 + self.weight_factors[2]*self.loss_omics1_ref + self.weight_factors[3]*self.loss_omics2_ref+self.weight_factors[4]*self.loss_translated_omics1_to_omics2+self.weight_factors[5]*self.loss_translated_omics2_to_omics1    

            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")

    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'GAN_Align': emb_combined.detach().cpu().numpy()
                }
        
        # Save translationModel
        # torch.save(self.model.translation_12.state_dict(), '/home/zxx/SMI/translationSet/HumanBrain_translation12.pth')
        # torch.save(self.model.translation_21.state_dict(), '/home/zxx/SMI/translationSet/HumanBrain_translation21.pth')
        # Save encoderModel
        # torch.save(self.model.encoder_omics1.state_dict(), 'encoder1.pth')
        # torch.save(self.model.encoder_omics2.state_dict(), 'encoder2.pth')
        # Save decoderModel
        # torch.save(self.model.decoder_omics1.state_dict(), 'decoder1.pth')
        # torch.save(self.model.decoder_omics2.state_dict(), 'decoder2.pth')
        # Save discriminatorModel
        # torch.save(self.model.discriminator_omics1.state_dict(), 'discriminator1.pth')  
        # torch.save(self.model.discriminator_omics2.state_dict(), 'discriminator2.pth')



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

        return loss_recon_omics1, loss_recon_omics2, loss_omics1_ref, loss_omics2_ref,loss_translated_omics1_to_omics2,loss_translated_omics2_to_omics1
    
    
    
        
    
    
      

    
        
    
    
