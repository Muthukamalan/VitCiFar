import torch
import torch.nn as nn


class PatchEmbedding(nn.Module): # Done
    """
    img_size: 1d size of each image (32 for CIFAR-10)
    patch_size: 1d size of each patch (img_size/num_patch_1d, 4 in this experiment)
    in_chans: input channel (3 for RGB images)
    emb_dim: flattened length for each token (or patch)
    """
    def __init__(self, img_size:int, patch_size:int, in_chans:int=3, emb_dim:int=48):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, 
            emb_dim, 
            kernel_size = patch_size, 
            stride = patch_size
        )

    def forward(self, x):
        with torch.no_grad():
            # x: [batch, in_chans, img_size, img_size]
            x = self.proj(x) # [batch, embed_dim, # of patches in a row, # of patches in a col], [batch, 48, 8, 8] in this experiment
            x = x.flatten(2) # [batch, embed_dim, total # of patches], [batch, 48, 64] in this experiment
            x = x.transpose(1, 2) # [batch, total # of patches, emb_dim] => Transformer encoder requires this dimensions [batch, number of words, word_emb_dim]
        return x


class TransformerEncoder(nn.Module): # Done
    def __init__(self, input_dim:int, mlp_hidden_dim:int, num_head:int=8, dropout:float=0.):
        # input_dim and head for Multi-Head Attention
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim) # LayerNorm is BatchNorm for NLP
        self.msa = MultiHeadSelfAttention(input_dim, n_heads=num_head)
        self.norm2 = nn.LayerNorm(input_dim)
        # Position-wise Feed-Forward Networks with GELU activation functions
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, input_dim),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.msa(self.norm1(x)) + x # add residual connection
        out = self.mlp(self.norm2(out)) + out # add another residual connection
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    dim: dimension of input and out per token features (emb dim for tokens)
    n_heads: number of heads
    qkv_bias: whether to have bias in qkv linear layers
    attn_p: dropout probability for attention
    proj_p: droupout probability last linear layer
    scale: scaling factor for attention (1/sqrt(dk))
    qkv: initial linear layer for the query, key, and value
    proj: last linear layer
    attn_drop, proj_drop: dropout layers for attn and proj
    """
    def __init__(self, dim:int, n_heads:int=8, qkv_bias:bool=True, attn_p:float=0.01, proj_p:float=0.01):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim # embedding dimension for input
        self.head_dim = dim // n_heads # d_q, d_k, d_v in the paper (int div needed to preserve input dim = output dim)
        self.scale = self.head_dim ** -0.5 # 1/sqrt(d_k)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) # lower linear layers in Figure 2 of the paper
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim) # upper linear layers in Figure 2 of the paper
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """
        Input and Output shape: [batch_size, n_patches + 1, dim]
        """
        batch_size, n_tokens, x_dim = x.shape # n_tokens = n_patches + 1 (1 is cls_token), x_dim is input dim

        # Sanity Check
        if x_dim != self.dim: # make sure input dim is same as concatnated dim (output dim)
            raise ValueError
        if self.dim != self.head_dim*self.n_heads: # make sure dim is divisible by n_heads
            raise ValueError(f"Input & Output dim should be divisible by Number of Heads")
        
        # Linear Layers for Query, Key, Value
        qkv = self.qkv(x) # (batch_size, n_patches+1, 3*dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim) # (batch_size, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (batch_size, n_heads, n_patches+1, head_dim)

        # Scaled Dot-Product Attention
        k_t = k.transpose(-2, -1) # K Transpose: (batch_size, n_heads, head_dim, n_patches+1)
        dot_product = (q @ k_t)*self.scale # Query, Key Dot Product with Scale Factor: (batch_size, n_heads, n_patches+1, n_patches+1)
        attn = dot_product.softmax(dim=-1) # Softmax: (batch_size, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn) # Attention Dropout: (batch_size, n_heads, n_patches+1, n_patches+1)
        weighted_avg = attn @ v # (batch_size, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (batch_size, n_patches+1, n_heads, head_dim)

        # Concat and Last Linear Layer
        weighted_avg = weighted_avg.flatten(2) # Concat: (batch_size, n_patches+1, dim)
        x = self.proj(weighted_avg) # Last Linear Layer: (batch_size, n_patches+1, dim)
        x = self.proj_drop(x) # Last Linear Layer Dropout: (batch_size, n_patches+1, dim)

        return x

class ViT(nn.Module): # Done
    def __init__(
            self,
            in_c:int=3, 
            num_classes:int=10, 
            img_size:int=32, 
            num_patch_1d:int=16, 
            dropout:float=0.1, 
            num_enc_layers:int=2, 
            hidden_dim:int=128, 
            mlp_hidden_dim:int=128//2, 
            num_head:int=4, 
            is_cls_token:bool=True
        ):
        super(ViT, self).__init__()
        """
        is_cls_token: are we using class token?
        num_patch_1d: number of patches in one row (or col), 3 in Figure 1 of the paper, 8 in this experiment
        patch_size: # 1d size (size of row or col) of each patch, 16 for ImageNet in the paper, 4 in this experiment
        flattened_patch_dim: Flattened vec length for each patch (4 x 4 x 3, each side is 4 and 3 color scheme), 48 in this experiment
        num_tokens: number of total patches + 1 (class token), 10 in Figure 1 of the paper, 65 in this experiment
        """
        self.is_cls_token = is_cls_token
        self.num_patch_1d = num_patch_1d
        self.patch_size = img_size//self.num_patch_1d
        num_tokens = (self.num_patch_1d**2)+1 if self.is_cls_token else (self.num_patch_1d**2)

        # Divide each image into patches
        self.images_to_patches = PatchEmbedding(
                                    img_size=img_size, 
                                    patch_size=img_size//num_patch_1d,
                                    emb_dim=num_patch_1d*num_patch_1d
                                )

        # Linear Projection of Flattened Patches
        self.lpfp = nn.Linear(num_patch_1d*num_patch_1d, hidden_dim) # 48 x 384 (384 is the latent vector size D in the paper)

        # Patch + Position Embedding (Learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if is_cls_token else None # learnable classification token with dim [1, 1, 384]. 1 in 2nd dim because there is only one class per each image not each patch
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)) # learnable positional embedding with dim [1, 65, 384]
        
        # Transformer Encoder
        enc_list = [TransformerEncoder(hidden_dim, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout, num_head=num_head) for _ in range(num_enc_layers)] # num_enc_layers is L in Transformer Encoder at Figure 1
        self.enc = nn.Sequential(*enc_list) # * should be adeed if given regular python list to nn.Sequential
        
        # MLP Head (Standard Classifier)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x): # x: [batch, 3, 32, 32]
        # Images into Patches (including flattening)
        out = self.images_to_patches(x) # [batch, 64, 48]
        # Linear Projection on Flattened Patches
        out = self.lpfp(out) # [batch, 64, 384]

        # Add Class Token and Positional Embedding
        if self.is_cls_token: 
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1) # [batch, 65, 384], added as extra learnable embedding
        out = out + self.pos_emb # [batch, 65, 384]

        # Transformer Encoder
        out = self.enc(out) # [batch, 65, 384]
        if self.is_cls_token:
            out = out[:,0] # [batch, 384]
        else:
            out = out.mean(1)

        # MLP Head
        out = self.mlp_head(out) # [batch, 10]
        return out 
    


import lightning as pl
from torchmetrics import Accuracy

class ViTLightning(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super(ViTLightning, self).__init__()
        self.vit = ViT(
            in_c=3, 
            num_classes=10, 
            img_size=32, 
            num_patch_1d=16, 
            dropout=0.1, 
            num_enc_layers=2, 
            hidden_dim=96, 
            mlp_hidden_dim=64, 
            num_head=8, 
            is_cls_token=True
        )
        self.train_acc = Accuracy('multiclass',num_classes=10)
        self.val_acc = Accuracy('multiclass',num_classes=10)
        self.test_acc = Accuracy('multiclass',num_classes=10)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.vit(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.train_acc(preds, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.test_acc(preds, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self.vit.parameters(), )
        num_epochs = self.trainer.max_epochs,
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=num_epochs,
            pct_start= .3,
            div_factor= 100,
            max_lr= 1e-3,
            three_phase= False,
            final_div_factor= 100,
            anneal_strategy='linear'
        )
        return {
            'optimizer':optimizer,
            'lr_scheduler':{
                'scheduler':scheduler,
                'monitor': "val_loss",
                "interval":"step",
                "frequency":1
            }
        }