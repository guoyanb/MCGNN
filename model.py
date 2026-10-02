# model.py
import torch.nn.functional as F
from dgl.ops import edge_softmax
from utils import *
import torch
import numpy as np
from cuda_wavelet import FirstLayerWaveletFeatureExtractor


class Subgraph_Fusion(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Subgraph_Fusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, z):
        w = self.project(z).mean(0)
        beta_ = torch.softmax(w, dim=0)
        beta = beta_.expand((z.shape[0],) + beta_.shape)
        return (beta * z).sum(1), beta_


class SemanticEncoder(nn.Module):
    def __init__(self, layer_num_heads, hidden_size, r_vec, etypes, device='cpu'):
        super(SemanticEncoder, self).__init__()
        self.num_heads = layer_num_heads
        self.hidden_size = hidden_size
        self.r_vec = r_vec
        self.etypes = etypes
        self.device = device

    def forward(self, edata):
        edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
        final_r_vec = torch.zeros([edata.shape[1], self.hidden_size // 2, 2]).to(self.device)
        r_vec = F.normalize(self.r_vec, p=2, dim=2)
        r_vec = torch.stack((r_vec, r_vec), dim=1)
        r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
        r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)
        final_r_vec[-1, :, 0] = 1
        for i in range(final_r_vec.shape[0] - 2, -1, -1):
            if self.etypes[i] is not None:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
            else:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
        for i in range(edata.shape[1] - 1):
            temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
            temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
            edata[:, i, :, 0] = temp1
            edata[:, i, :, 1] = temp2
        edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
        metapath_embedding = torch.mean(edata, dim=1)
        return metapath_embedding


class MessageAggregator(nn.Module):
    def __init__(self, num_heads, hidden_size, attn_drop, alpha, name, device='cpu'):
        super(MessageAggregator, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.device = device
        
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
            
        self.attn1 = nn.Linear(self.hidden_size, self.num_heads, bias=False).to(device)
        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        
        self.attn2 = nn.Parameter(torch.empty(size=(1, self.num_heads, self.hidden_size))).to(device)
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        
        self.name = name

    def forward(self, nodes, metapath_instances, metapath_embedding, features):
        h_ = []
        for i in range(len(nodes)):
            index = metapath_instances[metapath_instances[self.name] == nodes[i]].index.tolist()
            if index != []:
                node_metapath_embedding = metapath_embedding[index]  # (E,64)
                node_metapath_embedding = torch.cat([node_metapath_embedding] * self.num_heads, dim=1)  # (E,64*8)
                node_metapath_embedding = node_metapath_embedding.unsqueeze(dim=0)  # (1, E, 64*8)
                eft = node_metapath_embedding.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_size)  # (E, 8, 64)
                node_embedding = torch.vstack([features[i]] * len(index))  # (E, 64)
                a1 = self.attn1(node_embedding)
                a2 = (eft * self.attn2).sum(dim=-1)
                a = (a1 + a2).unsqueeze(dim=-1)
                a = self.leaky_relu(a)
                attention = F.softmax(a, dim=0)
                attention = self.attn_drop(attention)
                h = F.elu((attention * eft).sum(dim=0)).view(-1, self.hidden_size * self.num_heads)
                h_.append(h[0])
            else:
                node_embedding = torch.zeros(self.hidden_size * self.num_heads).to(self.device)
                h_.append(node_embedding)
        return torch.stack(h_, dim=0)


class MultiStageFourierConv(nn.Module):
    """Multi-stage Fourier convolution module"""
    def __init__(self, in_channels, out_channels, device='cpu', n_stages=2):
        super(MultiStageFourierConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.n_stages = n_stages
        
        # Multiple frequency-domain filters
        self.filter_real_layers = nn.ParameterList([
            nn.Parameter(torch.randn(out_channels, out_channels)) 
            for _ in range(n_stages)
        ])
        self.filter_imag_layers = nn.ParameterList([
            nn.Parameter(torch.randn(out_channels, out_channels))
            for _ in range(n_stages)
        ])
        
        # Initialization
        for i in range(n_stages):
            nn.init.kaiming_normal_(self.filter_real_layers[i])
            nn.init.kaiming_normal_(self.filter_imag_layers[i])
        
        # Input projection layer
        self.input_projection = None
        if in_channels != out_channels:
            self.input_projection = nn.Linear(in_channels, out_channels).to(device)
        
        # Output projection
        self.output_projections = nn.ModuleList([
            nn.Linear(out_channels, out_channels).to(device)
            for _ in range(n_stages)
        ])
        
        # Residual connection layers
        self.residual_projections = nn.ModuleList([
            nn.Linear(out_channels, out_channels).to(device) 
            for _ in range(n_stages)
        ])
        
        # Frequency attention mechanism - using the correct input dimension
        self.frequency_attention = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        ).to(device)
    
    def _fourier_transform_2d(self, x):
        """2D Fourier transform"""
        return torch.fft.fft(x, dim=1)
    
    def _inverse_fourier_transform_2d(self, x_freq):
        """2D inverse Fourier transform"""
        return torch.fft.ifft(x_freq, dim=1).real
    
    def _apply_frequency_filter(self, x_freq, stage_idx):
        """Apply frequency-domain filter"""
        real_part = x_freq.real
        imag_part = x_freq.imag
        
        filter_real_t = self.filter_real_layers[stage_idx].T
        filter_imag_t = self.filter_imag_layers[stage_idx].T
        
        filtered_real = torch.matmul(real_part, filter_real_t) - torch.matmul(imag_part, filter_imag_t)
        filtered_imag = torch.matmul(real_part, filter_imag_t) + torch.matmul(imag_part, filter_real_t)
        
        return torch.complex(filtered_real, filtered_imag)
    
    def _apply_frequency_attention(self, x_freq):
        """Apply frequency attention"""
        magnitude = torch.abs(x_freq)
        
        # Ensure input dimension is correct
        if magnitude.shape[1] != self.out_channels:
            # If dimensions do not match, re-project
            projection = nn.Linear(magnitude.shape[1], self.out_channels).to(self.device)
            magnitude_projected = projection(magnitude)
            attention_input = magnitude_projected.mean(dim=0, keepdim=True)
        else:
            attention_input = magnitude.mean(dim=0, keepdim=True)
        
        attention_weights = self.frequency_attention(attention_input)
        # Broadcast attention weights to all frequency components
        attention_weights = attention_weights.expand_as(x_freq.real)
        
        real_part = x_freq.real * attention_weights
        imag_part = x_freq.imag * attention_weights
        
        return torch.complex(real_part, imag_part)
    
    def forward(self, x, stage_idx=0, apply_attention=True):
        """Forward pass"""
        original_shape = x.shape
        
        # Flatten processing
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])
        
        # Input projection
        if self.input_projection is not None and stage_idx == 0:
            x = self.input_projection(x)
        elif x.shape[1] != self.out_channels:
            # Dynamically create projection layer
            self.input_projection = nn.Linear(x.shape[1], self.out_channels).to(self.device)
            x = self.input_projection(x)
        
        x_residual = x.clone()
        
        # Fourier transform to frequency domain
        x_freq = self._fourier_transform_2d(x)
        
        # Apply frequency attention
        if apply_attention:
            x_freq = self._apply_frequency_attention(x_freq)
        
        # Apply frequency-domain filter
        x_freq_filtered = self._apply_frequency_filter(x_freq, stage_idx)
        
        # Inverse Fourier transform back to time domain
        output = self._inverse_fourier_transform_2d(x_freq_filtered)
        
        # Output projection
        output = self.output_projections[stage_idx](output)
        
        # Residual connection
        if x_residual.shape[1] == self.out_channels:
            residual_proj = self.residual_projections[stage_idx](x_residual)
            output = output + residual_proj
        else:
            # If residual dimensions do not match, project first
            residual_projection = nn.Linear(x_residual.shape[1], self.out_channels).to(self.device)
            residual_proj = residual_projection(x_residual)
            residual_proj = self.residual_projections[stage_idx](residual_proj)
            output = output + residual_proj
        
        # Restore original shape
        if len(original_shape) > 2:
            output = output.view(original_shape[:-1] + (self.out_channels,))
        
        return output


class CrossTypeInteractionLearner(nn.Module):
    """Causal feature learner - learns causal relationships between nodes"""
    def __init__(self, hidden_size=128, num_heads=4, device='cpu'):
        super(CrossTypeInteractionLearner, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device = device
        
        # Causal attention mechanism - using the correct dimension
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,  # explicitly specify
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Input projection layer to handle dimension mismatch
        self.input_projection = nn.Linear(hidden_size * num_heads, hidden_size)
        
        # Causal graph learning
        self.causal_graph_learner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features):
        """Learn causal features between nodes"""
        # Get node types
        node_types = list(node_features.keys())
        n_types = len(node_types)
        
        # Collect all node features
        all_features = []
        for ntype in node_types:
            feat = node_features[ntype]
            if feat.numel() == 0:
                all_features.append(torch.zeros(1, self.hidden_size).to(self.device))
                continue
                
            # Get feature shape
            feat_shape = feat.shape
            
            # If 3D features (after batching), take the first sample
            if len(feat_shape) > 2:
                feat = feat.view(-1, feat_shape[-1])
            
            # If feature dimension does not match hidden dimension, projection is needed
            if feat.shape[-1] != self.hidden_size:
                # Dynamically create projection layer (if needed)
                if feat.shape[-1] == self.hidden_size * self.num_heads:
                    # If after the first layer (feature dimension expanded)
                    feat_projected = self.input_projection(feat.mean(dim=0, keepdim=True))
                else:
                    # Otherwise, create an adapted projection layer
                    projection = nn.Linear(feat.shape[-1], self.hidden_size).to(self.device)
                    feat_projected = projection(feat.mean(dim=0, keepdim=True))
            else:
                feat_projected = feat.mean(dim=0, keepdim=True)
            
            all_features.append(feat_projected)
        
        # If no features, return empty tensor
        if not all_features or all(all_feat.numel() == 0 for all_feat in all_features):
            return torch.zeros(n_types, self.hidden_size).to(self.device), \
                   torch.zeros(n_types, n_types).to(self.device)
        
        all_features = torch.cat(all_features, dim=0)  # [n_types, hidden_size]
        
        # Causal attention
        causal_features, _ = self.causal_attention(
            all_features.unsqueeze(0),  # [1, n_types, hidden_size]
            all_features.unsqueeze(0),
            all_features.unsqueeze(0)
        )
        causal_features = causal_features.squeeze(0)  # [n_types, hidden_size]
        
        # Generate causal graph
        causal_adj = torch.zeros(n_types, n_types).to(self.device)
        for i in range(n_types):
            for j in range(n_types):
                if i != j:
                    pair_features = torch.cat([causal_features[i], causal_features[j]], dim=0)
                    edge_prob = self.causal_graph_learner(pair_features.unsqueeze(0))
                    causal_adj[i, j] = edge_prob.squeeze()
        
        return causal_features, causal_adj


class FirstLayerFeatureLearner(nn.Module):
    """First-layer feature learner - fuses wavelet features with original features"""
    def __init__(self, in_size, hidden_size=128, wavelet_type='haar', device='cpu'):
        super(FirstLayerFeatureLearner, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        
        # Dedicated wavelet feature extractor for the first layer
        self.wavelet_extractor = FirstLayerWaveletFeatureExtractor(
            wavelet_type=wavelet_type,
            hidden_size=hidden_size,
            device=device
        )
        
        # Feature fusion gating mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Final feature projection
        self.final_projection = nn.Linear(hidden_size, hidden_size)
        
        # Causal feature learner
        self.causal_learner = CrossTypeInteractionLearner(
            hidden_size=hidden_size,
            device=device
        )
        
        # Fourier convolution (optional)
        self.fourier_conv = MultiStageFourierConv(
            hidden_size, hidden_size, device=device, n_stages=1
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, original_features):
        """First-layer feature learning, fusing wavelet features"""
        # 1. Extract wavelet features
        wavelet_features = self.wavelet_extractor(original_features)
        
        # 2. Fuse wavelet features with original features
        fused_features = {}
        
        for ntype, orig_feat in original_features.items():
            if orig_feat.numel() == 0:
                fused_features[ntype] = torch.zeros(0, self.hidden_size).to(self.device)
                continue
            
            # If original feature dimension differs from hidden dimension, project first
            if orig_feat.shape[-1] != self.hidden_size:
                projection = nn.Linear(orig_feat.shape[-1], self.hidden_size).to(self.device)
                orig_feat_proj = projection(orig_feat)
            else:
                orig_feat_proj = orig_feat
            
            # Get corresponding wavelet features
            if ntype in wavelet_features and wavelet_features[ntype].numel() > 0:
                wave_feat = wavelet_features[ntype]
                
                # Ensure wavelet feature dimension is correct
                if wave_feat.shape[-1] != self.hidden_size:
                    wave_projection = nn.Linear(wave_feat.shape[-1], self.hidden_size).to(self.device)
                    wave_feat = wave_projection(wave_feat)
                
                # Gated fusion
                concat_feat = torch.cat([orig_feat_proj, wave_feat], dim=-1)
                gate_weights = self.fusion_gate(concat_feat)
                
                # Weighted fusion
                fused = gate_weights * orig_feat_proj + (1 - gate_weights) * wave_feat
            else:
                fused = orig_feat_proj
            
            # 3. Optional: apply Fourier convolution
            fused = self.fourier_conv(fused, stage_idx=0)
            
            # 4. Layer normalization
            fused = self.layer_norm(fused)
            
            fused_features[ntype] = fused
        
        return fused_features


class SubsequentLayerFeatureLearner(nn.Module):
    """Subsequent-layer feature learner - uses only Fourier convolution and causal features"""
    def __init__(self, input_dim, hidden_size=128, device='cpu', n_stages=1):
        super(SubsequentLayerFeatureLearner, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = device
        self.n_stages = n_stages
        
        # Fourier convolution - using the correct input dimension
        self.fourier_conv = MultiStageFourierConv(
            input_dim, hidden_size, device=device, n_stages=n_stages
        )
        
        # Causal feature learner
        self.causal_learner = CrossTypeInteractionLearner(
            hidden_size=hidden_size,
            num_heads=4,  # can be adjusted as needed
            device=device
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Residual connection
        self.residual_connection = nn.Linear(input_dim, hidden_size)
        
    def forward(self, input_features, stage_idx=0):
        """Subsequent-layer feature learning"""
        enhanced_features = {}
        
        # Ensure stage_idx is within valid range
        if stage_idx >= self.n_stages:
            stage_idx = self.n_stages - 1
        
        for ntype, feat in input_features.items():
            if feat.numel() == 0:
                enhanced_features[ntype] = torch.zeros(0, self.hidden_size).to(self.device)
                continue
            
            # 1. Fourier convolution (using safe stage_idx)
            fourier_feat = self.fourier_conv(feat, stage_idx=stage_idx)
            
            # 2. Causal feature learning - ensure input format is correct
            # Create a dictionary containing only the current node type
            if fourier_feat.numel() > 0:
                # Ensure feature dimension is correct
                if fourier_feat.shape[-1] != self.hidden_size:
                    # If Fourier convolution output dimension is incorrect, project
                    projection = nn.Linear(fourier_feat.shape[-1], self.hidden_size).to(self.device)
                    fourier_feat_proj = projection(fourier_feat)
                else:
                    fourier_feat_proj = fourier_feat
                
                single_feat_dict = {ntype: fourier_feat_proj}
                causal_feat, _ = self.causal_learner(single_feat_dict)
                
                # Broadcast causal features to all nodes
                if causal_feat.numel() > 0:
                    mean_causal_feat = causal_feat.mean(dim=0, keepdim=True)
                    if fourier_feat_proj.shape[0] > 1:
                        causal_feat_expanded = mean_causal_feat.repeat(fourier_feat_proj.shape[0], 1)
                    else:
                        causal_feat_expanded = mean_causal_feat
                else:
                    causal_feat_expanded = torch.zeros_like(fourier_feat_proj)
            else:
                fourier_feat_proj = torch.zeros(0, self.hidden_size).to(self.device)
                causal_feat_expanded = torch.zeros(0, self.hidden_size).to(self.device)
            
            # 3. Fuse features
            if fourier_feat_proj.numel() > 0 and causal_feat_expanded.numel() > 0:
                concat_features = torch.cat([fourier_feat_proj, causal_feat_expanded], dim=1)
                fused_features = self.feature_fusion(concat_features)
                
                # 4. Residual connection
                if feat.shape[-1] == self.input_dim:
                    residual = self.residual_connection(feat)
                else:
                    # If input feature dimension does not match, use dynamic projection
                    residual_proj = nn.Linear(feat.shape[-1], self.hidden_size).to(self.device)
                    residual = residual_proj(feat)
                
                enhanced_features[ntype] = fused_features + residual
            else:
                enhanced_features[ntype] = torch.zeros(0, self.hidden_size).to(self.device)
        
        return enhanced_features

class MCGNN_Layer(nn.Module):
    def __init__(self, meta_paths, test_data, hidden_size, r_vec, layer_num_heads, 
                 dropout, etypes, name, device='cpu', is_first_layer=False, wavelet_type='haar'):
        super(MCGNN_Layer, self).__init__()
        self.num_heads = layer_num_heads
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.r_vec = r_vec
        self.etypes = etypes
        self.hidden_size = hidden_size
        self.test_data = test_data
        self.device = device
        self.is_first_layer = is_first_layer
        
        # Choose different feature learners based on whether it is the first layer
        if is_first_layer:
            self.feature_learner = FirstLayerFeatureLearner(
                in_size=hidden_size,
                hidden_size=hidden_size,
                wavelet_type=wavelet_type,
                device=device
            )
        else:
            # The input dimension of subsequent layers is hidden_size * num_heads
            self.feature_learner = SubsequentLayerFeatureLearner(
                input_dim=hidden_size * layer_num_heads,
                hidden_size=hidden_size,
                device=device,
                n_stages=1
            )
        
        # Semantic encoder
        self.semantic_encoder_layer = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.semantic_encoder_layer.append(
                SemanticEncoder(self.num_heads, self.hidden_size, self.r_vec, 
                              self.etypes[i], device).to(device))
        
        # Message aggregator
        self.message_aggregator_layer = nn.ModuleList()
        for i in name:
            self.message_aggregator_layer.append(
                MessageAggregator(self.num_heads, self.hidden_size, 
                                attn_drop=dropout, alpha=0.01, 
                                name=i, device=device).to(device))
        
        # Subgraph fusion
        self.subgraph_fusion = Subgraph_Fusion(
            in_size=self.hidden_size * self.num_heads, 
            hidden_size=self.hidden_size,
        ).to(device)
        
        self.separate_metapath_subgraph = Separate_subgraph()
        self.exclude_test = Prevent_leakage(self.test_data)
        
    def stack_embedding(self, embeddings):
        """Stack embeddings, ensuring all tensor dimensions are consistent"""
        if not embeddings:
            return torch.zeros(0, 0).to(self.device)
        
        # Check shapes of all embeddings
        shapes = [emb.shape for emb in embeddings if emb.numel() > 0]
        if not shapes:
            return torch.zeros(0, 0).to(self.device)
        
        # Find maximum dimensions
        max_nodes = max([shape[0] for shape in shapes])
        max_features = max([shape[1] for shape in shapes]) if len(shapes[0]) > 1 else 0
        
        # Adjust dimensions of each embedding
        adjusted_embeddings = []
        for emb in embeddings:
            if emb.numel() == 0:
                adjusted = torch.zeros(max_nodes, max_features).to(self.device)
            else:
                current_nodes, current_features = emb.shape
                # Adjust number of nodes
                if current_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - current_nodes, current_features).to(self.device)
                    adjusted = torch.cat([emb, padding], dim=0)
                elif current_nodes > max_nodes:
                    adjusted = emb[:max_nodes]
                else:
                    adjusted = emb
                
                # Adjust feature dimension
                if current_features < max_features:
                    padding = torch.zeros(max_nodes, max_features - current_features).to(self.device)
                    adjusted = torch.cat([adjusted, padding], dim=1)
                elif current_features > max_features:
                    adjusted = adjusted[:, :max_features]
            
            adjusted_embeddings.append(adjusted)
        
        # Stack
        if adjusted_embeddings:
            return torch.stack(adjusted_embeddings, dim=1)
        else:
            return torch.zeros(max_nodes, 0).to(self.device)
    
    def generate_metapath_instances(self, g, meta_path):
        edges = [g.edges(etype=f"{meta_path[j]}_{meta_path[j + 1]}") for j in range(len(meta_path) - 1)]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]
        df_0 = pd.DataFrame(edges[0], index=list(meta_path)[:2]).T
        df_1 = pd.DataFrame(edges[1], index=list(meta_path)[-2:]).T
        metapath_instances = pd.merge(df_0, df_1, how='inner')
        filt_metapath_instances = metapath_instances[['g', 'm', 'd']]
        filt_metapath_instances = self.exclude_test(filt_metapath_instances)
        metapath_instances = filt_metapath_instances[list(meta_path)]
        return metapath_instances

    def forward(self, g, h, stage_idx=0):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = self.separate_metapath_subgraph(g, meta_path)
        
        # Node feature learning (first layer includes wavelet, subsequent layers do not)
        if self.is_first_layer:
            enhanced_h = self.feature_learner(h)
        else:
            enhanced_h = self.feature_learner(h, stage_idx=stage_idx)
        
        semantic_embeddings = {'g': [], 'm': [], 'd': []}
        nodes_embeddings = {}
        
        # Metapath processing logic
        for i, meta_path in enumerate(self.meta_paths):
            edata_list = []
            new_g = self._cached_coalesced_graph[meta_path]
            metapath_instances = self.generate_metapath_instances(new_g, meta_path)
            
            for j in range(len(meta_path)):
                ntype = list(meta_path)[j]
                edata_list.append(
                    F.embedding(torch.tensor(metapath_instances.iloc[:, j]).to(self.device), 
                                enhanced_h[ntype]).unsqueeze(1))
            
            edata = torch.hstack(edata_list)
            metapathembedding = self.semantic_encoder_layer[i](edata)
            
            semantic_embeddings['g'].append(
                self.message_aggregator_layer[0](new_g.nodes('g').tolist(), metapath_instances, 
                                                 metapathembedding, enhanced_h['g']))
            semantic_embeddings['m'].append(
                self.message_aggregator_layer[1](new_g.nodes('m').tolist(), metapath_instances, 
                                                 metapathembedding, enhanced_h['m']))
            semantic_embeddings['d'].append(
                self.message_aggregator_layer[2](new_g.nodes('d').tolist(), metapath_instances, 
                                                 metapathembedding, enhanced_h['d']))

        for ntype in semantic_embeddings.keys():
            if ntype == 'g':
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], g_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 'm' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], m_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 'd' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], d_beta = self.subgraph_fusion(semantic_embeddings[ntype])
        
        return nodes_embeddings


class Predictor(nn.Module):
    """Predictor"""
    def __init__(self, hidden_size, num_heads, dropout, device='cpu'):
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device = device
        self.input_dim = hidden_size * num_heads * 3
        
        # Predictor structure
        self.predict = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size * 2),
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(True),
            
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def forward(self, h_concat):
        """Prediction"""
        prediction = self.predict(h_concat)
        return prediction


class MCGNN(nn.Module):
    def __init__(self, meta_paths, test_data, in_size, hidden_size, num_heads, 
                 dropout, etypes, device='cpu', wavelet_type='haar', num_layers=3):
        super(MCGNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device = device
        self.num_layers = num_layers
        
        # Basic feature transformation layers
        self.fc_g = nn.Sequential(
            nn.Linear(in_size['g'], hidden_size),
            nn.ReLU()
        ).to(device)
        
        self.fc_m = nn.Sequential(
            nn.Linear(in_size['m'], hidden_size),
            nn.ReLU()
        ).to(device)
        
        self.fc_d = nn.Sequential(
            nn.Linear(in_size['d'], hidden_size),
            nn.ReLU()
        ).to(device)
        
        # Relation vectors
        r_vec = nn.Parameter(torch.empty(size=(3, self.hidden_size // 2, 2))).to(device)
        nn.init.xavier_normal_(r_vec, gain=1.414)
        
        # Create multiple layers (first layer includes wavelet, subsequent layers do not)
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            is_first_layer = (layer_idx == 0)
            layer = MCGNN_Layer(
                meta_paths=meta_paths,
                test_data=test_data,
                hidden_size=hidden_size,
                r_vec=r_vec,
                layer_num_heads=num_heads,
                dropout=dropout,
                etypes=etypes,
                name=['g', 'm', 'd'], 
                device=device,
                is_first_layer=is_first_layer,
                wavelet_type=wavelet_type
            ).to(device)
            self.layers.append(layer)
        
        # Predictor
        self.predictor = Predictor(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        
        # Inter-layer fusion
        self.layer_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * num_heads, hidden_size * num_heads),
                nn.ReLU(),
                nn.LayerNorm(hidden_size * num_heads)
            ) for _ in range(num_layers - 1)
        ])
        
        # Initialize weights
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)
    
    def get_embed_map(self, features, embed_features, data):
        stack_embedding = {'g': list(), 'm': list(), 'd': list()}
        
        for i, ntype in enumerate(['g', 'm', 'd']):
            for j in range(len(data)):
                node_idx = int(data[j][i])
                if node_idx < len(embed_features[ntype]):
                    stack_embedding[ntype].append(embed_features[ntype][node_idx])
                else:
                    # If node index is out of range, use original features
                    stack_embedding[ntype].append(
                        torch.hstack([features[ntype][node_idx]] * self.num_heads).to(self.device))
            
            stack_embedding[ntype] = torch.stack(stack_embedding[ntype], dim=0)
        
        embedding_concat = torch.cat((stack_embedding['g'], stack_embedding['m'], stack_embedding['d']), dim=1)
        return embedding_concat
    
    def forward(self, g, inputs, data):
        # Perform initial transformation on features of each node type
        h_trans = {}
        h_trans['g'] = self.fc_g(inputs['g']).view(-1, self.hidden_size)
        h_trans['m'] = self.fc_m(inputs['m']).view(-1, self.hidden_size)
        h_trans['d'] = self.fc_d(inputs['d']).view(-1, self.hidden_size)
        
        # Multi-layer feature learning
        all_layer_embeddings = []
        current_features = h_trans
        
        for layer_idx, layer in enumerate(self.layers):
            # Pass through current layer
            layer_embed = layer(g, current_features, stage_idx=layer_idx)
            all_layer_embeddings.append(layer_embed)
            
            # If not the last layer, perform inter-layer fusion
            if layer_idx < self.num_layers - 1:
                # Use current layer output as next layer input
                current_features = {}
                for ntype in ['g', 'm', 'd']:
                    if ntype in layer_embed and layer_embed[ntype].numel() > 0:
                        # Apply inter-layer fusion
                        if layer_idx < len(self.layer_fusion):
                            current_features[ntype] = self.layer_fusion[layer_idx](layer_embed[ntype])
                        else:
                            current_features[ntype] = layer_embed[ntype]
                    else:
                        current_features[ntype] = torch.zeros(0, self.hidden_size * self.num_heads).to(self.device)
        
        # Use embedding from the last layer
        final_embed = all_layer_embeddings[-1]
        
        # Main prediction
        h_concat = self.get_embed_map(h_trans, final_embed, data)
        main_score = self.predictor(h_concat)
        
        return main_score