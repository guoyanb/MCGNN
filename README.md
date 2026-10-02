MCGNN: Multiscale Cross-Type Graph Neural Networks with Wavelet and Fourier Transforms for Direction-Aware
Gene-Microbe-Disease Association Mining

This repository contains the official implementation of MCGNN, a multiscale cross-type graph neural network for direction-aware gene–microbe–disease association mining. Mining multi-entity associations in heterogeneous
biological networks is fundamentally challenged by noise, sparsity, and the difficulty of disentangling directed dependencies from
spurious correlations. Existing methods typically treat spectral enhancement and directed interaction modeling as separate steps,
overlooking their complementarity. To bridge this gap, we propose the multiscale cross-type graph neural network (MCGNN), which
unifies spectral decomposition and directed dependency learning in a single heterogeneous graph learning framework. Following an enhance-then-propagate paradigm, MCGNN first enriches node features through a dual-domain spectral module: wavelet
decomposition captures multiresolution local details, while multistage Fourier convolutions with learnable complex filters
amplify globally salient frequency bands and suppress noise. A complex-valued rotation encoder preserves directed relational
semantics along biologically meaningful metapaths. Crucially, a cross-type directed interaction learner employs multi-head
attention to infer directed probabilistic dependencies among gene, microbe, and disease types, constructing an interpretable type-
level directed dependency graph. We emphasize that our framework learns directed probabilistic associations from
observational data to generate mechanistic hypotheses. Moreover, the joint wavelet–Fourier representation conserves complete
signal energy via Parseval’s identity, and establishes the gradient stability of the gated fusion mechanism. Extensive experiments on
triplet datasets show that MCGNN consistently outperforms state-of-the-art baselines. Ablation studies confirm the indispensable
and synergistic contributions of wavelet decomposition, Fourier filtering, and directed dependency learning.

File Structure
File	Description
cuda_wavelet.py	GPU-accelerated Haar wavelet transform, multi-level decomposition, and first-layer wavelet feature extractor.
data_process.py	Data loading, negative sampling, and generation of 5-fold CV and independent test splits.
main.py	Main entry point: runs 5-fold cross-validation and independent test.
model.py	MCGNN model definition: subgraph fusion, semantic encoder, message aggregator, multi-stage Fourier convolution, cross-type interaction learner, feature learners, MCGNN layers, and predictor.
train.py	Training loop with weighted MSE loss, NDCG/MRR evaluation, early stopping, and model checkpointing.
utils.py	Utility functions: loss, evaluation metrics (NDCG, MRR), heterogeneous graph construction, leakage prevention, subgraph separation, early stopping, and argument parser.


Data Preparation
Place the following data files under ./MCGNN/Data/:
mic_sim176.txt – microbe similarity matrix (tab-separated)
gene_sim_BP301.csv – gene functional similarity matrix (CSV)
dis_sim153.txt – disease semantic similarity matrix (tab-separated)
g_m_d_pos_pairs.txt – positive gene–microbe–disease triplets (space/tab-separated, columns: gene, microbe, disease)



bash
python data_process.py
This will create the following directories and CSV files:
./MCGNN/Data/CV_data/CV_1/ … CV_5/ with train_data_pos.csv, train_data_neg.csv, val_data_pos.csv, val_data_neg.csv
./MCGNN/Data/indepent_data/ with train_data_pos.csv, train_data_neg.csv, test_data_pos.csv, test_data_neg.csv

Usage
To run both 5-fold cross-validation and independent test:
python main.py


Output
Training results are printed to console and saved in ./Result/:
Enhanced_MCGNN_CV_Results.txt – 5-fold CV results
Enhanced_MCGNN_Indep_Results.txt – independent test results
Best model checkpoint saved as ./Result/best_model.pth

