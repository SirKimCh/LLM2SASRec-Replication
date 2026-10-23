"""Model registry over the ORIGINAL code (vendor/llmseqrec, unchanged). Hyper-parameters = the tuned configurations the
authors hard-coded in run_experiments.ipynb (commit 6464b39); only `cores` (15 -> machine) and `num_epochs` (smoke) differ.
Our additions: SASRec with the LLM2SASRec configuration but random init (A1), and FusedSASRec (A3)."""
import numpy as np
import tensorflow as tf
from keras import layers

from main.grurec.grurec import GRURec
from main.llm_based.similarity_model.llm_seq_sim import LLMSeqSim
from main.popularity.session_popular import SessionBasedPopular
from main.sknn.sknn import SessionBasedCF
from main.transformer.bert.bert import BERT
from main.transformer.bert.bert_with_embeddings import BERTWithEmbeddings
from main.transformer.sasrec.sasrec import SASRec
from main.transformer.sasrec.sasrec_with_embeddings import SASRecWithEmbeddings

NEURAL = dict(early_stopping_patience=2, is_verbose=True, pred_batch_size=5000, pred_seen=False, train_val_fraction=0.1)
GRU = {"N": 20, "activation": "relu", "emb_dim": 80, "fit_batch_size": 32, "hidden_dim": 32,
       "optimizer_kwargs": {"learning_rate": 0.0072, "weight_decay": 0.08929}}
SAS = {"L": 2, "N": 20, "activation": "relu", "drop_rate": 0.25, "emb_dim": 48, "fit_batch_size": 32, "h": 3,
       "optimizer_kwargs": {"learning_rate": 0.0048, "weight_decay": 0.09622}, "transformer_layer_kwargs": {"layout": "NFDR"}}
L2S = {"L": 1, "N": 20, "activation": "relu", "drop_rate": 0.0, "emb_dim": 352, "fit_batch_size": 416, "h": 2,
       "optimizer_kwargs": {"learning_rate": 0.0003, "weight_decay": 0.057}, "transformer_layer_kwargs": {"layout": "NFDR"}}
BERTC = {"L": 1, "N": 20, "activation": "gelu", "drop_rate": 0.4, "emb_dim": 192, "fit_batch_size": 256, "h": 2, "mask_prob": 0.9,
         "optimizer_kwargs": {"clipnorm": 92, "learning_rate": 0.0023, "weight_decay": 0.04088}, "transformer_layer_kwargs": {"layout": "FDRN"}}
L2B = {"L": 2, "N": 20, "activation": "gelu", "drop_rate": 0.45, "emb_dim": 448, "fit_batch_size": 224, "h": 1, "mask_prob": 0.65,
       "optimizer_kwargs": {"clipnorm": 5, "learning_rate": 0.0012, "weight_decay": 0.02354}, "transformer_layer_kwargs": {"layout": "FDRN"}}
VSKNN = {"decay": "harmonic", "idf_weighting": False, "k": 320, "sample_size": 1530, "sampling": "random", "similarity_measure": "dot",
         "filter_prompt_items": True, "is_verbose": False}
SEQSIM = {"batch_size": 500, "combination_decay": "harmonic", "embedding_combination_strategy": "mean", "similarity_measure": "cosine",
          "filter_prompt_items": True, "is_verbose": False, "max_session_length_for_decay_precomputation": 500}

NAMES = {"random": "Random", "llm2sasrec_shuf": "LLM2SASRec-shuffled", "popular": "Popular", "vsknn": "V-SKNN", "gru4rec": "GRU4Rec", "sasrec": "SASRec", "sasrec_l2s": "SASRec-L2Scfg",
         "bert4rec": "BERT4Rec", "llm2sasrec": "LLM2SASRec", "llm2bert4rec": "LLM2BERT4Rec", "llmseqsim": "LLMSeqSim", "fused": "Fused-SASRec"}
NEEDS_EMB = {"llm2sasrec", "llm2bert4rec", "llmseqsim", "fused", "llm2sasrec_shuf"}


class RandomRec:
    """Sanity baseline (requirement R3): uniform random items of the training catalogue, prompt items excluded, seeded."""

    def __init__(self, seed=42):
        self.seed = seed

    def train(self, train_data):
        self.items = np.unique(train_data["ItemId"].to_numpy())

    def predict(self, predict_data, top_k=10):
        rng, out = np.random.default_rng(self.seed), {}
        for sid, prompt in predict_data.items():
            cand = rng.choice(self.items, top_k + len(prompt), replace=False)
            out[sid] = cand[~np.isin(cand, prompt)][:top_k]
        return out


class ShuffledSASRecWithEmbeddings(SASRecWithEmbeddings):
    """Control for A1 (G2): LLM2SASRec whose LLM rows are permuted across items before the PCA (PCA is invariant to row
    order), so the initial table keeps exactly the same distribution/scale but loses the item-to-text semantics."""

    def __init__(self, shuffle_seed=42, **kw):
        self.shuffle_seed = shuffle_seed   # before super(): the vendor __init__ logs extract_config(self), which reads it back
        super().__init__(**kw)

    def train(self, train_data):
        E = SASRecWithEmbeddings.product_index_to_embedding
        SASRecWithEmbeddings.product_index_to_embedding = E[np.random.default_rng(self.shuffle_seed).permutation(len(E))]
        super().train(train_data)

    def name(self):
        return "LLM2SASRec-shuffled"


class FusedItemEmbedding(layers.Layer):
    """Frozen (L2-normalised) LLM table -> trainable linear adapter, plus a zero-initialised trainable ID embedding."""

    def __init__(self, table, emb_dim):
        super().__init__()
        self.table = tf.constant(table, dtype=tf.float32)
        self.adapter = layers.Dense(emb_dim, use_bias=False)
        self.ids = layers.Embedding(table.shape[0], emb_dim, embeddings_initializer="zeros")

    def call(self, x):
        return self.adapter(tf.gather(self.table, x)) + self.ids(x)

    @property
    def embeddings(self):
        """Full fused table (num_items + 1, emb_dim). The original ProjectionHead ties its output weights to
        `item_embedder.embeddings`, so the fused table must serve the output side as well (single-variable ablation)."""
        return self.adapter(self.table) + self.ids(tf.range(tf.shape(self.table)[0]))


class FusedSASRec(SASRecWithEmbeddings):
    """LLM2SASRec's model and configuration, but the item embedding is FusedItemEmbedding instead of a PCA initialisation."""

    def train(self, train_data):
        self.temp_model = SASRec(**dict(self.sasrec_config, num_epochs=0))   # builds the Keras model + id mapping, no training
        self.temp_model.train(train_data)
        order = list(dict(sorted(self.temp_model.id_reducer.id_lookup.items())).values())
        E = SASRecWithEmbeddings.product_index_to_embedding[[SASRecWithEmbeddings.product_id_to_index[i] for i in order]]
        E = E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-12)
        E = np.vstack([E, np.zeros((1, E.shape[1]))])                      # row for the mask/padding id
        fused = FusedItemEmbedding(E, self.emb_dim)
        self.temp_model.model.embedding_layer.item_emb = fused
        self.temp_model.model.projection_head.item_embedder = fused     # keep input and output tables tied, as in SASRec
        self.temp_model.model.compile_model()
        SASRec.train(self, train_data)

    def name(self):
        return "Fused-SASRec"


class SessionBasedCFInProcess(SessionBasedCF):
    """V-SKNN prediction without the multiprocessing Pool. The original passes the prompts to its workers through a module
    global (works with fork on Linux; with spawn on Windows the workers re-import the module and see None). Same algorithm,
    same arguments, the authors' own _predict_chunk — only run in this process."""

    def predict(self, predict_data, top_k=10):
        import main.sknn.sknn as S
        S.global_predict_data = predict_data
        return S.SessionBasedCF._predict_chunk(list(predict_data), self.k, self.sample_size, self.sampling, self.use_item_embeddings,
                                               self.prompt_session_emb_comb_strategy, self.sequential_weighting, self.sequential_filter,
                                               self.decay, self.similarity_measure, self.filter_prompt_items, self.last_n_items,
                                               self.idf_weighting, top_k)


def build(name, emb_path, emb_id, cores, epochs, item_dim=None, seed=42):
    """-> (model, config dict as used). LLM2BERT4Rec keeps the authors' LDA for PaLM (needs the 'class' column of their
    files); other embedding sources use PCA. LLMSeqSim reduces to 512 dims as in the paper, or keeps all when dim <= 512."""
    nn = dict(NEURAL, cores=cores, num_epochs=epochs)
    emb = {"product_embeddings_location": emb_path, "red_params": {}}
    if name == "random":
        cfg = {"seed": seed}; return RandomRec(seed), cfg
    if name == "llm2sasrec_shuf":
        cfg = dict(L2S, **nn, **emb, red_method="PCA"); return ShuffledSASRecWithEmbeddings(shuffle_seed=seed, **cfg), dict(cfg, shuffle_seed=seed)
    if name == "popular":
        cfg = {"cores": cores, "is_verbose": False}; return SessionBasedPopular(**cfg), cfg
    if name == "vsknn":
        cfg = dict(VSKNN, cores=cores, sample_random_state=seed); return SessionBasedCFInProcess(**cfg), cfg
    if name == "gru4rec":
        cfg = dict(GRU, **nn); return GRURec(**cfg), cfg
    if name == "sasrec":
        cfg = dict(SAS, **nn); return SASRec(**cfg), cfg
    if name == "sasrec_l2s":
        cfg = dict(L2S, **nn); return SASRec(**cfg), cfg
    if name == "bert4rec":
        cfg = dict(BERTC, **nn); return BERT(**cfg), cfg
    if name == "llm2sasrec":
        cfg = dict(L2S, **nn, **emb, red_method="PCA"); return SASRecWithEmbeddings(**cfg), cfg
    if name == "fused":
        cfg = dict(L2S, **nn, **emb, red_method="PCA"); return FusedSASRec(**cfg), cfg
    if name == "llm2bert4rec":
        cfg = dict(L2B, **nn, **emb, red_method="LDA" if emb_id == "palm" else "PCA"); return BERTWithEmbeddings(**cfg), cfg
    if name == "llmseqsim":
        red = min(512, item_dim or 512)
        cfg = dict(SEQSIM, cores=cores, dim_reduction_config={"normalize": True, "reduced_dim_size": red,
                                                                 "reduction_config": {"reduction_technique": "pca", "config": {}}})
        return LLMSeqSim(**cfg), cfg
    raise ValueError(name)
