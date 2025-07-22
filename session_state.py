import panel as pn
#import yaml
from anndata import read_h5ad
import io
import h5py
from functools import lru_cache

class SessionState:
    #def __init__(self, config_path: str = "config.yaml"):
    def __init__(self, adata):
        self.adata = adata
        self.session_loaded = False

        # placeholders maybe fill ? idk
        self.preprocess_results = None
        self.intermediate_results = None
        self.normalization_eval = None
        self.imputation_eval = None

    @staticmethod
    @pn.cache
    def initialize(adata) -> "SessionState":
        inst = SessionState(adata)
        inst._initialize_heavy()
        return inst

    def _initialize_heavy(self):
        if self.session_loaded:
            return self

        self.session_loaded = True

        return self
