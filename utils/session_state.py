import panel as pn

class SessionState:
    def __init__(self, adata):
        self.adata = adata
        self.session_loaded = False

    @staticmethod
    #@pn.cache
    def initialize(adata) -> "SessionState":
        inst = SessionState(adata)
        inst._initialize_heavy()
        return inst

    def _initialize_heavy(self):
        if self.session_loaded:
            return self

        self.session_loaded = True

        return self
