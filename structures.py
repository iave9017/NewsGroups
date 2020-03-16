class TokenInfo:
    def __init__(self):
        self.occList = []
        self.idf = 0.0
class TokenOccurence:
    def __init__(self,d,c):
        self.docID = d
        self.count = c
class DocRef:
    def __init__(self):
        self.f = np.nan
        self.flength = 0.0