from typing import Dict

from.admm_option import AdmmOption

class AdmmMetadata:

    def __init__(self, n_clusters, admmopt: AdmmOption) -> None:
        self.n_clusters = n_clusters
        self.admmopt: AdmmOption = admmopt


    def to_dict(self) -> Dict:
        admmopt_dict = {
            attr: getattr(self.admmopt, attr)
            for attr in dir(self.admmopt) if not attr.startswith('__')
        }

        return {
            'n_clusters': self.n_clusters,
            'admmopt': admmopt_dict
        }
