from typing import Dict

from.admm_option import AdmmOption

class AdmmMetadata:

    def __init__(self, clusters, admmopt: AdmmOption) -> None:
        self.clusters = clusters
        self.admmopt: AdmmOption = admmopt


    def to_dict(self) -> Dict:
        admmopt_dict = {
            attr: getattr(self.admmopt, attr)
            for attr in dir(self.admmopt) if not attr.startswith('__')
        }

        return {
            'clusters': self.clusters,
            'admmopt': admmopt_dict
        }
