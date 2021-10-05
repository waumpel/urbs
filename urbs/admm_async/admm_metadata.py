from typing import Dict, List

from.admm_option import AdmmOption

class AdmmMetadata:

    def __init__(self, clusters, admmopt: AdmmOption, flow_global_sizes: List[int]) -> None:
        self.clusters = clusters
        self.admmopt: AdmmOption = admmopt
        self.flow_global_sizes = flow_global_sizes


    def to_dict(self) -> Dict:
        admmopt_dict = {
            attr: getattr(self.admmopt, attr)
            for attr in dir(self.admmopt) if not attr.startswith('__')
        }

        return {
            'admmopt': admmopt_dict,
            'clusters': self.clusters,
            'flow_global_sizes': self.flow_global_sizes,
        }
