from .fedavg import FedAvg
from .fedprox import FedProx
from .scaffold import SCAFFOLD
from .feddyn import FedDyn
from .fedntd import FedNTD
from .fedvarp import FedVARP
from .fedlc import FedLC
from .feddecorr import FedDecorr
from .fedsol import FedSOL
from .fedacg import FedACG

from .localgc import LocalGC
from .globalgc import GlobalGC
from .gcfed import GCFed


def get_algorithm(algorithm_name, args, train_data, test_data):
    if algorithm_name == "fedavg":
        return FedAvg(args, train_data, test_data)
    elif algorithm_name == "fedprox":
        return FedProx(args, train_data, test_data)
    elif algorithm_name == "scaffold":
        return SCAFFOLD(args, train_data, test_data)
    elif algorithm_name == "feddyn":
        return FedDyn(args, train_data, test_data)
    elif algorithm_name == "fedntd":
        return FedNTD(args, train_data, test_data)
    elif algorithm_name == "fedvarp":
        return FedVARP(args, train_data, test_data)
    elif algorithm_name == "fedlc":
        return FedLC(args, train_data, test_data)
    elif algorithm_name == "feddecorr":
        return FedDecorr(args, train_data, test_data)
    elif algorithm_name == "fedsol":
        return FedSOL(args, train_data, test_data)
    elif algorithm_name == "fedacg":
        return FedACG(args, train_data, test_data)
    elif algorithm_name == "localgc":
        return LocalGC(args, train_data, test_data)
    elif algorithm_name == "globalgc":
        return GlobalGC(args, train_data, test_data)
    elif algorithm_name == "gcfed":
        return GCFed(args, train_data, test_data)

    else:
        raise ValueError(f"Algorithm {algorithm_name} not supported")
