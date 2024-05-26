import sys
sys.path.append("/Users/lihaoyang/Projects/srn-m/SRNet")
from nets import ModuleCGPNet, ModuleEQLNet
from sr_models import CGPModel, ImageCGPModel
from usr_models import DiffCGPModel, EQL, ImageEQL
from parameters import CGPParameter, EQLParameter
from functions import default_functions, add_sg_functions
from evolution import cgp_evo, diff_cgp_evo, cgp_layer_evo


srnets = {
    "MCGP_net": (CGPParameter, ModuleCGPNet),
    "MEQL_net": (EQLParameter, ModuleEQLNet)
}


evolutions = {
    "MCGP_net": cgp_evo,
    "layer": cgp_layer_evo
}


register_sr = {
    "cgp": CGPModel,
    "CGP": CGPModel,
    "ucgp": DiffCGPModel,
    "eql": EQL,
    "EQL": EQL,

    "ImageCGP": ImageCGPModel,
    "ImageEQL": ImageEQL
}

register_params = {
    "cgp": CGPParameter,
    "CGP": CGPParameter,
    "ucgp": CGPParameter,
    "eql": EQLParameter,
    "EQL": EQLParameter,

    "ImageCGP": CGPParameter,
    "ImageEQL": EQLParameter
}
