import ast
from argparse import ArgumentParser

from nle_utils.utils.utils import str2bool


def add_extra_params_demo(parser: ArgumentParser):
    """ """
    p = parser
    p.add_argument(
        "--demodir",
        default="demo_data/play_data",
        type=str,
        help="Directory path where data will be saved. " "Defaults to 'demo_data/play_data'.",
    )
    p.add_argument("--demopath", default=None, type=str, help="If exists we will continue playing the demo from it.")
    p.add_argument(
        "--demostep", default=-1, type=int, help="If demopath exists we will continue playing the demo from this step."
    )
    p.add_argument("--save_every_k", default=100, type=int, help="save checkpoint every kth step.")
