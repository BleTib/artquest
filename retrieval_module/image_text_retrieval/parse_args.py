import argparse

def parse():
    parser = argparse.ArgumentParser(description="Test alignCLIP.")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="configs/semart_retrieval.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--seed', type=int, default=42
                        , help='random seed for gpu.default:5')
    parser.add_argument('--num_gpus', type=int, default=1
                        , help='Number of gpus.')

    args = parser.parse_args()
    return args
