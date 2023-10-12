import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="source image filename")
    parser.add_argument("-t", "--target", type=str, help="target image filename")
    parser.add_argument("-o", "--output", type=str, help="output image filename")
    parser.add_argument("-m", "--mask",type=str, help="mask image filename")

    return parser
