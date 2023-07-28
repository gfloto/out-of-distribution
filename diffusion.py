import os 
import json     
import torch 
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_name', default='dev')
    parser.add_argument('--data_path', default='/drive2/ood/', help='train dataset, either odo or damage')

    args = parser.parse_args()

    # asserts
    assert args.test_name is not None, 'enter a test name'

    return args

def main():
    args = get_args()


if __name__ == '__main__':
    main()