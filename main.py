from util import check_path
from runmodel import RunModel
import logging


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='coil20', choices=['coil20', 'coil100', 'orl'])
    args = parser.parse_args()
    print(args)

    db = args.db
    check_path()
    logging.basicConfig(level=logging.DEBUG, filename='./results/' + db + '.log', filemode='a')

    run = RunModel(db)
    run.train_dsc()
