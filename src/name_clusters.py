import os
import sys
import csv
import argparse
sys.path.append(os.path.dirname(sys.path[0]))  # so that config can be imported from project root
from app import db
from app.models import OrderType


def run_manual():
    order_types = OrderType.query.all()

    print('%d cluster order types have been identified through clustering')
    print('Orders should be described in vernacular in addition to statistics')
    print('It is recommended that you view all clusters via heatmaps before describing')
    print('For this, see notebooks/develop/Cluster_Heatmaps.ipynb')

    for ot in order_types:
        stats = '\n'.join(['{}: {}'.format(i, v) for i, v in ot.__dict__.items()])
        print(stats + '\n')
        desc = input('Describe this cluster:\n\t')
        if not input:
            print('Leaving description unchanged and moving on')
            continue
        ot.desc = desc
        db.session.add(ot)

    if input('Want to commit these results? [Y/N]').lower() == 'y':
        db.session.commit()


def run_from_file(path):
    with open(path) as f:
        reader = csv.reader(f)
    header = reader.__next__()
    for line in reader:
        ot = OrderType.query.filter_by(label=line[0])
        ot.desc = line[1]
        db.session.add(ot)
    db.session.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Connect to database and edit description of each ordertype")
    parser.add_argument('-f', '--file', default=None,
                        help='Will read descriptions from file if provided'
                             '(file should be (label,desc) with header in .csv format')

    args = parser.parse_args()

    if args.file:
        run_from_file(args.file)
    else:
        run_manual()
