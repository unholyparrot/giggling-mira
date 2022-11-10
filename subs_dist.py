import argparse
import re
from functools import reduce

import numpy as np
import pandas as pd

from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm


def setup_args():
    parser = argparse.ArgumentParser(description="The calculation of the mutation distances " +
                                                 "according to the Nextclade output",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-first', '--first',
                        type=str, required=True,
                        help="Input table for the distances calculation, records stay for rows in matrix")

    parser.add_argument('-second', '--second',
                        type=str, required=False,
                        help="Secondary table, records stay for columns in matrix")

    parser.add_argument('-out', '--output',
                        type=str, required=True,
                        help="Pattern for the output, three files would be created here -- matrix, columns, rows.")

    parser.add_argument('-lb', '--lb',
                        type=int, required=False,
                        default=266, help="Left border for the ignored mutations")

    parser.add_argument('-rb', '--rb',
                        type=int, required=False,
                        default=29674, help="Right border for the ignored mutations")

    return parser.parse_args()


def main():
    args = setup_args()
    logger.add(f"{args.output}_records.log")
    logger.debug(f"Output would be like {args.output}_example.txt")
    logger.debug(f"Left border: {args.lb}, right border: {args.rb}")

    def clear_weak_mutations(record, lb=args.lb, rb=args.rb):
        to_ret = []
        splitted = record.split(",")
        for elem in splitted:
            if re.match(r'^[A-Z]\d{1,5}[A-Z]$', elem):
                if lb <= int(elem[1:-1]) <= rb:
                    to_ret.append(elem)
        return ",".join(to_ret)

    def get_functional_interval(elem):
        res = []
        if elem == '':
            elem = '99999'
        for cur in elem.split(','):
            interval = list(map(int, cur.split('-')))
            if len(interval) == 1:
                res.append(lambda arr, _value=interval[0]: arr == _value)
            elif len(interval) == 2:
                res.append(lambda arr, a=interval[0], b=interval[1]: np.logical_and(arr >= a, arr <= b))
            else:
                raise ValueError(f"Bad interval: {cur}")
        return lambda x: reduce(np.logical_or, [op(x) for op in res.copy()])

    def prepare_table(path, table_idx=1):
        df = pd.read_csv(path, sep="\t",
                         usecols=["seqName", "totalMissing", "substitutions", "missing"],
                         dtype=str).fillna("")

        df.drop(df[df["totalMissing"] == ""].index, inplace=True)  # убираем из таблицы плохие записи

        df.set_index("seqName", inplace=True)  # меняем индекс для облегчения взаимодействий

        df["substitutions"] = df["substitutions"].apply(clear_weak_mutations)  # чистим от "лишних" мутаций

        # создаем номера вместо записей о мутациях
        df['numbers'] = df['substitutions'].apply(
            lambda x: np.array([int(elem[1:-1]) for elem in x.split(',') if re.match(r'^[A-Z]\d{1,5}[A-Z]$', elem)]))
        # создаем интервалы
        df['func_intervals'] = df['missing'].apply(get_functional_interval)

        logger.debug(f"Table {table_idx} is read")

        return df

    def if_skipped_final(_row_1, _row_2):
        return _row_1['func_intervals'](_row_2['numbers']).sum() + _row_2['func_intervals'](_row_1['numbers']).sum()

    logger.debug(f"First table (rows): {args.first}")
    refer = prepare_table(args.first)  # читаем первую таблицу

    # ура, костыль на вторую таблицу
    if args.second:
        logger.debug(f"Second table (columns): {args.second}")
        target = prepare_table(args.second, table_idx=2)
    else:
        logger.debug(f"No second table, matrix would be squared")
        target = refer

    logger.info("Calculation of corrections, prepare your RAM and patience")
    # вычисляем матрицу коррекций скора
    correction_final = np.zeros((refer.shape[0], target.shape[0]))
    for idx_1, (_, row_1) in enumerate(tqdm(refer.iterrows(), total=len(refer))):
        for idx_2, (_, row_2) in enumerate(target.iterrows()):
            value = if_skipped_final(row_1, row_2)
            if value > 0:
                correction_final[idx_1][idx_2] = value

    correction_final = correction_final.astype(int)
    logger.success(f"Correction min: {np.min(correction_final)}, correction max: {np.max(correction_final)}")

    logger.debug("Creating model for distances")
    count_vect = CountVectorizer(tokenizer=lambda x: x.split(','))  # создаем модель

    logger.debug("Fitting all values to model")
    # передаем в неё вообще все значения
    count_vect.fit_transform(refer["substitutions"].values.tolist() + target["substitutions"].values.tolist())

    logger.debug("Passing table values to the model")
    # получаем реальные вектора в пространстве всех мутация для каждой записи
    data_a_counts = count_vect.transform(refer["substitutions"].values.tolist())
    data_b_counts = count_vect.transform(target["substitutions"].values.tolist())

    logger.info("Calculation of intersections, prepare your RAM")
    # вычисляем матрицу пересечений между сетами
    intersections = data_a_counts @ data_b_counts.T
    logger.success(f"Intersections matrix shape {intersections.shape}")

    logger.debug("Calculating the score")
    # получим размеры множества мутаций для каждого сиквенса
    len_a = np.array([len(cur_set) for cur_set in [elem.split(",") for elem in refer["substitutions"]]])
    len_b = np.array([len(cur_set) for cur_set in [elem.split(",") for elem in target["substitutions"]]])
    # получаем матрицу score
    dist_mat = len_a[..., None] + len_b[None, ...] - 2 * intersections

    logger.success(f"Distances min: {dist_mat.min()}, distances max: {dist_mat.max()}")

    dist_mat = dist_mat - correction_final

    logger.success(f"Scores min: {dist_mat.min()}, scores max: {dist_mat.max()}")

    logger.debug("Writing to files")
    as_list = dist_mat.tolist()
    with open(args.output + "_matrix.txt", "w") as wr:
        for row in tqdm(as_list):
            one_line = ";".join(map(str, row)) + "\n"
            wr.write(one_line)
    with open(args.output + "_columns.txt", "w") as wr:
        wr.write("\n".join(target.index))
        wr.write("\n")
    with open(args.output + "_rows.txt", "w") as wr:
        wr.write("\n".join(refer.index))
        wr.write("\n")


if __name__ == "__main__":
    logger.debug("Start")
    main()
