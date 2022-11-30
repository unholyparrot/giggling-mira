import argparse
from io import StringIO
import os
import re
from itertools import groupby
import linecache
import tempfile
import shutil
import textwrap

import numpy as np
import pandas as pd

from Bio.Align.Applications import MafftCommandline
from Bio.SeqIO import FastaIO

from tqdm.auto import tqdm
from loguru import logger


# TODO: добавить конвертацию fasta-файлов с записью их в /tmp
def setup_args():
    parser = argparse.ArgumentParser(description="The fixing if the sequences according to the " +
                                                 "distances matrix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-inp', '--input_p',
                        type=str, required=True,
                        help="""Path-like input pattern for the distance table, rows & cols. 
                        Indeed, it will search for the PROVIDED_PATH_PATTERN_columns.txt,
                        PROVIDED_PATH_PATTERN_rows.txt, PROVIDED_PATH_PATTERN_matrix.txt""")

    parser.add_argument('-to_fix', '--to_fix',
                        type=str, required=True,
                        help="Path to the FASTA with sequences to BE fixed. \n" +
                             "Please, provide only FASTA with the two-lines-for-a-record structure")

    parser.add_argument('-fix_kit', '--fix_kit',
                        type=str, required=True,
                        help="Path to the FASTA with the sequence that WILL fix. \n" +
                             "Please, provide only FASTA with the two-lines-for-a-record structure")

    parser.add_argument('-out', '--output',
                        type=str, required=True,
                        help="Patter for the output FASTA with fixed sequences and LOG files to be created")

    parser.add_argument('-ref_lb', '--ref_lb',
                        type=int, required=False,
                        default=266, help="Left border for the reference coding region")

    parser.add_argument('-ref_rb', '--ref_rb',
                        type=int, required=False,
                        default=28577, help="Right border for the reference coding region")

    parser.add_argument('-ref_path', '--ref_path',
                        type=str, required=False,
                        help="Path for the reference to be used")

    return parser.parse_args()


def all_equal(iterable):
    """
    Позаимствовано с StackOverflow, проверяет все переданные в iterable элементы на равенство
    :param iterable: перечисляемая величина
    :return: True, если все в iterable одинаковые, иначе False
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def filter_only(record):
    return record["level"].name == "TRACE"


def main():
    args = setup_args()
    logger.add(f"{args.output}_records.log")
    logger.add(f"{args.output}_trace.log", filter=filter_only, level="TRACE")
    logger.info(f"Output would be like {args.output}_example.txt")

    with open(args.input_p + "_rows.txt", "r") as fr:
        rows_names = fr.read().splitlines()
    with open(args.input_p + "_columns.txt", "r") as fr:
        cols_names = fr.read().splitlines()

    to_fix_dict = dict()
    fixing_dict = dict()

    logger.debug(f"TMP set for sequence's names is introduced")
    tmp_names_set = set(rows_names)

    with open(args.to_fix, "r") as fasta_ali:
        parser = FastaIO.SimpleFastaParser(fasta_ali)
        for idx, (seq_name, seq_body) in tqdm(enumerate(parser), desc="Indexing to_fix Fasta"):
            if seq_name in tmp_names_set:
                to_fix_dict[seq_name] = 2 * idx + 2  # добавляем индекс строки в файле выравнивания для linecache

    tmp_names_set = set(cols_names)

    with open(args.fix_kit, "r") as fasta_ali:
        parser = FastaIO.SimpleFastaParser(fasta_ali)
        for idx, (seq_name, seq_body) in tqdm(enumerate(parser), desc="Indexing fix_kit Fasta"):
            if seq_name in tmp_names_set:
                fixing_dict[seq_name] = 2 * idx + 2  # добавляем индекс строки в файле выравнивания для linecache

    tmp_names_set = None
    logger.debug(f"TMP set is now {tmp_names_set}")

    logger.debug("First heading from to_fix fasta: " + linecache.getline(args.to_fix, 1).rstrip('\n'))
    logger.debug("First heading from fix_kit fasta: " + linecache.getline(args.fix_kit, 1).rstrip('\n'))

    ref_path = args.ref_path if args.ref_path else os.path.join(os.path.split(__file__)[0], "reference.fasta")

    logger.info("Would be used as reference: " + ref_path)

    with open(ref_path, "r") as fasta_ali:
        parser = FastaIO.SimpleFastaParser(fasta_ali)
        for elem in parser:
            reference_name, reference_seq = elem[0], elem[1]

    logger.debug(f"Ref: {reference_name} with {len(reference_seq)}, stripping as ({args.ref_lb}:{args.ref_rb})")

    reference_seq = reference_seq[args.ref_lb - 1:args.ref_rb]

    logger.debug(f"Ref: {reference_name} with {len(reference_seq)} coding region length after stripping")

    var_temp_dir = tempfile.mkdtemp()

    logger.debug("Temporary directory for mafft usage: " + var_temp_dir)
    logger.debug("Remember to delete the tmp directory manually if the program crashes")

    # читаем DataFrame
    df = pd.DataFrame(np.loadtxt(args.input_p + "_matrix.txt", delimiter=";", dtype=float),
                      index=rows_names, columns=cols_names)
    logger.debug(f"Matrix desc: shape {df.shape}; min {np.min(df.to_numpy())}; max {np.max(df.to_numpy())}")

    # убираем все значения на "диагональных" элементах (которые могут иметь и иные порядки)
    for col_name in tqdm(df.columns, total=df.shape[1], desc="Cleaning up DF from collisions"):
        if col_name in df.index:
            df.loc[col_name, col_name] = np.nan

    # начинаем весьма постепенный парсинг
    logger.debug("Iteration over files, fixing and writing")

    out_fwr = open(args.output + "_fixed.fasta", "w")
    out_not_fwr = open(args.output + "_not_fixed.fasta", "w")

    for target in tqdm(df.index, desc="Fixing process and writing to file"):
        # получаем минимальное значение по массиву
        min_value = df.loc[target].sort_values(na_position='last')[0]
        # находим имена всех последовательностей с этим минимальным значением
        asv = df.loc[target][lambda x: x == min_value].index
        logger.trace(f">>{target}; score {min_value}; neighbours {len(asv)}")
        logger.trace(";".join(asv))
        # пишем в файл во временной директории референс, целевую последовательность, ближайшие последовательности
        with open(os.path.join(var_temp_dir, "tmp.fasta"), "w") as wrf:
            wrf.write(f">{reference_name}\n{reference_seq}\n")
            wrf.write(f">{target}\n{linecache.getline(args.to_fix, to_fix_dict[target])}")
            for close_seq in asv:
                wrf.write(f">{close_seq}\n{linecache.getline(args.fix_kit, fixing_dict[close_seq])}")
        # создаем объект для маффта
        mafft_cline = MafftCommandline(input=os.path.join(var_temp_dir, "tmp.fasta"))
        # запускаем объект для маффта
        stdout, stderr = mafft_cline()
        # создаем словарь для хранения выравниваний
        current_proceedings = dict()
        # читаем из stdout в буфер без записи в файл, а с записью в словарь хранения выравниваний
        parser = FastaIO.SimpleFastaParser(StringIO(stdout))
        for ali_name, ali_seq in parser:
            current_proceedings[ali_name] = ali_seq
        # создаем переменную для хранения интервала начала кодирующей части
        search_ali_start = re.match(r"^-+[a-z]", current_proceedings[reference_name])
        # создаем переменную для хранения интервала конца кодирующей части
        search_ali_end = re.search(r"[a-z]-+$", current_proceedings[reference_name])
        # получаем переменные для хранения начала и конца кодирующих частей
        wr_lb, wr_rb = search_ali_start.end() - 1, search_ali_end.start() + 1
        # ищем [nN]+ в кодирующей части целевой последовательности
        found_unc_s = [(m.start(0),
                       m.end(0)) for m in re.finditer("[ryswkmbdhvnRYSWKMBDHVN]+",
                                                      current_proceedings[target][wr_lb:wr_rb])]
        # производим замену выбранных интервалов
        resulting_seq_beginning = current_proceedings[target][:wr_lb]  # кусок, который не чиним
        under_construction = [_ for _ in current_proceedings[target][wr_lb:wr_rb]]  # кусок, который чиним
        resulting_seq_ending = current_proceedings[target][wr_rb:]  # кусок в конце, который не чиним
        # итерируемся по интервалам
        fixed_intervals = 0

        logger.trace(f"Found {len(found_unc_s)} N intervals")

        for cur_int_lb, cur_int_rb in found_unc_s:
            # проверяем, все ли ближайшие последовательности равны в этом интервале
            if all_equal([current_proceedings[x][wr_lb:wr_rb][cur_int_lb:cur_int_rb] for x in asv]):
                # если да, то записываем это в исправленную часть
                under_construction[cur_int_lb:cur_int_rb] = [_ for _ in current_proceedings[asv[0]][wr_lb:wr_rb][
                                                                        cur_int_lb:cur_int_rb]]
                fixed_intervals += 1
                if re.match("[nN]+", current_proceedings[target][wr_lb:wr_rb][cur_int_lb:cur_int_rb]):
                    logger.trace(f"({cur_int_lb}, {cur_int_rb}) fixed UNKNOWN")
                else:
                    logger.trace(f"({cur_int_lb}, {cur_int_rb}) fixed UNSURE")
            else:
                logger.trace(f"({cur_int_lb}, {cur_int_rb}) NOT FIXED")

        # фиксируем созданную исправленную часть
        resulting_seq = resulting_seq_beginning + "".join(under_construction) + resulting_seq_ending
        res_seq_to_wr = "\n".join(textwrap.wrap(resulting_seq.replace("-", "").upper(), 60))

        if fixed_intervals == len(found_unc_s):
            out_fwr.write(f">{target}|{fixed_intervals}/{len(found_unc_s)}|\n" + res_seq_to_wr + "\n")
        else:
            out_not_fwr.write(f">{target}|{fixed_intervals}/{len(found_unc_s)}|\n" + res_seq_to_wr + "\n")

    shutil.rmtree(var_temp_dir)
    logger.debug("Temporary directory was successfully deleted")


if __name__ == "__main__":
    logger.debug("Start")
    main()
