import argparse
import datetime
import os
import time
import re
import textwrap

from loguru import logger
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from Bio.SeqIO.FastaIO import SimpleFastaParser


def setup_args():
    m_parser = argparse.ArgumentParser(description="The GISAID data subset creation for RISBM calculations",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    m_parser.add_argument('-meta', '--meta',
                          type=str, required=True,
                          help="Input path for GISAID metadata table, TSV, see prefs.yaml")

    m_parser.add_argument('-fasta', '--fasta',
                          type=str, required=True,
                          help="Input path for GISAID sequences, FASTA")

    m_parser.add_argument('-out', '--output',
                          type=str, required=True,
                          help="Output patter, three files would be created here — metadata, fasta, logs.")

    m_parser.add_argument('-prefs', '--prefs',
                          type=str, required=False,
                          help="Input path for preferences file, YAML")

    return m_parser.parse_args()


if __name__ == "__main__":
    args = setup_args()

    out_fasta = os.path.join(args.out, "_sequences.fasta")
    out_meta = os.path.join(args.out, "_meta.tsv")
    out_logs = os.path.join(args.out, "_notes.log")

    logger.add(out_logs, level="DEBUG",
               colorize=True,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {line} - {message}",
               diagnose=False)

    logger.info("Start")

    logger.debug(f"INPUT META: {args.meta}")
    logger.debug(f"INPUT FASTA: {args.fasta}")

    logger.debug(f"INPUT PREFERENCES: {args.prefs}")

    logger.debug(f"OUT FASTA: {out_fasta}")
    logger.debug(f"OUT META: {out_meta}")
    logger.debug(f"OUT LOGS: {out_logs}")

    with open(args.prefs, "r") as fr:
        preferences = yaml.load(fr, Loader=yaml.SafeLoader)

    st_t = time.time()

    # parse_dates=['Collection date', 'Submission date']  можно добавить в чтение таблицы, если нужны даты
    df = pd.read_csv(args.meta, sep="\t", dtype=preferences["META_COLS"],
                     usecols=list(preferences["META_COLS"].keys()))

    logger.debug(f"DataFrame shape: {df.shape[0]}\t|\tBefore dropping empty indexes")

    df.drop_duplicates(subset=["Virus name"], keep=False, inplace=True)
    df.set_index('Virus name', inplace=True)

    df['N-Content'] = df['N-Content'].fillna(0).astype(float)
    logger.debug(f"DF is read and set within {time.time() - st_t}")

    print(df.head(3))
    logger.debug(f"DataFrame shape: {df.shape[0]}\t|\tAt start")

    df = df[
        (df['N-Content'] >= preferences['subs']['lb_n_content']) &
        (df['N-Content'] <= preferences['subs']['rb_n_content']) &
        (df['Sequence length'] >= preferences['subs']['lb_length']) &
        (df['Sequence length'] <= preferences['subs']['rb_length'])
        ]
    logger.debug(f"DataFrame shape: {df.shape[0]}\t|\tAfter NC and Length")

    df = df[
        (df["Collection date"].astype(np.datetime64) >= datetime.datetime.strptime(preferences['subs']['lb_date'],
                                                                                   "%Y-%m-%d")) &
        (df["Collection date"].astype(np.datetime64) <= datetime.datetime.strptime(preferences['subs']['rb_date'],
                                                                                   "%Y-%m-%d"))
        ]

    logger.debug(f"DataFrame shape: {df.shape[0]}\t|\tAfter collection date")

    df = df[df['Pango lineage'].str.contains(preferences['subs']['lineage_regex'], na=False)]
    logger.debug(f"DataFrame shape: {df.shape[0]}\t|\tAfter lineage")

    loc_includes = ~df["Location"].str.startswith("")

    if preferences['subs'].get('inc_location'):
        for elem in preferences['subs']['inc_location']:
            loc_includes += df['Location'].str.contains(elem, na=False)
    df = df[loc_includes].copy(deep=True)
    logger.debug(f"DataFrame shape: {df.shape[0]}\t|\tAfter including all locations")

    # Исключение локаций
    if preferences['subs'].get('exc_location'):
        for elem in preferences['subs']['exc_location']:
            df = df[~df['Location'].str.contains(elem, na=False)]
            logger.debug(f"DataFrame shape: {df.shape[0]}\t|\tAfter excluding location {elem}")

    asdi = df.to_dict('index')

    SEQ_PR = ">SARS-CoV-2/"

    # открываем файлы на чтение
    fa_wr = open(out_fasta, "w")
    me_wr = open(out_meta, "w")

    # пишем наименования
    columns_to_use = df.columns.tolist()
    me_wr.write("\t".join(['Virus name'] + columns_to_use) + "\n")

    st_t = time.time()
    logger.debug("ITERATION START")
    df_bar = tqdm(total=df.shape[0], position=0, desc="DF elements progress")
    fasta_bar = tqdm(position=1, desc="Fasta progress")

    with open(args.fasta, "r") as faf_r:
        parser = SimpleFastaParser(faf_r)
        for record in parser:  # получаем запись
            c_value = asdi.get(record[0].split("|")[0])  # ищем имя вируса в словаре
            if c_value:
                # создаем имя для последовательности
                loc_split = c_value['Location'].split("/")
                if len(loc_split) == 1:
                    loc_connected = loc_split[0] + "/GeoError"
                elif len(loc_split) == 2:
                    loc_connected = loc_split[1] + "/Unknown"
                else:
                    loc_connected = loc_split[1] + "/" + loc_split[2]
                loc_subbed = re.sub(r'[^a-zA-Z\d/]', "_", re.sub(r"[\s\-().\'`]", "", loc_connected))
                sequence_name = SEQ_PR + loc_subbed + "-" + c_value["Accession ID"] + "/" + c_value['Collection date']
                # пишем последовательность
                fa_wr.write(sequence_name + "\n" + '\n'.join(textwrap.wrap(record[1], 80)) + "\n")
                # составляем строку метаданных
                wr_str = record[0] + "\t" + "\t".join(map(str, [c_value[x] for x in columns_to_use])) + "\n"
                # пишем строку метаданных
                me_wr.write(wr_str)
                # обновляем прогресс бар по таблице
                df_bar.update(1)
            fasta_bar.update(1)
    logger.success(f"ITERATION OVER within {time.time() - st_t}")

    fa_wr.close()
    me_wr.close()
