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

    parser.add_argument('--dates_switch',
                        action='store_false', required=False,
                        help="Account for the dates from the sequence naming for the fixing process. \n" +
                             "Turns `False` if passed, no dates correction would be provided.")

    # чтобы не вводить файл настроек, но настраивать эту ерунду хочется быстро и легко
    parser.add_argument('--row_length',
                        type=int, required=False,
                        default=60, help="Style parameter for the length per row in output Fasta(s)")

    return parser.parse_args()


def create_date_f_name(seq_name):
    """
    Извлечение даты забора образца из полного наименования. Предполагается, что дата забора записана после слэша.
    В случае, если даты не найдется, то в качестве даты забора выступит абстрактно большая дата.
    Следует обратить внимание, что в качестве найденной даты будет использоваться первая найденная дата, т.е. если
    в наименование входят две даты последовательности, то будет использована первая из них. \n \n
    Случаи обработки обнаруженной даты по паттерну записи после слэша:
     1) Если дата нормальная -- 2021-07-12, то она и будет обработана в таком виде;
     2) Если дата урезана до месяца -- 2021-07, то в массив пойдет 2021-08-01;
     3) Если дата урезана до года -- 2021, то в массив пойдет 2021-12-31. \n \n
    :param seq_name: полное наименование последовательности;
    :return: объект даты.
    """
    to_ret = "2030-12-31"  # абстрактная большая дата
    pat = re.search(r"/\d{4}-\d{2}-\d{2}", seq_name)  # сперва ищем нормальную запись даты
    if pat:  # если находим, то обрабатываем её и готовим на возврат
        to_ret = np.datetime64(pat.group()[1:])
    else:  # если не находим, снижаем критерии до поиска месяца
        pat = re.search(r"/\d{4}-\d{2}", seq_name)
        if pat:  # если находим, то округляем до первого числа следующего месяца
            to_ret = np.datetime64(pat.group()[1:]) + np.timedelta64(1, 'M') + np.timedelta64(0, 'D')
        else:  # если не находим, вновь снижаем критерии к поиску
            pat = re.search(r"/\d{4}", seq_name)
            if pat:  # если находим, то округляем до 31 декабря этого года
                to_ret = np.datetime64(pat.group()[1:], 'Y') + np.timedelta64(11, 'M') + np.timedelta64(30, 'D')
            # если не находим ни одной даты по паттерну, то вернется изначальная абстрактно большая дата
    return to_ret


class DateFilterException(Exception):
    """
    Кастомный класс Exception для обработки ситуации, когда учет дат привел к отсутствию
    подходящих последовательностей для починки, а вмешиваться в общий алгоритм перебора последовательностей пагубно.
    """
    def __init__(self, seq_name, seq_date):
        self.name = seq_name
        self.date = seq_date
        super(Exception, self).__init__()


def all_equal(iterable):
    """
    Позаимствовано с StackOverflow, проверяет все переданные в iterable элементы на равенство
    :param iterable: перечисляемая величина
    :return: True, если все в iterable одинаковые, иначе False
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def filter_only(record):  # запись логов процесса починки
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
    
    # создаем списки дат для осуществления обоснованного выбора
    # вот это объявляем, чтобы даже если что-то пошло не так, просто вылетали ошибки отсутствия метода, а не NameSpace
    rows_dates, cols_dates = (None, None)
    # если все же ключ будет передан, здесь будет False и не запустится обработка дат
    if args.dates_switch:
        logger.debug("Fixing process will account for the dates from sequence names")
        logger.debug("Parsing dates")
        rows_dates = np.array([create_date_f_name(x) for x in rows_names])
        cols_dates = np.array([create_date_f_name(x) for x in cols_names])

    to_fix_dict = dict()
    fixing_dict = dict()

    var_temp_dir = tempfile.mkdtemp()

    logger.debug("Temporary directory is introduced: " + var_temp_dir)
    logger.warning("Remember to delete the tmp directory manually if the program crashes")

    logger.debug(f"Temporary set for sequences indexing is introduced")
    tmp_names_set = set(rows_names)

    to_fix_fasta_path = os.path.join(var_temp_dir, "to_fix_indexed.fasta")

    with open(to_fix_fasta_path, 'w') as tmp_fasta:
        with open(args.to_fix, "r") as fasta_ali:
            parser = FastaIO.SimpleFastaParser(fasta_ali)
            for idx, (seq_name, seq_body) in tqdm(enumerate(parser), desc="Indexing to_fix FASTA"):
                if seq_name in tmp_names_set:
                    to_fix_dict[seq_name] = 2 * idx + 2  # добавляем индекс строки в файле выравнивания для linecache
                    tmp_names_set.remove(seq_name)
                    tmp_fasta.write(f">{seq_name}\n{seq_body}\n")

    if len(tmp_names_set) > 0:
        warning_file = args.output + "_to_fix_not_found.txt"
        logger.warning(f"Did not found some sequences from rows in to_fix: {warning_file}")
        with open(warning_file, "w") as t_wr:
            t_wr.write("\n".join(tmp_names_set) + "\n")

    tmp_names_set = set(cols_names)

    fix_kit_fasta_path = os.path.join(var_temp_dir, "fix_kit_indexed.fasta")

    with open(fix_kit_fasta_path, 'w') as tmp_fasta:
        with open(args.fix_kit, "r") as fasta_ali:
            parser = FastaIO.SimpleFastaParser(fasta_ali)
            for idx, (seq_name, seq_body) in tqdm(enumerate(parser), desc="Indexing fix_kit Fasta"):
                if seq_name in tmp_names_set:
                    fixing_dict[seq_name] = 2 * idx + 2  # добавляем индекс строки в файле выравнивания для linecache
                    tmp_names_set.remove(seq_name)
                    tmp_fasta.write(f">{seq_name}\n{seq_body}\n")

    if len(tmp_names_set) > 0:
        warning_file = args.output + "_fix_kit_not_found.txt"
        logger.warning(f"Did not found some sequences from columns in fix_kit: {warning_file}")
        with open(warning_file, "w") as t_wr:
            t_wr.write("\n".join(tmp_names_set) + "\n")

    tmp_names_set = None
    logger.debug(f"Temporary set is now {tmp_names_set}")

    logger.debug("First heading from indexed to_fix fasta: " + linecache.getline(to_fix_fasta_path, 1).rstrip('\n'))
    logger.debug("First heading from indexed fix_kit fasta: " + linecache.getline(fix_kit_fasta_path, 1).rstrip('\n'))

    ref_path = args.ref_path if args.ref_path else os.path.join(os.path.split(__file__)[0], "reference.fasta")

    logger.info("Would be used as reference: " + ref_path)

    with open(ref_path, "r") as fasta_ali:
        parser = FastaIO.SimpleFastaParser(fasta_ali)
        for elem in parser:
            reference_name, reference_seq = elem[0], elem[1]

    logger.debug(f"Ref: {reference_name} with {len(reference_seq)}, stripping as ({args.ref_lb}:{args.ref_rb})")

    reference_seq = reference_seq[args.ref_lb - 1:args.ref_rb]

    logger.debug(f"Ref: {reference_name} with {len(reference_seq)} coding region length after stripping")

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
        try:
            # получаем минимальное значение по массиву
            min_value = np.nanmin(df.loc[target].values)
            # находим имена всех последовательностей с этим минимальным значением
            asv = df.loc[target][lambda x: x == min_value].index
            logger.trace(f">>{target}; score {min_value}; neighbours {len(asv)}")
            if args.dates_switch:
                # получаем целевую дату по индексу целевого имени последовательности
                target_date = rows_dates[df.index.get_loc(target)]
                # получаем даты по индексам имен вдоль массива наименований
                fixing_dates = cols_dates[df.columns.get_indexer(asv)]
                # фильтруем массив наименований в соответствии с примененным условием на значение даты
                dates_condition = np.where(fixing_dates < target_date)
                updated_asv = asv[dates_condition]
                if len(updated_asv) == 0:  # если получилось, что длинна массива ноль
                    # и выходим из итерации, эту последовательность мы не чиним
                    raise DateFilterException(target, target_date)
                elif len(updated_asv) > 10:
                    # если дат больше 10, то мы заранее и гарантированно обрезаем число хитов для починки,
                    # чтобы при дальнейшей обработке этого точно не произошло,
                    # что позволяет не менять функционал программы при ключе `dates_switch` False
                    asv = updated_asv[np.argsort(fixing_dates[dates_condition])[:10]]
                else:
                    # если записей не ноль и не более 10, то передаем как есть
                    asv = updated_asv
            # даже если ключ `dates_switch` передан, функционал работы починки не изменен
            # если мы не учитываем даты при починке, то мы просто выбираем 10 случайных последовательностей,
            # когда пул для починки очень большой
            if len(asv) > 10:
                asv = np.random.choice(asv, 10)
            logger.trace(";".join(asv))
            # пишем в файл во временной директории референс, целевую последовательность, ближайшие последовательности
            with open(os.path.join(var_temp_dir, "tmp.fasta"), "w") as wrf:
                wrf.write(f">{reference_name}\n{reference_seq}\n")
                wrf.write(f">{target}\n{linecache.getline(to_fix_fasta_path, to_fix_dict[target])}")
                for close_seq in asv:
                    wrf.write(f">{close_seq}\n{linecache.getline(fix_kit_fasta_path, fixing_dict[close_seq])}")
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
            # ищем [nN]+ или иные неоднозначно определенные нуклеотиды в кодирующей части целевой последовательности
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
            res_seq_to_wr = "\n".join(textwrap.wrap(resulting_seq.replace("-", "").upper(), args.row_length))
            # записываем в файл исправленных с учетом успеха починки всех интервалов
            if fixed_intervals == len(found_unc_s):
                out_fwr.write(f">{target}|{fixed_intervals}/{len(found_unc_s)}|\n" + res_seq_to_wr + "\n")
            else:
                out_not_fwr.write(f">{target}|{fixed_intervals}/{len(found_unc_s)}|\n" + res_seq_to_wr + "\n")
        except DateFilterException as e:  # значит не удалось пройти фильтр дат и починить
            # оповещаем об этом в логах процесса починки
            logger.trace(f"No fixer found for the {e.name} accounting for the date {e.date}")
            # пишем последовательность as is в файл неисправленных
            seq_for_wr = "\n".join(textwrap.wrap(linecache.getline(to_fix_fasta_path, to_fix_dict[target]),
                                                 args.row_length))
            out_not_fwr.write(f">{target}|-1/0|\n" + seq_for_wr + "\n")
        except (Exception, ) as e:
            # если любая другая ошибка возникла, оповещаем уже глобально
            logger.warning(f"Unexpected error with {target}")
            logger.error(e)
            # тем не менее пишем последовательность as is в файл неисправленных
            seq_for_wr = "\n".join(textwrap.wrap(linecache.getline(to_fix_fasta_path, to_fix_dict[target]),
                                                 args.row_length))
            out_not_fwr.write(f">{target}|-1/0|\n" + seq_for_wr + "\n")

    # закрываем файлы после записи
    out_fwr.close()
    out_not_fwr.close()
    # удаляем временную директорию
    shutil.rmtree(var_temp_dir)
    logger.debug("Temporary directory was successfully deleted")


if __name__ == "__main__":
    logger.debug("Start")
    main()
