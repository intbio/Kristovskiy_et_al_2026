import MDAnalysis as mda 
from MDAnalysis.analysis import align
import numpy as np
import io, pypdb
import scipy
from scipy.spatial.distance import cdist
import os.path
import multiprocessing
import queue
import re
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import nglview as nv
import random

pam_seq = {'SpCas9': {'PAM': r'\w{1}GG', 'PAM_len': 3}, 'SaCas9': {'PAM': r'\w{2}G[AG]{2}T', 'PAM_len': 6},
          'Nme2Cas9': {'PAM' : r'\w{4}CC', 'PAM_len': 6}, 'Nme1Cas9': {'PAM' : r'ATATGATT', 'PAM_len': 8},
          'AcCas9': {'PAM' : r'\w{3}CC', 'PAM_len': 5}, 'GoCas12': {'PAM': r'A', 'PAM_len': 1},
          'CasX': {'PAM' : r'CAT', 'PAM_len': 3}}
    
d_trans = {'A':'T','T':'A','G':'C','C':'G'}

PURINES = {'A', 'G', 'DA', 'DG'} # у них водородную связь в паре держит N1, у пиримидинов N3
COMPLEMENTARY = {('A', 'DT'), ('G', 'DC'), ('C', 'DG'), ('U', 'DA'), # гРНК - ДНК
                 ('DA', 'DT'), ('DT', 'DA'), ('DG', 'DC'), ('DC', 'DG')} # ДНК - ДНК
WC_HBOND = 3.3 # верхняя граница водородной связи N1-N3 в уотсон-криковской паре, A
MIN_RNA_PAIRS = 3 # меньше - считаем, что спаривание не опознано

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) # папки с данными лежат рядом с модулем

DNA_LIBRARIES = {40: 'pdb_files_40', 100: 'pdb_files_100'} # ансамбли конформеров линкерной ДНК: длина в п.н. -> папка
DNA_LIBRARY_SIZE = 10000 # в каждой папке файлы 0000.pdb .. 9999.pdb
DNA_LIBRARY_RETRIES = 50 # сколько раз тянуть заново, если попался файл чужой длины
SEAM_RETRIES = 20 # сколько раз пересобирать комплекс, если разошелся стык сшивки
# целевая геометрия фосфодиэфирной связи в B-ДНК (длины в A, углы в градусах)
PHOSPHODIESTER_O3P = 1.60 # длина связи O3'-P в B-ДНК, A
PHOSPHODIESTER_PO5 = 1.59 # длина связи P-O5' в B-ДНК, A
BOND_O5_C5 = 1.44 # O5'-C5'
BOND_P_OP = 1.48 # P-OP1 и P-OP2
ANGLE_C3_O3_P = 119.0 # C3'-O3'-P
ANGLE_O3_P_O5 = 104.0 # O3'-P-O5'
ANGLE_P_O5_C5 = 120.9 # P-O5'-C5'
ANGLE_OP_P_OP = 119.6 # OP1-P-OP2
ANGLE_O_P_OP = 108.0 # мостиковый кислород - P - несвязывающий кислород

PATH_CLEARANCE = 4.0 # на сколько прямая между измеряемыми концами должна отстоять
# от атомов, A. Ближе - считаем, что прямая идет сквозь структуру

SEAM_MIN = 1.55 # минимальная связь O3'-P на стыке, A. Короче - атомы налезли друг
# на друга, вьюер такую связь тоже не рисует: его окно примерно 1.53-2.33 A
SEAM_CUTOFF = 1.75 # максимальная связь O3'-P на стыке, A. Идеальная связь ~1.6.
# Вьюеры рисуют связь O-P по сумме ковалентных радиусов с допуском, это около 2.0 A,
# поэтому стык в 1.9-2.0 формально цел, но на экране выглядит разрывом. 1.75 - с запасом
MAX_PAM_PAM_DIST = max(DNA_LIBRARIES) - 3 # длиннее фрагмент вырезать не из чего

def open_structures(Cas1_model, Cas2_model): 
    '''функция для открытия необходимых структур Cas белков. 
    Cas1_model - PDB ID структуры первого Cas9 белка или путь до файла со структурой.
    Cas2_model - PDB ID структуры второго Cas9 белка или путь до файла со структурой.
    '''
    if Cas1_model == Cas2_model: #случай, если подаются модели двух одинаковых Cas белков
        if len(Cas1_model) == 4: #елсли название модели равно 4 символам -> это PDB ID
            Cas1_pdb_file_string = pypdb.get_pdb_file(Cas1_model)
            Cas1 = mda.Universe(io.StringIO(Cas1_pdb_file_string), format='PDB', in_memory = True)
        elif len(Cas1_model) > 4: #если название модели больше чем 4 символа -> это путь до файла
            Cas1 = mda.Universe(Cas1_model, in_memory = True)
        Cas2 = Cas1.copy() #создаем независимую копию Cas1 и кладем ее в переменную Cas2
    else: #случай, если подаются модели двух разных Cas белков
        if len(Cas1_model) == 4:
            Cas1_pdb_file_string = pypdb.get_pdb_file(Cas1_model)
            Cas1 = mda.Universe(io.StringIO(Cas1_pdb_file_string), format='PDB', in_memory = True)
        elif len(Cas1_model) > 4:
            Cas1 = mda.Universe(Cas1_model, in_memory = True)
        if len(Cas2_model) == 4:
            Cas2_pdb_file_string = pypdb.get_pdb_file(Cas2_model)
            Cas2 = mda.Universe(io.StringIO(Cas2_pdb_file_string), format='PDB', in_memory = True)
        elif len(Cas2_model) > 4:
            Cas2 = mda.Universe(Cas2_model, in_memory = True)
    return rename_strands(Cas1), rename_strands(Cas2) #возвращаем MDAnalysis Universe, соответствующие Cas1 и Cas2, у которых: белок - A, РНК - B, таргетная цепь - C, не таргетная - D


def rename_strands(model):
    """
    функция позволяющая правильно определить названия цепей
    model: Universe содержащая Cas
    """
    model = model.select_atoms('protein or nucleic') # выбираем только нуклеотиды и аминокислоты

    # Раскладываем цепи по типам. Идем по тем segid, что реально есть в файле, а не по
    # алфавиту A-Z: встречаются и строчные имена цепей, и они терялись.
    # Белок опознаем по "цепь целиком белковая", а не сравнением со всем белком сразу:
    # при двух копиях комплекса в файле ни одна цепь не равна всему белку, и обе
    # белковые цепи уезжали в список ДНК
    protein_names, RNA_names, DNA_names = [], [], []
    for seg in sorted(set(model.segids)):
        chain = model.select_atoms(f'segid {seg}')
        if not chain.n_atoms:
            continue
        if chain.select_atoms('protein').n_atoms == chain.n_atoms:
            protein_names.append(seg)
        elif 'U' in chain.residues.resnames: # если в цепи есть урацил - это гРНК
            RNA_names.append(seg)
        else: # иначе эта цепь - ДНК
            DNA_names.append(seg)

    if not protein_names or not RNA_names or len(DNA_names) < 2:
        raise ValueError(f'В структуре должен быть белок, гРНК и две цепи ДНК, а найдено: '
                         f'белок {protein_names}, РНК {RNA_names}, ДНК {DNA_names}')

    # в файле может лежать несколько копий комплекса - собираем одну по контактам,
    # начиная с первой белковой цепи
    protein_name = protein_names[0]
    RNA_name = closest_chain(model, f'segid {protein_name}', RNA_names)

    # таргетная цепь спарена с гРНК, нетаргетная вытеснена и с гРНК почти не контактирует
    model.select_atoms(f'segid {RNA_name}').segments.segids = 'b'
    contacts = find_target_strand(model, DNA_names)
    target_name = max(contacts, key=contacts.get)
    nontarget_name = min(contacts, key=contacts.get)

    # Остальные цепи ДНК - продолжение таргетной: в кристаллах ее нередко моделируют
    # двумя цепями, протоспейсерной и PAM-проксимальной. Выбрасывать их нельзя,
    # вместе с ними терялся и PAM
    extra_target = [name for name in DNA_names if name not in (target_name, nontarget_name)]

    dropped = ([name for name in protein_names if name != protein_name]
               + [name for name in RNA_names if name != RNA_name])
    if dropped:
        print(f'В файле несколько копий комплекса, взята одна: белок {protein_name}, '
              f'гРНК {RNA_name}, ДНК {target_name}/{nontarget_name}. Отброшено: {dropped}')
    if extra_target:
        print(f'Таргетная цепь собрана из нескольких: {target_name} + {extra_target}')

    # переименовываем
    model.select_atoms(f'segid {protein_name}').segments.segids = 'a'
    for name in [target_name] + extra_target:
        model.select_atoms(f'segid {name}').segments.segids = 'c'
    model.select_atoms(f'segid {nontarget_name}').segments.segids = 'd'
    model = model.select_atoms('segid a or segid b or segid c or segid d') # лишние копии долой

    model.select_atoms(f'segid a').segments.segids = 'A'
    model.select_atoms(f'segid b').segments.segids = 'B'
    model.select_atoms(f'segid c').segments.segids = 'C'
    model.select_atoms(f'segid d').segments.segids = 'D'
    
    return model

def find_target_strand(model, DNA_names):
    """
    функция, позволяющая определить таргетную цепь ДНК
    model: Universe содержащая Cas
    DNA_names: название цепей ДНК
    """
    around_seg_b = model.select_atoms('around 5 (segid b)', periodic=False) # выделяем область вокруг гРНК
    lst_segs = {}
    for name in DNA_names:
        lst_segs[name] = len(around_seg_b.select_atoms(f'segid {name}').residues.resnums) # записываем сколько атомов из цепей ДНК попало в это выделение
        # в выделение попадет больше таргетной ДНК - поэтому легко определить какая из них таргетная
    return lst_segs # возвращаем словарь


def closest_chain(model, around_what, candidates):
    """
    из candidates выбирает цепь, у которой больше всего остатков в контакте с around_what.
    Так одна копия комплекса собирается по контактам, а лишние копии и посторонние
    фрагменты ДНК отсеиваются. Раньше нетаргетная цепь бралась как минимум по словарю
    контактов - при трех цепях ДНК это давало произвольный результат
    model: AtomGroup со всеми цепями
    around_what: строка выделения, вокруг которой считаем контакты
    candidates: список segid, из которых выбираем
    """
    if len(candidates) == 1:
        return candidates[0]
    around = model.select_atoms(f'around 5 ({around_what})', periodic=False)
    contacts = {name: len(around.select_atoms(f'segid {name}').residues) for name in candidates}
    return max(contacts, key=contacts.get)


def swap_chains(model, name_1, name_2):
    """
    меняет местами имена двух цепей через временное имя.
    Нельзя писать segments.segids = ('D', 'C'): это требует ровно двух сегментов,
    а таргетная цепь может быть собрана из нескольких, и тогда падает
    """
    model.select_atoms(f'segid {name_1}').segments.segids = 'Z'
    model.select_atoms(f'segid {name_2}').segments.segids = name_1
    model.select_atoms('segid Z').segments.segids = name_2


def PAM_is_paired(Cas, PAM_index, PAM_len):
    """
    Настоящий PAM лежит в двуцепочечном участке, то есть спарен с таргетной цепью.
    Ложные совпадения мотива приходятся на неспаренный участок: в 7S4X из двух NGG
    у ложного нет ни одной уотсон-криковской пары, у настоящего - три.
    Одного совпадения последовательности мало: короткие вырожденные мотивы вроде NGG
    встречаются по нескольку раз, и наличия комплемента в цепи C тоже не хватает -
    он находится у обоих кандидатов
    """
    PAM_residues = Cas.select_atoms('segid D').residues[PAM_index : PAM_index + PAM_len]
    if not len(PAM_residues):
        return False
    return bool(watson_crick_pairs(PAM_residues.atoms, Cas.select_atoms('segid C')))


def find_PAM(Cas_name, chain_D, chain_C, Cas=None):
    '''find_PAM находит индексы первой нуклеотитдной пары из PAM-сайта.
    Поиск идет по регулярным выражениям солгасно pam_seq словарю.
    Cas_name - Модель каса для которой будет искаться PAM сайт в chain_D.
    chain_D - Последовательность нуклеотидов в D цепи (несущей PAM) от 5' к 3'.
    chain_C - Последовательнсть нуклеотидов в C цепи (комплементарной PAM) от 5' к 3'.
    Cas - структура, по ней проверяется спаренность кандидата. Без нее проверка
          пропускается, и при нескольких совпадениях мотива можно взять не тот сайт.
    '''
    rPAM_site = pam_seq[f'{Cas_name}']['PAM'] #вынимаем регулярное выражение для РАМ из словаря pam_seq
    PAM_len = pam_seq[f'{Cas_name}']['PAM_len'] #Записываем длину РАМ сайта

    # идем по всем позициям цепи D, а не берем первое совпадение re.findall:
    # findall пропускает перекрывающиеся совпадения, и настоящий PAM может оказаться
    # вторым из них (у Nme2Cas9 в 6JE3 промах ровно на один нуклеотид).
    # Настоящий PAM тот, у которого комплемент действительно есть в цепи C
    for Cas_PAM_index in range(len(chain_D)):
        PAM_match = re.match(rPAM_site, chain_D[Cas_PAM_index:]) #совпадение, привязанное к этой позиции
        if PAM_match is None:
            continue
        PAM_site = PAM_match.group()
        PAM_comp = ''.join([ d_trans[nuc] for nuc in PAM_site])[::-1] #получаем копмлиментраную последовательность к PAM и задаем направление 5>3
        if PAM_comp not in chain_C:
            continue
        if Cas is not None and not PAM_is_paired(Cas, Cas_PAM_index, PAM_len):
            continue # ложное совпадение мотива: такой участок с таргетной цепью не спарен
        PAM_comp_index = chain_C.index(PAM_comp) + PAM_len -1 #Находим индекс 1-го нуклеотида комплементраного РАМ для этого:
        # К индексу РАМ в С цепи добавляю его длину и -1 поскольку на С цепи ищу первый комплиментрный нуклеотид к 0-му на D цепи

        return Cas_PAM_index, PAM_comp_index, PAM_len

    raise ValueError(f'В цепи D не нашлось PAM для {Cas_name} (шаблон {rPAM_site}), '
                     f'комплемент которого есть в цепи C.\n  цепь D: {chain_D}\n  цепь C: {chain_C}')


def cut_and_renum_DNA_by_PAM(Cas_name, Cas):
    '''
    cut_and_renum_DNA_by_PAM: Обрезает DNA по PAM-сайту в направлении 5'>3' и перенумирует цепи относитепльно PAM-сайта
    Cas_name: Название CAS белка
    Cas: Модель CAS белка
    '''
    
    chain_D = ''.join(list(Cas.select_atoms('segid D').residues.resnames)).replace('D','') #сиквенс цепи D 
    chain_C = ''.join(list(Cas.select_atoms('segid C').residues.resnames)).replace('D','') #сиквенс цепи С 
    # print(chain_C)
    PAM_index, PAM_comp_index, PAM_len = find_PAM(Cas_name, chain_D, chain_C, Cas)
    
    Cas.select_atoms('segid D').residues.resnums = [num for num in np.arange(-PAM_index, len(chain_D) - PAM_index)]
    Cas.select_atoms('segid C').residues.resnums = [num for num in np.arange(-PAM_comp_index, len(chain_C) - PAM_comp_index)*(-1)] 
    
    Cas_chain_D_end = Cas.select_atoms('segid D').residues.resnums[-1]
    Cas_chain_C_end = Cas.select_atoms('segid C').residues.resnums[0]
    Cas = Cas.select_atoms('not (segid D and resnum %d:%d)'%(PAM_len + 1, Cas_chain_D_end)) 
    Cas = Cas.select_atoms('not (segid C and resnum %d:%d)'%(PAM_len + 1, Cas_chain_C_end))
    
    return Cas, PAM_index


def watson_crick_pairs(chain_1, chain_2, cutoff=WC_HBOND):
    """
    Находит уотсон-криковские пары между двумя цепями по водородной связи
    N1 пурина с N3 пиримидина. Расстояние между ними в паре около 2.9 A,
    это куда более надежный признак, чем близость сахаров: по C1'-C1' соседние
    по цепи остатки оказываются не дальше спаренных.
    Возвращает список пар (resnum в первой цепи, resnum во второй)
    """
    def hbond_atoms(chain):
        resnums, positions = [], []
        for res in chain.residues:
            name = 'N1' if res.resname.strip() in PURINES else 'N3'
            atom = res.atoms.select_atoms(f'name {name}')
            if len(atom):
                resnums.append(int(res.resnum))
                positions.append(atom.positions[0])

        return resnums, (np.array(positions) if positions else np.zeros((0, 3))), \
               [res.resname.strip() for res in chain.residues
                if len(res.atoms.select_atoms(f"name {'N1' if res.resname.strip() in PURINES else 'N3'}"))]

    nums_1, pos_1, names_1 = hbond_atoms(chain_1)
    nums_2, pos_2, names_2 = hbond_atoms(chain_2)
    if not len(pos_1) or not len(pos_2):
        return []
    close = cdist(pos_1, pos_2) < cutoff
    return [(nums_1[i], nums_2[j]) for i, j in zip(*np.nonzero(close))
            if (names_1[i], names_2[j]) in COMPLEMENTARY]


def renum_RNA(Cas, PAM_index):
    '''
    renum_RNA - перенумирует RNA
    Cas - модель CAS белка
    PAM_index - номер 0 нуклеотида в PAM-сайте D цепи

    Нумерация задается так, чтобы спаренные остатки получили одинаковые номера:
    нуклеотид спейсера гРНК стоит напротив нуклеотида протоспейсера с тем же номером.
    Нумеровать РНК просто по порядку нельзя - проверка на 12 структурах показала,
    что тогда номера расходятся с ДНК на 10-24 в зависимости от структуры,
    и совпадение в 5Y36 оказалось случайным.
    '''
    RNA = Cas.select_atoms('segid B')
    RNA.residues.resnums = np.arange(-PAM_index, len(RNA.residues) - PAM_index) #перенумеровали цепь B (RNA

    pairs = watson_crick_pairs(RNA, Cas.select_atoms('segid C'))
    if len(pairs) < MIN_RNA_PAIRS: # спаривание не опознано - оставляем нумерацию по порядку
        print(f'Спейсер гРНК не удалось сопоставить с протоспейсером: найдено пар {len(pairs)}. '
              f'Нумерация РНК оставлена позиционной и может не совпадать с нумерацией ДНК')
        return Cas

    # сдвигаем всю цепь на одну величину, а не каждый остаток по отдельности:
    # при вставке в спаривании поостаточная нумерация дала бы дубли и пропуски,
    # а на них рвется лента в программах визуализации
    offsets = [rna_num - dna_num for rna_num, dna_num in pairs]
    shift = max(set(offsets), key=offsets.count) # самый частый сдвиг
    RNA.residues.resnums = np.array(RNA.residues.resnums) - shift
    return Cas


def pick_DNA_library(DNA_len):
    """
    выбирает самую короткую библиотеку конформеров, из которой можно вырезать
    фрагмент длины DNA_len. Нужен запас минимум в 1 п.н. - на случайный сдвиг.
    DNA_len: требуемая длина фрагмента в парах оснований
    """
    for n_bp in sorted(DNA_LIBRARIES):
        if n_bp >= DNA_len + 1:
            return os.path.join(MODULE_DIR, DNA_LIBRARIES[n_bp])
    raise ValueError(f'Нужен фрагмент ДНК длины {DNA_len} п.н., '
                     f'а самая длинная библиотека - {max(DNA_LIBRARIES)} п.н. '
                     f'Максимальный PAM_PAM_distance = {MAX_PAM_PAM_DIST}')


def prepare_DNA(PAM_PAM_dist, DNA_model_name=None, start=None):
    """
    функция выдает структуру ДНК длины PAM_PAM_distance + 2
    PAM_PAM_dist: расстояние между PAM-сайтами в парах оснований
    DNA_model_name: номер конформера в библиотеке - int или строка вида '0042'.
                    None - случайный конформер (нужно для пересборки моделей,
                    пока белки не перестанут сталкиваться)
    start: сдвиг вырезаемого фрагмента от конца дуплекса, >= 1.
           None - случайный сдвиг
    """
    DNA_len = PAM_PAM_dist + 2
    library = pick_DNA_library(DNA_len)
    random_pick = DNA_model_name is None

    # в папках попадаются файлы чужой длины (в pdb_files_100 есть конформеры по 40 п.н.),
    # поэтому проверяем вытянутый файл по факту и, если он короткий, тянем следующий
    for attempt in range(DNA_LIBRARY_RETRIES):
        if random_pick:
            DNA_model_name = random.randint(0, DNA_LIBRARY_SIZE - 1) # рандомная ДНК из имеющхся

        DNA_path = os.path.join(library, f'{int(DNA_model_name):04d}.pdb')
        if not os.path.exists(DNA_path):
            raise FileNotFoundError(f'Нет файла с конформером ДНК: {DNA_path}')

        DNA = mda.Universe(DNA_path)
        DNA.select_atoms(f'segid A').segments.segids = 'C'
        DNA.select_atoms(f'segid B').segments.segids = 'D'

        MAX_LEN = len(DNA.select_atoms('segid C').residues.resnums) # максимальная длина ДНК
        if MAX_LEN >= DNA_len + 1:
            break
        if not random_pick: # конформер назвали явно - подменять его молча нельзя
            raise ValueError(f'В конформере {DNA_path} только {MAX_LEN} п.н., '
                             f'а нужен фрагмент {DNA_len} п.н.')
    else:
        raise RuntimeError(f'За {DNA_LIBRARY_RETRIES} попыток не нашлось конформера длиннее '
                           f'{DNA_len} п.н. в {os.path.basename(library)}')

    if start is None:
        start = random.randint(1, MAX_LEN - DNA_len)
    elif not 1 <= start <= MAX_LEN - DNA_len:
        raise ValueError(f'start = {start} вне диапазона 1..{MAX_LEN - DNA_len} '
                         f'для фрагмента длины {DNA_len} п.н. из {os.path.basename(library)}')

    select_C = ' '.join([str(i) for i in list(DNA.select_atoms('segid C').residues.resnums)[-(start + DNA_len) : -start]]) # запишем лишнюю часть ДНК цепи С
    select_D = ' '.join([str(i) for i in list(DNA.select_atoms('segid D').residues.resnums)[start : start + DNA_len]]) # запишем лишнюю часть ДНК цепи D

    DNA_model = DNA.select_atoms(f'(segid D and resnum {select_D}) or (segid C and resnum {select_C})') # отрезали лишнюю часть ДНК
    
    return DNA_model


def align_merge(model1, model2, orientation):
    """
    start_enumering - начало перенумеровки цепей ДНК в модели 2
    Функция строит комплекс из model1 и model2 в зависимости от orientation
    """
    #if orientation == 'PAM-in':
    model2_seq_D = np.array(model2.select_atoms('segid D').residues.resnums) # копии: ниже по ним считаем сдвиг
    model2_seq_C = np.array(model2.select_atoms('segid C').residues.resnums)
    
    # Опорный остаток - на самом краю цепи D: именно туда пристыковывается следующий
    # кусок. Если брать его внутри спаренной области, хвост длинной цепи остается
    # висеть и нумерация в собранной модели ломается
    model1_D_resnums = [int(num) for num in model1.select_atoms('segid D').residues.resnums]
    start_enumering = model1_D_resnums[-2] # 2PAM - 1 PAM_nuc 0
    #print(start_enumering)
    
    # Сдвигаем обе цепи на одну и ту же величину, а не нумеруем каждую заново от своей
    # длины. Нумерация из cut_and_renum_DNA_by_PAM уже согласована по парам оснований
    # (напротив N стоит N), и сдвиг ее сохраняет. Переназначение же давало верный
    # результат только при len(C) == len(D) - то есть на полностью спаренном дуплексе.
    # В кристаллах нетаргетная цепь обычно короче, и тогда номера C и D переставали
    # соответствовать парам: наложение шло по чужим остаткам.
    # Исключение - линкерная ДНК из библиотеки: там нумерация позиционная (цепь C это
    # 1..N, цепь D это N+1..2N), пары по ней не читаются. Ее нумеруем заново от длины,
    # и это корректно: линкер всегда полностью спаренный дуплекс, len(C) == len(D)
    model2_is_cas = bool(model2.select_atoms('protein').n_atoms)

    if model2_is_cas:
        # одно и то же аффинное преобразование для обеих цепей: равные номера остаются
        # равными, значит пары сохраняются при любом соотношении длин C и D.
        # D_dir учитывает, что после перестановки цепей в PAM-in цепь D идет по убыванию
        # Якорь берем по ОБЕИМ цепям, а не только по D. В кристаллах таргетная цепь
        # длиннее нетаргетной, и если считать сдвиг от конца D, то хвост цепи C
        # оказывается за якорем и наезжает номерами на цепь C первой половины -
        # нумерация в собранной модели переставала быть монотонной
        D_dir = 1 if model2_seq_D[-1] > model2_seq_D[0] else -1 # направление нумерации D по файлу
        both_chains = np.concatenate([model2_seq_D, model2_seq_C])
        D_first = both_chains.min() if D_dir == 1 else both_chains.max() # край всей ДНК model2
        sign = D_dir if orientation != 'PAM-out' else -D_dir
        new_D = start_enumering + sign * (model2_seq_D - D_first)
        new_C = start_enumering + sign * (model2_seq_C - D_first)
    elif orientation != 'PAM-out':
        new_D = np.arange(start_enumering, len(model2_seq_D) + start_enumering)
        new_C = np.arange(start_enumering, len(model2_seq_C) + start_enumering)[::-1]
    else:
        new_D = np.arange(start_enumering, -len(model2_seq_D) + start_enumering, -1)
        new_C = np.arange(start_enumering, -len(model2_seq_C) + start_enumering, -1)[::-1]

    model2.select_atoms('segid D').residues.resnums = new_D # перенумеровали цепь D model2
    model2.select_atoms('segid C').residues.resnums = new_C # перенумеровали цепь C model2

    # resnums = 8*[f'{res1}'] + 8*[f'{res1 + 1}'] + 8*[f'{res2}'] + 8*[f'{res2 - 1}'] #resnums остатков для align
    atom_names = ["O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"] #имена атомов для align
    second_resnum = start_enumering + 1 if orientation != 'PAM-out' else start_enumering - 1

    # Список выделений собираем динамически, оставляя только то, что есть РОВНО по
    # одному атому в обеих структурах. На асимметричном дуплексе нетаргетной цепи
    # в этом месте может уже не быть, и жесткий список давал разное число атомов:
    # alignto со strict=False тогда молча сопоставлял не те атомы
    atom_align_names_list = []
    for chain in ('D', 'C'):
        for resnum in (start_enumering, second_resnum):
            for name in atom_names:
                sel = f'segid {chain} and resnum {resnum} and name {name}'
                if len(model1.select_atoms(sel)) == 1 and len(model2.select_atoms(sel)) == 1:
                    atom_align_names_list.append(sel)
    if len(atom_align_names_list) < 8:
        raise ValueError(f'На стыке {start_enumering}/{second_resnum} нашлось всего '
                         f'{len(atom_align_names_list)} атомов для наложения - слишком мало')


    model1_model2_align = align.alignto(model2, model1,{'mobile':atom_align_names_list, 'reference':atom_align_names_list}, strict=False) #align model1 and model2
    
    if orientation != 'PAM-out':
        del_mod2_seg = ' '.join([str(i) for i in range(start_enumering, start_enumering + 1)])
        model2 = model2.select_atoms(f'not ((segid D and resnum {del_mod2_seg}) or (segid C and resnum {del_mod2_seg}))') # отрезали концевые нуклеотиды model2
        model1 = model1.select_atoms(f'not ((segid D and resnum {start_enumering + 1}) or (segid C and resnum {start_enumering + 1}))')
    
    else:
        del_mod2_seg = ' '.join([str(i) for i in range(start_enumering, start_enumering + 1)])
        model2 = model2.select_atoms(f'not ((segid D and resnum {del_mod2_seg}) or (segid C and resnum {del_mod2_seg}))') # отрезали концевые нуклеотиды model2
        model1 = model1.select_atoms(f'not ((segid D and resnum {start_enumering - 1}) or (segid C and resnum {start_enumering - 1}))')
    
    #model2.residues.resids = model2.residues.resnums
    
    if model2.select_atoms('protein'): #если model2 белок
        model1_model2_complex = mda.Merge(model1.select_atoms('segid A'), model1.select_atoms('segid B'),
                                          model2.select_atoms('segid a'), model2.select_atoms('segid b'),
                                          model1.select_atoms('segid D'), model2.select_atoms('segid D'),
                                          model2.select_atoms('segid C'), model1.select_atoms('segid C'))
    else: #если model2 ДНК
        # порядок аргументов Merge задает порядок остатков в цепи, а C и D антипараллельны:
        # для D сначала идет model1, для C - наоборот, иначе цепь C рвется на куски
        model1_model2_complex = mda.Merge(model1.select_atoms('segid A'), model1.select_atoms('segid B'),
                                          model1.select_atoms('segid D'), model2.select_atoms('segid D'),
                                          model2.select_atoms('segid C'), model1.select_atoms('segid C'))
    
    #return chains, atom_align_names_list, resnums, model1_model2_complex # возвращает сшитый комплекс model1, model2
    
    model1_model2_complex.residues.resids = model1_model2_complex.residues.resnums

    # segid мы переименовали, а chainID остался из исходных файлов - вьюеры и PDB-формат
    # смотрят именно на chainID, поэтому цепь выглядела рваной. Синхронизируем.
    if not hasattr(model1_model2_complex.atoms, 'chainIDs'):
        model1_model2_complex.add_TopologyAttr('chainIDs')
    model1_model2_complex.atoms.chainIDs = model1_model2_complex.atoms.segids

    return model1_model2_complex



def prepare_structures(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation): #возможно на вход лучше подавать словарь, где ключи это имя а хначение это структура
    '''Функция для начальной обработки структур Cas9 белков - удаление небелковой и ненуклеотидной части, переименовывание цепей, перенумеровка цепей РНК.
    Cas_name - название Cas белка (биологическое).
    Cas_name_1 - Название CAS белка
    Cas_name_2 - Название CAS белка
    Cas1_model - PDB ID структуры первого Cas9 белка или путь до файла со структурой.
    Cas2_model - PDB ID структуры второго Cas9 белка или путь до файла со структурой.
    orientation - способ позиционирования CAS-белков
    '''
    Cas1, Cas2 = open_structures(Cas1_model, Cas2_model) #используем функцию open_structure для загрузки структур Cas
    
    Cas1 = Cas1.select_atoms('protein or nucleic') #выбираем из Cas1 только белок и НК
    Cas2 = Cas2.select_atoms('protein or nucleic') #выбираем из Cas2 только белок и НК

    Cas1, Cas1_PAM_index = cut_and_renum_DNA_by_PAM(Cas_name_1, Cas1)
    Cas2, Cas2_PAM_index = cut_and_renum_DNA_by_PAM(Cas_name_2, Cas2)
    
    Cas1 = renum_RNA(Cas1, Cas1_PAM_index)
    Cas2 = renum_RNA(Cas2, Cas2_PAM_index)
    
    Cas2.select_atoms('segid A').segments.segids ='a' #переименовали цепь A Cas2 в a
    Cas2.select_atoms('segid B').segments.segids ='b' #переименовали цепь B Cas2 в b
    Cas1.residues.resids = Cas1.residues.resnums #назначение нумерации resnum в resid для Cas1
    Cas2.residues.resids = Cas2.residues.resnums 
    
    return Cas1, Cas2


def build_models(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation, r_PAM_PAM_distance_list, save, DNA_name = None, DNA_start = None, seam_cutoff = SEAM_CUTOFF, fix_seams = True, progress = True):
    '''Функция собирает комплекс из двух CAS-белков согласно указаным параметрам orientationPAM_PAM_distance

    Cas_name_1 - название Cas белка (биологическое)
    Cas_name_2 - название Cas белка (биологическое)
    Cas1_model - PDB ID структуры первого Cas9 белка или путь до файла со структурой.
    Cas2_model - PDB ID структуры второго Cas9 белка или путь до файла со структурой.
    orientation - {0:'PAM-direct',1:'PAM-in',2:'PAM-out'}.
    PAM_PAM_distance_list - list of n.p.
    save - создает папки куда сохраняет системы
    DNA_name - номер конформера линкерной ДНК, None - случайный на каждый комплекс
    DNA_start - сдвиг окна внутри конформера, None - случайный.
                для воспроизводимой сборки фиксировать надо оба: DNA_name и DNA_start
    seam_cutoff - максимальная связь O3'-P на стыке, A. Сборку с большим разрывом
                выбрасываем и повторяем со случайно вытянутым конформером
    fix_seams - чинить геометрию швов сшивки (fix_seam_geometry). False отдает
                модель как есть, со связями O3'-P длиной 0.4-2.3 A на швах
    progress - показывать прогрессбар. В рабочих процессах его отключают
    '''
    cas_or = {0:'PAM-direct',1:'PAM-in',2:'PAM-out'}
    
    orientation = cas_or[orientation]
    PAM1_len = pam_seq[f'{Cas_name_1}']['PAM_len']
    complex_dic = {}
    
    PAM_PAM_distance_list = []
    if isinstance(r_PAM_PAM_distance_list, int):
        PAM_PAM_distance_list.append(r_PAM_PAM_distance_list)
    else:
        PAM_PAM_distance_list = r_PAM_PAM_distance_list
    
    Cas1, Cas2 = prepare_structures(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation)
    
    min_dists = {'PAM-in': 0, 
                 'PAM-direct': abs(min(Cas1.select_atoms('segid C').residues.resnums) + 1),
                 'PAM-out': abs(min(Cas1.select_atoms('segid C').residues.resnums + 1) + 
                                min(Cas2.select_atoms('segid C').residues.resnums + 1))}
    
    if min(PAM_PAM_distance_list) < min_dists[orientation]:
        print(f'Вы ввели значение PAM_PAM_distance меньше допустимого = {min_dists[orientation]}')
        return None

    max_dist = MAX_PAM_PAM_DIST + min_dists[orientation] # сколько п.н. можно вырезать из библиотеки
    if max(PAM_PAM_distance_list) > max_dist:
        print(f'Вы ввели значение PAM_PAM_distance больше допустимого = {max_dist}')
        return None

    Cas2_u = mda.Merge(Cas2)
    Cas2_u.transfer_to_memory()

    # конформер линкера тянется случайно, и примерно в 15% случаев стык сшивки
    # расходится. Такую сборку выбрасываем и тянем заново - отбрасываем только
    # заведомо битые модели, лучшую из пула не выбираем, чтобы не портить статистику
    seam_tries = 1 if (DNA_name is not None and DNA_start is not None) else SEAM_RETRIES

    # перестановка цепей делается один раз до цикла, а не на каждой итерации:
    # внутри цикла она переключалась туда-обратно и на четных длинах списка
    # собиралась не та модель
    if orientation == 'PAM-in':
        swap_chains(Cas2, 'C', 'D')
    elif orientation == 'PAM-out':
        swap_chains(Cas1, 'C', 'D') #поменяли названия цепей для цепей C и D Cas1

    for PAM_PAM_distance in tqdm(PAM_PAM_distance_list ,desc = 'Кол-во собраных комплексов', disable = not progress):
        Cas2_c = Cas2_u.copy()
        for seam_attempt in range(1, seam_tries + 1):
            if orientation == 'PAM-direct':
                #Cas2_m = renum_cas2_DNA(Cas_name_1, Cas2_c, orientation, PAM_PAM_distance)
                DNA_m = prepare_DNA(PAM_PAM_distance - min_dists[orientation], DNA_name, DNA_start)
                #DNA_m = DNA.select_atoms('(segid D C and resnum %d:%d)'%(PAM1_len -1, PAM_PAM_distance + PAM1_len ))
                # берем ДНК с индексами нуклеотидов входящих в РАМ сайты cas1 cas2
                Cas1_DNA_complex = align_merge(Cas1, DNA_m, orientation)
                Cas1_cas2_complex = align_merge(Cas1_DNA_complex, Cas2, orientation)

            elif orientation == 'PAM-in':
                #Cas2_m = renum_cas2_DNA(Cas_name_1, Cas2_c, orientation, PAM_PAM_distance)
                DNA_m = prepare_DNA(PAM_PAM_distance, DNA_name, DNA_start)
                #DNA_m = DNA.select_atoms('(segid D C and resnum %d:%d)'%(PAM1_len -1, PAM_PAM_distance + PAM1_len ))
                # берем ДНК с индексами нуклеотидов входящих в РАМ сайты cas1 cas2
                Cas1_DNA_complex = align_merge(Cas1, DNA_m, orientation)
                Cas1_cas2_complex = align_merge(Cas1_DNA_complex, Cas2, orientation)

            elif orientation == 'PAM-out':
                DNA_m = prepare_DNA(PAM_PAM_distance - min_dists[orientation], DNA_name, DNA_start)
                Cas1_DNA_complex = align_merge(Cas1, DNA_m, orientation)
                Cas1_cas2_complex = align_merge(Cas1_DNA_complex, Cas2, orientation)

            if fix_seams:
                fix_seam_geometry(Cas1_cas2_complex, max_len = seam_cutoff) # чиним геометрию швов
            if not check_seams(Cas1_cas2_complex, seam_cutoff):
                break
        else:
            if seam_tries > 1:
                print(f'PAM_PAM_distance = {PAM_PAM_distance}: за {seam_tries} попыток стык так и '
                      f'не сошелся, худшая связь {max_backbone_gap(Cas1_cas2_complex):.2f} A')


        if save:
            how_to_save(Cas_name_1, Cas1_model, Cas_name_2, Cas2_model, Cas1_cas2_complex, orientation, PAM_PAM_distance)

        complex_dic[PAM_PAM_distance] = Cas1_cas2_complex
        

    return complex_dic


def model_tag(model):
    """
    Короткое имя структуры для имени файла: PDB ID остается как есть, а из пути
    к файлу берется только имя без папок и расширения. Иначе слэши из пути
    уезжали в имя файла и запись падала с FileNotFoundError
    """
    return os.path.splitext(os.path.basename(str(model)))[0]


def how_to_save(Cas_name_1, Cas1_model, Cas_name_2, Cas2_model, Cas1_cas2_complex, orientation, PAM_PAM_distance):

    """
    функция, сохраняющая модель из двух Cas
    """

    main_f = 'Cas_Cas_complex/' 
    if os.path.exists(main_f) == False:
        print(f'Папка {main_f} отсудствовала, но была создана')
        os.mkdir(main_f)
    if os.path.exists(f'{main_f}{orientation}') == False:
        print(f'Папка {orientation} отсудствовала, но была создана')
        os.mkdir(f'{main_f}{orientation}')
    if len(str(PAM_PAM_distance))<2:
        PAM_PAM = '0'+str(PAM_PAM_distance)
    else: PAM_PAM = PAM_PAM_distance
    path = f'{main_f}{orientation}/'
    f_name = (f'{path}{Cas_name_1}_{model_tag(Cas1_model)}_{Cas_name_2}_{model_tag(Cas2_model)}'
              f'_PAM_distans_{PAM_PAM}.pdb')
    Cas1_cas2_complex.atoms.write(f_name)
    print(f'file {f_name} writed')
    

def dist_endpoints(complex_model, param):
    """
    Координаты двух точек, между которыми меряется расстояние, и остатки, которым они
    принадлежат. Одно место, где это определено - get_dist, check_dist_path и show_dist
    берут точки отсюда, чтобы не разъехаться.
    param: "3'" - O3' на 3' концах гРНК, 'C' - C-концы белков, 'N' - N-концы белков
    Возвращает (точка_1, точка_2, ((цепь, resnum), (цепь, resnum)))
    """
    if param == "3'":
        chain_1, chain_2, atom_name = 'B', 'b', "O3'"
        resnum_1 = complex_model.select_atoms('segid B').residues.resnums[-1] # последний остаток РНК цепи B
        resnum_2 = complex_model.select_atoms('segid b').residues.resnums[-1] # последний остаток РНК цепи b
    elif param == 'N':
        chain_1, chain_2, atom_name = 'A', 'a', 'N'
        resnum_1 = complex_model.select_atoms('segid A').residues.resnums[0]
        resnum_2 = complex_model.select_atoms('segid a').residues.resnums[0]
    elif param == 'C':
        chain_1, chain_2, atom_name = 'A', 'a', 'C'
        resnum_1 = complex_model.select_atoms('segid A').residues.resnums[-1]
        resnum_2 = complex_model.select_atoms('segid a').residues.resnums[-1]
    else:
        raise ValueError(f'param должен быть "3\'", "C" или "N", а не {param!r}')

    point_1 = complex_model.select_atoms(f'segid {chain_1} and (resnum {resnum_1} and name {atom_name})').center_of_geometry()
    point_2 = complex_model.select_atoms(f'segid {chain_2} and (resnum {resnum_2} and name {atom_name})').center_of_geometry()
    return point_1, point_2, ((chain_1, int(resnum_1)), (chain_2, int(resnum_2)))


def check_dist_path(complex_model, param="3'", clearance=PATH_CLEARANCE):
    """
    Проверяет, не проходит ли прямая между измеряемыми точками сквозь структуру.
    get_dist меряет расстояние по прямой, а она запросто идет насквозь через ДНК -
    тогда это число ничего не говорит о том, достанет ли реальный линкер.

    clearance: насколько прямая должна отстоять от атомов, A. Меньше - считаем,
               что путь перекрыт
    Возвращает словарь: distance, clearance (минимальный зазор), blocked,
    blocked_by (какие цепи мешают), n_atoms (сколько атомов в коридоре)
    """
    point_1, point_2, ends = dist_endpoints(complex_model, param)

    atoms = complex_model.select_atoms('protein or nucleic')
    skip = ' or '.join(f'(segid {seg} and resnum {num})' for seg, num in ends)
    atoms = atoms.select_atoms(f'not ({skip})') # сами измеряемые остатки не мешают

    # расстояние от каждого атома до отрезка: проекция с зажимом на концах
    segment = point_2 - point_1
    length_sq = float(np.dot(segment, segment))
    if length_sq == 0 or not atoms.n_atoms:
        return {'distance': 0.0, 'clearance': float('inf'), 'blocked': False,
                'blocked_by': [], 'n_atoms': 0}
    shift = atoms.positions - point_1
    position_on_segment = np.clip(shift @ segment / length_sq, 0.0, 1.0)
    to_segment = shift - position_on_segment[:, None] * segment
    gaps = np.linalg.norm(to_segment, axis=1)

    in_the_way = gaps < clearance
    return {'distance': float(np.linalg.norm(segment)),
            'clearance': float(gaps.min()),
            'blocked': bool(in_the_way.any()),
            'blocked_by': sorted(set(atoms[in_the_way].chainIDs)) if in_the_way.any() else [],
            'n_atoms': int(in_the_way.sum())}


def show_dist(complex_model, param="3'", clearance=PATH_CLEARANCE, view=None, radius=1.2):
    """
    Рисует измеряемое расстояние прямо в nglview: цилиндр между концами, шарики на
    самих концах и подпись с длиной.
    Зеленый - прямая идет свободно. Красный - упирается в структуру, и тогда число
    по прямой мало что говорит: реальному линкеру пришлось бы идти в обход.
    view: можно передать уже готовый виджет, чтобы дорисовать поверх него
    """
    point_1, point_2, _ = dist_endpoints(complex_model, param)
    info = check_dist_path(complex_model, param, clearance)

    if view is None:
        view = nv.show_mdanalysis(complex_model.select_atoms('protein or nucleic'))

    colour = [0.9, 0.2, 0.2] if info['blocked'] else [0.2, 0.75, 0.3]
    start = [float(x) for x in point_1]
    end = [float(x) for x in point_2]
    view.shape.add_cylinder(start, end, colour, radius)
    view.shape.add_sphere(start, colour, radius * 2)
    view.shape.add_sphere(end, colour, radius * 2)

    middle = [float(x) for x in (point_1 + point_2) / 2]
    note = f"{param} {info['distance']:.1f}A"
    note += f" сквозь {','.join(info['blocked_by'])}" if info['blocked'] else ' свободно'
    view.shape.add_text(middle, [0, 0, 0], 4.0, note)
    return view


def get_dist(r_model_complex_list, param):
    """
    функция, находящая расстояние между 3' концами sgRNA двух Cas
    param - какие параметры измерять
    3' - расстояние между 3' концами
    C - расстояние между С концами
    N - расстояние между N концами

    ВНИМАНИЕ: меряется расстояние ПО ПРЯМОЙ, и она может идти насквозь через ДНК.
    Проверить это можно через check_dist_path, посмотреть глазами - через show_dist
    """
    distances = []
    model_complex_list = []
    
    if isinstance(r_model_complex_list, dict): # проверка на словарь
        for model_complex in r_model_complex_list.values():
            model_complex_list.append(model_complex) # переводим словарь в массив
    else:
        model_complex_list = r_model_complex_list
    
    if param == "3'":
        for model_complex in model_complex_list:
            last_seg_B = model_complex.select_atoms('segid B').residues.resnums[-1] # находим имена последнего остатка РНК цепи В
            last_seg_b = model_complex.select_atoms('segid b').residues.resnums[-1] # находим имена последнего остатка РНК цепи b

            seg_B_O3_coods = model_complex.select_atoms(f'segid B and (resnum {last_seg_B} and name O3\')').center_of_geometry() # находим координаты O3' на 3' конце РНК цепи В
            seg_b_O3_coods = model_complex.select_atoms(f'segid b and (resnum {last_seg_b} and name O3\')').center_of_geometry() # находим координаты O3' на 3' конце РНК цепи b

            distance = np.linalg.norm(seg_B_O3_coods - seg_b_O3_coods)
            distances.append(distance)
            
    elif param == 'N':
        for model_complex in model_complex_list:
            protein_A_N_end = model_complex.select_atoms('segid A').residues.resnums[0]
            protein_a_N_end = model_complex.select_atoms('segid a').residues.resnums[0]
            
            protein_A_N_end_coods = model_complex.select_atoms(f'segid A and (resnum {protein_A_N_end} and name N)').center_of_geometry()
            protein_a_N_end_coods = model_complex.select_atoms(f'segid a and (resnum {protein_a_N_end} and name N)').center_of_geometry()
            
            distance = np.linalg.norm(protein_A_N_end_coods - protein_a_N_end_coods)
            distances.append(distance)
    
    elif param == 'C':
        for model_complex in model_complex_list:
            protein_A_C_end = model_complex.select_atoms('segid A').residues.resnums[-1]
            protein_a_C_end = model_complex.select_atoms('segid a').residues.resnums[-1]
            
            protein_A_C_end_coods = model_complex.select_atoms(f'segid A and (resnum {protein_A_C_end} and name C)').center_of_geometry()
            protein_a_C_end_coods = model_complex.select_atoms(f'segid a and (resnum {protein_a_C_end} and name C)').center_of_geometry()
            
            distance = np.linalg.norm(protein_A_C_end_coods - protein_a_C_end_coods)
            distances.append(distance)

    return distances


def check_model(complex_model, bond_cutoff=SEAM_CUTOFF):
    """
    Полная проверка собранной модели - печатает, где именно рвется.
    Проверяет: состав цепей, совпадение segid и chainID, связность остова,
    пропуски в нумерации, спаривание цепей C и D.
    Возвращает список найденных проблем (пустой - значит все чисто).
    """
    problems = []

    print('ЦЕПИ:')
    for seg in sorted(set(complex_model.atoms.segids)):
        g = complex_model.select_atoms(f'segid {seg}')
        chains = sorted(set(g.chainIDs))
        kind = 'белок' if g.select_atoms('protein').n_atoms else 'НК'
        print(f'   segid {seg}: {len(g.residues):>5} ост. {kind:<6} chainID {chains}')
        if chains != [seg]:
            problems.append(f'segid {seg} имеет chainID {chains} - вьюер увидит разные цепи')

    print('\nОСТОВ НУКЛЕИНОВЫХ КИСЛОТ:')
    nucleic = complex_model.select_atoms('nucleic')
    for ch in sorted(set(nucleic.chainIDs)):
        res = nucleic.select_atoms(f'chainID {ch}').residues
        rn = [int(x) for x in res.resnums]
        breaks = []
        for res_1, res_2 in zip(res[:-1], res[1:]):
            O3 = res_1.atoms.select_atoms("name O3'")
            P = res_2.atoms.select_atoms('name P')
            if not len(O3) or not len(P):
                breaks.append((int(res_1.resnum), int(res_2.resnum), 'нет атома'))
                continue
            d = float(np.linalg.norm(O3.positions[0] - P.positions[0]))
            if d > bond_cutoff or d < SEAM_MIN:
                breaks.append((int(res_1.resnum), int(res_2.resnum), round(d, 2)))
        step = 1 if len(rn) > 1 and rn[-1] > rn[0] else -1
        jumps = [(rn[i], rn[i + 1]) for i in range(len(rn) - 1) if rn[i + 1] - rn[i] != step]
        print(f'   chainID {ch}: {len(rn):>4} ост., resnum {rn[0]}..{rn[-1]}, '
              f'разрывов {len(breaks)}, пропусков нумерации {len(jumps)}')
        for b in breaks[:5]:
            print(f'        разрыв между {b[0]} и {b[1]}: {b[2]} A')
            problems.append(f'chainID {ch}: разрыв остова {b[0]}->{b[1]} = {b[2]} A')
        for j in jumps[:5]:
            print(f'        пропуск нумерации: {j[0]} -> {j[1]}')
            problems.append(f'chainID {ch}: пропуск нумерации {j[0]}->{j[1]}')

    C = {int(r.resnum): r for r in complex_model.select_atoms('chainID C').residues}
    D = {int(r.resnum): r for r in complex_model.select_atoms('chainID D').residues}
    if C and D:
        print('\nСПАРИВАНИЕ ЦЕПЕЙ C и D (C1\'-C1\', спаренная пара ~10.5 A):')
        only_C = sorted(set(C) - set(D))
        only_D = sorted(set(D) - set(C))
        if only_C: problems.append(f'номера есть только в C: {only_C}')
        if only_D: problems.append(f'номера есть только в D: {only_D}')
        paired, apart = [], []
        for n in sorted(set(C) & set(D)):
            a = C[n].atoms.select_atoms("name C1'")
            b = D[n].atoms.select_atoms("name C1'")
            if not len(a) or not len(b): continue
            d = float(np.linalg.norm(a.positions[0] - b.positions[0]))
            (paired if 9.0 <= d <= 12.0 else apart).append(n)
        print(f'   спарено: {len(paired)} позиций')
        print(f'   расходятся: {len(apart)} позиций {apart[:30]}')
        print('   расхождение на отрицательных номерах - это R-петля, так и должно быть:')
        print('   там таргетная цепь спарена с гРНК, а нетаргетная вытеснена')
        if only_C or only_D:
            print(f'   ТОЛЬКО в C: {only_C}   ТОЛЬКО в D: {only_D}')

    print('\nИТОГ:', 'проблем не найдено' if not problems else f'проблем: {len(problems)}')
    for p in problems:
        print('   -', p)
    return problems


def backbone_bonds(complex_model):
    """
    все связи O3'-P между соседними нуклеотидами в цепях ДНК.
    Возвращает список (длина, цепь, resnum первого, resnum второго).
    Норма ~1.6 A, отклоняются только два стыка сшивки
    """
    bonds = []
    nucleic = complex_model.select_atoms('nucleic')
    for ch in ('C', 'D'):
        res = nucleic.select_atoms(f'chainID {ch}').residues
        for res_1, res_2 in zip(res[:-1], res[1:]):
            O3 = res_1.atoms.select_atoms("name O3'")
            P = res_2.atoms.select_atoms('name P')
            if len(O3) and len(P):
                bonds.append((float(np.linalg.norm(O3.positions[0] - P.positions[0])),
                              ch, int(res_1.resnum), int(res_2.resnum)))
    return bonds


def fix_seam_geometry(complex_model, min_len=SEAM_MIN, max_len=SEAM_CUTOFF, verbose=False):
    """
    Чинит геометрию швов сшивки: ставит фосфат так, чтобы связь O3'-P стала
    нормальной длины. Двигаются только три атома - P, OP1, OP2 - на доли ангстрема.
    Основания, сахара и оба белка остаются на месте, поэтому измеряемые расстояния
    между Cas (3'-3', C-C, N-N) не меняются.

    Правка нужна по существу, а не для красоты: наложение не может дать точный шов
    в принципе. У белок-связанной ДНК и у свободной B-ДНК из библиотеки разная
    локальная геометрия пар оснований, RMSD наложения выходит 2.2-2.4 A, и вся
    невязка садится на одну связь O3'-P. Собственная геометрия обеих половин при
    этом идеальна (родная связь 1.59-1.61 A).

    Возвращает список исправленных швов: (цепь, resnum, было, стало)
    """
    fixed = []
    for length, ch, resnum_1, resnum_2 in backbone_bonds(complex_model):
        if min_len <= length <= max_len:
            continue
        chain = complex_model.select_atoms(f'chainID {ch}')
        res_1 = chain.select_atoms(f'resnum {resnum_1}')
        res_2 = chain.select_atoms(f'resnum {resnum_2}')
        O3 = res_1.select_atoms("name O3'")
        P = res_2.select_atoms('name P')
        O5 = res_2.select_atoms("name O5'")
        if not (len(O3) and len(P) and len(O5)):
            continue

        C3 = res_1.select_atoms("name C3'")
        C5 = res_2.select_atoms("name C5'")
        OP = [res_2.select_atoms(f'name {name}') for name in ('OP1', 'OP2', 'O1P', 'O2P')]
        OP = [g for g in OP if len(g)]
        if not (len(C3) and len(C5) and len(OP) == 2):
            continue

        # Подвижны четыре атома: P, два несвязывающих кислорода и O5'. Все остальное -
        # сахара, основания, белки - закреплено, поэтому поправка никуда не расползается.
        # Двигать один фосфат недостаточно: он задает только длины связей, а угол
        # O3'-P-O5' тогда жестко определяется расстоянием O3'...O5' и выходит из нормы
        anchor_O3 = O3.positions[0].astype(float)
        anchor_C3 = C3.positions[0].astype(float)
        anchor_C5 = C5.positions[0].astype(float)
        start = np.concatenate([P.positions[0], OP[0].positions[0],
                                OP[1].positions[0], O5.positions[0]]).astype(float)

        # Отдаем вектор невязок, а не сумму квадратов: least_squares знает про структуру
        # задачи и сходится за полсотни вызовов вместо восьмисот у общего минимизатора
        def geometry_residuals(flat):
            p, op1, op2, o5 = flat[0:3], flat[3:6], flat[6:9], flat[9:12]
            def bond(x, y, target):
                return float(np.linalg.norm(x - y)) - target
            def angle(x, y, z, target):
                u, v = x - y, z - y
                cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
                deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                return (deg - target) / 50.0 # 50 град невязки по углу ~ 1 A по связи
            return np.array([
                bond(anchor_O3, p, PHOSPHODIESTER_O3P), bond(p, o5, PHOSPHODIESTER_PO5),
                bond(o5, anchor_C5, BOND_O5_C5),
                bond(p, op1, BOND_P_OP), bond(p, op2, BOND_P_OP),
                angle(anchor_C3, anchor_O3, p, ANGLE_C3_O3_P),
                angle(anchor_O3, p, o5, ANGLE_O3_P_O5),
                angle(p, o5, anchor_C5, ANGLE_P_O5_C5),
                angle(op1, p, op2, ANGLE_OP_P_OP),
                angle(anchor_O3, p, op1, ANGLE_O_P_OP), angle(anchor_O3, p, op2, ANGLE_O_P_OP),
                angle(o5, p, op1, ANGLE_O_P_OP), angle(o5, p, op2, ANGLE_O_P_OP)])

        result = scipy.optimize.least_squares(geometry_residuals, start, xtol=1e-10)
        P.positions = result.x[0:3]
        OP[0].positions = result.x[3:6]
        OP[1].positions = result.x[6:9]
        O5.positions = result.x[9:12]

        new_length = float(np.linalg.norm(result.x[0:3] - anchor_O3))
        shift = float(np.linalg.norm(result.x[0:3] - start[0:3]))
        fixed.append((ch, int(resnum_1), int(resnum_2), round(length, 2), round(new_length, 2)))
        if verbose:
            print(f"   шов {ch} {resnum_1}->{resnum_2}: O3'-P {length:.2f} -> {new_length:.2f} A, "
                  f'фосфат сдвинут на {shift:.2f} A')
    return fixed


def max_backbone_gap(complex_model):
    """максимальная связь O3'-P в цепях ДНК"""
    bonds = backbone_bonds(complex_model)
    return max(b[0] for b in bonds) if bonds else 0.0


def min_backbone_bond(complex_model):
    """минимальная связь O3'-P в цепях ДНК"""
    bonds = backbone_bonds(complex_model)
    return min(b[0] for b in bonds) if bonds else 0.0


def check_seams(complex_model, cutoff=SEAM_CUTOFF, min_len=SEAM_MIN):
    """
    Проверяет стыки сшитой ДНК в обе стороны: связь не должна быть ни слишком
    длинной, ни слишком короткой. Пересжатая связь - такой же разрыв, как и
    растянутая: вьюер рисует ее, только если длина попадает в окно примерно
    1.53-2.33 A (сумма ковалентных радиусов O и P с допуском). На шве атомы
    могут налезть друг на друга, и связь в 1.1 A на экране выглядит разрывом
    """
    bonds = backbone_bonds(complex_model)
    if not bonds:
        return False
    lengths = [b[0] for b in bonds]
    return max(lengths) > cutoff or min(lengths) < min_len


def check_clashing(complex_model):
    # функция, которая проверет сталкиваются ли касы или нет
    around_seg_A = complex_model.select_atoms('around 3.5 (segid A or segid B)') # выделяем область вокруг первого Cas 
    if len(around_seg_A.select_atoms('segid a or segid b').residues.resnums): # проверяем наличие атомов второго Cas в выделенной области
        return True
    else:
        return False
    
def how_many_atoms_clashing(complex_model):
    around_seg_A = complex_model.select_atoms('around 3.5 (segid A or segid B)') # выделяем область вокруг первого Cas 
    return len(around_seg_A.select_atoms('segid a or segid b')) # выводим количество атомов второго кас, попавших в выделение


STATS_BATCH = 8 # сколько дистанций рабочий процесс собирает за один вызов build_models.
# Больше - реже грузятся структуры Cas, но больше моделей висит в памяти разом


class _TqdmQueue:
    """
    Заглушка под очередь для последовательного режима: put двигает прогрессбар
    напрямую, без всякого межпроцессного обмена
    """
    def __init__(self, bar):
        self.bar = bar

    def put(self, _):
        self.bar.update(1)


def _collect_chunk(task):
    """
    Одна порция дистанций для отдельного процесса: собирает модели и меряет расстояния.
    Функция верхнего уровня, иначе multiprocessing не сможет ее передать в процесс.
    Наружу отдаются только числа, сами модели в процессе и остаются - передавать
    Universe между процессами и дорого, и незачем.
    progress_queue - сюда кладется единица на каждую собранную модель, по ним
    главный процесс двигает прогрессбар
    """
    (Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation,
     chunk, params, repeats, seed, progress_queue) = task
    random.seed(seed)

    values = {d: {param: [] for param in params} for d in chunk}
    blocked = {d: {param: [] for param in params} for d in chunk}
    clashing = {d: [] for d in chunk}

    for batch_start in range(0, len(chunk), STATS_BATCH): # режем на порции, чтобы не
        batch = chunk[batch_start : batch_start + STATS_BATCH] # держать в памяти все разом
        for _ in range(repeats):
            built = build_models(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model,
                                 orientation, list(batch), 0, progress=False)
            if built is None: # разнос вне допустимого диапазона
                continue
            for PAM_PAM_distance in batch:
                complex_model = built[PAM_PAM_distance]
                for param in params:
                    info = check_dist_path(complex_model, param)
                    values[PAM_PAM_distance][param].append(info['distance'])
                    blocked[PAM_PAM_distance][param].append(info['blocked'])
                clashing[PAM_PAM_distance].append(check_clashing(complex_model))
                if progress_queue is not None: # докладываем о каждой сборке, чтобы
                    progress_queue.put(1) # прогрессбар шел, а не стоял до конца

    # Складываем результат ПО КАЖДОЙ СБОРКЕ, а не усредненный. Усреднять по всем
    # подряд нельзя: часть моделей со столкновением, то есть физически невозможна,
    # и среднее по такой смеси ничего не описывает. Кто с чем - решается при отрисовке
    stats = {}
    for PAM_PAM_distance in chunk:
        if not clashing[PAM_PAM_distance]:
            continue
        stats[PAM_PAM_distance] = {
            param: {'values': values[PAM_PAM_distance][param],
                    'blocked': blocked[PAM_PAM_distance][param]}
            for param in params}
        stats[PAM_PAM_distance]['clashing'] = clashing[PAM_PAM_distance]
    return stats


def collect_dist_stats(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation,
                       PAM_PAM_distance_list, params=("3'",), repeats=5, processes=None):
    """
    Собирает статистику расстояний по диапазону разносов PAM-PAM.

    Конформер линкерной ДНК вытягивается случайно, поэтому на каждом разносе
    получается не одно число, а разброс - делаем repeats сборок и собираем все.

    Cas_name_1, Cas_name_2 - названия Cas белков
    Cas1_model, Cas2_model - PDB ID или пути до файлов
    orientation - {0:'PAM-direct', 1:'PAM-in', 2:'PAM-out'}
    PAM_PAM_distance_list - список разносов PAM-PAM, в парах оснований
    params - какие расстояния мерить: "3'", 'C', 'N'. Можно строкой для одного
             или списком/кортежем для нескольких
    repeats - сколько сборок на каждый разнос
    processes - во сколько процессов считать. None - по числу ядер, 1 - без
                распараллеливания. На коротких заданиях процессы не окупаются:
                каждый заново импортирует MDAnalysis, это около секунды

    Возвращает dict: разнос -> {param: {'values': [...], 'blocked': [...]},
                                'clashing': [...]}
    Все списки идут по сборкам и выровнены между собой: i-й элемент везде относится
    к i-й сборке. Так можно отделить модели со столкновением от остальных
    """
    if isinstance(params, str): # ("3'") без запятой это строка, а не кортеж -
        params = (params,) # иначе пошли бы перебирать ее по символам
    distances = list(PAM_PAM_distance_list)
    if not distances:
        return {}

    if processes is None:
        processes = min(os.cpu_count() or 1, len(distances))
        if len(distances) * repeats < 20: # мелкое задание, накладные расходы съедят выигрыш
            processes = 1

    total_builds = len(distances) * repeats

    if processes <= 1:
        with tqdm(total=total_builds, desc='Сборок посчитано') as bar:
            task = (Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation,
                    distances, params, repeats, random.randrange(10 ** 6), _TqdmQueue(bar))
            stats = _collect_chunk(task)
            bar.n = total_builds
            bar.refresh()
        return stats

    # раздаем дистанции вперемешку, чтобы тяжелые длинные не свалились в один процесс
    chunks = [distances[i::processes] for i in range(processes)]

    stats = {}
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        tasks = [(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation,
                  chunk, params, repeats, random.randrange(10 ** 6), progress_queue)
                 for chunk in chunks if chunk]

        with multiprocessing.Pool(len(tasks)) as pool:
            running = pool.map_async(_collect_chunk, tasks)
            # сами процессы считают долго, поэтому прогресс берем не из того, сколько
            # порций закончилось, а из очереди: воркеры кладут туда по единице на модель
            with tqdm(total=total_builds,
                      desc=f'Сборок посчитано ({len(tasks)} процессов)') as bar:
                while not running.ready() or not progress_queue.empty():
                    try:
                        progress_queue.get(timeout=0.2)
                        bar.update(1)
                    except queue.Empty:
                        pass
                bar.n = total_builds # разносы вне диапазона моделей не дают, добираем
                bar.refresh()
            for part in running.get():
                stats.update(part)
    return stats


def plot_dist_stats(stats, params=None, title='', ax=None, only_free=True):
    """
    Рисует график расстояний от разноса PAM-PAM по данным collect_dist_stats.

    only_free=True - в расчет идут ТОЛЬКО сборки без столкновений. Это главное:
    модели, где Cas-белки налезают друг на друга, физически невозможны, и
    усреднять их вместе с остальными нельзя. Там, где бесстолкновительных
    сборок не нашлось вовсе, на разносе не рисуется ничего - значит, такой
    разнос недостижим.

    На каждом разносе - свой box-plot по сборкам: прямоугольник от первого до
    третьего квартиля, усы до минимума и максимума, маркер на медиане. Между
    разносами точки не соединяются: каждый разнос считается сам по себе.
    Закрашенный маркер: прямая между концами идет свободно.
    Полый маркер: прямая перекрыта структурой, и тогда расстояние по прямой
    мало что говорит - реальному линкеру пришлось бы идти в обход.
    """
    if not stats:
        raise ValueError('Статистика пустая - нечего рисовать')
    distances = sorted(stats)
    if isinstance(params, str): # то же самое: одиночный параметр можно передать строкой
        params = (params,)
    if params is None:
        params = [key for key in stats[distances[0]] if key != 'clashing']
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    colours = {"3'": 'tab:red', 'C': 'tab:blue', 'N': 'tab:orange'}
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # для параметров не из colours
    x = np.array(distances, dtype=float)
    step = float(np.diff(x).min()) if len(x) > 1 else 1.0
    # при нескольких параметрах box-plot'ы одного разноса разводим в стороны,
    # иначе они встанут друг на друга
    slot = step * 0.6 / len(params)
    offsets = (np.arange(len(params)) - (len(params) - 1) / 2) * slot

    for i, (param, offset) in enumerate(zip(params, offsets)):
        samples, medians, blocked_frac = [], [], []
        for PAM_PAM_distance in distances:
            record = stats[PAM_PAM_distance]
            keep = np.array([not c for c in record['clashing']]) if only_free \
                else np.ones(len(record['clashing']), dtype=bool)
            taken = np.array(record[param]['values'])[keep]
            samples.append(taken)
            if not len(taken): # все сборки на этом разносе со столкновением
                medians.append(np.nan)
                blocked_frac.append(0.0)
                continue
            medians.append(np.median(taken))
            blocked_frac.append(float(np.mean(np.array(record[param]['blocked'])[keep])))
        medians = np.array(medians)
        blocked = np.array(blocked_frac) > 0.5
        drawn = ~np.isnan(medians)

        colour = colours.get(param) or cycle[i % len(cycle)]
        # квадратик в легенду вместо линии: точки разносов между собой не соединяем
        ax.plot([], [], 's', ms=8, color=colour, alpha=0.45, label=f'{param} - {param}')
        if drawn.any(): # box-plot на каждую точку, по сборкам этого разноса
            ax.boxplot([s for s, ok in zip(samples, drawn) if ok],
                       positions=x[drawn] + offset, widths=slot * 0.8,
                       whis=(0, 100), showfliers=False, # усы до минимума и максимума
                       patch_artist=True, manage_ticks=False, zorder=1,
                       boxprops=dict(facecolor=colour, alpha=0.25, edgecolor=colour, lw=0.8),
                       whiskerprops=dict(color=colour, lw=0.8),
                       capprops=dict(color=colour, lw=0.8),
                       medianprops=dict(color=colour, lw=1.2))
        ax.plot(x[drawn & ~blocked] + offset, medians[drawn & ~blocked], 'o', ms=5, color=colour)
        ax.plot(x[drawn & blocked] + offset, medians[drawn & blocked], 'o', ms=5,
                markerfacecolor='white', markeredgecolor=colour)

    # пояснения к маркерам - по одному разу, а не на каждое расстояние
    ax.plot([], [], 'o', ms=5, color='0.35', label='прямая идет свободно')
    ax.plot([], [], 'o', ms=5, markerfacecolor='white', markeredgecolor='0.35',
            label='прямая перекрыта структурой')

    ax.set_xlabel('разнос PAM-PAM, пар оснований')
    ax.set_ylabel('расстояние, A')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='best')
    label_every_point(ax, x)
    return ax


def label_every_point(ax, x):
    """
    Подписывает по оси X каждый посчитанный разнос, а не те деления, которые
    выберет matplotlib: разносы заданы списком, промежуточных значений между
    ними нет, и круглые деления вроде 50 или 75 могут вообще не считаться.
    """
    ax.set_xticks(x)
    ax.set_xticklabels([f'{value:g}' for value in x],
                       rotation=90 if len(x) > 18 else 0) # иначе подписи слипнутся
    ax.tick_params(axis='x', labelsize=7)
    return ax


def plot_clashing(stats, ax):
    """
    Панель с долей сборок, в которых Cas-белки столкнулись, по разносам PAM-PAM.
    Столбик до 100% означает, что на этом разносе бесстолкновительных моделей нет
    вовсе - в верхнем графике на этом разносе будет пусто
    """
    distances = sorted(stats)
    x = np.array(distances)
    clash = np.array([np.mean(stats[d]['clashing']) for d in distances]) * 100
    width = (x[1] - x[0]) * 0.85 if len(x) > 1 else 0.85
    ax.bar(x, clash, width=width, color='crimson', alpha=0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel('столкновения,\n% сборок', fontsize=8)
    ax.set_xlabel('разнос PAM-PAM, пар оснований')
    ax.grid(alpha=0.25, axis='y')
    ax.tick_params(labelsize=8)
    label_every_point(ax, x)
    return ax


def plot_dist(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model, orientation,
              PAM_PAM_distance_list, params=("3'",), repeats=5, save_png=None,
              processes=None, only_free=True):
    """
    Собирает статистику и сразу строит график: задаете белки, ориентацию и список
    разносов PAM-PAM, получаете график расстояний.

    Сверху - box-plot на каждом разносе по сборкам БЕЗ столкновений, снизу - какая
    доля сборок столкнулась. Там, где бесстолкновительных нет совсем, сверху пусто:
    такой разнос недостижим, и мерить там нечего.

    Пример:
        fig, stats = plot_dist('SpCas9', 'SpCas9', 'PDB/5Y36.pdb', 'PDB/5Y36.pdb',
                               2, range(42, 100, 4), params=("3'", 'C'), repeats=5)

    save_png - путь, куда сохранить картинку. None - не сохранять
    processes - во сколько процессов считать. None - по числу ядер
    only_free - считать разброс только по сборкам без столкновений. False вернет
                прежнее поведение, когда в среднее идут и невозможные модели
    Возвращает (fig, stats), чтобы можно было дорисовать свое или покопаться в данных
    """
    cas_or = {0: 'PAM-direct', 1: 'PAM-in', 2: 'PAM-out'}
    if isinstance(params, str):
        params = (params,)
    PAM_PAM_distance_list = list(PAM_PAM_distance_list)
    stats = collect_dist_stats(Cas_name_1, Cas_name_2, Cas1_model, Cas2_model,
                               orientation, PAM_PAM_distance_list, params, repeats,
                               processes)
    title = (f'{Cas_name_1} {Cas1_model}   {Cas_name_2} {Cas2_model}   '
             f'{cas_or[orientation]}   по {repeats} сборок на точку')
    fig, (ax, ax_clash) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [4, 1]})
    plot_dist_stats(stats, params, title, ax, only_free)
    ax.set_xlabel('')
    plot_clashing(stats, ax_clash)
    fig.tight_layout()
    if save_png:
        fig.savefig(save_png, dpi=130)
        print(f'график сохранен: {save_png}')
    return fig, stats
