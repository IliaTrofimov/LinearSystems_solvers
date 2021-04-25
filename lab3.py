import numpy as np
from numpy import linalg as lng

# =========================================================
# Реализация разряженной матрицы
class CrossMatr:
    """
    Разряженная матрица с заполнеными главной и побочной диагоналями и одним столбцом по середине
    """
    def __init__(self, main_diag, side_diag, column, filler=0):
        """
        :param main_diag: Элементы главной диагонали
        :param side_diag: Элементы побочной диагонали
        :param column: Элементы среднего столбца
        :param filler: Заполняет пустые места в матрице
        """
        if len(main_diag) != len(side_diag) or len(column) != len(side_diag):
            raise Exception(f"Размеры не входных матриц не равны: {len(main_diag)}, {len(side_diag)}, {len(column)}")
        self._size = len(main_diag)
        self.filler = filler
        self.main_diag = list()
        self.side_diag = list()
        self.column = list()

        for i in range(self._size):
            self.main_diag.append(main_diag[i])
            self.side_diag.append(side_diag[i])
            self.column.append(column[i])

    def to_ndarray(self):
        temp = np.ones((self.size, self.size)) * self.filler
        for i in range(self.size):
            temp[i, (self.size - 1) // 2] = self.column[i]
            temp[i, self.size - i - 1] = self.side_diag[i]
            temp[i, i] = self.main_diag[i]
        return temp

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return f"m: {np.transpose(self.main_diag)}\ns: {np.transpose(self.side_diag)}\nc: {np.transpose(self.column)}"

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if item[1] == 0:
                return self.main_diag[item[0]]
            elif item[1] == 1:
                return self.side_diag[item[0]]
            elif item[1] == 2:
                return self.column[item[0]]
            else:
                raise IndexError
        elif isinstance(item, int):
            return self.main_diag[item], self.side_diag[item], self.column[item]
        else:
            raise IndexError

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            if item[1] == 0:
                self.main_diag[item[0]] = value
            elif item[1] == 1:
                self.side_diag[item[0]] = value
            elif item[1] == 2:
                self.column[item[0]] = value
            else:
                raise IndexError
        elif isinstance(item, int) and isinstance(value, tuple):
            self.main_diag[item] = value[0]
            self.side_diag[item] = value[1]
            self.column[item] = value[2]
        else:
            raise IndexError

    def __str__(self):
        return self.to_ndarray().__str__()


# =========================================================
# Реализация необходимых методов и вспомогательные функции

def find_maxRow(matr: np.ndarray, row_start=0, col=0):
    """
    :param matr: Входная матрица
    :param row_start: Задаёт начальную строку поиска, предыдущие строки не будут обрабатываться
    :param col: Колонка, ао которой будет вестись поиск
    :return: Номер строки с максимальным элементом
    """
    pos_y = row_start
    for i in range(row_start, matr.shape[0]):
        if abs(matr[i, col]) > abs(matr[pos_y, col]):
            pos_y = i
    return pos_y


def get_LU(matr: np.ndarray):
    """ LU разложение
    :param matr: Входная матрица
    :return: Кортеж из матриц L и U
    """
    matr_u = matr.copy()
    matr_l = np.zeros(matr.shape, dtype=float)

    for j in range(0, matr.shape[0]):
        matr_l[j, j] = 1
        for i in range(j + 1, matr.shape[0]):
            matr_l[i, j] = matr_u[i, j] / matr_u[j, j]
            matr_u[i, j:] = matr_u[i, j:] - matr_u[j, j:] * matr_l[i, j]
    return matr_l, matr_u


def solve_LU(matr, b_vect: np.ndarray, matr_lu=None):
    """ Метод единственного деления с LU разложением
    :param matr: Матрица коэффициентов
    :param b_vect: Столбец свободных членов
    :param matr_lu: Если None, то matr будет преобразована в матрицу LU разложении, иначе
        matr не будет изменена, а LU разложение будет сохранено в matr_lu
    :return: Столбец решений
    """
    y_vect = np.copy(b_vect)
    rows = matr.shape[0]
    if matr_lu is None:
        matr_lu = matr
    else:
        matr_lu = matr.copy()

    for j in range(rows):
        for i in range(j + 1, rows):
            if matr_lu[i, j] != 0:
                key = matr_lu[i, j] / matr_lu[j, j]
                matr_lu[i, j + 1:] = matr_lu[i, j + 1:] - matr_lu[j, j + 1:] * key
                matr_lu[i, j] = key

    for i in range(rows):
        for j in range(0, i):
            y_vect[i] = y_vect[i] - matr_lu[i, j] * y_vect[j]

    for i in range(rows - 1, -1, -1):
        for j in range(i + 1, rows):
            y_vect[i] = y_vect[i] - matr_lu[i, j] * y_vect[j]
        y_vect[i] = y_vect[i] / matr_lu[i, i]
    return y_vect


def solve_select(matr, b_vect: np.ndarray):
    """ Метод частичного выбора с LU разложением
    :param matr: Матрица коэффициентов
    :param b_vect: Столбец свободных членов
    :return: Столбец решений
    """
    rows = matr.shape[0]
    y_vect = np.array(b_vect, dtype=float)
    matr_u = matr.copy()
    matr_l = np.zeros(matr.shape, dtype=float)

    for j in range(rows - 1):
        pos = find_maxRow(matr_u, j, j)
        matr_u[[j, pos]] = matr_u[[pos, j]]
        matr_l[[j, pos]] = matr_l[[pos, j]]
        y_vect[[j, pos]] = y_vect[[pos, j]]
        for i in range(j + 1, rows):
            key = matr_u[i, j] / matr_u[j, j]
            matr_u[i, j + 1:] = matr_u[i, j + 1:] - matr_u[j, j + 1:] * key
            matr_l[i, j] = key

    for i in range(rows):
        for j in range(0, i):
            y_vect[i] = y_vect[i] - matr_l[i, j] * y_vect[j]

    for i in range(rows - 1, -1, -1):
        for j in range(i + 1, rows):
            y_vect[i] = y_vect[i] - matr_u[i, j] * y_vect[j]
        y_vect[i] = y_vect[i] / matr_u[i, i]
    return y_vect


def solve_cross(matr: CrossMatr, b_vect: np.ndarray):
    """ Решает СЛАУ с разряженной матрицей-крестом
    :param matr: Матрица коэффициентов
    :param b_vect: Столбец свободных членов
    :return: Столбец решений
    """
    if not isinstance(matr, CrossMatr):
        raise ValueError("Неверный тип матрицы")

    x_vect = np.zeros((matr.size, 1), dtype=float)
    rows = matr.size - 1

    if matr.size % 2 == 0:
        r = (rows - 1) // 2
        det = matr[r, 0] * matr[r + 1, 0] - matr[r, 1] * matr[r + 1, 1]
        x_vect[r] = (b_vect[r] * matr[r + 1, 0] - b_vect[r + 1] * matr[r, 1]) / det
        x_vect[r + 1] = (b_vect[r] * matr[r, 0] - b_vect[r + 1] * matr[r + 1, 1]) / det
    else:
        r = rows // 2
        x_vect[r] = b_vect[r] / matr[r, 0]

    for i in range(r):
        det = matr[i, 0] * matr[rows - i, 0] - matr[i, 1] * matr[rows - i, 1]
        x1 = x_vect[r] * matr[i, 2]
        x2 = x_vect[r] * matr[rows - i, 2]
        x_vect[i] = ((b_vect[i] - x1) * matr[i, 0] - (b_vect[rows - i] - x2) * matr[i, 1]) / det
        x_vect[rows - i] = ((b_vect[rows - i] - x2) * matr[i, 0] - (b_vect[i] - x1) * matr[i, 1]) / det
    return x_vect


def solve_simpleIterations(matr, b_vect: np.ndarray, eps=10**(-6), get_iterations=False):
    """ Решает СЛАУ методом простой итерации с параметром
    :param matr: Матрица коэффициентов. Должна быть симметрической
    :param b_vect: Столбец свободных членов
    :param eps: Точность
    :param get_iterations: Отсавьте True, чтобы вернуть количество итераций помимо столбца решений
    :return: Столбец решений
    """
    eigs = lng.eigvals(matr)
    tau = 2 / (max(eigs) + min(eigs))
    q = (max(eigs) - min(eigs)) / (max(eigs) + min(eigs))
    matr_b = np.eye(matr.shape[0], dtype=float) - tau * matr

    c = b_vect.copy() * tau

    x = c.copy()
    xn = np.matmul(matr_b, x) + c
    n = 1

    while lng.norm(x - xn, np.inf) >= eps * (1 - q) / q:
        x = xn
        xn = np.matmul(matr_b, x) + c
        n += 1
    if get_iterations:
        return xn, n
    else:
        return xn

