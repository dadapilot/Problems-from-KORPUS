
'''
Решение задачи о восьми ферзях генетическим алгоритмом.
В данной программе реализован стандартный ГА:
- Бинарная кодировка признаков;
- Селекция реализована “колесом рулетки”;
- Используется оператор одноточечного скрещивания.
------------------------------------------------------------------
Каждый вещественный ген кодируется тремя бинарными генами.
Начальная популяция задается рандомно из последовательностей 0 и 1.
Фитнесс функция возвращает значение, обратно пропорциональное
количеству атак для каждого ферзя. Максимум атак в случае n=8 28.
'''

import numpy as np
import random

class Solver_8_Queens:

    def __init__(self, pop_size = 150, cross_prob = 0.6, mut_prob = 0.25, board_n = 8):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.board_n = board_n
        self.Population = self.generate_population()

    def solve(self, max_epochs = 1000, min_fitness = 0.999):

        fit = 0
        k=0
        fit_value = np.array(([self.fitness_func(chrms) for chrms in self.Population]))

        while fit < min_fitness and k <= max_epochs:

            prob = [val/sum(fit_value) for val in fit_value]   # вероятности для каждой хромосомы (сектор рулетки)

            self.reproduction(prob)

            ind_list_for_cross = [i for i in range(self.pop_size)]
            num_chrom = np.random.randint(self.pop_size, size=2)  # пара индексов для списка с номерами хромосом для кроссинговера

            for i in range(int(self.pop_size)-1):

                ind = self.crossingover(ind_list_for_cross[int(num_chrom[0])], # индекс потомка, который нужно удалить из списка
                                  ind_list_for_cross[int(num_chrom[1])])

                ind_list_for_cross = list(filter(lambda x: x != ind, ind_list_for_cross))  # удаляем из списка использованные индексы (не даем потомкам участвовать в скрещивании)
                num_chrom = np.random.randint(self.pop_size-(1+i), size=2)  # ограничеваем выбор индекса

            for i in range(self.pop_size):

                prob_m = random.random()

                if prob_m <= self.mut_prob:

                    self.Population[i] = self.mutation(self.Population[i])

            fit_value = np.array(([self.fitness_func(chrms) for chrms in self.Population]))
            k+=1
            fit = np.max(fit_value)

        ind_Ans = np.argwhere(fit_value == np.max(fit_value))[0]
        Ans = []
        out = [['+' for _ in range(self.board_n)] for _ in range(self.board_n)]

        for i in range(0, self.board_n*3, 3):
            Ans.append(self.from_binary(self.Population[ind_Ans][0][i:i+3]))
        for i, j in enumerate(Ans):
            out[i][j] = 'Q'
            print(''.join(out[i]))
        return (fit, k, ''.join([''.join(st) for st in out ]))

    def from_binary(self, num):
        res = 0
        for j, i in enumerate(num):
            res += 2 ** (len(num) - j - 1) * int(i)
        return res

    def generate_population(self):
        return np.random.randint(0, 2, (self.pop_size, self.board_n * 3)) # кодируем признак тремя генами

    def fitness_func(self, chromosome):

        attack = 0

        for idx in range(0, int(len(chromosome))+3, 3):
            for ind in range(3+idx, int(len(chromosome)), 3):

               if abs(self.from_binary(chromosome[idx:idx+3]) - self.from_binary(chromosome[ind:ind+3])) - abs(idx/3 - ind/3) == 0:

                    attack += 1        # штрафуем за диагональный бой

               elif self.from_binary(chromosome[idx:idx+3]) == self.from_binary(chromosome[ind:ind+3]):

                    attack += 1        # штрафуем за бой по гор-ли и верт-ли

        return 1/(attack+1)   # чем больше значение res, тем меньше вероятность (сектор рулетки)

    def reproduction(self, prob):
        # Рулетка
        integral = np.cumsum(prob)

        for i in range(self.pop_size):

            idx = random.random()
            k = np.argwhere(integral>=idx)[0]

            self.Population[i] = self.Population[k]

    def crossingover(self, num_chrom1, num_chrom2):

        prob_cross = random.random()    # для расчета вероятноси кроссинговера между двумя случайными хромосомами
        k = random.randint(1, self.board_n*3-1)
        child_choise = random.random()

        if prob_cross <= self.cross_prob:

            # выбираем одного потомка из двух результатов скрещивания
            if child_choise < 0.5:

               self.Population[num_chrom1][self.Population.shape[1] - k:] = self.Population[num_chrom2][:k]
               return num_chrom1

            else:

               self.Population[num_chrom2][self.Population.shape[1] - k:] = self.Population[num_chrom1][:k]
               return num_chrom2

        #если скрещивания не произошло, возвращаем одного из потомков с вер-ю 0.5
        else: return num_chrom2 if prob_cross > self.cross_prob/2 else num_chrom1

    def mutation(self, chroma):
        gen = random.randint(0, self.board_n * 3 - 1)
        chroma[gen] = 1 - chroma[gen]
        return chroma
