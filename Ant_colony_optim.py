import numpy as np;
import random;

# numpy.info(numpy.add);
alfa = 1;
betta = 1;
Q = 80;


def invdist(dist, size):
    n_dist = dist.astype(float);
    return 1./n_dist


class Ant(object):

    def __init__(self, size, dist, ndist, feromon):

        self.size = size;
        self.history = np.zeros(size).astype(int);
        self.feromon = feromon;
        self.lenghtpath = 0;
        self.distantion = dist;
        self.n_dist = ndist.astype(float);

    def antOdyssey(self):
        initval = random.randint(0, (self.size - 1));
        self.history[0] = initval;
        a = initval;
        # self.lenghtpath += 1/self.n_dist[0:5, a];
        self.n_dist[0:5, a] = 0;
        for i in range(self.size - 1):
            self.history[i + 1] = self.nextStepNum(a);
            a = self.history[i + 1];
        self.lenghtpath += self.distantion[self.history[self.size - 1]][initval];
        dtau = Q / self.lenghtpath;
        self.feromon[self.history[self.size - 1]][self.history[0]] += dtau;
        for i in range(self.size - 1):
            self.feromon[self.history[i]][self.history[i + 1]] += dtau;
        print("Result feromones:\n");
        print(self.feromon);
        return self.feromon

    def nextStepNum(self, currentNum):
        print("----------- ", currentNum, " ------------")
        common_divider = ((self.n_dist[currentNum][0:5] ** alfa) * (self.feromon[3, 0:5] ** betta)).sum(0);
        print(common_divider);
        P = ((self.n_dist[currentNum][0:5] ** alfa) * (self.feromon[3, 0:5] ** betta)) * 100 / common_divider;
        print("Probability density\n", P);
        PP = np.zeros(self.size);
        for i in range(self.size):
            for j in range(i):
                PP[i] += P[j];
        print("Probability integral\n", PP);
        for i in range(self.size - 1):
            PP[i] = PP[i + 1];
        PP[self.size - 1] = 100;
        print("Probability integral after\n", PP);
        a = random.randint(1, 100);
        print(a);
        nextNum = 0;
        i = 0;
        while a > PP[i]:
            i += 1;
            nextNum = i;
        print(nextNum);
        self.lenghtpath += self.distantion[currentNum][nextNum];
        self.n_dist[0:5, nextNum] = 0;
        print(self.n_dist);
        return nextNum;


if __name__ == "__main__":

    Msize = 5;
    # матрица расстояний
    D = np.array([[0, 2, 30, 9, 1],
                  [4, 0, 47, 7, 7],
                  [31, 33, 0, 33, 36],
                  [20, 13, 16, 0, 28],
                  [9, 36, 22, 22, 0]])

    # матрица феромонов на первом этапе все элементы равны 1
    F = np.arange(Msize * Msize).reshape(Msize, Msize);
    F[0:5, 0:5] = 1;
    # print("Feromon matrix = ", F);

    for i in range(20):
        antcopy = Ant(Msize, D, invdist(D, Msize), F);
        F += antcopy.antOdyssey();

    # antcopy.print();
    # antcopy.nextStepNum(0);
    # antcopy.antOdyssey();
    print("Result feromones:\n");
    print(F);

    Len = 0;
    idx = 0;
    for i in range(Msize):
        next_idx = F[idx].argmax();
        Len += D[idx][next_idx];
        idx = next_idx;
        print(idx, Len);

