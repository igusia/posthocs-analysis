import sys
import time
import multiprocessing as mp
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, chi2_contingency, chisquare

output = mp.Queue()
start = time.time()
interval = int(96/int(sys.argv[2]))

def posthocs(num, pos, output):
    """
    For each of the num iterations performs the following steps to return 7 different measures:
    1. Draws n numbers for X and for Y (two dimensions) from standard normal distribution
    2. Checks Pearson correlation between X and Y
    3. Checks nonparametric Spearman correlation between X and Y
    4. If two halves of X are non-empty, checks Ys in both groups with T-test and Mann-Whitney-Wilcoxon
    5. Same as 4. with X and Y swapped
    6. If each Cartesian quadrant is non-empty, checks Chi2
    7. If any of the 7 measures is statistically significant, increments value of positive tests
    erforms a set of statistical tests (Pearson, Spearman
    Counts false-positives for each of the tests: Pearson, Spearman,
    :param num: denotes how many iterations will be performed for each n
    :param pos: denotes id of a process
    :param output: queue to which results are sent
    :return: list of
    """
    counts=[]
    for n in range(pos*interval+5, pos*interval+5+interval):
        c0=c1=c2=c3=c4=c5=c6=0.0
        tests_pos = 0  # number of iterations with at least 1 test passed
        numX = 0  # number of iterations with all x-halves
        numY = 0  # number of iterations with all y-halves non-empty
        numchi2 = 0  # number of iterations with all quadrants non-empty
        for _ in range(num):
            any_positive = False  # indicates if any of the tests is positive in the iteration
            X = np.random.normal(size=n)
            Y = np.random.normal(size=n)
            # Cartesian quadrants
            q1 = list((x,y) for x,y in zip(X,Y) if x>=0 and y>=0)
            q2 = list((x,y) for x,y in zip(X,Y) if x<0 and y>=0)
            q3 = list((x,y) for x,y in zip(X,Y) if x<0 and y<0)
            q4 = list((x,y) for x,y in zip(X,Y) if x>=0 and y<0)
            # divinding into positive and negative halves for Xs and Ys
            dist_X_pos = q1+q4
            dist_X_neg = q2+q3
            dist_Y_pos = q1+q2
            dist_Y_neg = q3+q4
            # Pearson correlation
            if pearsonr(X,Y)[1] < 0.05:
                c0 += 1
                any_positive = True
            # nonparametric Spearman
            if spearmanr(X,Y)[1] < 0.05:
                c1 += 1
                any_positive = True
            # if both X-halves are non-empty, perform T-test and Mann-Whitney-Wilcoxon on Y values
            if len(dist_X_pos) and len(dist_X_neg):
                negXy = list(y for x,y in dist_X_neg)
                posXy = list(y for x,y in dist_X_pos)
                numX += 1
                if ttest_ind(posXy, negXy)[1] < 0.05:
                    c2 += 1
                    any_positive = True
                if mannwhitneyu(posXy, negXy, alternative='two-sided')[1] < 0.05:
                    c4 += 1
                    any_positive = True
            # same with Y-halves
            if len(dist_Y_pos) and len(dist_Y_neg):
                negYx = list(x for x,y in dist_Y_neg)
                posYx = list(x for x,y in dist_Y_pos)
                numY += 1
                if ttest_ind(posYx, negYx)[1] < 0.05:
                    c3 += 1
                    any_positive = True
                if mannwhitneyu(posYx, negYx, alternative='two-sided')[1] < 0.05:
                    c5 += 1
                    any_positive = True
            # chi2
            if len(dist_X_pos) and len(dist_X_neg) and len(dist_Y_pos) and len(dist_Y_neg):
                numchi2 += 1
                obs = np.array([[len(q1), len(q2)],[len(q4), len(q3)]])
                if chi2_contingency(obs)[1] < 0.05:
                    c6 += 1
                    any_positive = True
            if any_positive:
                tests_pos += 1
        counts.extend([[c0/num, c1/num, c2/numX, c3/numY, c4/numX, c5/numY, c6/numchi2, tests_pos]])
    output.put((pos, counts))

# takes number of iterations and number of CPUs used as parameters
processes = [mp.Process(target=posthocs, args=(int(sys.argv[1]), i, output)) for i in range(int(sys.argv[2]))]
for p in processes:
    p.start()

for p in processes:
    p.join()

# sort results by 1st element in a tuple (process number)
results = sorted([output.get() for p in processes], key=lambda x: x[0])
results_values = np.vstack([y for x,y in results])
np.savetxt('mp_results_{}.csv'.format(time.ctime()), results_values, delimiter=',')
print("--- %s seconds ---" % (time.time() - start))