import logging
from scipy.stats import zscore
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np

"""
Original programmer:  Maria Suarez-Diez
Date:   21/07/2015
Citation: Saccenti, Edoardo, Maria Suarez-Diez, Claudio Luchinat, Claudio Santucci, and Leonardo Tenori. 2014. “Probabilistic Networks of Blood 
          Metabolites in Healthy Subjects as Indicators of Latent Cardiovascular Risk.” Journal of Proteome Research (November 27). 
        doi:10.1021/pr501075r. http://dx.doi.org/10.1021/pr501075r.

Ported to Python by: Thomas Roder (roder.thomas@gmail.com)
Date:   24/11/2020
"""


# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


class PCLRC:
    """Probabilistic Context Likelihood of Relatedness on Correlation

    Quote:

    The PCLRC algorithm is based on the CLR algorithm and has been first introduced to reconstruct metabolite correlation networks. It combines the
    CLR approach replacing mutual information with correlation with iterative sampling. In each iteration, 70% of the samples from the full data
    set are randomly chosen to calculate the correlation matrix. Then, CLR is used to estimate possible associations and reconstruct a network,
    and then dynamical threshold is chosen so that the highest 30% of interactions are kept. The complete procedure (sampling and network
    generation) is iterated K times. The final network is constructed by assigning to every edge a weight defined as the frequency of the times
    that a particular edge was selected in the iterative process. This weight can be viewed as a probabilistic measurement of edge likeliness and
    can be interpreted as a confidence level on which to accept or reject the correlation between two pairs of metabolites. We set here K = 105.
    The algorithm outputs a weighted adjacency matrix whose entries are the probability of association for all possible X,Y metabolite pairs. We
    implemented the algorithm with both Pearson’s and Spearman’s correlation: the two variants are indicated with PCLRC-p and PCLRC-s, respectively.

    Source: Jahagirdar, Sanjeevan, Maria Suarez-Diez, and Edoardo Saccenti. Journal of proteome research 18.3 (2019): 1099-1113.

    Differences between this implementation and the R package:
        - CLR gives sliglty different output
        - application of a final threshold is optional
    """

    @staticmethod
    def pclrc(dataset: pd.DataFrame, threshold: float = None, n_iter=1000, frac=0.75, rank_thr=0.3, method='pearson', show_process=True
              ) -> pd.DataFrame:
        """performs  PCLRC and imposes a threshold to infer a network from a data matrix

        based on Saccenti et al 2015 J. Proteome Res., 2015, 14 (2)

        :param dataset: numeric matrix with the objects measurements. samples in rows, metabolites in cols.
        :param n_iter: number of iterations, default: 1000
        :param frac: fraction of the samples to be considered at each iteration, default: 0.75
        :param rank_thr: fraction of the predicted interactions to be kept at each iteration, default: 0.3
        :param threshold: threshold for an edge to be considered True, default: None (returns weighted matrix), default of R-package: 0.95
        :param method: method of correlation: ‘pearson’ (default), ‘kendall’, ‘spearman’ or 'bicor'
        :param show_process: print how many iterations have been calculated

        :returns: Adjacency matrix (weighted if no threshold was set) of interactions between the objects computed using the PCLRC algorithm.
        """
        # equivalent to PCLRC function in R
        matrix = PCLRC.__resample_clr(dataset,
                                      n_iter=n_iter,
                                      frac=frac,
                                      rank_thr=rank_thr,
                                      method=method,
                                      show_process=show_process
                                      )

        if threshold:
            if show_process:
                print(f'Applying threshold {threshold}...')
            matrix = PCLRC.apply_threshold(matrix, threshold=threshold)

        return pd.DataFrame(matrix, index=dataset.columns, columns=dataset.columns)

    @staticmethod
    def clr(matrix: np.ndarray) -> np.ndarray:
        """Context likelihood of relatedness (CLR) algorithm

        Based on Faith, Jeremiah J., et al. PLoS biol 5.1 (2007): e8.

        Note: This method does not give exactly the same output as the R method, the reason is that StandardScaler uses ddof=0 whereas R scale
        uses ddof=1. According to scikit-learn docs: "the choice of ddof is unlikely to affect model performance."

        :param matrix: (transformed) correlation matrix

        :returns: clr-adjusted weighed adjacency matrix
        """
        # faith 2007: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0050008
        # equivalent to computeCLRv2 function in R
        zc = matrix.astype(float)

        assert PCLRC.__is_symmetric(zc)

        np.fill_diagonal(zc, np.nan)

        scale(zc,
              axis=0,  # standardize each feature
              with_mean=True,  # center
              with_std=True,  # scale to unit variance
              copy=False  # inplace=True
              )

        zscore(zc, nan_policy='omit')

        zc = zc.clip(0)  # negative values are set to zero

        zr = zc.transpose()

        clr_z = np.sqrt(np.power(zr, 2) + np.power(zc, 2))

        np.fill_diagonal(clr_z, 0)

        return clr_z

    @staticmethod
    def __resample_clr(dataset: pd.DataFrame, method: str, n_iter: int, frac: float, rank_thr: float, show_process: bool) -> np.ndarray:
        """performs the PCLRC to infer probabilistic associations from a data matrix

        :param dataset: numeric matrix with the objects measurements. objects in rows, samples in cols.
        :param n_iter: number of iterations, default: 1000
        :param frac: fraction of the samples to be considered at each iteration, default: 0.75
        :param rank_thr: fraction of the predicted interactions to be kept at each iteration, default: 0.3
        :param method: method of correlation: ‘pearson’ (default), ‘kendall’, ‘spearman’ or 'bicor'
        :param show_process: print how many iterations have been calculated

        :returns: Weighed adjacency matrix of the network of interactions between the objects computed using the PCLRC algorithm
        """
        # equivalent to getPCLRC function in R
        n_samples = len(dataset.columns)

        n_valid_iterations = 0  # number of valid iterations (no NAs generated)

        # this table will store the number of times an interaction was selected
        table = np.zeros(shape=(n_samples, n_samples))

        for i in range(n_iter):
            if show_process:
                print(f'CLR iteration {i + 1} / {n_iter}', end='\r')

            subsample = dataset.sample(frac=frac, axis=0)

            # calculate correlation matrix
            if method == 'bicor':
                correlation_matrix = PCLRC.bicor(subsample.to_numpy())
            else:
                correlation_matrix = subsample.corr(method=method).to_numpy()

            # square
            correlation_matrix = np.square(correlation_matrix)

            # calculate CLR
            adj_subset = PCLRC.clr(correlation_matrix)

            if np.isnan(adj_subset).any():  # valid iteration
                logging.warning('Invalid iteration encountered: CLR output contained nan values.')
            else:
                # extract highest links
                out = PCLRC.__apply_quantile_threshold(matrix=adj_subset, rank_thr=rank_thr)
                # collect output
                table = table + out
                n_valid_iterations = n_valid_iterations + 1

        if show_process:
            print(f'{n_valid_iterations} out of {n_iter} iterations were valid.')

        return table / n_valid_iterations

    @staticmethod
    def __apply_quantile_threshold(matrix: np.ndarray, rank_thr: float = 0.3, verbose: bool = False) -> np.ndarray:
        """Obtains the links with the highest weight from an weighted adjacency matrix

        :param matrix: a weighted adjacency matrix (symmetric matrix)
        :param rank_thr: fraction of interactions to keep, default: 0.3
        :param verbose: print debugging messages

        :returns: binary matrix (0/1) with only the higher links having non null values
        """
        # equivalent to getHighestLinks function in R
        if np.nanmax(matrix) == 0:
            th = 0.01
            if verbose:
                print("null clr matrix \n")
        else:  # get the threshold
            if not rank_thr:
                th = np.nanmin(matrix[matrix > 0])
            else:
                # matrix[upper.tri(matrix)]
                th = np.nanquantile(matrix[np.triu_indices_from(matrix, k=1)], q=1 - rank_thr)
                if th == 0:
                    if verbose:
                        print("threshold too low,  min of the (non-null) matrix chosen instead\n")
                    th = min(matrix[matrix > 0])

        # set those below threshold to 0, rest to 1
        net = np.where(matrix < th, 0, 1)
        return net

    @staticmethod
    def bicor(matrix: np.ndarray) -> np.ndarray:
        """Calculate biweight midcorrelation

        It is median-based, rather than mean-based, thus is less sensitive to outliers, and can be a robust alternative to other similarity
        metrics, such as Pearson correlation or mutual information.

        Code taken from https://stackoverflow.com/a/61219867/6247327

        :param matrix: dataset
        :return: correlation matrix
        """
        n, m = matrix.shape
        matrix = matrix - np.median(matrix, axis=0, keepdims=True)
        v = 1 - (matrix / (9 * np.median(np.abs(matrix), axis=0, keepdims=True))) ** 2
        matrix = matrix * v ** 2 * (v > 0)
        norms = np.sqrt(np.sum(matrix ** 2, axis=0))
        return np.einsum('mi,mj->ij', matrix, matrix) / norms[:, None] / norms[None, :]

    @staticmethod
    def apply_threshold(matrix: np.ndarray, threshold: float = 0, is_positive: bool = True):
        """Impose a threshold on a weighed adjacency matrix to get connectivity matrix

        :param matrix: a weighted adjacency matrix (symmetric numeric matrix)
        :param threshold: numeric threshold to impose, default: 0
        :param is_positive: if the matrix is positive define. If not the abs value is considered, default: True
        :returns: binary (0/1) matrix
        """
        # equivalent to getAdjByThreshold function in R

        if is_positive:
            if np.any(matrix < 0):
                logging.warning("There should be no negative elements in matrix!")
            # set negative correlations and positive correlations below threshold to zero
            return np.where(matrix < threshold, 0, 1)
        else:
            # no differentiation between positive and negative correlations
            matrix = np.abs(matrix)
            return np.where(matrix < threshold, 0, 1)

    @staticmethod
    def __is_symmetric(a, tolerance=1e-8):
        return np.all(np.abs(a - a.T) < tolerance)


def test(test_dataset: str, expected_output: str):
    # ensure it gives the same output as R method
    test_dataset = pd.read_csv(test_dataset, index_col='Unnamed: 0')
    # 200 measurements of 20  metabolites
    expected_output = pd.read_csv(expected_output, index_col='Unnamed: 0')

    n_trials = 10
    successes = []
    for i in range(n_trials):
        _tmp = PCLRC.pclrc(
            test_dataset,
            method='pearson',
            n_iter=100,
            frac=0.75,
            rank_thr=0.3,
            threshold=0.95
        )

        differences = _tmp != expected_output
        differences = np.sum(differences.to_numpy())

        successes.append(bool(differences))

    print(f'{sum(successes)} out of {n_trials} gave the same result')
    assert sum(successes) > 0, 'the two algorithms should usually produce identical results'


if __name__ == '__main__':
    import os

    dir = os.path.dirname(__file__)
    test(test_dataset=f'{dir}/r_source/test_dataset.csv', expected_output=f'{dir}/r_source/pclr_output.csv')
