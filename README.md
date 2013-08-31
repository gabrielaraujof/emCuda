EMCuda
======

Parallel implementation of Expectation-Maximization algorithm for estimating Gaussian Mixture Models (GMMs) using the NVIDIA CUDA plataform.

EM Algorithm for GMM Estimation
-------------------------------

> For a normal mixture model with pre-specified model order **_g_**, the parameters to be estimated are the mixture component probabilities and the mixture component parameters. The most prevalent technique for optimising the parameters
given a set of n independent observations (the training data) is an iterative procedure known as the Expectation Maximisation (EM) algorithm. [(Webb, 2011)] [3]

> In statistics, an expectation–maximization (EM) algorithm is an iterative method for finding maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models, where the model depends on unobserved latent variables. The EM iteration alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step. These parameter-estimates are then used to determine the distribution of the latent variables in the next E step. [Wikipedia] [2]

Nvidia CUDA Plataform
---------------------

> CUDA is a parallel computing platform and programming model that makes using a GPU for general purpose computing simple and elegant. The developer still programs in the familiar C, C++, Fortran, or an ever expanding list of supported languages, and incorporates extensions of these languages in the form of a few basic keywords.
These keywords let the developer express massive amounts of parallelism and direct the compiler to the portion of the application that maps to the GPU. [NVIDIA Blog] [1]

TO DO List
----------

Please, check the [TO DO file] (TODO.md) for a list of suggested features and improvements for this project.

License
-------
The project is licensed under the General Public License **GPL v.3**. For the complete terms, please see the [License file] (LICENSE).


[1]: http://blogs.nvidia.com/2012/09/what-is-cuda-2/

[2]: http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm

[3]: http://books.google.com/books?isbn=1119952964
