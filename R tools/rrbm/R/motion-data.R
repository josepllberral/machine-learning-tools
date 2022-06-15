#' Fragment of the "Style Translation for Human Motion" dataset
#'
#' Data containing fragments of the Motion Dataset, from Eugene Hsu, Kari Pulli
#' and Jovan Popovic. The dataset contains three motion sequences, used in
#' Graham Taylor's work for Constrained RBMs.
#'
#' @docType data
#'
#' @usage data(motionfrag)
#'
#' @format List containing
#' \itemize{
#'   \item batchdata : numeric matrix 3826 x 49. Contains 3826 samples (with 49 features) from 3 sequences of captured motion. Data is normalized (see data_mean and data_std for original values reconstruction).
#'   \item seqlen : integer vector indicating the length of each sequence in batchdata.
#'   \item data_mean : numeric vector with the mean value for each feature.
#'   \item data_std : numeric vector with the standard deviation for each feature.
#' }
#'
#' @keywords datasets
#'
#' @references Eugene Hsu et al. (2005)
#' (\href{http://people.csail.mit.edu/ehsu/work/sig05stf/stf-sig2005.pdf}{Style Translation for Human Motion})
#' @references Graham Taylor et al. (2006)
#' (\href{http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/gwtaylor_nips.pdf}{Modeling Human Motion Using Binary Latent Variables})
#'
#' @source \href{http://people.csail.mit.edu/ehsu/work/sig05stf/}{Style Translation for Human Motion}
#' @source \href{http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat}{Original selected dataset file}
#'
#' @examples
#' data(motionfrag)
#'
#' \donttest{head(motionfrag$batchdata)}
#'
#' batchdata <- motionfrag$batchdata;
#' seqlen <- motionfrag$seqlen;
#'
#' idx <- 1;
#' first.sequence <- batchdata[idx:(idx + seqlen[1]),];
#' idx <- idx + seqlen[1];
#' second.sequence <- batchdata[idx:(idx + seqlen[2]),];
#' idx <- idx + seqlen[2];
#' third.sequence <- batchdata[idx:(idx + seqlen[3],];
#'
#' data_mean <- motionfrag$data_mean;
#' data_std <- motionfrag$data_std;
#' original.data <- t((t(batchdata) * data_std) + data_mean);
"motionfrag"
