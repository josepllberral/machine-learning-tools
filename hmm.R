###############################################################################
# HIDDEN MARKOV MODELS in R                                                   #
###############################################################################

## @author Josep Ll. Berral (Barcelona Supercomputing Center)

## @date 16 May 2018

## References:
## * Approach inspured on Chi-En Wu also Vince Buffalo implementations
##   https://github.com/jason2506/PythonHMM
##   https://github.com/vsbuffalo/hmmr

################################################################################
# HMM FUNCTIONS                                                                #
################################################################################

## Constructor method for HMMs
#  param states       : vector of states
#  param symbols      : vector of symbols
#  param states_dict  : vector <1:num_states>. Names are labels.
#  param symbols_dict : vector <1:num_symbols>. Names are labels.
#  param start_prob   : vector <num_states>. Default = NULL (equiprobability)
#  param trans_prob   : matrix [num_states x num_stats]. Default = NULL
#                       (equiprobability per stats|row)
#  param emit_prob    : matrix [num_states x num_symbols]. Default = NULL
#                       (equiprobability per symbols|row)
create_hmm <- function(states, symbols, states_dict, symbols_dict,
	start_prob = NULL, trans_prob = NULL, emit_prob = NULL)
{
	nstats <- length(states);
	nsymbs <- length(symbols);
	
	if (is.null(start_prob)) start_prob <- rep(1.0 / nstats, nstats);
	if (is.null(trans_prob)) trans_prob <- matrix(1.0 / nstats, nrow = nstats, ncol = nstats);
	if (is.null(emit_prob)) emit_prob <- matrix(1.0 / nsymbs, nrow = nstats, ncol = nsymbs);
	
	rownames(trans_prob) <- names(states_dict);
	colnames(trans_prob) <- names(states_dict);
	rownames(emit_prob) <- names(states_dict);
	colnames(emit_prob) <- names(symbols_dict);
	
	list(states = states, symbols = symbols, SP = start_prob,
		TP = trans_prob, EP = emit_prob, states_dict = states_dict,
		symbols_dict = symbols_dict, nstates = nstats,
		nsymbols = nsymbs);
}

## Forwards a SINGLE SYMBOL SEQUENCE through the HMM
#  param hmm           : the HMM to be used
#  param symb_sequence : vector, single sequence of symbols
#  returns             : matrix [Seq_Length x States]
forward_hmm <- function(hmm, symb_sequence)
{
	slen <- length(symb_sequence);
	if (slen == 0) return (NULL);
	
	alpha <- matrix(0, nrow = slen, ncol = hmm$nstates);
	alpha[1,] <- hmm$SP * hmm$EP[, symb_sequence[1]];

	for (i in 2:slen)
		for (state_to in 1:hmm$nstates)
			alpha[i, state_to] <- sum(alpha[i-1, ] * hmm$TP[, state_to]) * hmm$EP[state_to, symb_sequence[i]];

	dimnames(alpha) <- list(1:slen, rownames(hmm$EP));
	alpha;
}

## Backwards a SINGLE SYMBOL SEQUENCE through the HMM
#  param hmm           : the HMM to be used
#  param symb_sequence : vector, single sequence of symbols
#  returns             : matrix [Seq_Length x States]
backward_hmm <- function(hmm, symb_sequence)
{
	slen <- length(symb_sequence);
	if (slen == 0) return (NULL);
	
	beta <- matrix(0, nrow = slen, ncol = hmm$nstates);
	beta[slen, ] <- rep(1, hmm$nstates);
	
	for (i in (slen - 1):1)
		for (state_from in 1:hmm$nstates)
			beta[i, state_from] <- sum(beta[i+1,] * hmm$TP[state_from,] * hmm$EP[, symb_sequence[i+1]]);

	dimnames(beta) <- list(1:slen, rownames(hmm$EP));
	beta;
}

## Create initial Markov Chain from sample sequences
#  param sequences : list with two elements:
#       - stat_seq : list { vector of state sequences }
#       - symb_seq : list { vector of symbol sequences }
initialize_markov <- function(sequences)
{
	stat_seq <- sequences$states;
	symb_seq <- sequences$symbols;
	
	nseq <- length(stat_seq);
	states <- unique(unlist(stat_seq));
	symbols <- unique(unlist(symb_seq));

	# Create dictionary of values -> indices; Convert sequences to indices
	st_dict <- seq_along(states); names(st_dict) <- states;
	sb_dict <- seq_along(symbols); names(sb_dict) <- symbols;
	st_seq2 <- sapply(stat_seq, function(x) sapply(x, function(y) st_dict[[y]]));
	sb_seq2 <- sapply(symb_seq, function(x) sapply(x, function(y) sb_dict[[y]]));
	
	# Create Markov Chain matrices with probabilities
	ss_count <- rep(0, length(states));
	tp_count <- matrix(0, nrow = length(states), ncol = length(states));
	ep_count <- matrix(0, nrow = length(states), ncol = length(symbols));
	for (i in 1:nseq)
	{
		s <- st_seq2[[i]];
		b <- sb_seq2[[i]];
		ss_count[s[1]] <- ss_count[s[1]] + 1;
		for (j in 1:(length(s)-1))
		{
			tp_count[s[j], s[j + 1]] <- tp_count[s[j], s[j + 1]] + 1;
			ep_count[s[j], b[j]] <- ep_count[s[j], b[j]] + 1;
		}
	}
	ss_count <- ss_count / nseq;
	tp_count <- tp_count / rowSums(tp_count);
	ep_count <- ep_count / rowSums(ep_count);

	# Return dictionaries and matrices
	list(states = states, symbols = symbols, start_probs = ss_count,
		trans_probs = tp_count,	emit_probs = ep_count,
		states_dict = st_dict, symbols_dict = sb_dict);
}

## Use the forward algorithm to evaluate a given sequence of symbols
#  param hmm           : the HMM to be used
#  param symb_sequence : vector, single sequence of symbols
#  returns             : vector, probabilities per symbol
evaluate_hmm <- function(hmm, symb_sequence)
{
	slen <- length(symb_sequence);
        if (slen > 0)
        {
		alpha <- forward_hmm(hmm, hmm$symbols_dict[symb_sequence]);
		sum(alpha[slen,]);
	} else 0;
}

################################################################################
# HOW TO TRAIN YOUR HMM                                                        #
################################################################################

## Function to train the HMM (Interface for Supervised/Unsupervised)
#  param sequences  : list { states  = list{ state sequences },
#                           symbols = list{ symbol sequences } }
#  param supervised : whether the sequences contain symbols. Default = TRUE
#  param ...        : other parameters for training
#  returns          : trained HMM
train_hmm <- train.hmm <- function(sequences, supervised = TRUE, ...)
{
	f <- if (supervised) supervised_hmm else unsupervised_hmm;
	f(sequences, ...);
}

## Function to train the HMM Unsupervised
#  TODO -
unsupervised_hmm <- function(sequences, delta = 1e-4, smoothing = 0)
{										# TODO - Complete this functionality
	# ...
}

## Function to train the HMM Supervised. Supervised training function for HMM
#  using the Expectation-Maximization algorithm.
#  param sequences : list { states  = list{ state sequences },
#                           symbols = list{ symbol sequences } }
#  param delta     : specifies that the learning algorithm will stop when the
#                    difference of the log-likelihood between two consecutive
#                    iterations is less than delta. default = 0.0001
#  param smoothing : argument is used to avoid zero probability. default = 0
#  param rand_seed : random seed. Default = 1234
#  return          : a trained HMM
supervised_hmm <- function(sequences, delta = 1e-4, smoothing = 0,
	rand_seed = 1234)
{
	set.seed(rand_seed);
	start_time <- Sys.time();
	
	# Input checks for coherence in sequences
	if (length(sequences$states) != length(sequences$symbols))
	{
		message("Error: Different number of state and symbol sequences");
		return(NULL);
	}
	if (!all(sapply(sequences$symbols, length) == sapply(sequences$states, length)))
	{
		message("Error: Different lenght on a state/symbol sequences");
		return(NULL);
	}

	# Obtain initial Markov chain from sequences
	init_model <- initialize_markov(sequences);
	
	# Create the HMM object with initial matrices
	hmm <- create_hmm(init_model$states, init_model$symbols,
		init_model$states_dict, init_model$symbols_dict,
		init_model$start_prob, init_model$trans_prob,
		init_model$emit_prob);
	
	# Iterate to fit the HMM into training data
	old_likelihood <- mean(sapply(sequences$symbols, function(x) log(evaluate_hmm(hmm, x))));
	while (TRUE)
	{
		new_likelihood <- 0;
		for (symb in sequences$symbols)
		{
			hmm <- learn_hmm(hmm, symb, smoothing);
			new_likelihood <- new_likelihood + log(evaluate_hmm(hmm, symb));
		}
		new_likelihood <- new_likelihood / length(sequences$states);

		if (abs(new_likelihood - old_likelihood) < delta) break;
		old_likelihood <- new_likelihood;
	}

	end_time <- Sys.time();
	print(paste('Training took', (end_time - start_time), sep = " "));

	# Return the trained HMM
	class(hmm) <- c("hmm", class(hmm));
	hmm;
}

## Function to find the best state transition and emission probabilities, given
#  a sequence of symbols, updating the HMM.
#  param hmm            : the HMM to be trained
#  param symb_sequence  : vector of symbols
#  param smoothing      : additive smoothing. Default = 0
#  return               : the updated HMM
learn_hmm <- function(hmm, symb_sequence, smoothing = 0)
{
	slen <- length(symb_sequence);
	seq_aux <- hmm$symbols_dict[symb_sequence];
	
	alpha <- forward.hmm(hmm, seq_aux);
	beta  <- backward.hmm(hmm, seq_aux);
	gamma <- t(apply(alpha * beta, 1, function(x) { s <- sum(x); if (s > 0) x / s else x }));

	maux <- array(0, c(slen - 1, hmm$nstates, hmm$nstates));
	for (index in 1:(slen - 1))
	{
		for (sfrom in 1:hmm$nstates)
			for (sto in 1:hmm$nstates)
				maux[index, sfrom, sto] <- alpha[index, sfrom] * beta[index + 1, sto] * hmm$TP[sfrom, sto] * hmm$EP[sto, seq_aux[index + 1]];
		s <- sum(maux[index,,]);
		if (s > 0) maux[index,,] <- maux[index,,] / s;
	}

	# Update Start, Transition and Emission Probability vector and matrices
	for (state in 1:hmm$nstates)
	{
		hmm$SP[state] <- smoothing + gamma[1, state] / (1 + hmm$nstates * smoothing);

		gamma_sum <- sum(gamma[1:(slen - 1), state]);
		hmm$TP[state, ] <- if (gamma_sum > 0)
		{
			denominator <- gamma_sum + hmm$nstates * smoothing;
			sapply(1:hmm$nstates, function(s) smoothing + sum(maux[1:(slen - 1), state, s]) / denominator);			
		} else rep(0, hmm$nstates);
		
		gamma_sum <- gamma_sum + gamma[slen, state];

		emit_sum <- rep(0, hmm$nsymbols);
		for (index in 1:slen)
			emit_sum[seq_aux[index]] <- emit_sum[seq_aux[index]] + gamma[index, state];

		hmm$EP[state, ] <- if (gamma_sum > 0)
		{
			denominator <- gamma_sum + hmm$nsymbols * smoothing;
			(smoothing + emit_sum) / denominator;
		} else rep(0, hmm$nsymbols);
	}
	
	hmm;
}

################################################################################
# PREDICTING VALUES                                                            #
################################################################################

## Passes list of symbol sequences through the HMM, forward and backward
#  param hmm     : the HMM to be used
#  param newdata : list { vector of symbols }
#  return        : list { probabilities : list of matrices with probabilities, 
#                         max_states : list of vectors with max_states }
predict_hmm <- predict.hmm <- function(hmm, newdata)
{										# TODO - Check that newdata has only symbols in hmm dictionary
	# Get probabilities for each sequence
	probs <- lapply(newdata, function(i)
	{
		seq_aux <- hmm$symbols_dict[i];
		alpha <- forward_hmm(hmm, seq_aux);
		beta <- backward_hmm(hmm, seq_aux);
		alpha_beta <- alpha * beta;
		sweep(alpha_beta, 1, rowSums(alpha_beta), "/");
	});
	
	# Translate to states with max probability per input value
	states <- lapply(probs, function(x)
		apply(x, 1, function(y) 
			names(hmm$states_dict)[which(y == max(y))]
		)
	);
	
	list(probabilities = probs, max_states = states);
}

## Function to simulate observed sequences from a trained HMM
#  param hmm       : the HMM to be used
#  param n         : the length of the new sequence
#  param rand_seed : random seed. Default = 1234
simulate_hmm <- simulate.hmm <- function(hmm, n, rand_seed = 1234)
{
	set.seed(rand_seed);

	new_sts <- rep(0, n);
	new_seq <- rep(0, n);

	new_sts[1] <- sample(1:hmm$nstates, 1, prob = hmm$SP);
	new_seq[1] <- sample(1:hmm$nsymbols, 1, hmm$SP[new_sts[1]]);

	for (i in 2:n)
	{
		new_sts[i] <- sample(1:hmm$nstates,  1, prob = hmm$TP[new_sts[i-1],]);
		new_seq[i] <- sample(1:hmm$nsymbols, 1, prob = hmm$EP[new_sts[i-1],]);
	}

	names(new_sts) <- names(hmm$states_dict)[new_sts];
	names(new_seq) <- names(hmm$symbols_dict)[new_seq];
	
	list(symbols = new_seq, states = new_sts);
}

## Function to "decode" a sequence of symbols. Implementation of the Viterbi
#  algorithm.
#  param hmm           : the HMM to be used
#  param symb_sequence : the sequence to be decoded
decode_hmm <- decode.hmm <- function(hmm, symb_sequence)
{										# TODO - Check that newdata has only symbols in hmm dictionary
	slen <- length(symb_sequence);
	if (slen == 0) return (NULL);

	seq_aux <- hmm$symbols_dict[symb_sequence];
	delta <- hmm$SP * hmm$EP[, seq_aux[1]];

	pre <- matrix(0, nrow = slen - 1, ncol = hmm$nstates);
	for (index in 2:slen)
	{
		probs <- delta * hmm$TP;
		pre[index - 1,] <- apply(probs, 1, function(x) which(x == max(x))[1]);
#		pre[index - 1,] <- max.col(probs); # function max.col is stochastic
		delta <- apply(probs, 1, max) * hmm$EP[, seq_aux[index]];
	}

	max_state <- as.numeric(which(delta == max(delta))[1]);
	if (length(max_state) > 0)
	{
		result <- max_state;
		for (index in slen:2)
		{
			max_state <- pre[index - 1, max_state];
			result <- c(result, max_state);
		}
		hmm$states_dict[result];
	} else NULL;
}

################################################################################
# EXAMPLE                                                                      #
################################################################################

## Main Function - Program Entry Point
main <- function()
{
	# Dummy Example
	ta <- tb <- list();
	ta[[1]] <- c("A","C","B","A","C","D"); tb[[1]] <- c("Z","X","Z","Y","Y","Z");
	ta[[2]] <- c("C","A","A","B","D");     tb[[2]] <- c("X","X","Z","Y","Z");
	ta[[3]] <- c("B","E","D","D");         tb[[3]] <- c("X","Y","Z","Y")
	s_aux <- list(states = ta, symbols = tb);
	
	hmm1 <- train_hmm(sequences = s_aux, supervised = TRUE);
	print(hmm1);

	tc <- list();
	tc[[1]] <- c("Z","X","X","Z","Y","Y");
	tc[[2]] <- c("X","Y","Z","Y","Y");
	tc[[3]] <- c("X","Z","Y","Y");
	tc[[4]] <- c("Y","Z","X","X","Z");

	s_pred <- predict_hmm(hmm1, tc);
	print(s_pred);

	s_decoded <- decode_hmm(hmm1, c("Z","Y","X","Z"));
	print(s_decoded);
	
	s_simulated <- simulate_hmm(hmm1, 10);
	print(s_simulated);
	
	# Real Data Example							# TODO - Create a real example
	# ...
}
