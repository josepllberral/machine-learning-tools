################################################################################
# HIDDEN MARKOV MODELS in R                                                    #
################################################################################

## @author Josep Ll. Berral (Barcelona Supercomputing Center)

## @date 12 June 2018

## References:
## * Approach inspired on Chi-En Wu and Vince Buffalo implementations
##   https://github.com/jason2506/PythonHMM
##   https://github.com/vsbuffalo/hmmr
##   http://modelai.gettysburg.edu/2017/hmm/
##
## Lectures:
##   http://mlg.eng.cam.ac.uk/zoubin/course05/lect4time.pdf
##   http://karlstratos.com/notes/em_hmm_formulation.pdf

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
#  alpha = likelihood of being in state i at time t given symb_sequence
#  alpha(1,i)   <- SP[i] * EP[i,symb_1]
#  alpha(t+1,k) <- SUM_i(alpha(t,i) * TP[i,k]) * EP[k,symb_t+1]
#
#  param hmm           : the HMM to be used
#  param symb_sequence : vector, single sequence of symbols
#  returns alpha       : matrix [Seq_Length x States]
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
#  beta = probability of having symb_sequence when being in state i at time t
#  beta(L,j) <- 1
#  beta(t,j)   <- SUM_k(TP[k,j] * beta(t+1,k) * EP[k,symb_t+1])
#
#  param hmm           : the HMM to be used
#  param symb_sequence : vector, single sequence of symbols
#  returns beta        : matrix [Seq_Length x States]
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
#  param symb_sequences : list { vector of symbol sequences }
#  param stat_sequences : list { vector of state sequences }
initialize_markov <- function(symb_sequences, stat_sequences)
{
	stat_seq <- stat_sequences;
	symb_seq <- symb_sequences;
	
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

## Function to train the Hidden Markov Model.
#  - Supervised: Given a corpus (list of observations with known state
#    sequences), train an HMM using the Expectation-Maximization algorithm.
#  - Unsupervised: Given a corpus (list of observations with unknown state,
#    sequences), train an HMM using initial clustering, and the Baum-Welch
#    Expectation-Maximization algorithm maximizing the likelihood.
#  param symb_sequences : list { symbol sequences }
#  param stat_sequences : list { state sequences }. If NULL (default),
#                         unsupervised learning is performed. Default = NULL
#  param num_states     : in case of unsupervised learning, num_states is the
#                         number of states to infer. Default = 3
#  param delta          : specifies that the learning algorithm will stop when
#                         the difference of the log-likelihood between two
#                         consecutive iterations is less than delta.
#                         Default = 0.0001
#  param smoothing      : argument is used to avoid zero probability.
#                         default = 0
#  param rand_seed      : random seed. Default = 1234
#  return               : a trained HMM
train_hmm <- train.hmm <- function(symb_sequences, stat_sequences = NULL,
	num_states = 3, delta = 1e-4, smoothing = 0, rand_seed = 1234)
{
	set.seed(rand_seed);
	start_time <- Sys.time();
	
	if (!is.null(stat_sequences))
	{
		# Input checks for coherence in sequences
		if (length(stat_sequences) != length(symb_sequences))
		{
			message("Error: Different number of state and symbol sequences");
			return(NULL);
		}
		if (!all(sapply(symb_sequences, length) == sapply(stat_sequences, length)))
		{
			message("Error: Different lenght on a state/symbol sequences");
			return(NULL);
		}
		
	} else {
		if (is.na(num_states) || !is.numeric(num_states))
		{
			message("Error: Incorrect number of inferred states");
			return(NULL);
		}
		
		# Initial Dependent Mixture Model
		stat_sequences <- NULL; 					# TODO - Gausian Mixture/k-means/...?
	}
	symb_sequences <- lapply(symb_sequences, as.character);
	stat_sequences <- lapply(stat_sequences, as.character);
	
	# Obtain initial Markov chain from sequences
	init_model <- initialize_markov(symb_sequences, stat_sequences);
	
	# Create the HMM object with initial matrices
	hmm <- create_hmm(init_model$states, init_model$symbols,
		init_model$states_dict, init_model$symbols_dict,
		init_model$start_prob, init_model$trans_prob,
		init_model$emit_prob);	
	
	# Iterate to fit the HMM into training data
	old_likelihood <- mean(sapply(symb_sequences, function(x) log(evaluate_hmm(hmm, x))));
	while (TRUE)
	{
		new_likelihood <- 0;
		for (symb in symb_sequences)
		{
			hmm <- learn_unsup_hmm(hmm, symb, smoothing);
			new_likelihood <- new_likelihood + log(evaluate_hmm(hmm, symb));
		}
		new_likelihood <- new_likelihood / length(symb_sequences);

		if (abs(new_likelihood - old_likelihood) < delta) break;
		old_likelihood <- new_likelihood;
	}
	hmm$likelyhood <- old_likelihood;
	
	end_time <- Sys.time();
	print(paste('Training took', (end_time - start_time), sep = " "));

	# Return the trained HMM
	class(hmm) <- c("hmm", class(hmm));
	hmm;
	
}

## Function to find the best state transition and emission probabilities, given
#  a sequence of symbols, updating the HMM {start_probs, transition_matrix,
#  emision_matrix}.
#  param hmm            : the HMM to be trained
#  param symb_sequence  : vector of symbols
#  param smoothing      : additive smoothing. Default = 0
#  return               : the updated HMM
learn_hmm <- function(hmm, symb_sequence, smoothing = 0)
{
	slen <- length(symb_sequence);
	seq_aux <- hmm$symbols_dict[symb_sequence];
	
	# E-Step
	# alpha = likelihood of being in state i at time t given symbol_sequence_1:t
	# beta  = likelihood of producing symbol_sequence_t+1:L from state i at time t
	# gamma = probability of being in state i at time t given the symbol_sequence
	#
	# 	P^(symb_1:t | state_t = i) = alpha(t,i)
	# 	P^(symb_t+1:L | state_t = i) = beta(t,i)
	# 	P^(symb_1:L, state_t = i) = P^(symb_1:t, symb_t+1:L, state_t = i) = alpha(t,i) * beta(t,i) = gamma(t,i)
	#
	#	alpha(1,i)   <- SP[i] * EP[i,symb_1]
	#	alpha(t+1,i) <- SUM_k=[state](alpha(t,k) * TP[k,i]) * EP[i,symb_t+1]
	#	beta (t,i)   <- SUM_k=[state](TP[k,i] * beta(t+1,k) * EP[i,symb_t+1])
	# 	gamma(t,i)   <- alpha(t,i) * beta(t,i) / RS

	alpha <- forward_hmm(hmm, seq_aux);
	beta  <- backward_hmm(hmm, seq_aux);
	gamma <- t(apply(alpha * beta, 1, function(x) { s <- sum(x); if (s > 0) x / s else x }));

	# M-Step
	# M  = probability of moving from state i to j at time t and then emitting symbol_t+1
	# xi = expected number of transitions from state i to j being at time t
	#
	# 	P^(state_t+1 = j | state_t = i) * P^(symb_t+1 | state_t+1 = j) = TP[i,j] * EP[j,symb_t+1] = M_t[i,j]
	# 	P^(symb_1:L, state_t = i, state_t+1 = j) = 
	#	= P^(symb_1:t, symb_t+1:L, state_t = i, state_t+1 = j) =
	#	= P^(symb_1:t | state_t = i) * M_t[i,j] * P^(symb_t+2:L | state_t+1 = j) =
	#	= alpha(t,i) * M_t[i,j] * beta(t+1,j) = xi(t,i,j)
	#
	#	xi(t,i,j) <- alpha(t,i) * M_t[i,j] * beta(t+1,j) / RS
	
	xi <- array(0, c(slen - 1, hmm$nstates, hmm$nstates));
	for (index in 1:(slen - 1))
	{
		for (sfrom in 1:hmm$nstates)
			for (sto in 1:hmm$nstates)
				xi[index, sfrom, sto] <- alpha[index, sfrom] * hmm$TP[sfrom, sto] * hmm$EP[sto, seq_aux[index + 1]] * beta[index + 1, sto];
		xi[index,,] <- xi[index,,] / { s <- sum(xi[index,,]); if (s > 0) s else 1 };
	}

	# Expected Counts to Update the Start, Transition and Emission Probs.
	#
	# As this happens for each sequence, one by one:
	# -> [seq] = symb_sequence; #seq = 1; P^(seq = s) = 1
	#
	# P^(state_1 = i | seq = s) = P^(state_1 = i, seq = s) / P^(seq = s) = alpha(1,i) * beta(1,i) / P^(seq = s) = gamma(1,i) / P^(seq = s)
	# P^(state_t = i, state_t+1 = j | seq = s) = P^(state_t = i, state_t+1 = j, seq = s) / P^(seq = s) = xi(t,i,j) / P^(seq = s)
	# P^(state_t = i, symb_t = x | seq = s) = P^(state_t = i, symb_t = x, seq = s) / P^(seq = s) = SUM_t=[symb_t=x] gamma(t,i) / P^(seq = s)
	#
	# C^(state_1 = i) = SUM_s=[seq] P^(state_1 = i | seq = s) = gamma(1,i)
	# C^(state = i, state' = j) = SUM_s=[seq] SUM_t=1:L-1 P^(state_t = i, state_t+1 = j | seq = s) = SUM_t=1:L-1 xi(t,i,j)
	# C^(state = i, symb = x) = SUM_s=[seq] SUM_t=1:L P^(state_t = i, symb_t = x | seq = s) = SUM_t=[symb_t=x] gamma(t,i)
	#
	# C^(state_1) = SUM_h=[state] C^(state_1 = h) = #seq					Number of starts
	# C^(state = i)* = SUM_s=[seq] C^(state = i, seq = s) = SUM_t=1:L-1 gamma(t,i)		Expected number of emissions from state i to another state
	# C^(state = i)** = SUM_h=[state] C^(state = i, state' = h) = SUM_t=1:L gamma(t,i)	Expected number of emissions from state i
	#
	# SP[i]   <- P^(state_1 = i) = C^(state_1 = i) / C^(state_1) = gamma(1,i) / #seq
	# TP[i,j] <- P^(state' = j | state = i) = C^(state = i, state' = j) / C^(state = i)* = SUM_t=1:L-1 xi(t,i,j) / SUM_t=1:L-1 gamma(t,i)
	# EP[i,x] <- P^(symb = x | state = i) = C^(state = i, symb = x) / C^(state = i)** = SUM_t=[symb_t=x] gamma(t,i) / SUM_t=1:L gamma(t,i)
	
	for (state in 1:hmm$nstates)
	{
		# SP[i] = gamma(1,i)
		hmm$SP[state] <- smoothing + gamma[1, state] / (1 + hmm$nstates * smoothing);

		# TP[i,j] = SUM_t=1:L-1 xi(t,i,j) / SUM_t=1:L-1 gamma(i,t)
		gamma_sum <- sum(gamma[1:(slen - 1), state]);
		hmm$TP[state, ] <- if (gamma_sum > 0)
		{
			denominator <- gamma_sum + hmm$nstates * smoothing;
			sapply(1:hmm$nstates, function(s) smoothing + sum(xi[1:(slen - 1), state, s]) / denominator);			
		} else rep(0, hmm$nstates);
		
		# EP[i,x] = SUM_t=[symb_t=x] gamma(t,i) / SUM_t=1:L gamma(t,i)
		emit_sum <- rep(0, hmm$nsymbols);
		for (index in 1:slen)
			emit_sum[seq_aux[index]] <- emit_sum[seq_aux[index]] + gamma[index, state];
		gamma_sum <- gamma_sum + gamma[slen, state];
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
	st <- sy <- list();
	st[[1]] <- c("A","C","B","A","C","D"); sy[[1]] <- c("Z","X","Z","Y","Y","Z");
	st[[2]] <- c("C","A","A","B","D");     sy[[2]] <- c("X","X","Z","Y","Z");
	st[[3]] <- c("B","E","D","D");         sy[[3]] <- c("X","Y","Z","Y")
	
	hmm1 <- train_hmm(symb_sequences = sy, stat_sequences = st);
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
