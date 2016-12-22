# MNIST dataset functions
load_mnist <- function()
{
    load_image_file <- function(filename)
    {
        ret = list();
        f = file(filename,'rb');
        readBin(f,'integer',n=1,size=4,endian='big');
        ret$n = readBin(f,'integer',n=1,size=4,endian='big');
        nrow = readBin(f,'integer',n=1,size=4,endian='big');
        ncol = readBin(f,'integer',n=1,size=4,endian='big');
        x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F);
        ret$x = matrix(x, ncol=nrow*ncol, byrow=T);
        close(f);
        ret;
    }
    load_label_file <- function(filename)
    {
        f = file(filename,'rb');
        readBin(f,'integer',n=1,size=4,endian='big');
        n = readBin(f,'integer',n=1,size=4,endian='big');
        y = readBin(f,'integer',n=n,size=1,signed=F);
        close(f);
        y;
    }
    train <<- load_image_file('./datasets/train-images.idx3-ubyte');
    test <<- load_image_file('./datasets/t10k-images.idx3-ubyte');

    train$y <<- load_label_file('./datasets/train-labels.idx1-ubyte');
    test$y <<- load_label_file('./datasets/t10k-labels.idx1-ubyte');
}

show_digit <- function(arr784, col=gray(12:1/12), ...)
{
    image(matrix(arr784, nrow=28)[,28:1], col=col, ...);
}

# RBM Functions

visible_state_to_hidden_probabilities <- function(rbm_w, visible_state)
{
    1/(1 + exp(-rbm_w %*% visible_state));
}
 
hidden_state_to_visible_probabilities <- function(rbm_w, hidden_state)
{
    1/(1 + exp(-t(rbm_w) %*% hidden_state));
}
 
configuration_goodness <- function(rbm_w, visible_state, hidden_state)
{
    out <- 0;
    for (i in 1:dim(visible_state)[2])
    {
        out <- out + t(hidden_state[,i]) %*% rbm_w %*% visible_state[,i];
    }
    out/dim(visible_state)[2];
}
 
configuration_goodness_gradient <- function(visible_state, hidden_state)
{
    hidden_state %*% t(visible_state)/dim(visible_state)[2];
}
 
sample_bernoulli <- function(mat)
{
    dims <- dim(mat);
    matrix(rbinom(n = prod(dims), size = 1, prob = c(mat)), dims[1], dims[2]);
}

cd1 <- function(rbm_w, visible_data)
{
    visible_data <- sample_bernoulli(visible_data);
    
    H0 <- sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, visible_data));
    vh0 <- configuration_goodness_gradient(visible_data, H0);
    V1 <- sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, H0));
    H1 <- visible_state_to_hidden_probabilities(rbm_w, V1);

    vh1 <- configuration_goodness_gradient(V1, H1);
    vh0 - vh1;
}

rbm <- function(num_hidden, training_data, learning_rate, n_iterations, mini_batch_size=100, momentum=0.9, quiet=FALSE)
{
    n <- dim(training_data)[2];
    p <- dim(training_data)[1];
    if (n %% mini_batch_size != 0) stop("the number of test cases must be divisable by the mini_batch_size")
            
    model <- (matrix(runif(num_hidden*p),num_hidden,p) * 2 - 1) * 0.1;
    momentum_speed <- matrix(0, num_hidden, p);

    start_of_next_mini_batch <- 1;
    for (iteration_number in 1:n_iterations)
    {
        if (!quiet) cat("Iter", iteration_number, "\n");

        mini_batch <- training_data[, start_of_next_mini_batch:(start_of_next_mini_batch + mini_batch_size - 1)];
        start_of_next_mini_batch <- (start_of_next_mini_batch + mini_batch_size) %% n;
        gradient <- cd1(model, mini_batch);
        momentum_speed <- momentum * momentum_speed + gradient;
        model <- model + momentum_speed * learning_rate;
    }
    return(model);
}
