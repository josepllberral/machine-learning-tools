## R -d "valgrind --tool=memcheck --leak-check=full" --vanilla < vgtest-cnn.R > log-cnn.txt 2>&1

library(rcnn)

# CNN simple
train_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 1, 2, 3));
train_Y <- array(c(1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1), c(6, 2));
layers <- list(
    c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
    c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
    c('type' = "RELU", 'n_channels' = 4),
    c('type' = "FLAT", 'n_channels' = 4),
    c('type' = "LINE", 'n_visible' = 8, 'n_hidden' = 6, 'scale' = 0.1),
    #c('type' = "RBML", 'n_visible' = 8, 'n_hidden' = 6, 'scale' = 0.1, 'n_gibbs' = 3),
    c('type' = "LINE", 'n_visible' = 6, 'n_hidden' = 2, 'scale' = 0.1),
    c('type' = "SOFT", 'n_inputs' = 2)
);
cnn1 <- train.cnn(train_X, train_Y, layers, batch_size = 2);

test_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 1, 2, 3));
results <- predict(cnn1, test_X);

# CNN MNIST
data(mnist)

img_size <- c(28,28);

train <- mnist$train;
training_x <- array(train$x, c(nrow(train$x), 1, img_size)) / 255;
training_y <- binarization(train$y);

test <- mnist$test;
testing_x <- array(test$x, c(nrow(test$x), 1, img_size)) / 255;
testing_y <- binarization(test$y);

dataset <- training_x[1:1000,,,, drop=FALSE];
targets <- training_y[1:1000,, drop=FALSE];

newdata <- testing_x[1:1000,,,, drop=FALSE];

layers <- list(
    c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
    c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
    c('type' = "RELU", 'n_channels' = 4),
    c('type' = "CONV", 'n_channels' = 4, 'n_filters' = 16, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
    c('type' = "POOL", 'n_channels' = 16, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
    c('type' = "RELU", 'n_channels' = 16),
    c('type' = "FLAT", 'n_channels' = 16),
    c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 64, 'scale' = 0.1),
    c('type' = "LINE", 'n_visible' = 64, 'n_hidden' = 32, 'scale' = 0.1),
    #c('type' = "RBML", 'n_visible' = 64, 'n_hidden' = 32, 'scale' = 0.1, 'n_gibbs' = 4),
    c('type' = "RELV"),
    c('type' = "LINE", 'n_visible' = 32, 'n_hidden' = 10, 'scale' = 0.1),
    c('type' = "SOFT", 'n_inputs' = 10)
);

mnist_cnn <- train.cnn(dataset, targets, layers, batch_size = 10, training_epochs = 3, learning_rate = 1e-3, momentum = 0.8, rand_seed = 1234);

prediction <- predict(mnist_cnn, newdata);

mnist_cnn_up <- mnist_cnn
mnist_cnn_up <- train.cnn(dataset, targets, layers = NULL, batch_size = 10, training_epochs = 3, learning_rate = 1e-3, rand_seed = 1234, init_cnn = mnist_cnn_up);

rebuild <- pass_through.cnn(mnist_cnn, newdata);

# CNN MNIST WITH EVALUATOR
data(mnist)

img_size <- c(28,28);

train <- mnist$train;
training_x <- array(train$x, c(nrow(train$x), 1, img_size)) / 255;
training_y <- binarization(train$y);

test <- mnist$test;
testing_x <- array(test$x, c(nrow(test$x), 1, img_size)) / 255;
testing_y <- binarization(test$y);

dataset <- training_x[1:1000,,,, drop=FALSE];
targets <- training_y[1:1000,, drop=FALSE];

newdata <- testing_x[1:1000,,,, drop=FALSE];

layers <- list(
    c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
    c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
    c('type' = "RELU", 'n_channels' = 4),
    c('type' = "FLAT", 'n_channels' = 4),
    c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 10, 'scale' = 0.1),
    c('type' = "SOFT", 'n_inputs' = 10)
);
eval <- c('type' = "XENT");

mnist_cnn <- train.cnn(dataset, targets, layers, evaluator = eval, batch_size = 10, training_epochs = 3, learning_rate = 1e-3, momentum = 0.8, rand_seed = 1234);

prediction <- predict(mnist_cnn, newdata);

# DBN-CNN MNIST WITH EVALUATOR
data(mnist)

img_size <- c(28,28);

train <- mnist$train;
training_x <- array(train$x, c(nrow(train$x), 1, img_size)) / 255;
training_y <- binarization(train$y);

test <- mnist$test;
testing_x <- array(test$x, c(nrow(test$x), 1, img_size)) / 255;
testing_y <- binarization(test$y);

dataset <- training_x[1:1000,,,, drop=FALSE];
targets <- training_y[1:1000,, drop=FALSE];

newdata <- testing_x[1:1000,,,, drop=FALSE];

layers <- list(
    c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
    c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
    c('type' = "RELU", 'n_channels' = 4),
    c('type' = "FLAT", 'n_channels' = 4),
    c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 10, 'scale' = 0.1),
    c('type' = "SOFT", 'n_inputs' = 10)
);
eval <- c('type' = "RBML", 'n_visible' = 10, 'n_hidden' = 5, 'scale' = 0.1, 'n_gibbs' = 4);

mnist_dbn <- train.cnn(dataset, targets, layers, evaluator = eval, batch_size = 10, training_epochs = 3, learning_rate = 1e-3, momentum = 0.8, rand_seed = 1234);

prediction <- predict(mnist_dbn, newdata);

