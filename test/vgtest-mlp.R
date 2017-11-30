## R -d "valgrind --tool=memcheck --leak-check=full" --vanilla < vgtest-cnn.R > log.txt 2>&1

library(rcnn)

# MLP Simple
train_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 6));
train_Y <- array(c(1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1), c(6, 2));
layers <- list(
    c('type' = "LINE", 'n_visible' = 6, 'n_hidden' = 2, 'scale' = 0.1),
    c('type' = "RELV"),
    c('type' = "SOFT", 'n_inputs' = 2)
);
mlp1 <- train.cnn(train_X, train_Y, layers, batch_size = 2);

test_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 6));
results <- predict(mlp1, test_X);

# MLP MNIST
data(mnist)

train <- mnist$train;
training_x <- train$x / 255;
training_y <- binarization(train$y);

test <- mnist$test;
testing_x <- test$x / 255;
testing_y <- binarization(test$y);

dataset <- training_x[1:1000,, drop=FALSE];
targets <- training_y[1:1000,, drop=FALSE];

newdata <- testing_x[1:1000,, drop=FALSE];

layers <- list(
    c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 64, 'scale' = 0.1),
    c('type' = "RELV"),
    c('type' = "LINE", 'n_visible' = 64, 'n_hidden' = 10, 'scale' = 0.1),
    c('type' = "SOFT", 'n_inputs' = 10)
);

mnist_mlp <- train.cnn(dataset, targets, layers, batch_size = 10, training_epochs = 10, learning_rate = 1e-3, momentum = 0.8, rand_seed = 1234);

prediction <- predict(mnist_mlp, newdata);
