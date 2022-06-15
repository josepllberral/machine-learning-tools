## R -d "valgrind --tool=memcheck --leak-check=full" --vanilla < vgtest-rbm.R > log-rbm.txt 2>&1

library(rrbm)

train_X <- t(array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 6)));
rbm1 <- train.rbm(train_X, training_epochs = 10);
rbm2 <- train.rbm(train_X, training_epochs = 10, init_rbm=rbm1)

data(mnist)
     
training.num <- data.matrix(mnist$train$x)/255;
rbm_mnist <- train.rbm(n_hidden = 30, dataset = training.num,
                       learning_rate = 1e-3, training_epochs = 10,
                       batch_size = 10, momentum = 0.5);
     
rbm_mnist_update <- train.rbm(n_hidden = 30, dataset = training.num,
	                      learning_rate = 1e-3, training_epochs = 10,
                              batch_size = 10, momentum = 0.5,
                              init_rbm = rbm_mnist);

rbm_pred1 <- predict(rbm_mnist_update, training.num[1:10,]);

act1 <- forward.rbm(rbm_mnist_update, training.num[1:10,]);
recons1 <- backward.rbm(rbm_mnist_update, act1);
