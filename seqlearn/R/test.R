## RBM Test function
testing.rbm <- function()
{
	train_X <- t(array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 6)));
	rbm1 <- train.rbm(train_X);

	test_X <- t(array(c(1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0), c(6,2)));
	res <- predict(rbm1, test_X);
	res;
}

## CRBM Test function
testing.crbm <- function()
{
	dataset <- load_data('./datasets/motion.rds');

	crbm <- train.crbm (dataset$batchdata, dataset$seqlen, batch_size = 100, n_hidden = 100, delay = 6, training_epochs = 200, learning_rate = 1e-3, momentum = 0.5, rand_seed = 1234);

	res1 <- predict(crbm, dataset$batchdata);
	res2 <- forecast.crbm(crbm, dataset$batchdata[1:(crbm$delay+1),], 50, 30);

	list(predicted = res1, forecast = res2);
}
