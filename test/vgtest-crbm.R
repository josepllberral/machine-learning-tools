## R -d "valgrind --tool=memcheck --leak-check=full" --vanilla < vgtest-crbm.R > log.txt 2>&1

library(rrbm)

train_X <- t(array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0), c(6, 12)));
crbm1 <- train.crbm(train_X, seqlen = c(6, 6), delay = 2);
crbm2 <- train.crbm(train_X, seqlen = c(6, 6), delay = 2, init_crbm = crbm1);

data(motionfrag)
     
crbm_mfrag <- train.crbm(motionfrag$batchdata, motionfrag$seqlen,
		      batch_size = 100, n_hidden = 100, delay = 6,
		      training_epochs = 10, learning_rate = 1e-3,
		      momentum = 0.5, rand_seed = 1234);

crbm_mf2 <- train.crbm(motionfrag$batchdata, motionfrag$seqlen,
		      batch_size = 100, n_hidden = 100, delay = 6,
		      training_epochs = 10, learning_rate = 1e-3,
		      momentum = 0.5, rand_seed = 1234, init_crbm = crbm_mfrag);

crbm_pred1 <- predict(crbm_mf2, motionfrag$batchdata[1:motionfrag$seqlen[1],]);
crbm_pred2 <- forecast.crbm(crbm_mf2, motionfrag$batchdata[1:motionfrag$seqlen[1],], n_samples = 50);

act_1 <- forward.crbm(crbm_mf2, motionfrag$batchdata[1:motionfrag$seqlen[1],]);
rec_1 <- backward.crbm(crbm_mf2, act_1[(crbm_mf2$delay+1):nrow(act_1),], motionfrag$batchdata[1:motionfrag$seqlen[1]-1,]);
