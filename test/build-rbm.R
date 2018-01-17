## R --vanilla < test/build-rbm.R

library(devtools)
library(roxygen2)

setwd("rrbm")
document()
build()
setwd("..")
install("rrbm")
library(rrbm)
