## R --vanilla < test/build-cnn.R

library(devtools)
library(roxygen2)

setwd("rcnn")
document()
build()
setwd("..")
install("rcnn")
library(rcnn)
