.First.lib <-function (lib, pkg)
{
     library.dynam("rcnn", pkg, lib);
}

.onLoad <- function(libname = find.package("rcnn"), pkgname = "rcnn")
{
	library.dynam("rcnn", package = c("rcnn"), lib.loc = .libPaths());
}
