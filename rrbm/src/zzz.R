.First.lib <-function (lib, pkg)
{
     library.dynam("rrbm", pkg, lib);
}

.onLoad <- function(libname, pkgname)
{
	library.dynam("rrbm", package = c("rrbm"), lib.loc = .libPaths());
}
