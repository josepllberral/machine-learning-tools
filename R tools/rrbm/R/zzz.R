.First.lib <-function (lib, pkg)
{
     library.dynam("rrbm", pkg, lib);
}

.onLoad <- function(libname = find.package("rrbm"), pkgname = "rrbm")
{
	library.dynam("rrbm", package = c("rrbm"), lib.loc = .libPaths());
}
