# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = .
APIDOCDIR     = _apidoc
PACKAGEDIR    = ..
CONFDIR       = .
OPEN          = xdg-open
DEVNULL       = /dev/null
HTML          = html
INDEXLOCATION = $(shell pwd)/build/$(HTML)/index.html
EXCLUDEPATH   = ../*script*

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXAPIDOC) --force --output-dir "$(APIDOCDIR)" "$(PACKAGEDIR)" "$(EXCLUDEPATH)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS) $(O)
#	@if [ "$(HTML)" = "$@" ]; then \
#		$(OPEN) "$(INDEXLOCATION)" 1>$(DEVNULL) 2>$(DEVNULL); \
#	fi
