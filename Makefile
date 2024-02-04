SRC_CORR     := $(wildcard *_corr.ipynb)
TARGETS_CORR := $(patsubst %_corr.ipynb,%_nocorr.ipynb,$(SRC_CORR))
SRC_BASE     := $(wildcard *_base.ipynb)
TARGETS_BASE := $(patsubst %_base.ipynb,%_nocorr.ipynb,$(SRC_BASE))
TARGETS      := $(TARGETS_BASE) $(TARGETS_CORR)

all: $(TARGETS)

Sequences_nocorr.ipynb: Sequences_base.ipynb
	nbvariants --no-output $^ $@ keep
	nbvariants $^ $(patsubst %_nocorr.ipynb,%_corr.ipynb,$@) keep_corr

wrap-up_nocorr.ipynb: wrap-up_base.ipynb
	nbvariants --no-output $^ $@ keep
	nbvariants $^ $(patsubst %_nocorr.ipynb,%_corr.ipynb,$@) keep_corr

%_nocorr.ipynb: %_corr.ipynb
	nbvariants --no-output $^ $@ keep

clean:
	rm -f $(TARGETS)
