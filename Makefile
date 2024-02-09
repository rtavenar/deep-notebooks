SRC_CORR     := $(wildcard *_corr.ipynb)
TARGETS_CORR := $(patsubst %_corr.ipynb,%_nocorr.ipynb,$(SRC_CORR))
SRC_BASE     := $(wildcard *_base.ipynb)
TARGETS_BASE := $(patsubst %_base.ipynb,%_nocorr.ipynb,$(SRC_BASE))
TARGETS      := $(TARGETS_BASE) $(TARGETS_CORR)

all: $(TARGETS)

UNet_RemoteSensing_nocorr.ipynb: UNet_RemoteSensing_base.ipynb
	nbvariants --no-output $^ $@ keep_nocorr
	nbvariants $^ $(patsubst %_nocorr.ipynb,%_corr.ipynb,$@) keep_corr

%_nocorr.ipynb: %_corr.ipynb
	nbvariants --no-output $^ $@ keep

clean:
	rm -f $(TARGETS)

clean_artifacts:
	rm -fR __MACOSX
	rm -fR cats_and_dogs
	rm -fR vaihingen-cropped-small
	rm -fR cats_and_dogs.zip mini-vaihingen.zip pretrained_unet.h5

