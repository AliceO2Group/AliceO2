config_options.mak:
						cp config_options.sample config_options.mak

include						config_options.mak
include						config_common.mak

TARGET						= ca

ifeq ($(BUILD_CUDA), 1)
SUBTARGETS					+= libAliHLTTPCCAGPUSA
endif
SUBTARGETS_CLEAN				+= libAliHLTTPCCAGPUSA.*

ifeq ($(BUILD_OPENCL), 1)
SUBTARGETS					+= libAliHLTTPCCAGPUSAOpenCL
endif
SUBTARGETS_CLEAN				+= libAliHLTTPCCAGPUSAOpenCL.Q

CXXFILES					+= standalone.cxx \
						   $(HLTCA_STANDALONE_CXXFILES) \
						   $(HLTCA_MERGER_CXXFILES)

CPPFILES					+= cmodules/qconfig.cpp

ifeq ($(BUILD_EVENT_DISPLAY), 1)
CPPFILES					+= display/opengl.cpp
CONFIG_OPENGL				= 1
CONFIG_X11					= 1
DEFINES						+= BUILD_EVENT_DISPLAY
endif

ifeq ($(BUILD_QA), 1)
CPPFILES					+= qa/qa.cpp qa/genEvents.cpp
DEFINES						+= BUILD_QA
INCLUDEPATHSSYSTEM				+= $(shell root-config --incdir)
LIBSUSE						+= $(shell root-config --libs)
endif

ALLDEP						+= config_common.mak

o2:
						make CONFIGFILE=config_o2.mak -f makefile
