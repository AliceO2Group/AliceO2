#Architecture Settings
INTELARCH					= Host
GCCARCH						= native
MSVCFAVOR					= INTEL64
CUDAREGS					= 64
ARCHBITS					= 64

HIDEECHO					= @

CC_x86_64-pc-linux-gnu		= GCC
CC_i686-pc-cygwin			= ICC

INCLUDEPATHS				= . SliceTracker HLTHeaders Merger Base Global TRDTracking ITS dEdx TPCConvert DataCompression Common TPCFastTransformation display qa
DEFINES						= GPUCA_STANDALONE

EXTRAFLAGSGCC				+=
EXTRAFLAGSLINK				+= -rdynamic -Wl,--no-undefined -L .

ifeq ($(BUILD_DEBUG), 1)
COMPILER_FLAGS				= DBG
else
COMPILER_FLAGS				= OPT
endif
CONFIG_LTO					= 1

CONFIG_CPP					= c++17
CONFIG_CPP_CUDA				= c++14

ifeq ($(ARCH_CYGWIN), 1)
CONFIG_O2DIR				=
BUILD_QA					= 0
endif

ifeq ($(CONFIG_OPENMP), 1)
DEFINES						+= GPUCA_HAVE_OPENMP
endif

ifeq ($(CONFIG_VC), 1)
LIBSUSE						+= -lVc
else
DEFINES						+= GPUCA_NO_VC
endif

ifeq ($(LINK_ROOT), 0)
ifneq ($(CONFIG_O2DIR), )
$(warning Cannot use O2DIR without ROOT)
CONFIG_O2DIR =
endif
ifeq ($(BUILD_QA), 1)
$(warning Cannot build QA without ROOT)
BUILD_QA = 0
endif
endif

ifeq ($(CONFIG_O2DIR), )
ifeq ($(CONFIG_O2), 1)
$(warning Cannot build for O2 geometry wihout O2 dir)
CONFIG_O2 = 0
endif
endif

ifneq (${CONFIG_O2DIR}, )
DEFINES						+= HAVE_O2HEADERS
INCLUDEPATHS					+= ${CONFIG_O2DIR}/Common/Constants/include \
								${CONFIG_O2DIR}/Common/MathUtils/include \
								${CONFIG_O2DIR}/DataFormats/common/include \
								${CONFIG_O2DIR}/Detectors/TPC/base/include \
								${CONFIG_O2DIR}/DataFormats/Detectors/TPC/include \
								${CONFIG_O2DIR}/DataFormats/common/include \
								${CONFIG_O2DIR}/Detectors/TRD/base/include \
								${CONFIG_O2DIR}/Detectors/TRD/base/src \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/include \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include \
								${CONFIG_O2DIR}/DataFormats/Detectors/ITSMFT/ITS/include \
								${CONFIG_O2DIR}/DataFormats/Reconstruction/include \
								${CONFIG_O2DIR}/DataFormats/simulation/include \
								${CONFIG_O2DIR}/Detectors/Base/src \
								${CONFIG_O2DIR}/Detectors/Base/include \
								${CONFIG_O2DIR}/DataFormats/Detectors/Common/include
endif

ifeq ($(CONFIG_O2), 1)
DEFINES						+= GPUCA_TPC_GEOMETRY_O2
endif

ifeq ($(BUILD_CUDA), 1)
DEFINES						+= CUDA_ENABLED
endif
ifeq ($(BUILD_OPENCL), 1)
DEFINES						+= OPENCL1_ENABLED
endif
ifeq ($(BUILD_HIP), 1)
DEFINES						+= HIP_ENABLED
endif
ifeq ($(BUILD_EVENT_DISPLAY), 1)
DEFINES						+= BUILD_EVENT_DISPLAY
endif
ifeq ($(BUILD_QA), 1)
DEFINES						+= BUILD_QA
endif


ALLDEP						+= config_common.mak config_options.mak
