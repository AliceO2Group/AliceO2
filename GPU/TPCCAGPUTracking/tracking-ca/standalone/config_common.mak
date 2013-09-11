#Architecture Settings
INTELARCH					= Host
GCCARCH						= native
MSVCFAVOR					= INTEL64
CUDAVERSION					= 20
CUDAREGS					= 64
ARCHBITS					= 64

HIDEECHO					= @

CONFIG_OPENMP				= 1

CC_x86_64-pc-linux-gnu		= GCC
CC_i686-pc-cygwin			= ICC

INCLUDEPATHS				= include code base merger-ca cagpubuild
DEFINES						= HLTCA_STANDALONE BUILD_GPU

EXTRAFLAGSGCC				= -Weffc++

COMPILER_FLAGS				= OPT
CONFIG_LTO					= 1