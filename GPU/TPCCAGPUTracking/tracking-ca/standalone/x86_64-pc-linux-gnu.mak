CUDAPATH					= /opt/cuda
CUDASDKPATH					= $(CUDAPATH)/sdk

GCC							= c++
CC							= $(GCC) $(GCCFLAGS64)
CCCUDA						= c++ $(GCCFLAGS64) -x c++
ASM							= yasm
ASMPRE						= $(GCC)
LINK						= $(GCC) -Wall -m64 -ggdb
NVCC						= nvcc

CUDALIBS					= -lcudart $(CUDASDKPATH)/lib/libcutil.a

EXECUTABLE					= $(TARGET)

COMPILEOUTPUT				= -o $@
LINKOUTPUT					= -o $@
COMPILEONLY					= -c
PRECOMPILEONLY				= -blakfa

INCLUDEPATHSUSE				= $(GCCINCLUDEPATHS)
DEFINESUSE					= $(GCCDEFINES)

LIBPATHS					= -L$(CUDAPATH)/lib

NVCCARCHS					:= `for i in $(CUDAVERSION); do echo -n -gencode arch=compute_$$i,code=sm_$$i\ ;done`

COMMONINCLUDEPATHS			= "$(CUDASDKPATH)/common/inc"