#Compilation Output Control
HIDEECHO					= @
HIDEVARS					= 1

CUDAPATH					= c:/Utility/Speeches/cuda
CUDASDKPATH					= $(CUDAPATH)/sdk

CALLVC						= $(HIDEECHO) cmd /C callvc.bat

PATH						= $(CYGWINPATH)/bin:$(CYGWINPATH)/usr/bin:$(WINPATH):$(WINPATH)/system32

CC							= $(ICC64) $(INTELFLAGS64) $(CFLAGS64) 
LINK						= $(ICCLINK64)
CCCUDA						= $(MSCC64) $(VSNETFLAGS64) $(CFLAGS64) /TP
ASM							= $(MASM64)
ASMPRE						= $(MSCC32)
GCC							= "$(GCCPATH)/bin/g++.exe"
NVCC						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(CUDAPATH)/bin/nvcc"

MULTITHREADGCC				= -mthreads -D_MT

LIBS						= kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib ddraw.lib vfw32.lib winmm.lib amstrmid.lib dxguid.lib msacm32.lib
LIBSCYGWIN					= libgcc.a libstdc++.a libmingw32.a libgcov.a /LIBPATH:"$(GCCPATH)/lib"
CUDALIBS					= cudart.lib

EXECUTABLE					= $(TARGET).exe

COMMONINCLUDEPATHS			= "c:\utility\speeches\sdk\directx\include" "$(CUDAPATH)/include" "$(CUDASDKPATH)/common/inc"
LIBPATHS					= /LIBPATH:"c:\utility\speeches\cuda\lib" /LIBPATH:"c:\utility\speeches\sdk\directx\lib\x64" /LIBPATH:"$(ICCPATH)/lib/intel64"

COMPILEOUTPUT				= /Fo"$@"
LINKOUTPUT					= /Out:"$@"
COMPILEONLY					= /c
PRECOMPILEONLY				= /EP 

INCLUDEPATHSUSE				= $(VSINCLUDEPATHS)
DEFINESUSE					= $(VSDEFINES)

DEFINESARCH					= "WIN32"

NVCCARCHS					:= `for i in $(CUDAVERSION); do echo -n -gencode arch BAT_SPECIAL_EQ compute_$$i BAT_SPECIAL_KOMMA code BAT_SPECIAL_EQ sm_$$i\ ;done`

LINKFLAGSUSE				= $(LINKFLAGS64)