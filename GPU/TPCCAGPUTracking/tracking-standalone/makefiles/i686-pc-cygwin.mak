#Set these Compiler Paths and Variables to your needs! 
VSPATH							:= ${VS120COMNTOOLS}../..
VSPATH10						:= ${VS100COMNTOOLS}../..
VSPATH9							:= ${VS90COMNTOOLS}../..
VSPATH6							:= c:/Utility/Speeches/Visual Studio 6
ICCPATH							:= ${ICPP_COMPILER14}
VECTORCPATH						:= c:/Utility/speeches/Codeplay
WINPATH							:= /cygdrive/c/Windows
CUDAPATH						:= $(CUDA_PATH)/
AMDPATH							:= $(AMDAPPSDKROOT)/
CUDASDKPATH						:= $(CUDAPATH)
DIRECTXPATH						:= $(DXSDK_DIR)
QTPATH							:= C:/Utility/Speeches/Qt/5.2.1/msvc2012_64_opengl

ifeq ($(GCC32), )
GCC32							= i686-pc-mingw32-c++.exe
endif
ifeq ($(GCC64), )
GCC64							= x86_64-w64-mingw32-c++.exe -B makefiles -w
endif

ICCPATH32						= $(ICCPATH)bin/ia32
ICCPATH64						= $(ICCPATH)bin/intel64

ICC32							= $(HIDEECHOA) $(CALLVC) "$(ICCPATH)bin/iclvars_ia32.bat" $(HIDEVARS) "$(ICCPATH32)/icl.exe"
ICC64							= $(HIDEECHOA) $(CALLVC) "$(ICCPATH)bin/iclvars_intel64.bat" $(HIDEVARS) "$(ICCPATH64)/icl.exe"
MSCC32							= $(HIDEECHOA) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/cl.exe"
MSCC64							= $(HIDEECHOA) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvars64.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/amd64/cl.exe"
MSCC1032						= $(HIDEECHOA) $(CALLVC) "$(VSPATH10)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH10)/vc/bin/cl.exe"
MSCC1064						= $(HIDEECHOA) $(CALLVC) "$(VSPATH10)/vc/bin/amd64/vcvars64.bat" $(HIDEVARS) "$(VSPATH10)/vc/bin/amd64/cl.exe"
MASM32							= $(HIDEECHOA) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/ml.exe"
MASM64							= $(HIDEECHOA) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvars64.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/amd64/ml64.exe"
VCC32							= $(HIDEECHOA) "$(VECTORCPATH)/vectorc86.exe"

MSLINK32GCC						= $(HIDEECHOA) $(CALLVC) "$(ICCPATH)bin/iclvars_ia32.bat" $(HIDEVARS) "$(VSPATH8)/VC/bin/link.exe"
MSLINK32						= $(HIDEECHOA) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/VC/bin/link.exe"
MSLINK64						= $(HIDEECHOA) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvars64.bat" $(HIDEVARS) "$(VSPATH)/VC/bin/amd64/link.exe"
ICCLINK32						= $(HIDEECHOA) $(CALLVC) "$(ICCPATH)bin/iclvars_ia32.bat" $(HIDEVARS) "$(ICCPATH32)/xilink.exe" -quseenv
ICCLINK64						= $(HIDEECHOA) $(CALLVC) "$(ICCPATH)bin/iclvars_intel64.bat" $(HIDEVARS) "$(ICCPATH64)/xilink.exe" -quseenv

#Linker Optionss
LINKFLAGSCOMMON					= /fixed:no /nologo /subsystem:console /incremental:no /debug $(MULTITHREADLIBS) /MANIFEST:NO $(HOARD) /pdb:"$(WORKPATH)/$(TARGET).pdb"
LINKFLAGS32						= $(LINKFLAGSCOMMON) /machine:I386
LINKFLAGS64						= $(LINKFLAGSCOMMON) /machine:X64

#Common Compiler Options
PREHEADER						= /Fp"$@.pch" /Fd"$@.pdb"
CFLAGSCOMMON					= $(PREHEADER) /nologo /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /W3 $(MULTITHREAD)
CFLAGS32						= $(CFLAGSCOMMON)
CFLAGS64						= $(CFLAGSCOMMON) /D "_WIN64" /D "_AMD64_" /D "_X64_" 
DEBUGFLAGS						= /EHs /Zi /Od /D "DEBUG_RUNTIME"

GCCFLAGS32						+= -mrtd

INTELQPROF						= 
#/Qprof_gen, /Qprof_use

#Intel Compiler Options
INTELFLAGSOPT					= /Oa /Ow /Qansi-alias /Ob2 /Ot /Oi /GA /G7 /O3 /Ox /Qvec_report0 /Qopt-prefetch /Q$(INTELARCH) /Gs0 /debug:minimal
# /Qguide  /Qopt-report:2 /Qvec-report:5
ifeq ($(CONFIG_LTO), 1)
INTELFLAGSOPT					+= /Qipo
INTELLINKIPO					= /Qipo-c /Qipo-fo
else
INTELFLAGSOPT					+= /Qip
endif
INTELFLAGSDBG					= /Od /Zi
INTELFLAGSBASE					= /EHsc /D "INTEL_RUNTIME" /Qprof_dir$(WORKPATH) $(MULTITHREAD) $(INTELQPROF)
INTELFLAGSCOMMON				= $(INTELFLAGSBASE) $(INTELFLAGSUSE)
INTELFLAGS32					= $(INTELFLAGSCOMMON) /Oy /Gr
INTELFLAGS64					= $(INTELFLAGSCOMMON)
# /Zd /Zi /Qvec_report0 

#VectorC Compiler Options
VECTORCOPTIMIZED				= /ssecalls /optimize 10 /max /target p4 /autoinline 4096 /vc /Ob2 /Oi /Ot
VECTORCSTANDARD					= /optimize 0 /novectors /vc /Ob0
VECTORCFLAGS					= /nologo /noprogress /vserror /cpp /mslibs $(VECTORCSTANDARD) /c /D "VECTORC_RUNTIME" $(MULTITHREAD) /I"$(VSPATH6)/VC98/include" $(VC8INCLUDES)

#Visual Studio Compiler Options
VSNETFLAGSOPT					= /EHs /O2 /Ox /Oi /Ot /Oy /GA /Ob2 /Zi /Qfast_transcendentals $(MSOPENMP)
VSNETFLAGSDBG					= /Od /Zi
VSNETFLAGSCOMMON				= /D "VSNET_RUNTIME" $(VSNETFLAGSUSE) $(EXTRAFLAGSMSCC) /EHsc
VSNETFLAGS32					= $(VSNETFLAGSCOMMON)
VSNETFLAGS64					= $(VSNETFLAGSCOMMON) /favor:$(MSVCFAVOR)

ifeq ("$(CONFIG_OPENMP)", "1")
INTELFLAGSCOMMON				+= /Qopenmp
VSNETFLAGSCOMMON				+= /openmp
endif

ifeq ($(GCCARCH), )
GCCARCHA						= -march=native -msse4.2
else
GCCARCHA						= -march=$(GCCARCH) -msse4.2
endif

#Compilation Output Control
ifneq ("$(VERBOSE)", "1")
HIDEECHOB						= @
ifndef HIDEECHO
HIDEECHOA						= @
else ifneq ($(HIDEECHOA), "-")
HIDEECHOA						= $(HIDEECHO)
endif
endif
ifndef HIDEVARS
HIDEVARS						= 1
endif

CALLVC							= $(HIDEECHOA) cmd /C "makefiles\callvc.bat"

PATH							:= /bin:/usr/bin:$(WINPATH):$(WINPATH)/system32:$(PATH)

ifeq ($(ARCHBITS), 64)
ICC								= $(ICC64) $(INTELFLAGS64) $(CFLAGS64)
CCDBG							= $(ICC64) $(INTELFLAGSBASE) $(INTELFLAGSDBG) $(CFLAGS64) $(DEBUGFLAGS)
ICCLINK							= $(ICCLINK64) $(LINKFLAGS64)
MSCC							= $(MSCC64) $(VSNETFLAGS64) $(CFLAGS64)
MSLINK							= $(MSLINK64) $(LINKFLAGS64)
GCC								= $(GCC64) $(GCCFLAGS64) $(GCCFLAGSCOMMON) $(GCCFLAGSUSE)
MASM							= $(MASM64)
CCCUDA							= $(MSCC64) /TP $(VSNETFLAGS64) $(CFLAGS64)
LIBPATHAMD						= /LIBPATH:"$(AMDPATH)lib" /LIBPATH:"$(AMDPATH)lib/x86_64"
LIBPATHCUDA						= /LIBPATH:"$(CUDAPATH)lib/x64" /LIBPATH:"$(CUDASDKPATH)common/lib/x64"
LIBPATHDIRECTX					= /LIBPATH:"$(DIRECTXPATH)lib/x64"
LIBPATHSUSE						= /LIBPATH:"$(ICCPATH)compiler/lib/intel64"
LINKFLAGSARCH					= /machine:X64
else
ICC								= $(ICC32) $(INTELFLAGS32) $(CFLAGS32)
CCDBG							= $(MSCC32) $(CFLAGS32) $(DEBUGFLAGS)
ICCLINK							= $(ICCLINK32) $(LINKFLAGS32)
MSCC							= $(MSCC32) $(VSNETFLAGS32) $(CFLAGS32) /Gr
MSLINK							= $(MSLINK32) $(LINKFLAGS32)
MSLINKGCC						= $(MSLINK32GCC) $(LINKFLAGS32)
VCC								= $(VCC32) /outfile $@ $(VECTORCFLAGS) $(CFLAGS32)
GCC								= $(GCC32) $(GCCFLAGS32) $(GCCFLAGSCOMMON) $(GCCFLAGSUSE)
MASM							= $(MASM32)
CCCUDA							= $(MSCC1032) $(VSNETFLAGS32) $(CFLAGS32) /TP /Gd
LIBPATHAMD						= /LIBPATH:"$(AMDPATH)lib" /LIBPATH:"$(AMDPATH)lib/x86"
LIBPATHCUDA						= /LIBPATH:"$(CUDAPATH)lib/win32" /LIBPATH:"$(CUDASDKPATH)common/lib/Win32"
LIBPATHDIRECTX					= /LIBPATH:"$(DIRECTXPATH)lib/x86"
LIBPATHSUSE						= /LIBPATH:"$(ICCPATH)compiler/lib/ia32"
LINKFLAGSARCH					= /machine:I386
endif
QTUIC							= $(QTPATH)/bin/uic.exe
QTMOC							= $(QTPATH)/bin/moc.exe

LIBPATHSUSE						+= $(LIBPATHS:%=/LIBPATH:%)

ifeq ($(CC_i686-pc-cygwin), ICC)
CC								= $(ICC)
ifeq ($(CPPFILES_GCC), )
LINK							= $(ICCLINK)
else
LINK							= $(MSLINK)
endif
else ifeq ($(CC_i686-pc-cygwin), GCC)
CC								= $(GCC)
LINK							= $(GCC)
else
CC								= $(MSCC)
LINK							= $(MSLINK)
endif
GCC3264							= $(GCC)
CC_SELECTED						= $(CC_i686-pc-cygwin)

ASM								= $(MASM)
ASMPRE							= $(MSCC32)
NVCC							= $(HIDEECHOA) $(CALLVC) "$(VSPATH10)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(CUDAPATH)bin/nvcc"

MULTITHREADGCC					= -mthreads -D_MT

LIBSUSE							= kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib

ifneq ($(CPPFILES_GCC), )
GCCLIBPATH						:= $(shell cygpath -m `$(GCC) -print-libgcc-file-name | sed -e s/libgcc.a//` `$(GCC) -print-sysroot`/mingw/lib)
LIBSUSE							+= $(GCCLIBPATH:%=/LIBPATH:"%") libgcc.a libstdc++.a libmingw32.a libgcc_eh.a
#libmingwex.a
#libmsvcrt.a
#libgcov.a libmingwex.a
endif

OPENCLLIB						= OpenCL.lib
ifeq ("$(CONFIG_OPENCL)", "1")
LIBSUSE							+= $(OPENCLLIB)
endif

ifeq ("$(CONFIG_CAL)", "1")
ifeq ($(ARCHBITS), 64)
LIBSUSE							+= aticalcl64.lib aticalrt64.lib
else
LIBSUSE							+= aticalcl.lib aticalrt.lib
endif
endif

ifeq ("$(CONFIG_DIRECTX)", "1")
LIBSUSE							+= ddraw.lib dxguid.lib dxerr.lib
COMMONINCLUDEPATHS				+= "$(DIRECTXPATH)include"
LIBPATHSUSE						+= $(LIBPATHDIRECTX)
endif

ifeq ("$(CONFIG_VIDEO_EDIT)", "1")
LIBSUSE							+= amstrmid.lib msacm32.lib vfw32.lib winmm.lib
endif

ifeq ("$(CONFIG_OPENGL)", "1")
LIBSUSE							+= opengl32.lib glu32.lib
endif

ifeq ("$(CONFIG_QT)", "1")
LIBSUSE							+= Qt5Gui.lib Qt5Core.lib Qt5Widgets.lib
COMMONINCLUDEPATHS				+= $(QTPATH)/include $(QTPATH)/include/QtGui $(QTPATH)/include/QtCore $(QTPATH)/include/QtWidgets $(WORKPATH)/qt
LIBPATHSUSE						+= /LIBPATH:$(QTPATH)/lib
endif

LIBSUSE							+= $(LIBS:%=%.lib)

ifeq ($(TARGETTYPE), LIB)
LINKTARGETTYPE					= /DLL
EXECUTABLE						= $(TARGET).dll
else
LINKTARGETTYPE					=
EXECUTABLE						= $(TARGET).exe
endif

ifeq ("$(CONFIG_OPENCL)", "1")
ifeq ("$(CONFIG_OPENCL_VERSION)", "AMD")
COMMONINCLUDEPATHS				+= "$(AMDPATH)include"
LIBPATHSUSE						+= $(LIBPATHAMD)
endif
ifeq ("$(CONFIG_OPENCL_VERSION)", "NVIDIA")
COMMONINCLUDEPATHS				+= "$(CUDAPATH)include"
LIBPATHSUSE						+= $(LIBPATHCUDA)
endif
ifeq ("$(CONFIG_OPENCL_VERSION)", "Intel")
#COMMONINCLUDEPATHS				+= ""
endif
ifeq ("$(CONFIG_OPENCL_VERSION)", "All")
COMMONINCLUDEPATHS				+= "$(AMDPATH)include"
COMMONINCLUDEPATHS				+= "$(CUDAPATH)include"
#COMMONINCLUDEPATHS				+= ""
LIBPATHSUSE						+= $(LIBPATHAMD)
endif
endif

ifeq ("$(CONFIG_CUDA)", "1")
COMMONINCLUDEPATHS				+= "$(CUDAPATH)include" "$(CUDASDKPATH)common/inc"
LIBPATHSUSE						+= $(LIBPATHCUDA)
ifneq ($(CUFILES), )
LIBSUSE							+= cudart.lib cuda.lib
ifeq ($(CONFIG_CUDA_DC), 1)
LIBSUSE							+= cudadevrt.lib
endif
ifeq ("$(CONFIG_OPENGL)", "1")
ifeq ($(ARCHBITS), 64)
LIBSUSE							+= freeglut.lib glew64.lib
else
LIBSUSE							+= freeglut.lib glew32.lib
endif
endif
endif
endif

ifeq ("$(CONFIG_CAL)", "1")
COMMONINCLUDEPATHS				+= "$(AMDPATH)/include/CAL"
LIBPATHSUSE						+= $(LIBPATHAMD)
endif

ifeq ($(CC_i686-pc-cygwin), GCC)
COMPILEOUTPUT					= -o $@
LINKOUTPUT						= -o $@
COMPILEONLY						= -c
ASMONLY							= -c
PRECOMPILEONLY					= -x c++ -E
INCLUDEPATHSUSE					= $(GCCINCLUDEPATHS)
DEFINESUSE						= $(GCCDEFINES)
else
INCLUDEPATHSUSE					= $(VSINCLUDEPATHS)
DEFINESUSE						= $(VSDEFINES)
COMPILEOUTPUTBASE				= /Fo
COMPILEOUTPUT					= $(COMPILEOUTPUTBASE)"$@"
LINKOUTPUT						= /Out:"$@"
COMPILEONLY						= /c
ASMONLY							= /c
PRECOMPILEONLY					= /EP 
endif
OBJ								= obj

DEFINESARCH						= "WIN32"

NVCCARCHS						:= `for i in $(CUDAVERSION); do echo -n -gencode arch BAT_SPECIAL_EQ compute_$$i BAT_SPECIAL_KOMMA code BAT_SPECIAL_EQ sm_$$i\ ;done`
NVCC_GREP						= "^#line\|^$$"
