#Set these Compiler Paths and Variables to your needs! 
VSPATH						= c:/Utility/Speeches/Visual Studio 9
VSPATH8						= c:/Utility/Speeches/Visual Studio 8
VSPATH6						= c:/Utility/Speeches/Visual Studio 6
ICCPATH						= c:/Utility/speeches/intel/compiler/11.0/074/cpp
GCCPATH						= c:/Utility/Speeches/gcc
CYGWINPATH					= /cygdrive/c/Utility/Cygwin
WINPATH						= /cygdrive/c/Windows

ICCPATH32					= $(ICCPATH)/bin/ia32
ICCPATH64					= $(ICCPATH)/bin/intel64

ICC32						= $(HIDEECHO) $(CALLVC) "$(ICCPATH32)/iclvars_ia32.bat" $(HIDEVARS) "$(ICCPATH32)/icl.exe"
ICC64						= $(HIDEECHO) $(CALLVC) "$(ICCPATH64)/iclvars_intel64.bat" $(HIDEVARS) "$(ICCPATH64)/icl.exe"
MSCC32						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/cl.exe"
MSCC64						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvarsamd64.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/amd64/cl.exe"
MASM32						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/ml.exe"
MASM64						= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvarsamd64.bat" $(HIDEVARS) "$(VSPATH)/vc/bin/amd64/ml64.exe"
VCC							= $(HIDEECHO) "c:/Utility/speeches/Codeplay/vectorc86.exe"

MSLINK32GCC					= $(HIDEECHO) $(CALLVC) "$(ICCPATH32)/iclvars_ia32.bat" $(HIDEVARS) "$(VSPATH8)/VC/bin/link.exe"
MSLINK32					= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/vcvars32.bat" $(HIDEVARS) "$(VSPATH)/VC/bin/link.exe"
MSLINK64					= $(HIDEECHO) $(CALLVC) "$(VSPATH)/vc/bin/amd64/vcvarsamd64.bat" $(HIDEVARS) "$(VSPATH)/VC/bin/amd64/link.exe"
ICCLINK32					= $(HIDEECHO) $(CALLVC) "$(ICCPATH32)/iclvars_ia32.bat" $(HIDEVARS) "$(ICCPATH32)/xilink.exe" -quseenv
ICCLINK64					= $(HIDEECHO) $(CALLVC) "$(ICCPATH64)/iclvars_intel64.bat" $(HIDEVARS) "$(ICCPATH64)/xilink.exe" -quseenv

ARCH						:= $(shell sort <<< $$MACHTYPE)
ARCHFILE					= $(ARCH).mak
ALLDEP						= makefile config.mak $(ARCHFILE)

WORKPATH					= release/$(ARCH)

#GCC Compiler Options
GCCFLAGSOPT					= -O3 -ftree-vectorize -march=core2 -msse3 -fkeep-inline-functions -fweb -frename-registers -minline-all-stringops -funit-at-a-time -mfpmath=sse -ftracer -finline-limit=1200 -funroll-loops -fpeel-loops -fprefetch-loop-arrays -finline-functions
GCCFLAGSDBG					= -O0
GCCFLAGSCOMMON				= $(MULTITHREADGCC) -pipe -DGCC_RUNTIME $(GCCPROF) $(GCCFLAGSUSE) -ggdb -Wno-write-strings
GCCFLAGS32					= $(GCCFLAGSCOMMON) -m32 -mrtd 
GCCFLAGS64					= $(GCCFLAGSCOMMON) -m64 -D"_AMD64_" -D"_X64_"

#Multithread Options
MULTITHREAD					= /MT
MULTITHREADLIBS				= /nodefaultlib:libc.lib

NVCCFLAGSOPT				= --use_fast_math --maxrregcount $(CUDAREGS) -O4 -Xptxas -v -Xptxas -O4 $(NVCCARCHS)
NVCCFLAGSDBG				= --device-emulation --maxrregcount $(CUDAREGS) -Xptxas -v -Xptxas -O0 -O0 $(NVCCARCHS)

#Linker Optionss
LINKFLAGSCOMMON				= /fixed:no /nologo /subsystem:console /incremental:no /debug $(MULTITHREADLIBS) $(DNDVERSION) /MANIFEST:NO $(HOARD) /pdb:"$(WORKPATH)/$(TARGET).pdb"
LINKFLAGS32					= $(LINKFLAGSCOMMON) /machine:I386
LINKFLAGS64					= $(LINKFLAGSCOMMON) /machine:X64
#HOARD						= usewinhoard.obj winhoard.lib

#Common Compiler Options
PREHEADER					= /Fp"$@.pch" /Fd"$@.pdb"
CFLAGSCOMMON				= $(PREHEADER) /nologo /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /W3 $(MULTITHREAD)
CFLAGS32					= $(CFLAGSCOMMON)
CFLAGS64					= $(CFLAGSCOMMON) /D "_WIN64" /D "_AMD64_" /D "_X64_" 
DEBUGFLAGS					= /EHs /Zi /Od /D "DEBUG_RUNTIME"

#/Qprof_gen, /Qprof_use
INTELQPROF					= 
INTELOPENMP					= /Qopenmp /Qopenmp-link:static /Qparallel
MSOPENMP					= 
#/openmp
#-fprofile-arcs, -fbranch-probabilities
GCCPROF						= 

#Intel Compiler Options
INTELFLAGSOPT				= /Oa /Ow /Ob2 /Ot /Oi /GA /G7 /O3 /Qipo /Qvec_report1 /Qopt-prefetch /Qax$(INTELARCH).2 /Gs0 /Zd $(INTELOPENMP)
INTELFLAGSDBG				= /Od /Zi /Qopenmp-stubs
INTELFLAGSCOMMON			= /EHsc /D "INTEL_RUNTIME" /Qvc9 /Qprof_dir$(WORKPATH) $(MULTITHREAD) $(INTELQPROF) $(INTELFLAGSUSE)
INTELFLAGS32				= $(INTELFLAGSCOMMON) /Oy /Gr /Og
INTELFLAGS64				= $(INTELFLAGSCOMMON)
# /Zd /Zi /Qvec_report0 

#VectorC Compiler Options
VECTORCOPTIMIZED			= /ssecalls /optimize 10 /max /target p4 /autoinline 4096 /vc /Ob2 /Oi /Ot
VECTORCSTANDARD				= /optimize 0 /novectors /vc /Ob0
VECTORCFLAGS				= /nologo /noprogress /vserror /cpp /mslibs $(VECTORCSTANDARD) /c /D "VECTORC_RUNTIME" $(MULTITHREAD) /I"$(VSPATH6)/VC98/include" $(VC8INCLUDES)

#Visual Studio Compiler Options
VSNETFLAGSOPT				= /EHs /O2 /Ox /Oi /Ot /Oy /GA /Gr /Ob2 /fp:fast /Qfast_transcendentals $(MSOPENMP)
VSNETFLAGSDBG				= /Od /Zi
VSNETFLAGSCOMMON			=  /D "VSNET_RUNTIME" $(VSNETFLAGSUSE) /EHsc
VSNETFLAGS32				= $(VSNETFLAGSCOMMON)
VSNETFLAGS64				= $(VSNETFLAGSCOMMON) /favor:INTEL64

include						config.mak
include						$(ARCHFILE)

CUDAINCLUDEPATHS			= $(INCLUDEPATHSUSE:%=--compiler-options %)
CUDADEFINES					= $(DEFINESUSE:%=--compiler-options %)

OBJ64FILES					= $(CPPFILES:%.cpp=$(WORKPATH)/%.obj) $(CXXFILES:%.cxx=$(WORKPATH)/%.cxobj)
CU64OBJFILES				= $(CUFILES:%.cu=$(WORKPATH)/%.cuobj)
ASM64OBJFILES				= $(ASMFILES:%.asm=$(WORKPATH)/%.asmobj)

CPPDEPENDS					= $(CPPFILES:%.cpp=$(WORKPATH)/%.d) $(CXXFILES:%.cxx=$(WORKPATH)/%.cxd)
CUDEPENDS					= $(CUFILES:%.cu=$(WORKPATH)/%.cud)
ASMDEPENDS					= $(ASMFILES:%.asm=$(WORKPATH)/%.asmd)

GCCINCLUDEPATHS				= $(INCLUDEPATHS:%=-I%) $(COMMONINCLUDEPATHS:%=-I%)
VSINCLUDEPATHS				= $(INCLUDEPATHS:%=/I%) $(COMMONINCLUDEPATHS:%=/I%)

GCCDEFINES					= $(DEFINES:%=-D%) $(DEFINESARCH:%=-D%)
VSDEFINES					= $(DEFINES:%=/D%) $(DEFINESARCH:%=/D%)

LIBFILES					= $(LIBS) $(CUDALIBS)

all:						$(EXECUTABLE)

.SECONDARY:					$(CUFILES:%.cu=$(WORKPATH)/%.cucpp) $(ASMFILES:%.asm=$(WORKPATH)/%.asmcpp)

$(EXECUTABLE):				$(OBJ64FILES) $(CU64OBJFILES) $(ASM64OBJFILES) $(ALLDEP)
							$(LINK) $(LIBPATHS) $(OBJ64FILES) $(CU64OBJFILES) $(ASM64OBJFILES) $(LIBFILES) $(LINKFLAGSUSE) $(LINKOUTPUT)
							./build.sh

$(WORKPATH)/%.obj:			%.cpp $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.cxobj:		%.cxx $(ALLDEP)
							$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.cuobj:		$(WORKPATH)/%.cucpp $(ALLDEP)
							$(CCCUDA) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(COMPILEONLY) $< $(COMPILEOUTPUT)
							
$(WORKPATH)/%.asmobj:		$(WORKPATH)/%.asmcpp $(ALLDEP)
							$(ASM) $(COMPILEOUTPUT) $(COMPILEONLY) $<
							
$(WORKPATH)/%.asmcpp:		%.asm $(ALLDEP)
							$(MSCC32) $(PRECOMPILEONLY) /D "_AMD64_" $< > $@

$(WORKPATH)/%.cucpp:		%.cu $(ALLDEP)
							$(NVCC) $(NVCCFLAGSUSE) $(CUDAINCLUDEPATHS) $(CUDADEFINES) --cuda --output-file "$@" $<
							#patch -p0 < cucpp.patch

clean:		
							rm -f *.plg *.dpi $(EXECUTABLE) x64/release/* *.cucpp *.cubin *.gpu *.ptx *.linkinfo *.ii cuda.compute_* $(CPPDEPENDS) $(CUDEPENDS) $(ASMDEPENDS) $(OBJ64FILES) $(CU64OBJFILES) $(ASM64OBJFILES) $(CUFILES:%.cu=$(WORKPATH)/%.cucpp) $(ASMFILES:%.asm=$(WORKPATH)/%.asmcpp)

SAVEDIR						= releases/`date +%F`-BUILD-`cat buildnr`							
backup:						
							$(HIDEECHO) mkdir $(SAVEDIR)
							cp *.cpp *.h makefile buildnr *.sh *.bat *.conf *.cu $(SAVEDIR)

#Dependancies
$(WORKPATH)/%.cud:			%.cu
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC) $(GCCINCLUDEPATHS) $(GCCDEFINES) -D__CUDACC__ -x c -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.cuobj $(WORKPATH)/$*.cucpp $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.asmd:			%.asm
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC) $(GCCINCLUDEPATHS) $(GCCDEFINES) -x c -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.asmobj $(WORKPATH)/$*.asmcpp $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.d:			%.cpp
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC) $(GCCINCLUDEPATHS) $(GCCDEFINES) -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.obj $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;
							
$(WORKPATH)/%.cxd:			%.cxx
							$(HIDEECHO)mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` && \
							$(GCC) $(GCCINCLUDEPATHS) $(GCCDEFINES) -MM $< | \
							sed 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/$*.cxobj $@ : ,g' | \
							sed 's,c:/,/cygdrive/c/,g' > \
							$@;

DEPENDS						= $(CPPDEPENDS) $(CUDEPENDS) $(ASMDEPENDS)

include $(DEPENDS)