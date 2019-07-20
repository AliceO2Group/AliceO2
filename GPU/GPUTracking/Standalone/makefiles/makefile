all								: all_tmp

SHELL							:= /bin/bash
ARCH							:= $(shell echo $$MACHTYPE)
ARCHCHK							:= $(shell if [ -a makefiles/$(ARCH).mak ]; then echo -n 1; else echo -n 0; fi)
ifeq ($(ARCHCHK), 0)
$(warning Unknown Architecture: $(ARCH), defaulting to x86_64-pc-linux-gnu)
ARCH							:= x86_64-pc-linux-gnu
endif
ifeq ($(ARCH), i686-pc-cygwin)
ARCH_CYGWIN						:= 1
endif
ifeq ($(ARCH), x86_64-unknown-cygwin)
ARCH_CYGWIN						:= 1
endif

ARCHFILE						= $(ARCH).mak

ifeq ($(CONFIGFILE), )
CONFIGFILE						= config.mak
CLEANRELEASEDIR					= release
endif

ifeq ($(BUILDSCRIPT), )
BUILDSCRIPT						= build.sh
endif

#GCC Compiler Options
GCCFLAGSOPT						= -O3 $(GCCFLAGSARCH) -fweb -frename-registers -minline-all-stringops -mfpmath=sse -ftracer -funroll-loops -fpeel-loops -fprefetch-loop-arrays -ffast-math -fno-stack-protector
GCCFLAGSDBG						= -O0
CLANGFLAGSOPT					= -O3 $(GCCFLAGSARCH) -minline-all-stringops -funroll-loops -ffast-math -fno-stack-protector $(EXTRAFLAGSCLANG)
CLANGFLAGSDBG					= $(GCCFLAGSDBG) $(EXTRAFLAGSCLANG)
GCCWARNINGS						= -Wall -Wextra -Wshadow -Wno-unused-function -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-write-strings
GCCFLAGSCOMMON					= $(MULTITHREADGCC) -pipe -DGCC_RUNTIME $(GCCPROF) $(GCCWARNINGS) $(EXTRAFLAGSGCC)
CLANGFLAGSCOMMON				= $(MULTITHREADGCC) -pipe -DGCC_RUNTIME -DCLANG_RUNTIME $(GCCPROF) $(GCCWARNINGS) $(EXTRAFLAGSGCC)
GCCFLAGS32						= -m32
GCCFLAGS64						= -m64 -D"_AMD64_" -D"_X64_"

GCCPROF							=
#-fprofile-arcs, -fbranch-probabilities

#Multithread Options
MULTITHREAD						= /MT
MULTITHREADLIBS					= /nodefaultlib:libc.lib

NVCCFLAGSOPT					= --use_fast_math --maxrregcount $(CUDAREGS) -O4 -Xptxas -v -Xptxas -O4 -Xcompiler -O4 -m$(ARCHBITS) $(NVCCARCHS) --expt-relaxed-constexpr
NVCCFLAGSDBG					= --maxrregcount $(CUDAREGS) -Xptxas -v -Xptxas -O0 -O0 -G -m$(ARCHBITS) $(NVCCARCHS) --expt-relaxed-constexpr

TARGETTYPE						= EXECUTABLE

WORKPATHSUFFIX					= $(TARGETTYPE)_$(TARGET)
include							$(CONFIGFILE)
WORKPATH						= release/$(ARCH)_$(ARCHBITS)$(WORKPATHSUFFIX)
ifeq ($(CONFIG_OPENCL_VERSION), )
CONFIG_OPENCL_VERSION			= All
endif
ifeq ($(COMPILER_FLAGS), DBG)
INTELFLAGSUSE					= $(INTELFLAGSDBG)
VSNETFLAGSUSE					= $(VSNETFLAGSDBG)
GCCFLAGSUSE						= $(GCCFLAGSDBG)
CLANGFLAGSUSE					= $(CLANGFLAGSDBG)
NVCCFLAGSUSE					= $(NVCCFLAGSDBG)
OPENCL_DEF_OPTIONS				= "-O0" #"-g"
CONFIG_LTO						= 0
endif
ifeq ($(COMPILER_FLAGS), OPT)
INTELFLAGSUSE					= $(INTELFLAGSOPT)
VSNETFLAGSUSE					= $(VSNETFLAGSOPT)
GCCFLAGSUSE						= $(GCCFLAGSOPT)
CLANGFLAGSUSE					= $(CLANGFLAGSOPT)
NVCCFLAGSUSE					= $(NVCCFLAGSOPT)
OPENCL_DEF_OPTIONS				= "-O3"
DEFINES							+= NDEBUG
endif
ifeq ($(GCCCUDA), )
GCCCUDA							= $(GCC3264)
endif
CCCUDAOPTS						+= -Wno-parentheses

MKDIR							= $(HIDEECHOB) mkdir -p `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'` `echo $@ | sed 's,/[a-zA-Z0-9._-]*$$,,'`

ALLDEP							+= makefiles/makefile $(CONFIGFILE) makefiles/$(ARCHFILE) config.mak
include							makefiles/$(ARCHFILE)
ifneq ($(LINK_OVERRIDE), )
LINK							= $(LINK_OVERRIDE)
endif
HIPCC							= hipcc
ifneq ($(HIP_TARGET), )
HIPCC							+= --amdgpu-target=$(HIP_TARGET)
endif

GCCFLAGSARCH					+= $(GCCARCHA)
ifneq ($(CONFIG_CPP), )
GCCFLAGSCOMMON					+= -std=$(CONFIG_CPP)
CLANGFLAGSCOMMON				+= -std=$(CONFIG_CPP)
else
ifeq ($(CONFIG_CPP11), 1)
GCCFLAGSCOMMON					+= -std=c++11
CLANGFLAGSCOMMON				+= -std=c++11
endif
endif
ifneq ($(CONFIG_CPP_CUDA), )
NVCCFLAGSOPT					+= -Xcompiler "--std $(CONFIG_CPP_CUDA)"
NVCCFLAGSDBG					+= -Xcompiler "--std $(CONFIG_CPP_CUDA)"
endif

ifeq ($(CONFIG_OPENMP), 1)
GCCFLAGSCOMMON					+= -fopenmp
CLANGFLAGSCOMMON				+= -fopenmp
GCCLINK							+= -fopenmp
endif

ifeq ($(TARGETPATH), )
COPIED_EXECUTABLE				= $(EXECUTABLE)
TARGETPATH						= .
else
COPIED_EXECUTABLE				= $(TARGETPATH)$(EXECUTABLE)
$(COPIED_EXECUTABLE)			: $(EXECUTABLE)
								cp $(EXECUTABLE) $(COPIED_EXECUTABLE)
endif

all_tmp:						$(SUBTARGETS:%=subbuild/%.mak) $(COPIED_EXECUTABLE)

main:							$(COPIED_EXECUTABLE)

run								: all
								cd $(TARGETPATH) && ./$(EXECUTABLE)


subbuild/%.mak:
								+$(MAKE) CONFIGFILE=config_`echo $@ | sed s,subbuild/,,` BUILDSCRIPT=config_`echo $@ | sed s,subbuild/,, | sed s,mak,sh,` -f makefile

CUDAINCLUDEPATHS				= --compiler-options '$(INCLUDEPATHSUSE)'
CUDADEFINES						= $(DEFINESUSE)

DEPENDS							:= $(CUFILES:%.cu=$(WORKPATH)/cu/%.d) $(CLFILES:%.cl=$(WORKPATH)/cl/%.d) $(ASMFILES:%.asm=$(WORKPATH)/asm/%.d) $(CPPFILES_DBG:%.cpp=$(WORKPATH)/dbg/%.d) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/vcc/%.d) \
									$(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.d) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.d) $(CFILES:%.c=$(WORKPATH)/c/%.d) \
									$(CPPFILES_MSCC:%.cpp=$(WORKPATH)/mscc/%.d) $(CPPFILES_CLANG:%.cpp=$(WORKPATH)/clang/%.d) $(CPPFILES_ICC:%.cpp=$(WORKPATH)/icc/%.d) $(CPPFILES_GCC:%.cpp=$(WORKPATH)/gcc/%.d) $(QTFILES:%.ui=$(WORKPATH)/qt/%.qtd) \
									$(HIPFILES:%.hip.cxx=$(WORKPATH)/hip/%.d)
CPPFILES						+= $(QTFILES:%.ui=$(WORKPATH)/cpp/%.$(OBJ)) $(QTCPPFILES)

OBJFILES						:= $(CUFILES:%.cu=$(WORKPATH)/cu/%.$(OBJ)) $(ASMFILES:%.asm=$(WORKPATH)/asm/%.$(OBJ)) $(CPPFILES_DBG:%.cpp=$(WORKPATH)/dbg/%.$(OBJ)) $(CPPFILES_VCC:%.cpp=$(WORKPATH)/vcc/%.$(OBJ)) $(RESOURCEFILES:%=$(WORKPATH)/res/%.$(OBJ)) \
 									$(CLFILES:%.cl=$(WORKPATH)/cl/%.$(OBJ)) $(QTFILES:%.ui=$(WORKPATH)/qt/%_moc.$(OBJ)) $(QTCPPFILES:%.cpp=$(WORKPATH)/qt/%_moccpp.$(OBJ)) $(HIPFILES:%.hip.cxx=$(WORKPATH)/hip/%.$(OBJ))

CPPFILES_ICC					:= $(CPPFILES_ICC:%.cpp=$(WORKPATH)/icc/%.$(OBJ))
CPPFILES_GCC					:= $(CPPFILES_GCC:%.cpp=$(WORKPATH)/gcc/%.$(OBJ))
CPPFILES_MSCC					:= $(CPPFILES_MSCC:%.cpp=$(WORKPATH)/mscc/%.$(OBJ))
CPPFILES_CLANG					:= $(CPPFILES_CLANG:%.cpp=$(WORKPATH)/clang/%.$(OBJ))
ifeq ($(CC_SELECTED), ICC)
CPPFILES_ICC					+= $(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.$(OBJ)) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.$(OBJ)) $(CFILES:%.c=$(WORKPATH)/c/%.$(OBJ))
CCWITHLINK						= /link
endif
ifeq ($(CC_SELECTED), MSCC)
CPPFILES_MSCC					+= $(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.$(OBJ)) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.$(OBJ)) $(CFILES:%.c=$(WORKPATH)/c/%.$(OBJ))
CCWITHLINK						= /link
endif
ifeq ($(CC_SELECTED), GCC)
CPPFILES_GCC					+= $(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.$(OBJ)) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.$(OBJ)) $(CFILES:%.c=$(WORKPATH)/c/%.$(OBJ))
endif
ifeq ($(CC_SELECTED), clang)
CPPFILES_CLANG					+= $(CPPFILES:%.cpp=$(WORKPATH)/cpp/%.$(OBJ)) $(CXXFILES:%.cxx=$(WORKPATH)/cxx/%.$(OBJ)) $(CFILES:%.c=$(WORKPATH)/c/%.$(OBJ))
endif

ifeq ($(CONFIG_LTO), 1)
ifneq ("0$(CPPFILES_ICC)", "0")
OBJFILES						+= $(WORKPATH)/make_lto_icc/icclto_$(TARGET).$(OBJ)
endif
OBJFILES						+= $(CPPFILES_MSCC) $(CPPFILES_GCC) $(CPPFILES_CLANG)
else
OBJFILES						+= $(CPPFILES_ICC) $(CPPFILES_MSCC) $(CPPFILES_GCC) $(CPPFILES_CLANG)
endif

ifneq ($(CONFIG_GDB), 0)
GCCFLAGSOPT						+= -ggdb
GCCFLAGSDBG						+= -ggdb
CLANGFLAGSOPT					+= -ggdb
endif

ifeq ($(CONFIG_CUDA_DC), 1)
OBJFILES						+= $(WORKPATH)/make_cuda_device_link/dl_$(TARGET).$(OBJ)
NVCCFLAGSDC						= -dc
endif

ifeq ($(ARCH_CYGWIN), 1)
ifeq ($(INCLUDEPATHS)$(COMMONINCLUDEPATHS), )
GCCINCLUDEPATHS					=
else
GCCINCLUDEPATHSA				:= $(shell cygpath -u $(INCLUDEPATHS))
GCCINCLUDEPATHSB				:= $(shell cygpath -u $(COMMONINCLUDEPATHS))
GCCINCLUDEPATHS					:= $(GCCINCLUDEPATHSA:%=-I%) $(GCCINCLUDEPATHSB:%=-isystem%)
endif
else
GCCINCLUDEPATHS					:= $(INCLUDEPATHS:%=-I%) $(COMMONINCLUDEPATHS:%=-isystem %) $(INCLUDEPATHSSYSTEM:%=-isystem %)
endif
VSINCLUDEPATHS					:= $(INCLUDEPATHS:%=/I%) $(COMMONINCLUDEPATHS:%=/I%)

ifeq ($(ARCHBITS), 64)
DEFINES							+= _64BIT
endif
GCCDEFINES						:= $(DEFINES:%=-D%) $(DEFINESARCH:%=-D%)
VSDEFINESTMP					:= $(DEFINES:%=/D%) $(DEFINESARCH:%=/D%)
VSDEFINES						:= $(subst =, BAT_SPECIAL_EQ ,$(VSDEFINESTMP))

LIBFILES						= $(LIBSUSE)

.SECONDARY:						$(CUFILES:%.cu=$(WORKPATH)/cu/%.cpp) $(ASMFILES:%.asm=$(WORKPATH)/asm/%.cpp) $(RESOURCEFILES:%=$(WORKPATH)/res/%.$(OBJ)) $(CLFILES:%.cl=$(WORKPATH)/cl/%.$(OBJ)) $(QTFILES:%.ui=$(WORKPATH)/qt/%_moc.$(OBJ)) \
									$(QTFILES:%.ui=$(WORKPATH)/qt/%_moc.cpp) $(QTFILES:%.ui=$(WORKPATH)/qt/%_ui.h) $(CLFILES:%.cl=$(WORKPATH)/cl/%.clbin)

$(EXECUTABLE):					$(EXTRADEPS) $(OBJFILES) $(EXTRAOBJFILES) $(ALLDEP)
								$(LINK) $(EXTRAFLAGSLINK) $(LIBPATHSUSE) $(OBJFILES) $(EXTRAOBJFILES) $(LIBFILES) $(LINKOUTPUT) $(LINKTARGETTYPE)
								$(HIDEECHOA) if [ -e "$(BUILDSCRIPT)" ]; then ./$(BUILDSCRIPT); fi

$(WORKPATH)/cpp/%.$(OBJ):		%.cpp $(ALLDEP)
								$(MKDIR)
								$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)

$(WORKPATH)/cxx/%.$(OBJ):		%.cxx $(ALLDEP)
								$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)

$(WORKPATH)/c/%.$(OBJ):			%.c $(ALLDEP)
								$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)

$(WORKPATH)/cu/%.$(OBJ):		$(WORKPATH)/cu/%.cpp $(ALLDEP)
								$(CCCUDA) $(CCCUDAOPTS) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)

$(WORKPATH)/cu/%.cpp:			%.cu $(ALLDEP)
								$(NVCC) $(NVCCFLAGSUSE) $(NVCCFLAGSDC) $(CUDAINCLUDEPATHS) $(CUDADEFINES) $(FILEFLAGS$<) --cuda --output-file "$@" $<
								
$(WORKPATH)/hip/%.$(OBJ):		%.hip.cxx $(ALLDEP)
								$(HIPCC) $(CLANGFLAGSUSE) $(CLANGFLAGSCOMMON) $(GCCINCLUDEPATHS) $(GCCDEFINES) $(COMPILETARGETTYPE) $(FILEFLAGS$<) -Wno-unused-command-line-argument -Wno-invalid-constexpr -Wno-unused-private-field -c $< -o $@

$(WORKPATH)/asm/%.$(OBJ):		$(WORKPATH)/asm/%.asm $(ALLDEP)
								$(ASM) $(COMPILEOUTPUT) $(ASMONLY) $<

$(WORKPATH)/asm/%.asm:			%.asm $(ALLDEP)
								$(CC) $(PRECOMPILEONLY) $(FILEFLAGS$<) $(DEFINESUSE) $< > $@

$(WORKPATH)/dbg/%.$(OBJ):		%.cpp $(ALLDEP)
								$(CCDBG) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/icc/%.$(OBJ):		%.cpp $(ALLDEP)
								$(ICC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/vcc/%.$(OBJ):		%.cpp $(ALLDEP)
								$(VCC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $<
$(WORKPATH)/gcc/%.$(OBJ):		%.cpp $(ALLDEP)
								$(GCC) $(GCCINCLUDEPATHS) $(GCCDEFINES) $(FILEFLAGS$<) -c $< -o $@
$(WORKPATH)/mscc/%.$(OBJ):		%.cpp $(ALLDEP)
								$(MSCC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/clang/%.$(OBJ):		%.cpp $(ALLDEP)
								$(CLANG) $(GCCINCLUDEPATHS) $(GCCDEFINES) $(FILEFLAGS$<) -c $< -o $@


$(WORKPATH)/qt/%_ui.h:			%.ui $(ALLDEP)
								$(MKDIR)
								$(QTUIC) -o $@ $<

$(WORKPATH)/qt/%_moc.cpp:		%.h $(WORKPATH)/qt/%_ui.h $(ALLDEP)
								$(QTMOC) -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -o $@ $<

$(WORKPATH)/qt/%_moccpp.cpp:	%.h $(ALLDEP)
								$(QTMOC) -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -o $@ $<

$(WORKPATH)/qt/%.$(OBJ):		$(WORKPATH)/qt/%.cpp $(ALLDEP)
								$(CC) $(INCLUDEPATHSUSE) $(DEFINESUSE) $(FILEFLAGS$<) $(COMPILEONLY) $< $(COMPILEOUTPUT)

$(WORKPATH)/_makefiles_support_opencl.$(OBJ):	makefiles/makefile_opencl_compiler.cxx $(ALLDEP)
								$(HIDEECHOB) mkdir -p $(WORKPATH)
								$(CC) $(INCLUDEPATHSUSE) $(COMPILEONLY) $< $(COMPILEOUTPUT)
$(WORKPATH)/_makefiles_support_opencl.exe:	$(WORKPATH)/_makefiles_support_opencl.$(OBJ) $(ALLDEP)
								$(LINK) $(LIBPATHSUSE) $< $(LINKOUTPUT) $(LINKFLAGSARCH) $(OPENCLLIB)

$(WORKPATH)/cl/%.clbin:			%.cl $(WORKPATH)/_makefiles_support_opencl.exe $(ALLDEP)
								$(HIDEECHOA) $(OPENCL_ENVIRONMENT) ./$(WORKPATH)/_makefiles_support_opencl.exe -output-file $@ $< -- -I. $(GCCINCLUDEPATHS:-isystem=-I) $(GCCDEFINES) $(OPENCL_DEF_OPTIONS) $(OPENCL_OPTIONS) > /dev/null

$(WORKPATH)/cl/%.$(OBJ):		$(WORKPATH)/cl/%.clbin $(ALLDEP)
								@echo $<
								$(HIDEECHOA) sed -e "s&FILENAMENORMAL&$<&g" -e "s&FILENAMEMOD&_makefile_opencl_program_`echo $< | sed -e "s&$(WORKPATH)/cl/&&g" -e "s&\.clbin&\.cl&g" -e "s&\.&_&g" -e "s&/&_&g" -e "s& &_&g"`&g" makefiles/include.S | $(GCC3264) -c -x assembler -o $@ -
$(WORKPATH)/res/%.$(OBJ):		% $(ALLDEP)
								@echo $<
								$(MKDIR)
								$(HIDEECHOA) sed -e "s&FILENAMENORMAL&$<&g" -e "s&FILENAMEMOD&_resource_`echo $< | sed -e "s&\.&_&g" -e "s&/&_&g" -e "s& &_&g"`&g" makefiles/include.S | $(GCC3264) -c -x assembler -o $@ -



$(WORKPATH)/make_lto_icc/icclto_$(TARGET).$(OBJ):	$(CPPFILES_ICC) $(ALLDEP)
								$(HIDEECHOA) mkdir -p $(WORKPATH)/make_lto_icc
								$(ICCLINK) $(INTELLINKIPO)$@ $(CPPFILES_ICC)

$(WORKPATH)/make_cuda_device_link/dl_$(TARGET).$(OBJ):	$(CUFILES:%.cu=$(WORKPATH)/cu/%.$(OBJ)) $(ALLDEP)
								$(HIDEECHOA) mkdir -p $(WORKPATH)/make_cuda_device_link
								$(NVCC) $(NVCCFLAGSUSE) -dlink $(CUFILES:%.cu=$(WORKPATH)/cu/%.$(OBJ)) -o $@ -lcudadevrt

clean:							$(SUBTARGETS:%=subclean/%.mak)
								rm -Rf *.plg *.dpi *.exp *.lib $(EXECUTABLE) $(COPIED_EXECUTABLE) x64/release/* *.cubin *.gpu *.ptx *.linkinfo *.ii cuda.compute_* $(DEPENDS) $(OBJFILES) $(CUFILES:%.cu=$(WORKPATH)/cu/%.cpp) $(ASMFILES:%.asm=$(WORKPATH)/asm/%.asm) $(CLEANRELEASEDIR) $(SUBTARGETS_CLEAN)

subclean/%.mak:
								+export CONFIGFILE=config_`echo $@ | sed s,subclean/,,` && $(MAKE) -f makefile clean

SAVEDIR							= releases/`date +%F`-BUILD-`cat buildnr`
backup:
								mkdir $(SAVEDIR)
								cp *.cpp *.h makefile buildnr *.sh *.bat *.conf *.cu $(SAVEDIR)

#Dependencies

DEPMKDIR						= $(MKDIR) &&
DEPGCC							= $(GCC3264) $(GCCFLAGSARCH) $(GCCINCLUDEPATHS) $(GCCDEFINES) -x c++ -fPIC -MM $<
DEPSED1							= sed -e ':a;N;$$!ba;s/\n/ /g' | sed -e
DEPSED2							= -e 's,\\,/,g' \
								-e 's,[cC]:/,/cygdrive/c/,g' > \
								$@;


$(WORKPATH)/qt/%.qtd:			%.cpp %.ui $(WORKPATH)/qt/%_ui.h $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/cpp/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/cl/%.d:				%.cl $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -I$(GCCPATH)/include -D__OPENCL__ | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/cl/$*.clbin $(WORKPATH)/cl/$*.cl $@ : ,g' $(DEPSED2)

$(WORKPATH)/cu/%.d:				%.cu $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -I$(GCCPATH)/include -D_MSC_VER=1700 -D__CUDACC__ | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/cu/$*.$(OBJ) $(WORKPATH)/cu/$*.cpp $@ : ,g' $(DEPSED2)

$(WORKPATH)/hip/%.d:			%.hip.cxx $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) `hipconfig -C` | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/hip/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/asm/%.d:			%.asm $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/asm/$*.$(OBJ) $(WORKPATH)/asm/$*.asm $@ : ,g' $(DEPSED2)

$(WORKPATH)/cpp/%.d:			%.cpp $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/cpp/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/cxx/%.d:			%.cxx $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/cxx/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/c/%.d:				%.c $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/c/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/dbg/%.d:			%.cpp $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -DDEBUG_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/dbg/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/icc/%.d:			%.cpp $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -DINTEL_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/icc/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/vcc/%.d:			%.cpp $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -DVECTORC_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/vcc/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/gcc/%.d:			%.cpp $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -DGCC_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/gcc/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/mscc/%.d:			%.cpp $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -DVSNET_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/mscc/$*.$(OBJ) $@ : ,g' $(DEPSED2)

$(WORKPATH)/clang/%.d:			%.cpp $(ALLDEP)
								$(DEPMKDIR) $(DEPGCC) -DCLANG_RUNTIME | $(DEPSED1) 's,^[a-zA-Z0-9._-]*[ ]*:,$(WORKPATH)/clang/$*.$(OBJ) $@ : ,g' $(DEPSED2)

ifneq ($(CHECK_DEPENDENCIES), 0)
include $(DEPENDS)
endif
