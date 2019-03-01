#ifndef GPURECONSTRUCTIONINCLUDES_H
#define GPURECONSTRUCTIONINCLUDES_H

//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#ifndef WIN32
#include <sys/syscall.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sched.h>
#endif

#include "GPUTPCDef.h"
#include "GPUTPCGPUConfig.h"
#include "GPULogging.h"

#include <iostream>
#include <fstream>

#if defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_GPULIBRARY)
#include "AliHLTDefinitions.h"
#include "AliHLTSystem.h"
#endif

#include "GPUReconstructionIncludesITS.h"

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

#endif
