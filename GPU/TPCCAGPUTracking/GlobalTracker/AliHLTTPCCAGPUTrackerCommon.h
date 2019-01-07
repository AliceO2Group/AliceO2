//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#ifdef WIN32
#else
#include <sys/syscall.h>
#include <semaphore.h>
#include <fcntl.h>
#endif
#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGPUConfig.h"

#if defined(GPUCA_STANDALONE) & !defined(WIN32)
#include <sched.h>
#endif

#include <iostream>
#include <fstream>

#include "MemoryAssignmentHelpers.h"

#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPULIBRARY)
#include "AliHLTDefinitions.h"
#include "AliHLTSystem.h"
#endif
