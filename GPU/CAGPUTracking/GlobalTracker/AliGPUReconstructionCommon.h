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

#include "AliGPUTPCDef.h"
#include "AliGPUTPCGPUConfig.h"
#include "AliCAGPULogging.h"

#include <iostream>
#include <fstream>

#if defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_GPULIBRARY)
#include "AliHLTDefinitions.h"
#include "AliHLTSystem.h"
#endif

#include "TPCFastTransform.h"

#ifdef HAVE_O2HEADERS
#include "ITStracking/TrackerTraitsCPU.h"
#include "TRDBase/TRDGeometryFlat.h"
#else
namespace o2 { namespace ITS { class TrackerTraits {}; class TrackerTraitsCPU : public TrackerTraits {}; }}
namespace o2 { namespace trd { class TRDGeometryFlat {public: void clearInternalBufferUniquePtr() {}}; }}
#endif
using namespace o2::ITS;
using namespace o2::trd;

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1
