#ifndef CLUSTERNATIVEACCESSEXT_H
#define CLUSTERNATIVEACCESSEXT_H

#include "AliGPUTPCSettings.h"
#include "AliGPUTPCGPUConfig.h"

#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/ClusterNative.h"
#else
namespace o2 { namespace TPC { struct ClusterNative {}; struct ClusterNativeAccessFullTPC {const ClusterNative* clusters[36][GPUCA_ROW_COUNT]; unsigned int nClusters[36][GPUCA_ROW_COUNT];}; }}
#endif

struct ClusterNativeAccessExt : public o2::TPC::ClusterNativeAccessFullTPC
{
	unsigned int clusterOffset[36][GPUCA_ROW_COUNT];
};

#endif
