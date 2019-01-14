//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAO2INTERFACE_H
#define ALIHLTTPCCAO2INTERFACE_H

//Some defines denoting that we are compiling for O2
#ifndef GPUCA_O2_LIB
#define GPUCA_O2_LIB
#endif
#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif

class AliGPUReconstruction;
class AliGPUCAConfiguration;
#include "AliGPUTPCGMMergedTrack.h"
#include "AliGPUTPCGMMergedTrackHit.h"
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform;}}
namespace o2 { namespace TPC { struct ClusterNativeAccessFullTPC; struct ClusterNative;}}
#include <memory>
#include "AliGPUTPCClusterData.h"

class AliGPUTPCO2Interface
{
public:
	AliGPUTPCO2Interface();
	~AliGPUTPCO2Interface();
	
	int Initialize(const AliGPUCAConfiguration& config, std::unique_ptr<ali_tpc_common::tpc_fast_transformation::TPCFastTransform>&& fastTrans);
	int Initialize(const char* options, std::unique_ptr<ali_tpc_common::tpc_fast_transformation::TPCFastTransform>&& fastTrans);
	void Deinitialize();
	
	int RunTracking(const o2::TPC::ClusterNativeAccessFullTPC* inputClusters, const AliGPUTPCGMMergedTrack* &outputTracks, int &nOutputTracks, const AliGPUTPCGMMergedTrackHit* &outputTrackClusters);
	void Cleanup();
	
	bool GetParamContinuous() {return(fContinuous);}
	void GetClusterErrors2( int row, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const;

private:
	AliGPUTPCO2Interface(const AliGPUTPCO2Interface&);
	AliGPUTPCO2Interface &operator=( const AliGPUTPCO2Interface& );
	
	bool fInitialized;
	bool fDumpEvents;
	bool fContinuous;
	
	std::unique_ptr<AliGPUReconstruction> mRec;
	std::unique_ptr<AliGPUCAConfiguration> mConfig;
};

#endif
