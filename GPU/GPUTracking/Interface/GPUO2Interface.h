// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2Interface.h
/// \author David Rohr

#ifndef GPUO2INTERFACE_H
#define GPUO2INTERFACE_H

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

class GPUReconstruction;
class GPUChainTracking;
class GPUO2InterfaceConfiguration;
class GPUDisplayBackendGlfw;
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform;}}
namespace o2 { namespace TPC { struct ClusterNativeAccessFullTPC; struct ClusterNative;}}
#include <memory>

class GPUTPCO2Interface
{
public:
	GPUTPCO2Interface();
	~GPUTPCO2Interface();
	
	int Initialize(const GPUO2InterfaceConfiguration& config, std::unique_ptr<ali_tpc_common::tpc_fast_transformation::TPCFastTransform>&& fastTrans);
	int Initialize(const char* options, std::unique_ptr<ali_tpc_common::tpc_fast_transformation::TPCFastTransform>&& fastTrans);
	void Deinitialize();
	
	int RunTracking(const o2::TPC::ClusterNativeAccessFullTPC* inputClusters, const GPUTPCGMMergedTrack* &outputTracks, int &nOutputTracks, const GPUTPCGMMergedTrackHit* &outputTrackClusters);
	void Cleanup();
	
	bool GetParamContinuous() {return(fContinuous);}
	void GetClusterErrors2( int row, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const;

private:
	GPUTPCO2Interface(const GPUTPCO2Interface&);
	GPUTPCO2Interface &operator=( const GPUTPCO2Interface& );
	
	bool fInitialized = false;
	bool fDumpEvents = false;
	bool fContinuous = false;
	
	std::unique_ptr<GPUReconstruction> mRec;
	GPUChainTracking* mChain = nullptr;
	std::unique_ptr<GPUO2InterfaceConfiguration> mConfig;
	std::unique_ptr<GPUDisplayBackendGlfw> mDisplayBackend;
};

#endif
