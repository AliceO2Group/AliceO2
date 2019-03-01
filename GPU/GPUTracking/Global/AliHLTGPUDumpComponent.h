// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliHLTGPUDumpComponent.h
/// \author David Rohr

#ifndef ALIHLTGPUDUMPCOMPONENT_H
#define ALIHLTGPUDUMPCOMPONENT_H

#include "GPUCommonDef.h"
#include "AliHLTProcessor.h"

class GPUReconstruction;
class GPUChainTracking;
class GPUTPCClusterData;
class AliTPCcalibDB;
class AliTPCRecoParam;
#include "AliRecoParam.h"
class AliTPCTransform;
namespace ali_tpc_common { namespace tpc_fast_transformation {
	class TPCFastTransform;
	class TPCFastTransformManager;
}}

class AliHLTGPUDumpComponent : public AliHLTProcessor
{
  public:
	static const unsigned int NSLICES = 36;
	static const unsigned int NPATCHES = 6;
	
	AliHLTGPUDumpComponent();

	AliHLTGPUDumpComponent( const AliHLTGPUDumpComponent& ) CON_DELETE;
	AliHLTGPUDumpComponent& operator=( const AliHLTGPUDumpComponent& ) CON_DELETE;

	virtual ~AliHLTGPUDumpComponent();

	const char* GetComponentID() ;
	void GetInputDataTypes( vector<AliHLTComponentDataType>& list )  ;
	AliHLTComponentDataType GetOutputDataType() ;
	virtual void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier ) ;
	AliHLTComponent* Spawn() ;

  protected:
	int DoInit( int argc, const char** argv );
	int DoDeinit();
	int Reconfigure( const char* cdbEntry, const char* chainId );
	int DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks,
		AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr,
		AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks );

  private:

	float fSolenoidBz;
	GPUReconstruction* fRec;
	GPUChainTracking* fChain;
	GPUTPCClusterData* fClusterData[NSLICES];
	ali_tpc_common::tpc_fast_transformation::TPCFastTransformManager* fFastTransformManager;
	AliTPCcalibDB* fCalib;
	AliTPCRecoParam* fRecParam;
	AliRecoParam fOfflineRecoParam;
	AliTPCTransform* fOrigTransform;
	bool fIsMC;
};

#endif
