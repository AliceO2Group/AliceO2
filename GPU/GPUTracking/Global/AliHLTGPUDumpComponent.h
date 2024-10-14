// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

class AliTPCcalibDB;
class AliTPCRecoParam;
#include "AliRecoParam.h"
class AliTPCTransform;
namespace GPUCA_NAMESPACE
{
namespace gpu
{
class TPCFastTransform;
class TPCFastTransformManager;
class GPUReconstruction;
class GPUChainTracking;
class GPUTPCClusterData;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

class AliHLTGPUDumpComponent : public AliHLTProcessor
{
 public:
  static const uint32_t NSLICES = 36;
  static const uint32_t NPATCHES = 6;

  AliHLTGPUDumpComponent();

  AliHLTGPUDumpComponent(const AliHLTGPUDumpComponent&) CON_DELETE;
  AliHLTGPUDumpComponent& operator=(const AliHLTGPUDumpComponent&) CON_DELETE;

  virtual ~AliHLTGPUDumpComponent();

  const char* GetComponentID();
  void GetInputDataTypes(vector<AliHLTComponentDataType>& list);
  AliHLTComponentDataType GetOutputDataType();
  virtual void GetOutputDataSize(uint64_t& constBase, double& inputMultiplier);
  AliHLTComponent* Spawn();

 protected:
  int32_t DoInit(int argc, const char** argv);
  int32_t DoDeinit();
  int32_t Reconfigure(const char* cdbEntry, const char* chainId);
  int32_t DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks);

 private:
  float fSolenoidBz;
  GPUCA_NAMESPACE::gpu::GPUReconstruction* fRec;
  GPUCA_NAMESPACE::gpu::GPUChainTracking* fChain;
  GPUCA_NAMESPACE::gpu::TPCFastTransformManager* fFastTransformManager;
  AliTPCcalibDB* fCalib;
  AliTPCRecoParam* fRecParam;
  AliRecoParam fOfflineRecoParam;
  AliTPCTransform* fOrigTransform;
  bool fIsMC;
  int64_t fInitTimestamp;
};

#endif
