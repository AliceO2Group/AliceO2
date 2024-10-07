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

/// \file AliHLTTPCClusterStatComponent.h
/// \author David Rohr

#ifndef GPUTPCCLUSTERSTAT_H
#define GPUTPCCLUSTERSTAT_H

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"
#include "AliOptionParser.h"

class AliHLTExternalTrackParam;
class AliHLTTPCRawCluster;
class AliHLTTPCClusterXYZ;
namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUParam;
}
} // namespace GPUCA_NAMESPACE

class AliHLTTPCClusterStatComponent : public AliHLTProcessor, public AliOptionParser
{
 public:
  /** standard constructor */
  AliHLTTPCClusterStatComponent();
  /** destructor */
  virtual ~AliHLTTPCClusterStatComponent();

  static const uint32_t NSLICES = 36;
  static const uint32_t NPATCHES = 6;

  struct AliHLTTPCTrackHelperStruct {
    int32_t fID;
    const AliHLTExternalTrackParam* fTrack;
    float fResidualPad;
    float fResidualTime;
    bool fFirstHit;
    int64_t fAverageQMax;
    int64_t fAverageQTot;
  };

  // interface methods of base class
  const char* GetComponentID() { return "TPCClusterStat"; };
  void GetInputDataTypes(AliHLTComponentDataTypeList& list);
  AliHLTComponentDataType GetOutputDataType();
  void GetOutputDataSize(uint64_t& constBase, double& inputMultiplier);
  AliHLTComponent* Spawn() { return new AliHLTTPCClusterStatComponent; }

  static void TransformReverse(int32_t slice, int32_t row, float y, float z, float padtime[]);
  static void TransformForward(int32_t slice, int32_t row, float pad, float time, float xyz[]);

  void PrintDumpClustersScaled(int32_t is, int32_t ip, AliHLTTPCRawCluster& cluster, AliHLTTPCClusterXYZ& clusterTransformed, AliHLTTPCTrackHelperStruct& clusterTrack);

 protected:
  // interface methods of base class
  int32_t DoInit(int argc, const char** argv);
  int32_t DoDeinit();
  int32_t DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t& size, AliHLTComponentBlockDataList& outputBlocks);

  using AliHLTProcessor::DoEvent;
  int32_t ProcessOption(TString option, TString value);

 private:
  /** copy constructor prohibited */
  AliHLTTPCClusterStatComponent(const AliHLTTPCClusterStatComponent&);
  /** assignment operator prohibited */
  AliHLTTPCClusterStatComponent& operator=(const AliHLTTPCClusterStatComponent&);

  GPUCA_NAMESPACE::gpu::GPUParam* mSliceParam;

  int32_t fTotal, fEdge, fSplitPad, fSplitTime, fSplitPadTime, fSplitPadOrTime, fAssigned; //!

  int32_t fCompressionStudy;    //!
  int32_t fPrintClusters;       //!
  int32_t fPrintClustersScaled; //!
  int32_t fDumpClusters;        //!
  int32_t fAggregate;           //!
  int32_t fSort;                //!
  int32_t fEvent;

  FILE* fp;

 protected:
  ClassDef(AliHLTTPCClusterStatComponent, 0);
};
#endif
