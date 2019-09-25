// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  static const unsigned int NSLICES = 36;
  static const unsigned int NPATCHES = 6;

  struct AliHLTTPCTrackHelperStruct {
    int fID;
    const AliHLTExternalTrackParam* fTrack;
    float fResidualPad;
    float fResidualTime;
    bool fFirstHit;
    long long int fAverageQMax;
    long long int fAverageQTot;
  };

  // interface methods of base class
  const char* GetComponentID() { return "TPCClusterStat"; };
  void GetInputDataTypes(AliHLTComponentDataTypeList& list);
  AliHLTComponentDataType GetOutputDataType();
  void GetOutputDataSize(unsigned long& constBase, double& inputMultiplier);
  AliHLTComponent* Spawn() { return new AliHLTTPCClusterStatComponent; }

  static void TransformReverse(int slice, int row, float y, float z, float padtime[]);
  static void TransformForward(int slice, int row, float pad, float time, float xyz[]);

  void PrintDumpClustersScaled(int is, int ip, AliHLTTPCRawCluster& cluster, AliHLTTPCClusterXYZ& clusterTransformed, AliHLTTPCTrackHelperStruct& clusterTrack);

 protected:
  // interface methods of base class
  int DoInit(int argc, const char** argv);
  int DoDeinit();
  int DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t& size, AliHLTComponentBlockDataList& outputBlocks);

  using AliHLTProcessor::DoEvent;
  int ProcessOption(TString option, TString value);

 private:
  /** copy constructor prohibited */
  AliHLTTPCClusterStatComponent(const AliHLTTPCClusterStatComponent&);
  /** assignment operator prohibited */
  AliHLTTPCClusterStatComponent& operator=(const AliHLTTPCClusterStatComponent&);

  GPUCA_NAMESPACE::gpu::GPUParam* mSliceParam;

  int fTotal, fEdge, fSplitPad, fSplitTime, fSplitPadTime, fSplitPadOrTime, fAssigned; //!

  int fCompressionStudy;    //!
  int fPrintClusters;       //!
  int fPrintClustersScaled; //!
  int fDumpClusters;        //!
  int fAggregate;           //!
  int fSort;                //!
  int fEvent;

  FILE* fp;

 protected:
  ClassDef(AliHLTTPCClusterStatComponent, 0);
};
#endif
