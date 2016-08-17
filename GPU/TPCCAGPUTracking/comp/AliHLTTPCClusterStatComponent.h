#ifndef ALIHLTTPCCLUSTERSTAT_H
#define ALIHLTTPCCLUSTERSTAT_H
//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"
#include "AliOptionParser.h"

class AliHLTTPCReverseTransformInfoV1;
class AliHLTExternalTrackParam;
class AliHLTTPCCAParam;
class AliHLTTPCRawCluster;
class AliHLTTPCClusterXYZ;

using namespace AliZMQhelpers;

class AliHLTTPCReverseTransformInfoV1;
class AliHLTExternalTrackParam;
class AliHLTTPCCAParam;
class AliHLTTPCRawCluster;
class AliHLTTPCClusterXYZ;

class AliHLTTPCClusterStatComponent : public AliHLTProcessor, public AliOptionParser
{
 public:
  /** standard constructor */
  AliHLTTPCClusterStatComponent();
  /** destructor */
  virtual ~AliHLTTPCClusterStatComponent();

  struct AliHLTTPCTrackHelperStruct
  {
    int fID;
    const AliHLTExternalTrackParam* fTrack;
    float fResidualPad;
    float fResidualTime;
    bool fFirstHit;
    long long int fAverageQMax;
    long long int fAverageQTot;
  };

  // interface methods of base class
  const char* GetComponentID() {return "TPCClusterStat";};
  void GetInputDataTypes(AliHLTComponentDataTypeList& list);
  AliHLTComponentDataType GetOutputDataType();
  void GetOutputDataSize(unsigned long& constBase, double& inputMultiplier);
  AliHLTComponent* Spawn() {return new AliHLTTPCClusterStatComponent;}

  static void TransformReverse(int slice, int row, float y, float z, float padtime[], const AliHLTTPCReverseTransformInfoV1* revInfo, bool applyCorrection = false);
  static void TransformForward(int slice, int row, float pad, float time, float xyz[], const AliHLTTPCReverseTransformInfoV1* revInfo, bool applyCorrection = false);
  
  void PrintDumpClustersScaled(int is, int ip, AliHLTTPCRawCluster &cluster, AliHLTTPCClusterXYZ &clusterTransformed, AliHLTTPCTrackHelperStruct &clusterTrack);

 protected:
  // interface methods of base class
  int DoInit(int argc, const char** argv);
  int DoDeinit();
  int DoEvent( const AliHLTComponentEventData& evtData,
		       const AliHLTComponentBlockData* blocks, 
		       AliHLTComponentTriggerData& trigData,
		       AliHLTUInt8_t* outputPtr, 
		       AliHLTUInt32_t& size,
		       AliHLTComponentBlockDataList& outputBlocks );

  using AliHLTProcessor::DoEvent;
  int ProcessOption(TString option, TString value);
  
  

 private:
  /** copy constructor prohibited */
  AliHLTTPCClusterStatComponent(const AliHLTTPCClusterStatComponent&);
  /** assignment operator prohibited */
  AliHLTTPCClusterStatComponent& operator=(const AliHLTTPCClusterStatComponent&);
  
  AliHLTTPCCAParam* fSliceParam;
  float fPolinomialFieldBz[6];

  int fTotal, fSplitPad, fSplitTime, fSplitPadTime, fSplitPadOrTime, fAssigned; //!

  int fPrintClusters; //!
  int fPrintClustersScaled; //!
  int fDumpClusters; //!
  int fAggregate; //!
  int fEvent;
  
  FILE* fp;

protected:

  ClassDef(AliHLTTPCClusterStatComponent, 0)
};
#endif
