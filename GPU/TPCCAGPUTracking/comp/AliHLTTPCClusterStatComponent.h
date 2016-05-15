#ifndef ALIHLTTPCCLUSTERSTAT_H
#define ALIHLTTPCCLUSTERSTAT_H
//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"
#include "AliZMQhelpers.h"

class AliHLTTPCClusterStatComponent : public AliHLTProcessor, public AliOptionParser
{
 public:
  /** standard constructor */
  AliHLTTPCClusterStatComponent();
  /** destructor */
  virtual ~AliHLTTPCClusterStatComponent();

  // interface methods of base class
  const char* GetComponentID() {return "TPCClusterStat";};
  void GetInputDataTypes(AliHLTComponentDataTypeList& list);
  AliHLTComponentDataType GetOutputDataType();
  void GetOutputDataSize(unsigned long& constBase, double& inputMultiplier);
  AliHLTComponent* Spawn() {return new AliHLTTPCClusterStatComponent;}

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

  int fTotal, fSplitPad, fSplitTime, fSplitPadTime, fSplitPadOrTime; //!

  int fPrintClusters; //!
  int fPrintClustersScaled; //!

protected:

  ClassDef(AliHLTTPCClusterStatComponent, 0)
};
#endif
