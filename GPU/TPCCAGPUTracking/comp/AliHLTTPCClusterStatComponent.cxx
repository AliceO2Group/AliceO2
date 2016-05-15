// $Id$

//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Mikolaj Krzewicki <mikolaj.krzewicki@cern.ch>         *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

//  @file   AliHLTTPCClusterStatComponent.cxx
//  @author David Rohr <drohr@cern.ch>
// 

#include "AliHLTTPCClusterStatComponent.h"
#include "AliHLTTPCDefinitions.h"
#include "AliHLTTPCRawCluster.h"

ClassImp(AliHLTTPCClusterStatComponent)

AliHLTTPCClusterStatComponent::AliHLTTPCClusterStatComponent() : AliHLTProcessor()
, fTotal(0)
, fSplitPad(0)
, fSplitTime(0)
, fSplitPadTime(0)
, fSplitPadOrTime(0)
, fPrintClusters(0)
, fPrintClustersScaled(0)
, fDumpClusters(0)
{
}

AliHLTTPCClusterStatComponent::~AliHLTTPCClusterStatComponent()
{
}

void AliHLTTPCClusterStatComponent::GetInputDataTypes(AliHLTComponentDataTypeList& list)
{
  list.push_back(AliHLTTPCDefinitions::fgkRawClustersDataType | kAliHLTDataOriginTPC);
}

AliHLTComponentDataType AliHLTTPCClusterStatComponent::GetOutputDataType()
{
  return kAliHLTDataTypeHistogram|kAliHLTDataOriginOut;
}

void AliHLTTPCClusterStatComponent::GetOutputDataSize(unsigned long& constBase, double& inputMultiplier)
{
    constBase = 2000000;
    inputMultiplier = 0.0;
}

int AliHLTTPCClusterStatComponent::ProcessOption(TString option, TString value)
{
    int iResult = 0;

    if (option.EqualTo("print-clusters"))
    {
	fPrintClusters = 1;
    }
    else if (option.EqualTo("print-clusters-scaled"))
    {
	fPrintClustersScaled = 1;
    }
    else if (option.EqualTo("dump-clusters"))
    {
	fDumpClusters = 1;
    }
    else
    {
        HLTError("invalid option: %s", value.Data());
        return -EINVAL;
    }
    return iResult;
}

int AliHLTTPCClusterStatComponent::DoInit(int argc, const char** argv)
{
    int iResult=0;

    if (ProcessOptionString(GetComponentArgs())<0)
    {
        HLTFatal("wrong config string! %s", GetComponentArgs().c_str());
        return -EINVAL;
    }
    
    if (fDumpClusters) if ((fp = fopen("clusters.dump", "w+b")) == NULL) return -1;

    return iResult;
}

int AliHLTTPCClusterStatComponent::DoDeinit()
{
    if (fDumpClusters) fclose(fp);
    return 0;
}

int AliHLTTPCClusterStatComponent::DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& /*trigData*/, AliHLTUInt8_t* /*outputPtr*/, AliHLTUInt32_t& /*size*/, AliHLTComponentBlockDataList& /*outputBlocks*/)
{
    int iResult=0;

    if (!IsDataEvent()) {return iResult;}

    fTotal = fSplitPad = fSplitTime = fSplitPadTime = fSplitPadOrTime = 0;
    int slice, patch;

    int nBlocks = evtData.fBlockCnt;

    for (int is = 0;is < 36;is++) for (int ip = 0;ip < 6;ip++)
    for (int ndx=0; ndx<nBlocks; ndx++)
    {
        const AliHLTComponentBlockData* iter = blocks+ndx;

        if (iter->fDataType == (AliHLTTPCDefinitions::fgkRawClustersDataType | kAliHLTDataOriginTPC)) //Size of HLT-TPC clusters (uncompressed from HWCF after HWCFDecoder, not yet transformed)
        {
    	    slice = AliHLTTPCDefinitions::GetMinSliceNr(iter->fSpecification);
            patch = AliHLTTPCDefinitions::GetMinPatchNr(iter->fSpecification);
            
            if (slice != is || patch != ip) continue;

	    AliHLTTPCRawClusterData* clusters = (AliHLTTPCRawClusterData*)(iter->fPtr);
    	    for(UInt_t iCluster=0;iCluster<clusters->fCount;iCluster++)
    	    {
        	AliHLTTPCRawCluster &cluster = clusters->fClusters[iCluster];
        	fTotal++;
		if (cluster.GetFlagSplitPad()) fSplitPad++;
		if (cluster.GetFlagSplitTime()) fSplitTime++;
		if (cluster.GetFlagSplitAny()) fSplitPadOrTime++;
		if (cluster.GetFlagSplitPad() && cluster.GetFlagSplitTime()) fSplitPadTime++;
		
		if (fPrintClusters) HLTImportant("Slice %d, Patch %d, Row %d, Pad %.2f, Time %.2f, SPad %.2f, STime %.2f, QMax %d, QTot %d, SplitPad %d, SplitTime %d",
		    slice, patch, (int) cluster.GetPadRow(), cluster.GetPad(), cluster.GetTime(), cluster.GetSigmaPad2(), cluster.GetSigmaTime2(), (int) cluster.GetQMax(), (int) cluster.GetCharge(), (int) cluster.GetFlagSplitPad(), (int) cluster.GetFlagSplitTime());
		
		AliHLTUInt64_t pad64=0;
		if (!isnan(cluster.GetPad())) pad64=(AliHLTUInt64_t)round(cluster.GetPad()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kPad].fScale);
		
		AliHLTUInt64_t time64=0;
        	if (!isnan(cluster.GetTime())) time64=(AliHLTUInt64_t)round(cluster.GetTime()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kTime].fScale);
        	
        	AliHLTUInt64_t sigmaPad64=0;
        	if (!isnan(cluster.GetSigmaPad2())) sigmaPad64=(AliHLTUInt64_t)round(cluster.GetSigmaPad2()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fScale);
        	
        	AliHLTUInt64_t sigmaTime64=0;
        	if (!isnan(cluster.GetSigmaTime2())) sigmaTime64=(AliHLTUInt64_t)round(cluster.GetSigmaTime2()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fScale);
        	
        	if (sigmaPad64 >= (unsigned)1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fBitLength)
	            sigmaPad64 = (1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fBitLength)-1;
		if (sigmaTime64 >= (unsigned)1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fBitLength)
        	    sigmaTime64 = (1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fBitLength)-1;
        	
        	if (fDumpClusters)
        	{
        	    int dumpVals[11] = {slice, patch, cluster.GetPadRow(), pad64, time64, sigmaPad64, sigmaTime64, cluster.GetQMax(), cluster.GetCharge(), cluster.GetFlagSplitPad(), cluster.GetFlagSplitTime()};
        	    fwrite(dumpVals, sizeof(int), 11, fp);
        	}
		
		if (fPrintClustersScaled)
		{
        	    HLTImportant("Slice %d, Patch %d, Row %d, Pad %d, Time %d, SPad %d, STime %d, QMax %d, QTot %d, SplitPad %d, SplitTime %d",
			slice, patch, (int) cluster.GetPadRow(), (int) pad64, (int) time64, (int) sigmaPad64, (int) sigmaTime64, (int) cluster.GetQMax(), (int) cluster.GetCharge(), (int) cluster.GetFlagSplitPad(), (int) cluster.GetFlagSplitTime());
		}
	    }
        }
    }

    HLTImportant("Total %d SplitPad %d SplitTime %d SplitPadTime %d SplitPadOrTime %d", fTotal, fSplitPad, fSplitTime, fSplitPadTime, fSplitPadOrTime);

    return iResult;
}
