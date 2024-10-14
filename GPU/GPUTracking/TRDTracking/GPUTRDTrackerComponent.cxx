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

/// \file GPUTRDTrackerComponent.cxx
/// \brief A TRD tracker processing component for the GPU

/// \author Ole Schmidt

#include "TSystem.h"
#include "TTimeStamp.h"
#include "TObjString.h"
#include "TClonesArray.h"
#include "TObjArray.h"
#include "AliESDEvent.h"
#include "AliHLTErrorGuard.h"
#include "AliHLTDataTypes.h"
#include "GPUTRDGeometry.h"
#include "GPUTRDTracker.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTrackerComponent.h"
#include "GPUTRDSpacePoint.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDTrackletLabels.h"
#include "AliHLTTRDDefinitions.h"
#include "AliHLTTPCDefinitions.h"
#include "GPUTRDTrackPoint.h"
#include "AliHLTGlobalBarrelTrack.h"
#include "AliExternalTrackParam.h"
#include "AliHLTExternalTrackParam.h"
#include "AliHLTTrackMCLabel.h"
#include "GPUTRDTrackData.h"
#include "AliGeomManager.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUSettings.h"
#include <map>
#include <vector>
#include <algorithm>

using namespace GPUCA_NAMESPACE::gpu;

ClassImp(GPUTRDTrackerComponent);

GPUTRDTrackerComponent::GPUTRDTrackerComponent()
  : fTracker(0x0), fGeo(0x0), fTrackList(0x0), fDebugTrackOutput(false), fVerboseDebugOutput(false), fRequireITStrack(false), fBenchmark("TRDTracker")
{
}

GPUTRDTrackerComponent::GPUTRDTrackerComponent(const GPUTRDTrackerComponent&) : fTracker(0x0), fGeo(0x0), fRec(0x0), fChain(0x0), fTrackList(0x0), AliHLTProcessor(), fDebugTrackOutput(false), fVerboseDebugOutput(false), fRequireITStrack(false), fBenchmark("TRDTracker")
{
  // see header file for class documentation
  HLTFatal("copy constructor untested");
}

GPUTRDTrackerComponent& GPUTRDTrackerComponent::operator=(const GPUTRDTrackerComponent&)
{
  // see header file for class documentation
  HLTFatal("assignment operator untested");
  return *this;
}

GPUTRDTrackerComponent::~GPUTRDTrackerComponent() { delete fTracker; }

const char* GPUTRDTrackerComponent::GetComponentID() { return "TRDTracker"; }

void GPUTRDTrackerComponent::GetInputDataTypes(std::vector<AliHLTComponentDataType>& list)
{
  list.clear();
  list.push_back(kAliHLTDataTypeTrack | kAliHLTDataOriginITS);
  // list.push_back( kAliHLTDataTypeTrack|kAliHLTDataOriginTPC );
  list.push_back(AliHLTTPCDefinitions::TracksOuterDataType() | kAliHLTDataOriginTPC);
  list.push_back(kAliHLTDataTypeTrackMC | kAliHLTDataOriginTPC);
  list.push_back(AliHLTTRDDefinitions::fgkTRDTrackletDataType);
  list.push_back(AliHLTTRDDefinitions::fgkTRDMCTrackletDataType);
}

AliHLTComponentDataType GPUTRDTrackerComponent::GetOutputDataType() { return kAliHLTMultipleDataType; }

int32_t GPUTRDTrackerComponent::GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList)
{
  // see header file for class documentation
  tgtList.clear();
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDTrackDataType | kAliHLTDataOriginTRD);
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDTrackPointDataType | kAliHLTDataOriginTRD);
  tgtList.push_back(kAliHLTDataTypeTObject | kAliHLTDataOriginTRD);
  return tgtList.size();
}

void GPUTRDTrackerComponent::GetOutputDataSize(uint64_t& constBase, double& inputMultiplier)
{
  // define guess for the output data size
  constBase = 1000;     // minimum size
  inputMultiplier = 2.; // size relative to input
}

AliHLTComponent* GPUTRDTrackerComponent::Spawn()
{
  // see header file for class documentation
  return new GPUTRDTrackerComponent;
}

int32_t GPUTRDTrackerComponent::ReadConfigurationString(const char* arguments)
{
  // Set configuration parameters for the TRD tracker component from the string

  int32_t iResult = 0;
  if (!arguments) {
    return iResult;
  }

  TString allArgs = arguments;
  TString argument;

  TObjArray* pTokens = allArgs.Tokenize(" ");

  int32_t nArgs = pTokens ? pTokens->GetEntries() : 0;

  for (int32_t i = 0; i < nArgs; i++) {
    argument = ((TObjString*)pTokens->At(i))->GetString();
    if (argument.IsNull()) {
      continue;
    }

    if (argument.CompareTo("-debugOutput") == 0) {
      fDebugTrackOutput = true;
      fVerboseDebugOutput = true;
      HLTInfo("Tracks are dumped in the GPUTRDTrackGPU format");
      continue;
    }

    if (argument.CompareTo("-requireITStrack") == 0) {
      fRequireITStrack = true;
      HLTInfo("TRD tracker requires seeds (TPC tracks) to have an ITS match");
      continue;
    }

    HLTError("Unknown option \"%s\"", argument.Data());
    iResult = -EINVAL;
  }
  delete pTokens;

  return iResult;
}

// #################################################################################
int32_t GPUTRDTrackerComponent::DoInit(int argc, const char** argv)
{
  // see header file for class documentation

  int32_t iResult = 0;
  if (fTracker) {
    return -EINPROGRESS;
  }

  fBenchmark.Reset();
  fBenchmark.SetTimer(0, "total");
  fBenchmark.SetTimer(1, "reco");

  if (AliGeomManager::GetGeometry() == nullptr) {
    AliGeomManager::LoadGeometry();
  }

  fTrackList = new TList();
  if (!fTrackList) {
    return -ENOMEM;
  }
  fTrackList->SetOwner(kFALSE);

  TString arguments = "";
  for (int32_t i = 0; i < argc; i++) {
    if (!arguments.IsNull()) {
      arguments += " ";
    }
    arguments += argv[i];
  }

  iResult = ReadConfigurationString(arguments.Data());

  GPUSettingsGRP cfgGRP;
  cfgGRP.solenoidBzNominalGPU = GetBz();
  GPUSettingsRec cfgRec;
  GPUSettingsProcessing cfgDeviceProcessing;
  GPURecoStepConfiguration cfgRecoStep;
  cfgRecoStep.steps = GPUDataTypes::RecoStep::NoRecoStep;
  cfgRecoStep.inputs.clear();
  cfgRecoStep.outputs.clear();
  fRec = GPUReconstruction::CreateInstance("CPU", true);
  fRec->SetSettings(&cfgGRP, &cfgRec, &cfgDeviceProcessing, &cfgRecoStep);
  fChain = fRec->AddChain<GPUChainTracking>();

  fGeo = new GPUTRDGeometry();
  if (!fGeo) {
    return -ENOMEM;
  }
  if (!GPUTRDGeometry::CheckGeometryAvailable()) {
    HLTError("TRD geometry not available");
    return -EINVAL;
  }
  fTracker = new GPUTRDTrackerGPU();
  if (!fTracker) {
    return -ENOMEM;
  }
  if (fVerboseDebugOutput) {
    fTracker->EnableDebugOutput();
  }
  fRec->RegisterGPUProcessor(fTracker, false);
  fChain->SetTRDGeometry(reinterpret_cast<o2::trd::GeometryFlat*>(fGeo));
  if (fRec->Init()) {
    return -EINVAL;
  }

  return iResult;
}

// #################################################################################
int32_t GPUTRDTrackerComponent::DoDeinit()
{
  // see header file for class documentation
  delete fTracker;
  fTracker = 0x0;
  delete fGeo;
  fGeo = 0x0;
  return 0;
}

// #################################################################################
int32_t GPUTRDTrackerComponent::DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& /*trigData*/, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t& size, std::vector<AliHLTComponentBlockData>& outputBlocks)
{
  // process event

  if (!IsDataEvent()) {
    return 0;
  }

  if (evtData.fBlockCnt <= 0) {
    HLTWarning("no blocks in event");
    return 0;
  }

  fBenchmark.StartNewEvent();
  fBenchmark.Start(0);

  AliHLTUInt32_t maxBufferSize = size;
  size = 0; // output size

  int32_t iResult = 0;

  if (fTrackList->GetEntries() != 0) {
    fTrackList->Clear(); // tracks are owned by GPUTRDTrackerGPU
  }

  int32_t nBlocks = evtData.fBlockCnt;

  const AliHLTTracksData* tpcData = nullptr;
  const AliHLTTracksData* itsData = nullptr;
  const AliHLTTrackMCData* tpcDataMC = nullptr;

  std::vector<GPUTRDTrackGPU> tracksTPC;
  std::vector<int32_t> tracksTPCId;

  bool hasMCtracklets = false;

  int32_t nTrackletsTotal = 0;
  int32_t nTrackletsTotalMC = 0;
  const GPUTRDTrackletWord* tracklets = nullptr;
  const GPUTRDTrackletLabels* trackletsMC = nullptr;

  for (int32_t iBlock = 0; iBlock < nBlocks; iBlock++) {
    if (blocks[iBlock].fDataType == (kAliHLTDataTypeTrack | kAliHLTDataOriginITS) && fRequireITStrack) {
      itsData = (const AliHLTTracksData*)blocks[iBlock].fPtr;
      fBenchmark.AddInput(blocks[iBlock].fSize);
    } else if (blocks[iBlock].fDataType == (AliHLTTPCDefinitions::TracksOuterDataType() | kAliHLTDataOriginTPC)) {
      tpcData = (const AliHLTTracksData*)blocks[iBlock].fPtr;
      fBenchmark.AddInput(blocks[iBlock].fSize);
    } else if (blocks[iBlock].fDataType == (kAliHLTDataTypeTrackMC | kAliHLTDataOriginTPC)) {
      tpcDataMC = (const AliHLTTrackMCData*)blocks[iBlock].fPtr;
      fBenchmark.AddInput(blocks[iBlock].fSize);
    } else if (blocks[iBlock].fDataType == (AliHLTTRDDefinitions::fgkTRDTrackletDataType)) {
      tracklets = reinterpret_cast<const GPUTRDTrackletWord*>(blocks[iBlock].fPtr);
      nTrackletsTotal = blocks[iBlock].fSize / sizeof(GPUTRDTrackletWord);
      fBenchmark.AddInput(blocks[iBlock].fSize);
    } else if (blocks[iBlock].fDataType == (AliHLTTRDDefinitions::fgkTRDMCTrackletDataType)) {
      hasMCtracklets = true;
      trackletsMC = reinterpret_cast<const GPUTRDTrackletLabels*>(blocks[iBlock].fPtr);
      nTrackletsTotalMC = blocks[iBlock].fSize / sizeof(GPUTRDTrackletLabels);
      fBenchmark.AddInput(blocks[iBlock].fSize);
    }
  }

  if (tpcData == nullptr) {
    HLTInfo("did not receive any TPC tracks. Skipping event");
    return 0;
  }

  if (nTrackletsTotal == 0) {
    HLTInfo("did not receive any TRD tracklets. Skipping event");
    return 0;
  }

  if (hasMCtracklets && nTrackletsTotal != nTrackletsTotalMC) {
    HLTError("the numbers of input tracklets does not match the number of input MC labels for them");
    return -EINVAL;
  }

  // copy tracklets into temporary vector to allow for sorting them (the input array is const)
  std::vector<GPUTRDTrackletWord> trackletsTmp(nTrackletsTotal);
  for (int32_t iTrklt = 0; iTrklt < nTrackletsTotal; ++iTrklt) {
    trackletsTmp[iTrklt] = tracklets[iTrklt];
  }

  int32_t nTPCtracks = tpcData->fCount;
  std::vector<bool> itsAvail(nTPCtracks, false);
  if (itsData) {
    // look for ITS tracks with >= 2 hits
    int32_t nITStracks = itsData->fCount;
    const AliHLTExternalTrackParam* currITStrack = itsData->fTracklets;
    for (int32_t iTrkITS = 0; iTrkITS < nITStracks; iTrkITS++) {
      if (currITStrack->fNPoints >= 2) {
        itsAvail.at(currITStrack->fTrackID) = true;
      }
      uint32_t dSize = sizeof(AliHLTExternalTrackParam) + currITStrack->fNPoints * sizeof(uint32_t);
      currITStrack = (AliHLTExternalTrackParam*)(((Byte_t*)currITStrack) + dSize);
    }
  }
  std::map<int32_t, int32_t> mcLabels;
  if (tpcDataMC) {
    // look for TPC track MC labels
    int32_t nMCtracks = tpcDataMC->fCount;
    for (int32_t iMC = 0; iMC < nMCtracks; iMC++) {
      const AliHLTTrackMCLabel& lab = tpcDataMC->fLabels[iMC];
      mcLabels[lab.fTrackID] = lab.fMCLabel;
    }
  }
  const AliHLTExternalTrackParam* currOutTrackTPC = tpcData->fTracklets;
  for (int32_t iTrk = 0; iTrk < nTPCtracks; iTrk++) {
    // store TPC tracks (if required only the ones with >=2 ITS hits)
    if (itsData != nullptr && !itsAvail.at(currOutTrackTPC->fTrackID)) {
      continue;
    }
    GPUTRDTrackGPU t(*currOutTrackTPC);
    int32_t mcLabel = -1;
    if (tpcDataMC) {
      if (mcLabels.find(currOutTrackTPC->fTrackID) != mcLabels.end()) {
        mcLabel = mcLabels[currOutTrackTPC->fTrackID];
      }
    }
    tracksTPC.push_back(t);
    tracksTPCId.push_back(currOutTrackTPC->fTrackID);
    uint32_t dSize = sizeof(AliHLTExternalTrackParam) + currOutTrackTPC->fNPoints * sizeof(uint32_t);
    currOutTrackTPC = (AliHLTExternalTrackParam*)+(((Byte_t*)currOutTrackTPC) + dSize);
  }

  if (fVerboseDebugOutput) {
    HLTInfo("TRDTrackerComponent received %i tracklets\n", nTrackletsTotal);
  }

  fTracker->SetGenerateSpacePoints(true);
  fTracker->Reset();
  fChain->mIOPtrs.nMergedTracks = tracksTPC.size();
  fChain->mIOPtrs.nTRDTracklets = nTrackletsTotal;
  fChain->mIOPtrs.nTRDTriggerRecords = 1;
  uint8_t trigRecMaskDummy[1] = {1};
  fChain->mIOPtrs.trdTrigRecMask = &(trigRecMaskDummy[0]);
  fRec->PrepareEvent();
  fRec->SetupGPUProcessor(fTracker, true);

  std::sort(trackletsTmp.begin(), trackletsTmp.end());
  fChain->mIOPtrs.trdTracklets = &(trackletsTmp[0]);

  // loop over all tracks
  for (uint32_t iTrack = 0; iTrack < tracksTPC.size(); ++iTrack) {
    fTracker->LoadTrack(tracksTPC[iTrack], tracksTPCId[iTrack]);
  }

  fBenchmark.Start(1);
  fChain->DoTRDGPUTracking<1>(fTracker);
  fBenchmark.Stop(1);

  GPUTRDTrackGPU* trackArray = fTracker->Tracks();
  int32_t nTracks = fTracker->NTracks();
  GPUTRDSpacePoint* spacePoints = fTracker->SpacePoints();

  // TODO delete fTrackList since it only works for TObjects (or use compiler flag after tests with GPU track type)
  // for (int32_t iTrack=0; iTrack<nTracks; ++iTrack) {
  //  fTrackList->AddLast(&trackArray[iTrack]);
  //}

  // push back GPUTRDTracks for debugging purposes
  if (fDebugTrackOutput) {
    PushBack(fTrackList, (kAliHLTDataTypeTObject | kAliHLTDataOriginTRD), 0x3fffff);
  }
  // push back AliHLTExternalTrackParam (default)
  else {

    AliHLTUInt32_t blockSize = GPUTRDTrackData::GetSize(nTracks);
    if (size + blockSize > maxBufferSize) {
      HLTWarning("Output buffer exceeded for tracks");
      return -ENOSPC;
    }

    GPUTRDTrackData* outTracks = (GPUTRDTrackData*)(outputPtr);
    outTracks->fCount = 0;
    int32_t assignedTracklets = 0;

    for (int32_t iTrk = 0; iTrk < nTracks; ++iTrk) {
      GPUTRDTrackGPU& t = trackArray[iTrk];
      if (t.getNtracklets() == 0) {
        continue;
      }
      assignedTracklets += t.getNtracklets();
      GPUTRDTrackDataRecord& currOutTrack = outTracks->fTracks[outTracks->fCount];
      t.ConvertTo(currOutTrack);
      outTracks->fCount++;
    }

    AliHLTComponentBlockData resultData;
    FillBlockData(resultData);
    resultData.fOffset = size;
    resultData.fSize = blockSize;
    resultData.fDataType = AliHLTTRDDefinitions::fgkTRDTrackDataType;
    outputBlocks.push_back(resultData);
    fBenchmark.AddOutput(resultData.fSize);

    size += blockSize;
    outputPtr += resultData.fSize;

    blockSize = 0;

    // space points calculated from tracklets

    blockSize = sizeof(GPUTRDTrackPointData) + sizeof(GPUTRDTrackPoint) * nTrackletsTotal;

    if (size + blockSize > maxBufferSize) {
      HLTWarning("Output buffer exceeded for space points");
      return -ENOSPC;
    }

    GPUTRDTrackPointData* outTrackPoints = (GPUTRDTrackPointData*)(outputPtr);
    outTrackPoints->fCount = nTrackletsTotal;

    { // fill array with 0 for a case..
      GPUTRDTrackPoint empty;
      empty.fX[0] = 0;
      empty.fX[1] = 0;
      empty.fX[2] = 0;
      empty.fVolumeId = 0;
      for (int32_t i = 0; i < nTrackletsTotal; ++i) {
        outTrackPoints->fPoints[i] = empty;
      }
    }

    for (int32_t i = 0; i < nTrackletsTotal; ++i) {
      const GPUTRDSpacePoint& sp = spacePoints[i];
      GPUTRDTrackPoint* currOutPoint = &outTrackPoints->fPoints[i];
      currOutPoint->fX[0] = sp.getX(); // x in sector coordinates
      currOutPoint->fX[1] = sp.getY(); // y in sector coordinates
      currOutPoint->fX[2] = sp.getZ(); // z in sector coordinates
      int32_t detId = trackletsTmp[i].GetDetector();
      int32_t layer = detId % 6;                                     // TRD layer number for given detector
      int32_t modId = (detId / 18) * 5 + ((detId % 30) / 6);         // global TRD stack number [0..89]
      int32_t volId = (UShort_t(9 + layer) << 11) | UShort_t(modId); // taken from AliGeomManager::LayerToVolUID(). AliGeomManager::ELayerID(AliGeomManager::kTRD1) == 9
      currOutPoint->fVolumeId = volId;
    }
    AliHLTComponentBlockData resultDataSP;
    FillBlockData(resultDataSP);
    resultDataSP.fOffset = size;
    resultDataSP.fSize = blockSize;
    resultDataSP.fDataType = AliHLTTRDDefinitions::fgkTRDTrackPointDataType | kAliHLTDataOriginTRD;
    outputBlocks.push_back(resultDataSP);
    fBenchmark.AddOutput(resultData.fSize);
    size += blockSize;
    outputPtr += resultDataSP.fSize;

    HLTInfo("TRD tracker: output %d tracks (%d assigned tracklets) and %d track points", outTracks->fCount, assignedTracklets, outTrackPoints->fCount);
  }

  fBenchmark.Stop(0);
  HLTInfo(fBenchmark.GetStatistics());

  return iResult;
}

// #################################################################################
int32_t GPUTRDTrackerComponent::Reconfigure(const char* cdbEntry, const char* chainId)
{
  // see header file for class documentation

  int32_t iResult = 0;
  TString cdbPath;
  if (cdbEntry) {
    cdbPath = cdbEntry;
  } else {
    cdbPath = "HLT/ConfigGlobal/";
    cdbPath += GetComponentID();
  }

  AliInfoClass(Form("reconfigure '%s' from entry %s%s", chainId, cdbPath.Data(), cdbEntry ? "" : " (default)"));
  iResult = ConfigureFromCDBTObjString(cdbPath);

  return iResult;
}
