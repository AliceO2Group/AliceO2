// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackletReaderComponent.cxx
/// \brief A pre-processing component for TRD tracking/trigger data on FEP-level

/// \author Felix Rettig, Stefan Kirsch, Ole Schmidt

#include <cstdlib>
#include "AliLog.h"
#include "AliHLTDataTypes.h"
#include "AliHLTTRDDefinitions.h"
#include "GPUTRDTrackletReaderComponent.h"
#include "AliRawReaderMemory.h"
#include "AliTRDrawStream.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDTrackletLabels.h"
#include "AliTRDtrackletWord.h"
#include "AliTRDtrackletMCM.h"
#include "TTree.h"
#include "TEventList.h"
#include "AliRunLoader.h"
#include "AliLoader.h"
#include "AliDataLoader.h"

using namespace GPUCA_NAMESPACE::gpu;

ClassImp(GPUTRDTrackletReaderComponent)

#define LogError(...)               \
  {                                 \
    HLTError(__VA_ARGS__);          \
    if (fDebugLevel >= 1) {         \
      DbgLog("ERROR", __VA_ARGS__); \
    }                               \
  }
#define LogInfo(...)               \
  {                                \
    HLTInfo(__VA_ARGS__);          \
    if (fDebugLevel >= 1) {        \
      DbgLog("INFO", __VA_ARGS__); \
    }                              \
  }
#define LogInspect(...)               \
  {                                   \
    HLTDebug(__VA_ARGS__);            \
    if (fDebugLevel >= 1) {           \
      DbgLog("INSPECT", __VA_ARGS__); \
    }                                 \
  }
#define LogDebug(...)               \
  {                                 \
    if (fDebugLevel >= 1) {         \
      HLTInfo(__VA_ARGS__);         \
      DbgLog("DEBUG", __VA_ARGS__); \
    }                               \
  }

  GPUTRDTrackletReaderComponent::GPUTRDTrackletReaderComponent()
  : AliHLTProcessor(), fDebugLevel(0), fEventId(fgkInvalidEventId), fTrackletArray(nullptr), fRawReaderMem(nullptr), fRawReaderTrd(nullptr)
{
  // constructor
}

GPUTRDTrackletReaderComponent::~GPUTRDTrackletReaderComponent()
{
  // destructor
}

const char* GPUTRDTrackletReaderComponent::GetComponentID() { return "TRDTrackletReader"; }

void GPUTRDTrackletReaderComponent::GetInputDataTypes(vector<AliHLTComponentDataType>& list)
{
  list.push_back(kAliHLTDataTypeDDLRaw | kAliHLTDataOriginTRD);
  list.push_back(kAliHLTDataTypeAliTreeD | kAliHLTDataOriginTRD);
}

AliHLTComponentDataType GPUTRDTrackletReaderComponent::GetOutputDataType() { return kAliHLTMultipleDataType; }

int GPUTRDTrackletReaderComponent::GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList)
{
  tgtList.clear();
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDTrackletDataType);
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDMCTrackletDataType);
  return tgtList.size();
}

void GPUTRDTrackletReaderComponent::GetOutputDataSize(unsigned long& constBase, double& inputMultiplier)
{
  constBase = 5000000;
  inputMultiplier = 0;
}

void GPUTRDTrackletReaderComponent::GetOCDBObjectDescription(TMap* const /*targetMap*/) {}

AliHLTComponent* GPUTRDTrackletReaderComponent::Spawn() { return new GPUTRDTrackletReaderComponent; }

int GPUTRDTrackletReaderComponent::Reconfigure(const char* /*cdbEntry*/, const char* /*chainId*/) { return 0; }

int GPUTRDTrackletReaderComponent::ReadPreprocessorValues(const char* /*modules*/) { return 0; }

int GPUTRDTrackletReaderComponent::ScanConfigurationArgument(int argc, const char** argv)
{

  if (argc <= 0) {
    return 0;
  }

  unsigned short iArg = 0;
  TString argument(argv[iArg]);

  if (!argument.CompareTo("-debug")) {
    if (++iArg >= argc) {
      return -EPROTO;
    }
    argument = argv[iArg];
    fDebugLevel = argument.Atoi();
    LogInfo("debug level set to %d.", fDebugLevel);
    return 2;
  }

  return 0;
}

int GPUTRDTrackletReaderComponent::DoInit(int argc, const char** argv)
{

  int iResult = 0;

  do {

    fRawReaderMem = new AliRawReaderMemory;
    if (!fRawReaderMem) {
      iResult = -ENOMEM;
      break;
    }

    fTrackletArray = new TClonesArray("AliTRDtrackletWord", 1000);
    if (!fTrackletArray) {
      iResult = -ENOMEM;
      break;
    }

    fRawReaderTrd = new AliTRDrawStream(fRawReaderMem);
    if (!fRawReaderTrd) {
      iResult = -ENOMEM;
      break;
    }

    fRawReaderTrd->SetTrackletArray(fTrackletArray);

    // Disable raw reader error messages that could flood HLT logbook
    AliLog::SetClassDebugLevel("AliTRDrawStream", 0);
    fRawReaderTrd->SetErrorDebugLevel(AliTRDrawStream::kLinkMonitor, 1);

  } while (0);

  if (iResult < 0) {

    if (fRawReaderTrd) {
      delete fRawReaderTrd;
    }
    fRawReaderTrd = nullptr;

    if (fRawReaderMem) {
      delete fRawReaderMem;
    }
    fRawReaderMem = nullptr;

    if (fTrackletArray) {
      delete fTrackletArray;
    }
    fTrackletArray = nullptr;
  }

  vector<const char*> remainingArgs;
  for (int i = 0; i < argc; ++i) {
    remainingArgs.push_back(argv[i]);
  }

  if (argc > 0) {
    ConfigureFromArgumentString(remainingArgs.size(), &(remainingArgs[0]));
  }

  return iResult;
}

int GPUTRDTrackletReaderComponent::DoDeinit()
{

  if (fRawReaderTrd) {
    delete fRawReaderTrd;
  }
  fRawReaderTrd = nullptr;

  if (fRawReaderMem) {
    delete fRawReaderMem;
  }
  fRawReaderMem = nullptr;

  if (fTrackletArray) {
    delete fTrackletArray;
  }
  fTrackletArray = nullptr;

  return 0;
}

// void GPUTRDTrackletReaderComponent::DbgLog(const char* prefix, const char* msg){
//  AliHLTEventID_t eventNumber = fEventId;
//  int runNumber = -1;
//  HLTInfo("TRDGM %s-%s: [PRE] %s%s",
//   (runNumber >= 0) ? Form("%06d", runNumber) : "XXXXXX",
//   (eventNumber != fgkInvalidEventId) ? Form("%05llu", eventNumber) : "XXXXX",
//   (strlen(prefix) > 0) ? Form("<%s> ", prefix) : "", msg);
//}

void GPUTRDTrackletReaderComponent::DbgLog(const char* prefix, ...)
{
#ifdef __TRDHLTDEBUG
  AliHLTEventID_t eventNumber = fEventId;
  int runNumber = -1;
  printf("TRDHLTGM %s-X-%s: [PRE] %s", (runNumber >= 0) ? Form("%06d", runNumber) : "XXXXXX", (eventNumber != fgkInvalidEventId) ? Form("%05llu", eventNumber) : "XXXXX", (strlen(prefix) > 0) ? Form("<%s> ", prefix) : "");
#endif
  va_list args;
  va_start(args, prefix);
  char* fmt = va_arg(args, char*);
  vprintf(fmt, args);
  printf("\n");
  va_end(args);
}

int GPUTRDTrackletReaderComponent::DoEvent(const AliHLTComponentEventData& hltEventData, AliHLTComponentTriggerData& /*trigData*/)
{

  fEventId = hltEventData.fEventID;

  HLTInfo("### START DoEvent [event id: %llu, %d blocks, size: %d]", hltEventData.fEventID, hltEventData.fBlockCnt, hltEventData.fStructSize);

  // event processing function
  int iResult = 0;

  fTrackletArray->Clear();
  fRawReaderMem->ClearBuffers();

  if (!IsDataEvent()) { // process data events only
    HLTInfo("### END   DoEvent [event id: %llu, %d blocks, size: %d] (skipped: no data event)", hltEventData.fEventID, hltEventData.fBlockCnt, hltEventData.fStructSize);
    return iResult;
  }

  std::vector<GPUTRDTrackletWord> outputTrkls;
  std::vector<GPUTRDTrackletLabels> outputTrklsMC;

  { // read raw data

    TString infoStr("");
    unsigned int sourceSectors = 0;

    // loop over all incoming TRD raw data blocks
    for (const AliHLTComponentBlockData* pBlock = GetFirstInputBlock(kAliHLTDataTypeDDLRaw | kAliHLTDataOriginTRD); pBlock != nullptr && iResult >= 0; pBlock = GetNextInputBlock()) {

      int trdSector = -1;

      // determine sector from block specification
      for (unsigned pos = 0; pos < 8 * sizeof(AliHLTUInt32_t); pos++) {
        if (pBlock->fSpecification & (0x1 << pos)) {
          if (trdSector >= 0) {
            HLTWarning("Cannot uniquely identify DDL number from specification, skipping data block %s 0x%08x", DataType2Text(pBlock->fDataType).c_str(), pBlock->fSpecification);
            trdSector = -1;
            break;
          }
          trdSector = pos;
        }
      }
      if (trdSector < 0) {
        continue;
      }

      // add data block to rawreader
      infoStr += Form("%02d, ", trdSector);
      sourceSectors |= pBlock->fSpecification;
      if (!fRawReaderMem->AddBuffer((unsigned char*)pBlock->fPtr, pBlock->fSize, trdSector + 1024)) {
        LogError("Could not add buffer of data block  %s, 0x%08x to rawreader", DataType2Text(pBlock->fDataType).c_str(), pBlock->fSpecification);
        continue;
      }
    } // loop over all incoming TRD raw data blocks

    if (sourceSectors) {
      infoStr.Remove(infoStr.Length() - 2, 2);
      LogDebug("preprocessing raw data from sectors: %s...", infoStr.Data());

      // extract header info and TRD tracklets from raw data
      fRawReaderTrd->ReadEvent();

      // read and process TRD tracklets
      int nTracklets = fTrackletArray->GetEntriesFast();

      HLTInfo("There are %i tracklets in this event\n", nTracklets);
      for (int iTracklet = 0; iTracklet < nTracklets; ++iTracklet) {
        GPUTRDTrackletWord trkl = *((AliTRDtrackletWord*)fTrackletArray->At(iTracklet));
        trkl.SetId(iTracklet);
        outputTrkls.push_back(trkl);
      }
      LogDebug("pushing data for sectors: 0x%05x", sourceSectors);
    }
    fRawReaderMem->ClearBuffers();
  }

  { // loop over all incoming TRD MC tracklets data blocks

    for (const TObject* iter = GetFirstInputObject(kAliHLTDataTypeAliTreeD | kAliHLTDataOriginTRD); iter != nullptr; iter = GetNextInputObject()) {
      TTree* trackletTree = dynamic_cast<TTree*>(const_cast<TObject*>(iter));
      if (!trackletTree) {
        HLTFatal("No Tracklet Tree found");
        return -EINVAL;
      }

      TBranch* trklbranch = trackletTree->GetBranch("mcmtrklbranch");
      if (!trklbranch) {
        HLTFatal("No tracklet branch found in tracklet tree");
        return -EINVAL;
      }
      int nTracklets = trklbranch->GetEntries();
      HLTInfo("Input tree with %d TRD MCM tracklets", nTracklets);

      //-----------------------------------
      // Deploy same hack as in ITS Clusterizer
      AliRunLoader* pRunLoader = AliRunLoader::Instance();
      if (!pRunLoader) {
        HLTError("failed to get global runloader instance");
        return -ENOSYS;
      }
      pRunLoader->GetEvent(GetEventCount());
      const char* loaderType = "TRDLoader";
      AliLoader* pLoader = pRunLoader->GetLoader(loaderType);
      if (!pLoader) {
        HLTError("can not get loader \"%s\" from runloader", loaderType);
        return -ENOSYS;
      }
      pLoader->LoadDigits("read");
      AliDataLoader* dataLoader = pLoader->GetDataLoader("tracklets");
      if (dataLoader) {
        trackletTree = dataLoader->Tree();
        dataLoader->Load("read");
      } else {
        HLTWarning("TRD tracklet loader not found");
      }
      trklbranch = trackletTree->GetBranch("mcmtrklbranch");
      if (!trklbranch) {
        HLTFatal("No tracklet branch found in tracklet tree");
        return -EINVAL;
      }
      if (trklbranch->GetEntries() != nTracklets) {
        HLTFatal("Incorrect number of tracklets in tree");
        return -EINVAL;
      }
      //-----------------------------------

      AliTRDtrackletMCM* trkl = 0x0;
      trklbranch->SetAddress(&trkl);

      for (int iTracklet = 0; iTracklet < nTracklets; iTracklet++) {
        int nbytes = trklbranch->GetEntry(iTracklet, 1);
        if (!trkl || nbytes <= 0) {
          HLTWarning("Can not read entry from tracklet branch");
          continue;
        }
        GPUTRDTrackletWord hltTrkl = *trkl;
        hltTrkl.SetId(iTracklet);
        outputTrkls.push_back(hltTrkl);
        GPUTRDTrackletLabels trklMC;
        trklMC.mLabel[0] = trkl->GetLabel(0);
        trklMC.mLabel[1] = trkl->GetLabel(1);
        trklMC.mLabel[2] = trkl->GetLabel(2);
        outputTrklsMC.push_back(trklMC);
      }
    }
  }

  if (outputTrkls.size() > 0) {
    iResult = PushBack(&outputTrkls[0], outputTrkls.size() * sizeof(outputTrkls[0]), AliHLTTRDDefinitions::fgkTRDTrackletDataType, 0);
  }
  if (outputTrklsMC.size() > 0) {
    iResult = PushBack(&outputTrklsMC[0], outputTrklsMC.size() * sizeof(outputTrklsMC[0]), AliHLTTRDDefinitions::fgkTRDMCTrackletDataType, 0);
  }

  HLTInfo("### END   DoEvent [event id: %llu, %d blocks, size: %d, output tracklets: %d]", hltEventData.fEventID, hltEventData.fBlockCnt, hltEventData.fStructSize, outputTrkls.size());

  return iResult;
}
