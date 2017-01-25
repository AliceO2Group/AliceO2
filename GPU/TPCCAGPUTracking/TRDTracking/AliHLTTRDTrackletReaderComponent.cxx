// $Id$
//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

/// @file   AliHLTTRDTrackletReaderComponent.cxx
/// @author Felix Rettig, Stefan Kirsch
/// @date   2012-08-16
/// @brief  A pre-processing component for TRD tracking/trigger data on FEP-level
/// @ingroup alihlt_trd_components

#include <cstdlib>
#include "AliLog.h"
#include "AliHLTDataTypes.h"
#include "AliHLTTRDDefinitions.h"
#include "AliHLTTRDTrackletReaderComponent.h"
#include "AliRawReaderMemory.h"
#include "AliTRDrawStream.h"
#include "AliHLTTRDTrackletWord.h"
#include "AliTRDtrackletWord.h"
#include "TTree.h"
#include "TEventList.h"

ClassImp(AliHLTTRDTrackletReaderComponent)

#define LogError( ... ) { HLTError(__VA_ARGS__); if (fDebugLevel >= 1) { DbgLog("ERROR", __VA_ARGS__); } }
#define LogInfo( ... ) { HLTInfo(__VA_ARGS__); if (fDebugLevel >= 1) { DbgLog("INFO", __VA_ARGS__); } }
#define LogInspect( ... ) { HLTDebug(__VA_ARGS__); if (fDebugLevel >= 1) { DbgLog("INSPECT", __VA_ARGS__); } }
#define LogDebug( ... ) { if (fDebugLevel >= 1) { HLTInfo(__VA_ARGS__); DbgLog("DEBUG", __VA_ARGS__); } }

AliHLTTRDTrackletReaderComponent::AliHLTTRDTrackletReaderComponent() :
  AliHLTProcessor(),
  fDebugLevel(0),
  fEventId(fgkInvalidEventId),
  fTrackletArray(NULL),
  fRawReaderMem(NULL),
  fRawReaderTrd(NULL)
{
  // constructor
}

AliHLTTRDTrackletReaderComponent::~AliHLTTRDTrackletReaderComponent() {
  // destructor
}

const char* AliHLTTRDTrackletReaderComponent::GetComponentID() {
  return "TRDTrackletReader";
}

void AliHLTTRDTrackletReaderComponent::GetInputDataTypes( vector<AliHLTComponentDataType>& list) {
  list.push_back(kAliHLTDataTypeDDLRaw | kAliHLTDataOriginTRD);
  list.push_back(kAliHLTDataTypeAliTreeD|kAliHLTDataOriginTRD);
}

AliHLTComponentDataType AliHLTTRDTrackletReaderComponent::GetOutputDataType() {
  return kAliHLTMultipleDataType;
}

int AliHLTTRDTrackletReaderComponent::GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList) {
  tgtList.clear();
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDTrackletDataType);
  return tgtList.size();
}

void AliHLTTRDTrackletReaderComponent::GetOutputDataSize(unsigned long& constBase, double& inputMultiplier ) {
  constBase = 5000000;
  inputMultiplier = 0;
}

void AliHLTTRDTrackletReaderComponent::GetOCDBObjectDescription( TMap* const /*targetMap*/) {
}

AliHLTComponent* AliHLTTRDTrackletReaderComponent::Spawn(){
  return new AliHLTTRDTrackletReaderComponent;
}

int AliHLTTRDTrackletReaderComponent::Reconfigure(const char* /*cdbEntry*/, const char* /*chainId*/) {
  return 0;
}

int AliHLTTRDTrackletReaderComponent::ReadPreprocessorValues(const char* /*modules*/){
  return 0;
}

int AliHLTTRDTrackletReaderComponent::ScanConfigurationArgument(int argc, const char** argv){

  if (argc <= 0)
    return 0;

  UShort_t iArg = 0;
  TString argument(argv[iArg]);

  if (!argument.CompareTo("-debug")){
    if (++iArg >= argc) return -EPROTO;
    argument = argv[iArg];
    fDebugLevel = argument.Atoi();
    LogInfo("debug level set to %d.", fDebugLevel);
    return 2;
  }

  return 0;
}

int AliHLTTRDTrackletReaderComponent::DoInit(int argc, const char** argv){

  int iResult = 0;

  do {

    fRawReaderMem = new AliRawReaderMemory;
    if (!fRawReaderMem) {
      iResult=-ENOMEM;
      break;
    }

    fTrackletArray = new TClonesArray("AliTRDtrackletWord", 1000);
    if (!fTrackletArray) {
      iResult=-ENOMEM;
      break;
    }

    fRawReaderTrd = new AliTRDrawStream(fRawReaderMem);
    if (!fRawReaderTrd) {
      iResult=-ENOMEM;
      break;
    }

    fRawReaderTrd->SetTrackletArray(fTrackletArray);

    // Disable raw reader error messages that could flood HLT logbook
    AliLog::SetClassDebugLevel("AliTRDrawStream", 0);
    fRawReaderTrd->SetErrorDebugLevel(AliTRDrawStream::kLinkMonitor, 1);

  } while (0);

  if (iResult < 0) {

    if (fRawReaderTrd) delete fRawReaderTrd;
    fRawReaderTrd = NULL;

    if (fRawReaderMem) delete fRawReaderMem;
    fRawReaderMem = NULL;

    if (fTrackletArray) delete fTrackletArray;
    fTrackletArray = NULL;

  }

  vector<const char*> remainingArgs;
  for (int i = 0; i < argc; ++i)
    remainingArgs.push_back(argv[i]);

  if (argc > 0)
    ConfigureFromArgumentString(remainingArgs.size(), &(remainingArgs[0]));

  return iResult;
}

int AliHLTTRDTrackletReaderComponent::DoDeinit() {

  if (fRawReaderTrd) delete fRawReaderTrd;
  fRawReaderTrd = NULL;

  if (fRawReaderMem) delete fRawReaderMem;
  fRawReaderMem = NULL;

  if (fTrackletArray) delete fTrackletArray;
  fTrackletArray = NULL;

  return 0;
}

//void AliHLTTRDTrackletReaderComponent::DbgLog(const char* prefix, const char* msg){
//  AliHLTEventID_t eventNumber = fEventId;
//  int runNumber = -1;
//  printf("TRDGM %s-%s: [PRE] %s%s\n",
// 	 (runNumber >= 0) ? Form("%06d", runNumber) : "XXXXXX",
// 	 (eventNumber != fgkInvalidEventId) ? Form("%05llu", eventNumber) : "XXXXX",
// 	 (strlen(prefix) > 0) ? Form("<%s> ", prefix) : "", msg);
//}


void AliHLTTRDTrackletReaderComponent::DbgLog(const char* prefix, ...){
#ifdef __TRDHLTDEBUG
  AliHLTEventID_t eventNumber = fEventId;
  int runNumber = -1;
  printf("TRDHLTGM %s-X-%s: [PRE] %s",
 	 (runNumber >= 0) ? Form("%06d", runNumber) : "XXXXXX",
 	 (eventNumber != fgkInvalidEventId) ? Form("%05llu", eventNumber) : "XXXXX",
 	 (strlen(prefix) > 0) ? Form("<%s> ", prefix) : "");
#endif
  va_list args;
  va_start(args, prefix);
  char* fmt = va_arg(args, char*);
  vprintf(fmt, args);
  printf("\n");
  va_end(args);
}


int AliHLTTRDTrackletReaderComponent::DoEvent(const AliHLTComponentEventData& hltEventData,
					    AliHLTComponentTriggerData& /*trigData*/) {

  fEventId = hltEventData.fEventID;

  HLTInfo("### START DoEvent [event id: %llu, %d blocks, size: %d]",
	   hltEventData.fEventID, hltEventData.fBlockCnt, hltEventData.fStructSize);

  // event processing function
  int iResult = 0;

  fTrackletArray->Clear();
  fRawReaderMem->ClearBuffers();

  if (!IsDataEvent()) { // process data events only
    HLTInfo("### END   DoEvent [event id: %llu, %d blocks, size: %d] (skipped: no data event)",
	    hltEventData.fEventID, hltEventData.fBlockCnt, hltEventData.fStructSize);
    return iResult;
  }

  std::vector<AliHLTTRDTrackletWord> outputTrkls;
  
  { // read raw data
    
    TString infoStr("");
    UInt_t sourceSectors = 0;

    // loop over all incoming TRD raw data blocks
    for (const AliHLTComponentBlockData* pBlock = GetFirstInputBlock(kAliHLTDataTypeDDLRaw | kAliHLTDataOriginTRD);
	 pBlock != NULL && iResult >= 0;
	 pBlock = GetNextInputBlock()) {

      int trdSector = -1;

      // determine sector from block specification
      for (unsigned pos = 0; pos < 8*sizeof(AliHLTUInt32_t); pos++) {
	if (pBlock->fSpecification & (0x1 << pos)) {
	  if (trdSector >= 0) {
	    HLTWarning("Cannot uniquely identify DDL number from specification, skipping data block %s 0x%08x",
		       DataType2Text(pBlock->fDataType).c_str(),
		       pBlock->fSpecification);
	    trdSector = -1;
	    break;
	  }
	  trdSector = pos;
	}
      }
      if (trdSector < 0) continue;

      // add data block to rawreader
      infoStr += Form("%02d, ", trdSector);
      sourceSectors |= pBlock->fSpecification;
      if(!fRawReaderMem->AddBuffer((UChar_t*) pBlock->fPtr, pBlock->fSize, trdSector + 1024)){
	LogError("Could not add buffer of data block  %s, 0x%08x to rawreader",
		 DataType2Text(pBlock->fDataType).c_str(),
		 pBlock->fSpecification);
	continue;
      }
    } // loop over all incoming TRD raw data blocks

    if (sourceSectors){
      infoStr.Remove(infoStr.Length() - 2, 2);
      LogDebug("preprocessing raw data from sectors: %s...", infoStr.Data());
 
      // extract header info and TRD tracklets from raw data
      fRawReaderTrd->ReadEvent();

      // read and process TRD tracklets
      int nTracklets = fTrackletArray->GetEntriesFast();

      HLTInfo("There are %i tracklets in this event\n", nTracklets);
      for (int iTracklet = 0; iTracklet < nTracklets; ++iTracklet){
	AliHLTTRDTrackletWord trkl = *((AliTRDtrackletWord*)fTrackletArray->At(iTracklet));
	trkl.SetId(iTracklet);
	outputTrkls.push_back(trkl);
      } 
      LogDebug("pushing data for sectors: 0x%05x", sourceSectors);
    }  
    fRawReaderMem->ClearBuffers();
  }


  { // loop over all incoming TRD MC tracklets data blocks

    for ( const TObject *iter = GetFirstInputObject( kAliHLTDataTypeAliTreeD|kAliHLTDataOriginTRD ); iter != NULL; iter = GetNextInputObject() ) {  
      TTree *trackletTree = dynamic_cast<TTree*>(const_cast<TObject*>( iter ) );
      if(!trackletTree){
	HLTFatal("No Tracklet Tree found");
	return -EINVAL;
      }    
      TBranch *trklbranch = trackletTree->GetBranch("mcmtrklbranch");
      if (!trklbranch ) {
	HLTFatal("No tracklet branch found in tracklet tree");
	return -EINVAL;     
      }
      Int_t nTracklets = trklbranch->GetEntries();
      HLTInfo("Input tree with %d TRD MCM tracklets", nTracklets );

      AliTRDtrackletMCM *trkl = 0x0;
      trklbranch->SetAddress(&trkl);
      
      for (Int_t iTracklet = 0; iTracklet < nTracklets; iTracklet++) {
	int nbytes = trklbranch->GetEntry(iTracklet,1);
	if( !trkl || nbytes<=0 ){
	  //HLTWarning("Can not read entry %d of %d from tracklet branch", &iTracklet, &nTracklets);
	  HLTWarning("Can not read entry from tracklet branch");
	  continue;
	}
	AliHLTTRDTrackletWord hltTrkl = *trkl;
	hltTrkl.SetId(iTracklet);
	outputTrkls.push_back(hltTrkl);	        
      }
    }    
  }
  

 if( outputTrkls.size()>0 ){
   iResult = PushBack(&outputTrkls[0], outputTrkls.size() * sizeof(outputTrkls[0]), AliHLTTRDDefinitions::fgkTRDTrackletDataType, 0);  
 }

 HLTInfo("### END   DoEvent [event id: %llu, %d blocks, size: %d, output tracklets: %d]",
	 hltEventData.fEventID, hltEventData.fBlockCnt, hltEventData.fStructSize, outputTrkls.size() );   

 return iResult;
}
