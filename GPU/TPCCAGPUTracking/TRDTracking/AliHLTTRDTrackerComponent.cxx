// $Id: AliHLTTRDTrackerComponent.cxx 2016-05-22 16:08:40Z marten $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Marten Ole Schmidt <ole.schmidt@cern.ch>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

///  @file   AliHLTTRDTrackerComponent.cxx
///  @author Marten Ole Schmidt <ole.schmidt@cern.ch>
///  @date   May 2016
///  @brief  A TRD tracker processing component for the HLT


/////////////////////////////////////////////////////
//                                                 //
// a TRD tracker processing component for the HLT  //
//                                                 //
/////////////////////////////////////////////////////

#include "TSystem.h"
#include "TTimeStamp.h"
#include "TObjString.h"
#include "TClonesArray.h"
#include "TObjArray.h"
#include "AliESDEvent.h"
#include "AliHLTErrorGuard.h"
#include "AliHLTDataTypes.h"
#include "AliHLTTRDGeometry.h"
#include "AliHLTTRDTracker.h"
#include "AliHLTTRDTrack.h"
#include "AliHLTTRDTrackerComponent.h"
#include "AliHLTTRDTrackletWord.h"
#include "AliHLTTRDTrackletLabels.h"
#include "AliHLTTRDDefinitions.h"
#include "AliHLTTPCDefinitions.h"
#include "AliHLTTRDTrackPoint.h"
#include "AliHLTGlobalBarrelTrack.h"
#include "AliExternalTrackParam.h"
#include "AliHLTExternalTrackParam.h"
#include "AliHLTTrackMCLabel.h"
#include "AliHLTTRDTrackData.h"
#include "AliGeomManager.h"
#include <map>
#include <vector>
#include <algorithm>


ClassImp(AliHLTTRDTrackerComponent)

AliHLTTRDTrackerComponent::AliHLTTRDTrackerComponent() :
  fTracker(0x0),
  fGeo(0x0),
  fTrackList(0x0),
  fDebugTrackOutput(false),
  fVerboseDebugOutput(false),
  fRequireITStrack(false),
  fBenchmark("TRDTracker")
{
}

AliHLTTRDTrackerComponent::AliHLTTRDTrackerComponent( const AliHLTTRDTrackerComponent& )
  :
  fTracker(0x0),
  fGeo(0x0),
  fTrackList(0x0),
  AliHLTProcessor(),
  fDebugTrackOutput(false),
  fVerboseDebugOutput(false),
  fRequireITStrack(false),
  fBenchmark("TRDTracker")
{
  // see header file for class documentation
  HLTFatal( "copy constructor untested" );
}

AliHLTTRDTrackerComponent& AliHLTTRDTrackerComponent::operator=( const AliHLTTRDTrackerComponent& )
{
  // see header file for class documentation
  HLTFatal( "assignment operator untested" );
  return *this;
}

AliHLTTRDTrackerComponent::~AliHLTTRDTrackerComponent() {
  delete fTracker;
}

const char* AliHLTTRDTrackerComponent::GetComponentID() {
  return "TRDTracker";
}

void AliHLTTRDTrackerComponent::GetInputDataTypes( std::vector<AliHLTComponentDataType>& list) {
  list.clear();
  list.push_back( kAliHLTDataTypeTrack|kAliHLTDataOriginITS );
  //list.push_back( kAliHLTDataTypeTrack|kAliHLTDataOriginTPC );
  list.push_back( AliHLTTPCDefinitions::TracksOuterDataType()|kAliHLTDataOriginTPC);
  list.push_back( kAliHLTDataTypeTrackMC|kAliHLTDataOriginTPC );
  list.push_back( AliHLTTRDDefinitions::fgkTRDTrackletDataType );
  list.push_back( AliHLTTRDDefinitions::fgkTRDMCTrackletDataType );
}

AliHLTComponentDataType AliHLTTRDTrackerComponent::GetOutputDataType() {
  return kAliHLTMultipleDataType;
}

int AliHLTTRDTrackerComponent::GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList)
{
  // see header file for class documentation
  tgtList.clear();
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDTrackDataType|kAliHLTDataOriginTRD);
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDTrackPointDataType|kAliHLTDataOriginTRD);
  tgtList.push_back(kAliHLTDataTypeTObject|kAliHLTDataOriginTRD);
  return tgtList.size();
}

void AliHLTTRDTrackerComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier ) {
  // define guess for the output data size
  constBase = 1000;       // minimum size
  inputMultiplier = 2.; // size relative to input
}

AliHLTComponent* AliHLTTRDTrackerComponent::Spawn() {
  // see header file for class documentation
  return new AliHLTTRDTrackerComponent;
}

int AliHLTTRDTrackerComponent::ReadConfigurationString(  const char* arguments )
{
  // Set configuration parameters for the TRD tracker component from the string

  int iResult = 0;
  if ( !arguments ) return iResult;

  TString allArgs = arguments;
  TString argument;

  TObjArray* pTokens = allArgs.Tokenize( " " );

  int nArgs =  pTokens ? pTokens->GetEntries() : 0;

  for ( int i = 0; i < nArgs; i++ ) {
    argument = ( ( TObjString* )pTokens->At( i ) )->GetString();
    if ( argument.IsNull() ) continue;

    if ( argument.CompareTo("-debugOutput") == 0 ) {
      fDebugTrackOutput = true;
      fVerboseDebugOutput = true;
      HLTInfo( "Tracks are dumped in the HLTTRDTrack format" );
      continue;
    }

    if ( argument.CompareTo("-requireITStrack") == 0 ) {
      fRequireITStrack = true;
      HLTInfo( "TRD tracker requires seeds (TPC tracks) to have an ITS match" );
      continue;
    }

    HLTError( "Unknown option \"%s\"", argument.Data() );
    iResult = -EINVAL;

  }
  delete pTokens;

  return iResult;
}


// #################################################################################
int AliHLTTRDTrackerComponent::DoInit( int argc, const char** argv ) {
  // see header file for class documentation

  int iResult=0;
  if ( fTracker ) {
    return -EINPROGRESS;
  }

  fBenchmark.Reset();
  fBenchmark.SetTimer(0,"total");
  fBenchmark.SetTimer(1,"reco");

  if(AliGeomManager::GetGeometry()==NULL) {
    AliGeomManager::LoadGeometry();
  }

  fTrackList = new TList();
  if (!fTrackList) {
    return -ENOMEM;
  }
  fTrackList->SetOwner(kFALSE);

  TString arguments = "";
  for ( int i = 0; i < argc; i++ ) {
    if ( !arguments.IsNull() ) arguments += " ";
    arguments += argv[i];
  }

  iResult = ReadConfigurationString( arguments.Data() );

  fGeo = new AliHLTTRDGeometry();
  if (!fGeo) {
    return -ENOMEM;
  }
  if (!AliHLTTRDGeometry::CheckGeometryAvailable()) {
    HLTError("TRD geometry not available");
    return -EINVAL;
  }
  fTracker = new AliHLTTRDTracker();
  if (!fTracker) {
    return -ENOMEM;
  }
  if (fVerboseDebugOutput) {
    fTracker->EnableDebugOutput();
  }
  fTracker->Init(fGeo);

  return iResult;
}



// #################################################################################
int AliHLTTRDTrackerComponent::DoDeinit() {
  // see header file for class documentation
  delete fTracker;
  fTracker = 0x0;
  delete fGeo;
  fGeo = 0x0;
  return 0;
}

// #################################################################################
int AliHLTTRDTrackerComponent::DoEvent
(
  const AliHLTComponentEventData& evtData,
  const AliHLTComponentBlockData* blocks,
  AliHLTComponentTriggerData& /*trigData*/,
  AliHLTUInt8_t* outputPtr,
  AliHLTUInt32_t& size,
  std::vector<AliHLTComponentBlockData>& outputBlocks )
{
  // process event

  if (!IsDataEvent()) return 0;

  if ( evtData.fBlockCnt <= 0 ) {
    HLTWarning( "no blocks in event" );
    return 0;
  }

  fBenchmark.StartNewEvent();
  fBenchmark.Start(0);

  AliHLTUInt32_t maxBufferSize = size;
  size = 0; // output size

  int iResult=0;

  if (fTrackList->GetEntries() != 0) {
    fTrackList->Clear(); // tracks are owned by AliHLTTRDTracker
  }

  int nBlocks = evtData.fBlockCnt;

  AliHLTTracksData *tpcData = NULL;
  AliHLTTracksData *itsData = NULL;
  AliHLTTrackMCData *tpcDataMC = NULL;

  std::vector< HLTTRDTrack > tracksTPC;
  std::vector< int > tracksTPCLab;
  std::vector< int > tracksTPCId;

  bool hasMCtracklets = false;

  int nTrackletsTotal = 0;
  int nTrackletsTotalMC = 0;
  AliHLTTRDTrackletWord *tracklets = NULL;
  AliHLTTRDTrackletLabels *trackletsMC = NULL;

  for (int iBlock = 0; iBlock < nBlocks; iBlock++) {
    if (blocks[iBlock].fDataType == (kAliHLTDataTypeTrack | kAliHLTDataOriginITS) && fRequireITStrack) {
      itsData = (AliHLTTracksData*) blocks[iBlock].fPtr;
      fBenchmark.AddInput(blocks[iBlock].fSize);
    }
    else if (blocks[iBlock].fDataType == (AliHLTTPCDefinitions::TracksOuterDataType() | kAliHLTDataOriginTPC)) {
      tpcData = (AliHLTTracksData*) blocks[iBlock].fPtr;
      fBenchmark.AddInput(blocks[iBlock].fSize);
    }
    else if (blocks[iBlock].fDataType == (kAliHLTDataTypeTrackMC|kAliHLTDataOriginTPC)) {
      tpcDataMC = (AliHLTTrackMCData*) blocks[iBlock].fPtr;
      fBenchmark.AddInput(blocks[iBlock].fSize);
    }
    else if (blocks[iBlock].fDataType == (AliHLTTRDDefinitions::fgkTRDTrackletDataType)) {
      tracklets = reinterpret_cast<AliHLTTRDTrackletWord*>( blocks[iBlock].fPtr );
      nTrackletsTotal = blocks[iBlock].fSize / sizeof(AliHLTTRDTrackletWord);
      fBenchmark.AddInput(blocks[iBlock].fSize);
    }
    else if (blocks[iBlock].fDataType == (AliHLTTRDDefinitions::fgkTRDMCTrackletDataType)) {
      hasMCtracklets = true;
      trackletsMC = reinterpret_cast<AliHLTTRDTrackletLabels*>( blocks[iBlock].fPtr );
      nTrackletsTotalMC = blocks[iBlock].fSize / sizeof(AliHLTTRDTrackletLabels);
      fBenchmark.AddInput(blocks[iBlock].fSize);
    }
  }

  if (tpcData == NULL) {
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

  int nTPCtracks = tpcData->fCount;
  std::vector<bool> itsAvail(nTPCtracks, false);
  if (itsData) {
    // look for ITS tracks with >= 2 hits
    int nITStracks = itsData->fCount;
    AliHLTExternalTrackParam *currITStrack = itsData->fTracklets;
    for (int iTrkITS = 0; iTrkITS < nITStracks; iTrkITS++) {
      if (currITStrack->fNPoints >= 2) {
        itsAvail.at(currITStrack->fTrackID) = true;
      }
      unsigned int dSize = sizeof(AliHLTExternalTrackParam) + currITStrack->fNPoints * sizeof(unsigned int);
      currITStrack = (AliHLTExternalTrackParam*) ( ((Byte_t*) currITStrack) + dSize);
    }
  }
  std::map<int,int> mcLabels;
  if (tpcDataMC) {
    // look for TPC track MC labels
    int nMCtracks = tpcDataMC->fCount;
    for (int iMC = 0; iMC < nMCtracks; iMC++) {
      AliHLTTrackMCLabel &lab = tpcDataMC->fLabels[iMC];
      mcLabels[lab.fTrackID] = lab.fMCLabel;
    }
  }
  AliHLTExternalTrackParam *currOutTrackTPC = tpcData->fTracklets;
  for (int iTrk = 0; iTrk < nTPCtracks; iTrk++) {
    // store TPC tracks (if required only the ones with >=2 ITS hits)
    if (itsData != NULL && !itsAvail.at(currOutTrackTPC->fTrackID)) {
      continue;
    }
    HLTTRDTrack t(*currOutTrackTPC);
    int mcLabel = -1;
    if (tpcDataMC) {
      if (mcLabels.find(currOutTrackTPC->fTrackID) != mcLabels.end()) {
        mcLabel = mcLabels[currOutTrackTPC->fTrackID];
      }
    }
    tracksTPC.push_back( t );
    tracksTPCId.push_back( currOutTrackTPC->fTrackID );
    tracksTPCLab.push_back( mcLabel );
    unsigned int dSize = sizeof(AliHLTExternalTrackParam) + currOutTrackTPC->fNPoints * sizeof(unsigned int);
    currOutTrackTPC = (AliHLTExternalTrackParam*) + ( ((Byte_t*) currOutTrackTPC) + dSize );
  }




  if (fVerboseDebugOutput) {
    HLTInfo("TRDTrackerComponent received %i tracklets\n", nTrackletsTotal);
  }

  fTracker->Reset();
  fTracker->StartLoadTracklets(nTrackletsTotal);

  // loop over all tracklets
  for (int iTracklet=0; iTracklet<nTrackletsTotal; ++iTracklet){
    if (!hasMCtracklets) {
	    fTracker->LoadTracklet(tracklets[iTracklet]);
    }
    else {
	    fTracker->LoadTracklet(tracklets[iTracklet], trackletsMC[iTracklet].fLabel);
    }
  }

  fBenchmark.Start(1);
  fTracker->DoTracking(&(tracksTPC[0]), &(tracksTPCLab[0]), tracksTPC.size());
  fBenchmark.Stop(1);

  HLTTRDTrack *trackArray = fTracker->Tracks();
  int nTracks = fTracker->NTracks();
  AliHLTTRDTracker::AliHLTTRDSpacePointInternal *spacePoints = fTracker->SpacePoints();

  // TODO delete fTrackList since it only works for TObjects (or use compiler flag after tests with HLT track type)
  //for (int iTrack=0; iTrack<nTracks; ++iTrack) {
  //  fTrackList->AddLast(&trackArray[iTrack]);
  //}

  // push back AliHLTTRDTracks for debugging purposes
  if (fDebugTrackOutput) {
    PushBack(fTrackList, (kAliHLTDataTypeTObject | kAliHLTDataOriginTRD), 0x3fffff);
  }
  // push back AliHLTExternalTrackParam (default)
  else {

    AliHLTUInt32_t blockSize = AliHLTTRDTrackData::GetSize( nTracks );
    if (size + blockSize > maxBufferSize) {
      HLTWarning( "Output buffer exceeded for tracks" );
      return -ENOSPC;
    }

    AliHLTTRDTrackData* outTracks = ( AliHLTTRDTrackData* )( outputPtr );
    outTracks->fCount = 0;

    for (int iTrk=0; iTrk<nTracks; ++iTrk) {
      HLTTRDTrack &t = trackArray[iTrk];
      if (t.GetNtracklets() == 0) continue;
      AliHLTTRDTrackDataRecord &currOutTrack = outTracks->fTracks[outTracks->fCount];
      t.ConvertTo(currOutTrack);
      outTracks->fCount++;
    }

    AliHLTComponentBlockData resultData;
    FillBlockData( resultData );
    resultData.fOffset = size;
    resultData.fSize = blockSize;
    resultData.fDataType = AliHLTTRDDefinitions::fgkTRDTrackDataType;
    outputBlocks.push_back( resultData );
    fBenchmark.AddOutput(resultData.fSize);

    size += blockSize;
    outputPtr += resultData.fSize;

    blockSize = 0;

    // space points calculated from tracklets

    blockSize = sizeof(AliHLTTRDTrackPointData) + sizeof(AliHLTTRDTrackPoint) * nTrackletsTotal;

    if (size + blockSize > maxBufferSize) {
      HLTWarning( "Output buffer exceeded for space points" );
      return -ENOSPC;
    }

    AliHLTTRDTrackPointData* outTrackPoints = ( AliHLTTRDTrackPointData* )( outputPtr );
    outTrackPoints->fCount = nTrackletsTotal;

    { // fill array with 0 for a case..
      AliHLTTRDTrackPoint empty;
      empty.fX[0] = 0;
      empty.fX[1] = 0;
      empty.fX[2] = 0;
      empty.fVolumeId = 0;
      for (int i=0; i<nTrackletsTotal; ++i) {
	outTrackPoints->fPoints[i] = empty;
      }
    }

    for (int i=0; i<nTrackletsTotal; ++i) {
      const AliHLTTRDTracker::AliHLTTRDSpacePointInternal &sp = spacePoints[i];
      int id = sp.fId;
      if( id<0 || id>=nTrackletsTotal ){
	HLTError("Internal error: wrong space point index %d", id );
      }
      AliHLTTRDTrackPoint *currOutPoint = &outTrackPoints->fPoints[id];
      currOutPoint->fX[0] = sp.fR; // x in sector coordinates
      currOutPoint->fX[1] = sp.fX[0]; // y in sector coordinates
      currOutPoint->fX[2] = sp.fX[1]; // z in sector coordinates
      currOutPoint->fVolumeId = sp.fVolumeId;
    }
    AliHLTComponentBlockData resultDataSP;
    FillBlockData( resultDataSP );
    resultDataSP.fOffset = size;
    resultDataSP.fSize = blockSize;
    resultDataSP.fDataType = AliHLTTRDDefinitions::fgkTRDTrackPointDataType|kAliHLTDataOriginTRD;
    outputBlocks.push_back( resultDataSP );
    fBenchmark.AddOutput(resultData.fSize);
    size += blockSize;
    outputPtr += resultDataSP.fSize;

    HLTInfo("TRD tracker: output %d tracks and %d track points",outTracks->fCount, outTrackPoints->fCount);
  }

  fBenchmark.Stop(0);
  HLTInfo( fBenchmark.GetStatistics() );

  return iResult;
}

// #################################################################################
int AliHLTTRDTrackerComponent::Reconfigure(const char* cdbEntry, const char* chainId) {
  // see header file for class documentation

  int iResult=0;
  TString cdbPath;
  if (cdbEntry) {
    cdbPath=cdbEntry;
  } else {
    cdbPath="HLT/ConfigGlobal/";
    cdbPath+=GetComponentID();
  }

  AliInfoClass(Form("reconfigure '%s' from entry %s%s", chainId, cdbPath.Data(), cdbEntry?"":" (default)"));
  iResult=ConfigureFromCDBTObjString(cdbPath);

  return iResult;
}
