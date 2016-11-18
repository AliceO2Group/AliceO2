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
#include "AliHLTTRDTracker.h"
#include "AliHLTTRDtrack.h"
#include "AliHLTTRDTrackerComponent.h"
#include "AliHLTTRDTrackletWord.h"
#include "AliHLTTRDDefinitions.h"
#include "AliHLTTRDSpacePoint.h"
#include "AliHLTGlobalBarrelTrack.h"
#include "AliExternalTrackParam.h"
#include "AliHLTExternalTrackParam.h"
#include "AliHLTTrackMCLabel.h"
#include <map>


ClassImp(AliHLTTRDTrackerComponent)

AliHLTTRDTrackerComponent::AliHLTTRDTrackerComponent() :
  fTracker(0x0),
  fTrackList(0x0),
  fDebugTrackOutput(false)
{
}

AliHLTTRDTrackerComponent::AliHLTTRDTrackerComponent( const AliHLTTRDTrackerComponent& )
  :
  fTracker(0x0),
  fTrackList(0x0),
  AliHLTProcessor(),
  fDebugTrackOutput(false)
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
  list.push_back( kAliHLTDataTypeTrack|kAliHLTDataOriginTPC );
  list.push_back( kAliHLTDataTypeTrackMC|kAliHLTDataOriginTPC );
  list.push_back( AliHLTTRDDefinitions::fgkTRDTrackletDataType );
}

AliHLTComponentDataType AliHLTTRDTrackerComponent::GetOutputDataType() {
  return kAliHLTMultipleDataType;
}

int AliHLTTRDTrackerComponent::GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList)
{
  // see header file for class documentation
  tgtList.clear();
  tgtList.push_back(kAliHLTDataTypeTrack|kAliHLTDataOriginTRD);
  tgtList.push_back(AliHLTTRDDefinitions::fgkTRDSpacePointDataType);
  tgtList.push_back(kAliHLTDataTypeTObject|kAliHLTDataOriginTRD);
  return tgtList.size();
}

void AliHLTTRDTrackerComponent::GetOutputDataSize( ULong_t& constBase, Double_t& inputMultiplier ) {
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
      HLTWarning( "Tracks are dumped in the AliHLTTRDtrack format" );
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
  printf("AliHLTTRDTrackerComponent::DoInit\n");

  int iResult=0;
  if ( fTracker ) return -EINPROGRESS;

  fTrackList = new TList();
  if (!fTrackList)
    return -ENOMEM;
  fTrackList->SetOwner(kFALSE);

  fTracker = new AliHLTTRDTracker();
  fTracker->Init();

  TString arguments = "";
  for ( int i = 0; i < argc; i++ ) {
    if ( !arguments.IsNull() ) arguments += " ";
    arguments += argv[i];
  }

  iResult = ReadConfigurationString( arguments.Data() );

  return iResult;
}



// #################################################################################
int AliHLTTRDTrackerComponent::DoDeinit() {
  // see header file for class documentation
  delete fTracker;
  fTracker = 0x0;
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

  AliHLTUInt32_t maxBufferSize = size;
  size = 0; // output size  

  int iResult=0;

  if (fTrackList->GetEntries() != 0) {
    fTrackList->Clear(); // tracks are owned by AliHLTTRDTracker
  }

  int nBlocks = evtData.fBlockCnt;

  std::vector< AliExternalTrackParam > tracksTPC;
  std::vector< int > tracksTPCLab;
  std::vector< int > tracksTPCId;

  int nTrackletsTotal = 0;
  AliHLTTRDTrackletWord *tracklets = NULL;


  for (int ndx=0; ndx<nBlocks && iResult>=0; ndx++) {

    if (ndx > 2)
      HLTWarning("unexpected number of blocks (%i) for this component, expected 2", ndx);

    const AliHLTComponentBlockData* iter = blocks+ndx;

    // Look for (and store if present) MC information

    std::map<int,int> mcLabels; // mapping of TPC track ID <-> MC track label
    for (const AliHLTComponentBlockData *pBlock = GetFirstInputBlock(kAliHLTDataTypeTrackMC|kAliHLTDataOriginTPC); pBlock != 0x0; pBlock = GetNextInputBlock()) {
      AliHLTTrackMCData *dataPtr = reinterpret_cast<AliHLTTrackMCData*>( pBlock->fPtr );
      if (sizeof(AliHLTTrackMCData) + dataPtr->fCount * sizeof(AliHLTTrackMCLabel) == pBlock->fSize) {
        for (unsigned int il=0; il < dataPtr->fCount; ++il) {
          AliHLTTrackMCLabel &lab = dataPtr->fLabels[il];
          mcLabels[lab.fTrackID] = lab.fMCLabel;
        }
      }
      else {
        HLTWarning("data mismatch in block %s (0x%08x): count %d, size %d -> ignoring track MC information", DataType2Text(pBlock->fDataType).c_str(), pBlock->fSpecification, dataPtr->fCount, pBlock->fSize);
      }
    }

    // Read TPC tracks

    if( iter->fDataType == ( kAliHLTDataTypeTrack|kAliHLTDataOriginTPC ) ){
      AliHLTTracksData* dataPtr = ( AliHLTTracksData* ) iter->fPtr;
      int nTracks = dataPtr->fCount;
      AliHLTExternalTrackParam* currOutTrack = dataPtr->fTracklets;
      for( int itr=0; itr<nTracks; itr++ ){
	      AliHLTGlobalBarrelTrack t(*currOutTrack);
	      tracksTPC.push_back( t );
	      tracksTPCId.push_back( currOutTrack->fTrackID );
	      unsigned int dSize = sizeof( AliHLTExternalTrackParam ) + currOutTrack->fNPoints * sizeof( unsigned int );
	      currOutTrack = ( AliHLTExternalTrackParam* )( (( Byte_t * )currOutTrack) + dSize );
      }
    }

    // Read TRD tracklets

    if ( iter->fDataType == (AliHLTTRDDefinitions::fgkTRDTrackletDataType) ){
      tracklets = reinterpret_cast<AliHLTTRDTrackletWord*>( iter->fPtr );
      nTrackletsTotal = iter->fSize / sizeof(AliHLTTRDTrackletWord);
    }

  }// end read input blocks

  printf("TRDTrackerComponent recieved %i tracklets\n", nTrackletsTotal);

  fTracker->Reset();
  fTracker->StartLoadTracklets(nTrackletsTotal);

  // loop over all tracklets
  for (int iTracklet=0; iTracklet<nTrackletsTotal; ++iTracklet){
	  fTracker->LoadTracklet(tracklets[iTracklet]);
  }

  fTracker->DoTracking(&(tracksTPC[0]), &(tracksTPCLab[0]), tracksTPC.size());

  AliHLTTRDtrack *trackArray = fTracker->Tracks();
  int nTracks = fTracker->NTracks();
  AliHLTTRDTracker::AliHLTTRDSpacePointInternal *spacePoints = fTracker->SpacePoints();

  if (nTracks >= 2000) {
    HLTWarning("Too many tracks in AliHLTTRDTracker (%i), skipping the last %i tracks", nTracks, nTracks-2000);
  }
  for (int iTrack=0; iTrack<nTracks; ++iTrack) {
    fTrackList->AddLast(&trackArray[iTrack]);
  }

  // push back AliHLTTRDtracks for debugging purposes
  if (fDebugTrackOutput) {
    PushBack(fTrackList, (kAliHLTDataTypeTObject | kAliHLTDataOriginTRD), 0x3fffff);
  }
  // push back AliHLTExternalTrackParam (default)
  else {
 
    if (size + sizeof(AliHLTTracksData) > maxBufferSize) {
      HLTWarning( "Output buffer exceeded for AliHLTTracksData" );
      return -ENOSPC;
    }

    AliHLTTracksData* outTracks = ( AliHLTTracksData* )( outputPtr );
    outTracks->fCount = 0;
     
    AliHLTExternalTrackParam* currOutTrack = outTracks->fTracklets;

    AliHLTUInt32_t blockSize = sizeof(AliHLTTracksData);

    for (int iTrk=0; iTrk<nTracks; ++iTrk) {
      unsigned int dSize = sizeof(AliHLTExternalTrackParam) + 6 * sizeof( unsigned int );
      if (size + blockSize + dSize > maxBufferSize) {
        HLTWarning( "Output buffer exceeded for tracks" );
        return -ENOSPC;
      }
      AliHLTTRDtrack &t = trackArray[iTrk];
      currOutTrack->fAlpha = t.GetAlpha();
      currOutTrack->fX = t.GetX();
      currOutTrack->fY = t.GetY();
      currOutTrack->fZ = t.GetZ();
      currOutTrack->fLastX = 0;
      currOutTrack->fLastY = 0;
      currOutTrack->fLastZ = 0;
      currOutTrack->fq1Pt = t.GetSigned1Pt();
      currOutTrack->fSinPsi = t.GetSnp();
      currOutTrack->fTgl = t.GetTgl();
      for( int i=0; i<15; i++ ) currOutTrack->fC[i] = t.GetCovariance()[i];
      currOutTrack->fTrackID = t.GetTPCtrackId();
      currOutTrack->fFlags = 0;
      currOutTrack->fNPoints = 6;      
      for ( int i = 0; i <6; i++ ){
	currOutTrack->fPointIDs[ i ] = t.GetTracklet( i );        
      }
      dSize = sizeof(AliHLTExternalTrackParam) + currOutTrack->fNPoints * sizeof( unsigned int );
      blockSize += dSize;
      currOutTrack = ( AliHLTExternalTrackParam* )( (( Byte_t * )currOutTrack) + dSize );
      outTracks->fCount++;
    }

    AliHLTComponentBlockData resultData;
    FillBlockData( resultData );
    resultData.fOffset = size;
    resultData.fSize = blockSize;
    resultData.fDataType = kAliHLTDataTypeTrack|kAliHLTDataOriginTRD;
    outputBlocks.push_back( resultData );

    size += blockSize;
    outputPtr += resultData.fSize;
    
    blockSize = 0;

    // space points calculated from tracklets

    blockSize = sizeof(AliHLTTRDSpacePointData) + sizeof(AliHLTTRDSpacePoint) * nTrackletsTotal;

    if (size + blockSize > maxBufferSize) {
      HLTWarning( "Output buffer exceeded for space points" );
      return -ENOSPC;
    }

    AliHLTTRDSpacePointData* outSpacePoints = ( AliHLTTRDSpacePointData* )( outputPtr );
    outSpacePoints->fCount = nTrackletsTotal;

    { // fill array with 0 for a case..
      AliHLTTRDSpacePoint empty;
      empty.fX[0] = 0;
      empty.fX[1] = 0;
      empty.fX[2] = 0;      
      for (int i=0; i<nTrackletsTotal; ++i) {
	outSpacePoints->fPoints[i] = empty;
      }
    }

    for (int i=0; i<nTrackletsTotal; ++i) {
      const AliHLTTRDTracker::AliHLTTRDSpacePointInternal &sp = spacePoints[i];
      int id = sp.fId;
      if( id<0 || id>=nTrackletsTotal ){
	HLTError("Internal error: wrong space point index %d", id );
      }
      AliHLTTRDSpacePoint *currOutPoint = &outSpacePoints->fPoints[id];
      currOutPoint->fX[0] = sp.fX[0];
      currOutPoint->fX[1] = sp.fX[1];
      currOutPoint->fX[2] = sp.fX[2];
    }
    AliHLTComponentBlockData resultDataSP;
    FillBlockData( resultDataSP );
    resultDataSP.fOffset = size;
    resultDataSP.fSize = blockSize;
    resultData.fDataType = AliHLTTRDDefinitions::fgkTRDSpacePointDataType;
    outputBlocks.push_back( resultDataSP );
    size += blockSize;

    HLTInfo("TRD tracker: output %d tracks and %d tracklets (spacepoints)",outTracks->fCount, outSpacePoints->fCount);
  }

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
