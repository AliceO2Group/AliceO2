// @(#) $Id$
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
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


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// a TPC tracker processing component for the HLT based on CA by Ivan Kisel  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "AliHLTTPCCATrackerComponent.h"
#include "AliHLTTPCGeometry.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTArray.h"

#include "AliHLTTPCRawCluster.h"
#include "AliHLTTPCClusterXYZ.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCGeometry.h"
#include "AliHLTTPCDefinitions.h"
#include "AliExternalTrackParam.h"
#include "TMath.h"
#include "AliCDBEntry.h"
#include "AliCDBManager.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliRunLoader.h"
#include "AliHeader.h"
#include "TBranch.h"
#include "TTree.h"
#include "AliStack.h"
#include "AliMCParticle.h"
#include "AliTrackReference.h"
#include "AliHLTTPCCAMCInfo.h"
#include "AliHLTTPCGMMergedTrackHit.h"
#include "TPDGCode.h"
#if __GNUC__>= 3
using namespace std;
#endif

#include "../tracking-standalone/include/standaloneSettings.h"

const AliHLTComponentDataType AliHLTTPCCADefinitions::fgkTrackletsDataType = AliHLTComponentDataTypeInitializer( "CATRACKL", kAliHLTDataOriginTPC );

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp( AliHLTTPCCATrackerComponent )

  AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent()
  :
  fTracker( NULL ),
  fClusterData( NULL ),
  fMinSlice( 0 ),
  fSliceCount( fgkNSlices ),
  fSolenoidBz( 0 ),
  fMinNTrackClusters( -1 ),
  fMinTrackPt(MIN_TRACK_PT_DEFAULT),
  fClusterZCut( 500. ),
  fNeighboursSearchArea( 0 ), 
  fClusterErrorCorrectionY(0), 
  fClusterErrorCorrectionZ(0),
  fBenchmark("CATracker"), 
  fAllowGPU( 0),
  fGPUHelperThreads(-1),
  fCPUTrackers(0),
  fGlobalTracking(0),
  fGPUDeviceNum(-1),
  fGPULibrary(""),
  fGPUStuckProtection(0),
  fAsync(0),
  fDumpEvent(0),
  fDumpEventNClsCut(0),
  fSearchWindowDZDR(0.),
  fAsyncProcessor()
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt
  for( int i=0; i<fgkNSlices; i++ ){
    fSliceOutput[i] = NULL;
  }
}

AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent( const AliHLTTPCCATrackerComponent& )
  :
AliHLTProcessor(),
  fTracker( NULL ),
  fClusterData( NULL ),
  fMinSlice( 0 ),
  fSliceCount( fgkNSlices ),
  fSolenoidBz( 0 ),
  fMinNTrackClusters( -1 ),
  fMinTrackPt( MIN_TRACK_PT_DEFAULT ),
  fClusterZCut( 500. ),
  fNeighboursSearchArea(0),
  fClusterErrorCorrectionY(0), 
  fClusterErrorCorrectionZ(0),
  fBenchmark("CATracker"),
  fAllowGPU( 0),
  fGPUHelperThreads(-1),
  fCPUTrackers(0),
  fGlobalTracking(0),
  fGPUDeviceNum(-1),
  fGPULibrary(""),
  fGPUStuckProtection(0),
  fAsync(0),
  fDumpEvent(0),
  fDumpEventNClsCut(0),
  fSearchWindowDZDR(0.),
  fAsyncProcessor()
{
  // see header file for class documentation
  for( int i=0; i<fgkNSlices; i++ ){
    fSliceOutput[i] = NULL;
  }
  HLTFatal( "copy constructor untested" );
}

AliHLTTPCCATrackerComponent& AliHLTTPCCATrackerComponent::operator=( const AliHLTTPCCATrackerComponent& )
{
  // see header file for class documentation
  for( int i=0; i<fgkNSlices; i++ ){
    fSliceOutput[i] = NULL;
  }
  HLTFatal( "assignment operator untested" );
  return *this;
}

AliHLTTPCCATrackerComponent::~AliHLTTPCCATrackerComponent()
{
  // see header file for class documentation
  if (fTracker) delete fTracker;
  if (fClusterData) delete[] fClusterData;
}

//
// Public functions to implement AliHLTComponent's interface.
// These functions are required for the registration process
//

const char* AliHLTTPCCATrackerComponent::GetComponentID()
{
  // see header file for class documentation
  return "TPCCATracker";
}

void AliHLTTPCCATrackerComponent::GetInputDataTypes( vector<AliHLTComponentDataType>& list )
{
  // see header file for class documentation
  list.clear();
  list.push_back( AliHLTTPCDefinitions::RawClustersDataType() );
  list.push_back( AliHLTTPCDefinitions::ClustersXYZDataType() );
  list.push_back( AliHLTTPCDefinitions::AliHLTDataTypeClusterMCInfo() );
}

AliHLTComponentDataType AliHLTTPCCATrackerComponent::GetOutputDataType()
{
  // see header file for class documentation
  return AliHLTTPCCADefinitions::fgkTrackletsDataType;
}

void AliHLTTPCCATrackerComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier )
{
  // define guess for the output data size
  constBase = 10000;       // minimum size
  inputMultiplier = 0.6; // size relative to input
}

AliHLTComponent* AliHLTTPCCATrackerComponent::Spawn()
{
  // see header file for class documentation
  return new AliHLTTPCCATrackerComponent;
}

void AliHLTTPCCATrackerComponent::SetDefaultConfiguration()
{
  // Set default configuration for the CA tracker component
  // Some parameters can be later overwritten from the OCDB

  fSolenoidBz = -5.00668;
  fMinNTrackClusters = -1;
  fMinTrackPt = MIN_TRACK_PT_DEFAULT;
  fClusterZCut = 500.;
  fNeighboursSearchArea = 0;
  fClusterErrorCorrectionY = 0;
  fClusterErrorCorrectionZ = 0;
  fBenchmark.Reset();
  fBenchmark.SetTimer(0,"total");
  fBenchmark.SetTimer(1,"reco");
}

int AliHLTTPCCATrackerComponent::ReadConfigurationString(  const char* arguments )
{
  // Set configuration parameters for the CA tracker component from the string

  int iResult = 0;
  if ( !arguments ) return iResult;

  TString allArgs = arguments;
  TString argument;
  int bMissingParam = 0;

  TObjArray* pTokens = allArgs.Tokenize( " " );

  int nArgs =  pTokens ? pTokens->GetEntries() : 0;

  for ( int i = 0; i < nArgs; i++ ) {
    argument = ( ( TObjString* )pTokens->At( i ) )->GetString();
    if ( argument.IsNull() ) continue;

    if ( argument.CompareTo( "-solenoidBz" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      HLTWarning("argument -solenoidBz is deprecated, magnetic field set up globally (%f)", GetBz());
      continue;
    }

    if ( argument.CompareTo( "-minNClustersOnTrack" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fMinNTrackClusters = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      HLTInfo( "minNClustersOnTrack set to: %d", fMinNTrackClusters );
      continue;
    }

    if ( argument.CompareTo( "-minTrackPt" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fMinTrackPt = ( ( TObjString* )pTokens->At( i ) )->GetString().Atof();
      HLTInfo( "minTrackPt set to: %f", fMinTrackPt );
      continue;
    }

    if ( argument.CompareTo( "-clusterZCut" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fClusterZCut = TMath::Abs( ( ( TObjString* )pTokens->At( i ) )->GetString().Atof() );
      HLTInfo( "ClusterZCut set to: %f", fClusterZCut );
      continue;
    }

    if ( argument.CompareTo( "-neighboursSearchArea" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fNeighboursSearchArea = TMath::Abs( ( ( TObjString* )pTokens->At( i ) )->GetString().Atof() );
      HLTInfo( "NeighboursSearchArea set to: %f", fNeighboursSearchArea );
      continue;
    }

    if ( argument.CompareTo( "-errorCorrectionY" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fClusterErrorCorrectionY = ( ( TObjString* )pTokens->At( i ) )->GetString().Atof();
      HLTInfo( "Cluster Y error correction factor set to: %f", fClusterErrorCorrectionY );
      continue;
    }

    if ( argument.CompareTo( "-errorCorrectionZ" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fClusterErrorCorrectionZ = ( ( TObjString* )pTokens->At( i ) )->GetString().Atof();
      HLTInfo( "Cluster Z error correction factor set to: %f", fClusterErrorCorrectionZ );
      continue;
    }

    if (argument.CompareTo( "-allowGPU" ) == 0) {
      fAllowGPU = 1;
      HLTInfo( "Will try to run tracker on GPU" );
      continue;
    }

    if (argument.CompareTo( "-GlobalTracking" ) == 0) {
      fGlobalTracking = 1;
      HLTInfo( "Global Tracking Activated" );
      continue;
    }

    if (argument.CompareTo( "-DumpEvent" ) == 0) {
      fDumpEvent = 1;
      HLTImportant( "Dumping Events for Debugging" );
      continue;
    }

    if ( argument.CompareTo( "-DumpEventNClsCut" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fDumpEventNClsCut = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      HLTInfo( "Dump Event NCls cut set to: %d", fDumpEventNClsCut );
      continue;
    }

    if ( argument.CompareTo( "-GPUHelperThreads" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fGPUHelperThreads = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      HLTInfo( "Number of GPU Helper Threads set to: %d", fGPUHelperThreads );
      continue;
    }

    if ( argument.CompareTo( "-CPUTrackers" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fCPUTrackers = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      HLTInfo( "Number of CPU Trackers set to: %d", fCPUTrackers );
      continue;
    }

    if ( argument.CompareTo( "-SearchWindowDZDR" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fSearchWindowDZDR = ( ( TObjString* )pTokens->At( i ) )->GetString().Atof();
      HLTInfo( "Search Window DZDR set to: %f", fSearchWindowDZDR );
      continue;
    }


    if ( argument.CompareTo( "-GPUDeviceNum" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fGPUDeviceNum = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      HLTInfo( "Using GPU Device Number %d", fGPUDeviceNum );
      continue;
    }

    if ( argument.CompareTo( "-GPULibrary" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fGPULibrary = ( ( TObjString* )pTokens->At( i ) )->GetString();
      continue;
    }

    if ( argument.CompareTo( "-GPUStuckProtection" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fGPUStuckProtection = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      continue;
    }

	if ( argument.CompareTo( "-AsyncGPUStuckProtection" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fAsync = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      continue;
    }

    HLTError( "Unknown option \"%s\"", argument.Data() );
    iResult = -EINVAL;
  }
  delete pTokens;

  if ( bMissingParam ) {
    HLTError( "Specifier missed for parameter \"%s\"", argument.Data() );
    iResult = -EINVAL;
  }

  return iResult;
}


int AliHLTTPCCATrackerComponent::ReadCDBEntry( const char* cdbEntry, const char* chainId )
{
  // see header file for class documentation

  const char* defaultNotify = "";

  if ( !cdbEntry ) {
    cdbEntry = "HLT/ConfigTPC/TPCCATracker";
    defaultNotify = " (default)";
    chainId = 0;
  }

  HLTInfo( "configure from entry \"%s\"%s, chain id %s", cdbEntry, defaultNotify, ( chainId != NULL && chainId[0] != 0 ) ? chainId : "<none>" );
  AliCDBEntry *pEntry = AliCDBManager::Instance()->Get( cdbEntry );//,GetRunNo());

  if ( !pEntry ) {
    HLTError( "cannot fetch object \"%s\" from CDB", cdbEntry );
    return -EINVAL;
  }

  TObjString* pString = dynamic_cast<TObjString*>( pEntry->GetObject() );

  if ( !pString ) {
    HLTError( "configuration object \"%s\" has wrong type, required TObjString", cdbEntry );
    return -EINVAL;
  }

  HLTInfo( "received configuration object string: \"%s\"", pString->GetString().Data() );

  return  ReadConfigurationString( pString->GetString().Data() );
}


int AliHLTTPCCATrackerComponent::Configure( const char* cdbEntry, const char* chainId, const char *commandLine )
{
  // Configure the component
  // There are few levels of configuration,
  // parameters which are set on one step can be overwritten on the next step

  //* read hard-coded values
  SetDefaultConfiguration();

  //* read the default CDB entry
  int iResult1 = ReadCDBEntry( NULL, chainId );

  //* read magnetic field
  fSolenoidBz = GetBz();

  //* read the actual CDB entry if required
  int iResult2 = ( cdbEntry ) ? ReadCDBEntry( cdbEntry, chainId ) : 0;

  //* read extra parameters from input (if they are)
  int iResult3 = 0;

  if ( commandLine && commandLine[0] != '\0' ) {
    HLTInfo( "received configuration string from HLT framework: \"%s\"", commandLine );
    iResult3 = ReadConfigurationString( commandLine );
  }

  if (fTracker) ConfigureSlices();

  return iResult1 ? iResult1 : ( iResult2 ? iResult2 :  iResult3  );
}

void AliHLTTPCCATrackerComponent::ConfigureSlices()
{
  // Initialize the tracker slices
  for (int slice = 0;slice < fgkNSlices;slice++)
  {
    int iSec = slice;
    float inRmin = 83.65;
    //    float inRmax = 133.3;
    //    float outRmin = 133.5;
    float outRmax = 247.7;
    float plusZmin = 0.0529937;
    float plusZmax = 249.778;
    float minusZmin = -249.645;
    float minusZmax = -0.0799937;
    float dalpha = 0.349066;
    float alpha = 0.174533 + dalpha * iSec;

    bool zPlus = ( iSec < 18 );
    float zMin =  zPlus ? plusZmin : minusZmin;
    float zMax =  zPlus ? plusZmax : minusZmax;
    //TPCZmin = -249.645, ZMax = 249.778
    //    float rMin =  inRmin;
    //    float rMax =  outRmax;
    int nRows = AliHLTTPCGeometry::GetNRows();

    float padPitch = 0.4;
    float sigmaZ = 0.228808;

    float *rowX = new float [nRows];
    for ( int irow = 0; irow < nRows; irow++ ) {
      rowX[irow] = AliHLTTPCGeometry::Row2X( irow );
    }

    AliHLTTPCCAParam param;

    param.Initialize( iSec, nRows, rowX, alpha, dalpha,
      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, fSolenoidBz );
    param.SetHitPickUpFactor( 2 );
    if( fNeighboursSearchArea>0 ) param.SetNeighboursSearchArea( fNeighboursSearchArea );
    if( fClusterErrorCorrectionY>1.e-4 ) param.SetClusterError2CorrectionY( fClusterErrorCorrectionY*fClusterErrorCorrectionY );
    if( fClusterErrorCorrectionZ>1.e-4 ) param.SetClusterError2CorrectionZ( fClusterErrorCorrectionZ*fClusterErrorCorrectionZ );
    param.SetMinNTrackClusters( fMinNTrackClusters );
    param.SetMinTrackPt( fMinTrackPt );
    param.SetSearchWindowDZDR(fSearchWindowDZDR);
    param.LoadClusterErrors();

    param.Update();
    fTracker->InitializeSliceParam( slice, param );
    delete[] rowX;
  }
}

void* AliHLTTPCCATrackerComponent::TrackerInit(void* par)
{
    fMinSlice = 0;
    fSliceCount = fgkNSlices;
    //Create tracker instance and set parameters
    fTracker = new AliHLTTPCCATrackerFramework(fAllowGPU, fGPULibrary, fGPUDeviceNum);
    if ( fAllowGPU && fTracker->GetGPUStatus() < 2 ) {
      HLTError("GPU Tracker requested but unavailable, aborting.");
      return((void*) -1);
    }
    fClusterData = new AliHLTTPCCAClusterData[fgkNSlices];
    if (fGPUHelperThreads != -1)
    {
      char cc[256] = "HelperThreads";
      fTracker->SetGPUTrackerOption(cc, fGPUHelperThreads);
    }
    {
      char cc[256] = "CPUTrackers";
      fTracker->SetGPUTrackerOption(cc, fCPUTrackers);
      char cc2[256] = "SlicesPerCPUTracker";
      fTracker->SetGPUTrackerOption(cc2, 1);
    }
    if (fGlobalTracking)
    {
      char cc[256] = "GlobalTracking";
      fTracker->SetGPUTrackerOption(cc, 1);
    }
    if (fGPUStuckProtection)
    {
      char cc[256] = "StuckProtection";
      fTracker->SetGPUTrackerOption(cc, fGPUStuckProtection);
    }

    ConfigureSlices();
    return(NULL);
}

int AliHLTTPCCATrackerComponent::DoInit( int argc, const char** argv )
{
  if ( fTracker ) return EINPROGRESS;

  // Configure the CA tracker component
  TString arguments = "";
  for ( int i = 0; i < argc; i++ ) {
    if ( !arguments.IsNull() ) arguments += " ";
    arguments += argv[i];
  }

  int retVal = Configure( NULL, NULL, arguments.Data() );
  if (retVal == 0)
  {
    if (fAsync)
    {
      if (fAsyncProcessor.Initialize(1)) return(-ENODEV);
      void* initRetVal;
      if (fAsyncProcessor.InitializeAsyncMemberTask(this, &AliHLTTPCCATrackerComponent::TrackerInit, NULL, &initRetVal) != 0) return(-ENODEV);
      if (initRetVal) return(-ENODEV);
    }
    else
    {
      if (TrackerInit(NULL) != NULL) return(-ENODEV);
    }
  }

  return(retVal);
}

void* AliHLTTPCCATrackerComponent::TrackerExit(void* par)
{
    if (fTracker) delete fTracker;
    fTracker = NULL;
    if (fClusterData) delete[] fClusterData;
    fClusterData = NULL;
	return(NULL);
}

int AliHLTTPCCATrackerComponent::DoDeinit()
{
  // see header file for class documentation
  if (fAsync)
  {
    void* initRetVal = NULL;
    fAsyncProcessor.InitializeAsyncMemberTask(this, &AliHLTTPCCATrackerComponent::TrackerExit, NULL, &initRetVal);
    fAsyncProcessor.Deinitialize();
  }
  else
  {
    TrackerExit(NULL);
  }
  return 0;
}

int AliHLTTPCCATrackerComponent::Reconfigure( const char* cdbEntry, const char* chainId )
{
  // Reconfigure the component from OCDB .
  return Configure( cdbEntry, chainId, NULL );
}

int AliHLTTPCCATrackerComponent::DoEvent
  (
  const AliHLTComponentEventData& evtData,
  const AliHLTComponentBlockData* blocks,
  AliHLTComponentTriggerData& /*trigData*/,
  AliHLTUInt8_t* outputPtr,
  AliHLTUInt32_t& size,
  vector<AliHLTComponentBlockData>& outputBlocks )
{
  //* process event
  if (!fTracker)
  {
    HLTError( "CATracker not initialized properly" );
    return -ENOENT;
  }
  
  AliHLTTPCTrackerWrapperData tmpPar;
  tmpPar.fEvtData = &evtData;
  tmpPar.fBlocks = blocks;
  tmpPar.fOutputPtr = outputPtr;
  tmpPar.fSize = &size;
  tmpPar.fOutputBlocks = &outputBlocks;
  
  static int trackerTimeout = 0;
  if (trackerTimeout)
  {
    size = 0;
    return(0);
  }
  
  int retVal;
  if (fAsync)
  {
    void* asyncRetVal = NULL;
    if (fAsyncProcessor.InitializeAsyncMemberTask(this, &AliHLTTPCCATrackerComponent::TrackerDoEvent, &tmpPar, &asyncRetVal, fAsync) != 0)
    {
      HLTError( "Tracking timed out, disabling this tracker instance" );
      trackerTimeout = 1;
      size = 0;
      return(-ENODEV);
    }
    else
    {
      retVal = (int) (size_t) asyncRetVal;
    }
  }
  else
  {
    retVal = (int) (size_t) TrackerDoEvent(&tmpPar);
  }
  return(retVal);
}
  
void* AliHLTTPCCATrackerComponent::TrackerDoEvent(void* par)
{
  AliHLTTPCTrackerWrapperData* tmpPar = (AliHLTTPCTrackerWrapperData*) par;
  
  const AliHLTComponentEventData& evtData = *(tmpPar->fEvtData);
  const AliHLTComponentBlockData* blocks = tmpPar->fBlocks;
  AliHLTUInt8_t* outputPtr = tmpPar->fOutputPtr;
  AliHLTUInt32_t& size = *(tmpPar->fSize);
  vector<AliHLTComponentBlockData>& outputBlocks = *(tmpPar->fOutputBlocks);
  
  AliHLTUInt32_t maxBufferSize = size;
  size = 0; // output size

  if ( GetFirstInputBlock( kAliHLTDataTypeSOR ) || GetFirstInputBlock( kAliHLTDataTypeEOR ) ) {
    return 0;
  }

  fBenchmark.StartNewEvent();
  fBenchmark.Start(0);

  //Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "CA::DoEvent()" );
  if ( evtData.fBlockCnt <= 0 ) {
    HLTWarning( "no blocks in event" );
    return 0;
  }

  //Prepare everything for all slices
  const AliHLTTPCClusterMCData* clusterLabels[36][6] = {NULL};
  const AliHLTTPCClusterXYZData* clustersXYZ[36][6] = {NULL};
  const AliHLTTPCRawClusterData* clustersRaw[36][6] = {NULL};
  bool labelsPresent = false;
  
  for ( unsigned long ndx = 0; ndx < evtData.fBlockCnt; ndx++ )
  {
    const AliHLTComponentBlockData &pBlock = blocks[ndx];
    int slice = AliHLTTPCDefinitions::GetMinSliceNr(pBlock);
    int patch = AliHLTTPCDefinitions::GetMinPatchNr(pBlock);
    if ( pBlock.fDataType == AliHLTTPCDefinitions::RawClustersDataType() )
    {
      clustersRaw[slice][patch] = (const AliHLTTPCRawClusterData*) pBlock.fPtr;
    }
    else if ( pBlock.fDataType == AliHLTTPCDefinitions::ClustersXYZDataType() )
    {
      clustersXYZ[slice][patch] = (const AliHLTTPCClusterXYZData*) pBlock.fPtr;
    }
    else if ( pBlock.fDataType == AliHLTTPCDefinitions::AliHLTDataTypeClusterMCInfo() )
    {
      clusterLabels[slice][patch] = (const AliHLTTPCClusterMCData*) pBlock.fPtr;
      labelsPresent = true;
    }
  }
  
  int nClustersTotal = 0;
  for (int islice = 0;islice < fSliceCount;islice++)
  {
    int slice = fMinSlice + islice;
    int nClustersSliceTotal = 0;
    for (int patch = 0;patch < 6;patch++)
    {
      if (clustersXYZ[slice][patch]) nClustersSliceTotal += clustersXYZ[slice][patch]->fCount;
    }
    if (nClustersSliceTotal > 500000)
    {
      HLTWarning( "Too many clusters in tracker input: Slice %d, Number of Clusters %d, slice not included in tracking", slice, nClustersSliceTotal );
      fClusterData[islice].StartReading(slice, 0);
    }
    else if (nClustersSliceTotal == 0)
    {
      fClusterData[islice].StartReading(slice, 0);
    }
    else
    {
      fClusterData[islice].StartReading( slice, nClustersSliceTotal );
      AliHLTTPCCAClusterData::Data* pCluster = fClusterData[islice].Clusters();
      for (int patch = 0;patch < 6;patch++)
      {
        if (clustersXYZ[slice][patch] != NULL && clustersRaw[slice][patch] != NULL)
        {
          const AliHLTTPCClusterXYZData& clXYZ = *clustersXYZ[slice][patch];
          const AliHLTTPCRawClusterData& clRaw = *clustersRaw[slice][patch];
          
          if (clXYZ.fCount != clRaw.fCount)
          {
            HLTError("Number of entries in raw and xyz clusters are not mached %d vs %d", clXYZ.fCount, clRaw.fCount);
            continue;
          }

          const int firstRow = AliHLTTPCGeometry::GetFirstRow(patch);
          for (int ic = 0;ic < clXYZ.fCount;ic++)
          {
            const AliHLTTPCClusterXYZ &c = clXYZ.fClusters[ic];
            const AliHLTTPCRawCluster &cRaw = clRaw.fClusters[ic];
            if ( c.GetZ() > fClusterZCut || c.GetZ() < -fClusterZCut) continue;
            if ( c.GetX() <1.f ) continue; // cluster xyz position was not calculated for whatever reason
            pCluster->fId = AliHLTTPCGeometry::CreateClusterID( slice, patch, ic );
            pCluster->fX = c.GetX();
            pCluster->fY = c.GetY();
            pCluster->fZ = c.GetZ();
            pCluster->fRow = firstRow + cRaw.GetPadRow();
            pCluster->fFlags = cRaw.GetFlags();
            if (cRaw.GetSigmaPad2() < kAlmost0 || cRaw.GetSigmaTime2() < kAlmost0) pCluster->fFlags |= AliHLTTPCGMMergedTrackHit::flagSingle;
            pCluster->fAmp = cRaw.GetCharge();
#ifdef HLTCA_FULL_CLUSTERDATA
            pCluster->fPad = cRaw.GetPad();
            pCluster->fTime = cRaw.GetTime();
            pCluster->fAmpMax = cRaw.GetQMax();
            pCluster->fSigmaPad2 = cRaw.GetSigmaPad2();
            pCluster->fSigmaTime2 = cRaw.GetSigmaTime2();
#endif
            pCluster++;
          }
        }
      }
      fClusterData[islice].SetNumberOfClusters(pCluster - fClusterData[islice].Clusters());
      nClustersTotal += fClusterData[islice].NumberOfClusters();
      HLTDebug("Read %d->%d hits for slice %d", nClustersSliceTotal, fClusterData[islice].NumberOfClusters(), slice );
    }
  }
  
  if (fDumpEvent && nClustersTotal > fDumpEventNClsCut && fSliceCount == 36)
  {
    static int nEvent = 0;
    std::ofstream out;
    char filename[256];

    if (nEvent == 0)
    {
        sprintf(filename, "config.dump");
        out.open(filename, std::ofstream::binary);
        hltca_event_dump_settings eventSettings;
        eventSettings.setDefaults();
        eventSettings.solenoidBz = fTracker->Param(0).BzkG();
        eventSettings.constBz = false;
        out.write((char*) &eventSettings, sizeof(eventSettings));
        out.close();
    }

    sprintf(filename, HLTCA_EVDUMP_FILE ".%d.dump", nEvent++);
    out.open(filename, std::ofstream::binary);
    if (!out.fail())
    {
      //Write cluster data
      for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ )
      {
        fClusterData[iSlice].WriteEvent( out );
      }
    
      if (labelsPresent)
      {
        //Write cluster labels
        std::vector<AliHLTTPCClusterMCLabel> labels;
        for (int iSlice = 0;iSlice < 36;iSlice++)
        {
          AliHLTTPCCAClusterData::Data* pCluster = fClusterData[iSlice].Clusters();
          for (int iPatch = 0;iPatch < 6;iPatch++)
          {
            if (clusterLabels[iSlice][iPatch] == NULL || clustersXYZ[iSlice][iPatch] == NULL || clusterLabels[iSlice][iPatch]->fCount != clustersXYZ[iSlice][iPatch]->fCount) continue;
            const AliHLTTPCClusterXYZData& clXYZ = *clustersXYZ[iSlice][iPatch];
            for (int ic = 0;ic < clXYZ.fCount;ic++)
            {
              if (pCluster->fId != AliHLTTPCGeometry::CreateClusterID(iSlice, iPatch, ic)) continue;
              labels.push_back(clusterLabels[iSlice][iPatch]->fLabels[ic]);
              pCluster++;
            }
          }
        }
        
        if (!labels.size() || labels.size() != nClustersTotal)
        {
          printf("Error getting cluster MC labels\n");
        }
        else
        {
          out.write((const char*) labels.data(), labels.size() * sizeof(labels[0]));
          
          //Write MC tracks
          bool OK = false;
          do
          {
            AliRunLoader* rl = AliRunLoader::Instance();
            if (rl == NULL) {printf("RL\n"); break;}
            
            rl->LoadKinematics();
            rl->LoadTrackRefs(); 
            
            int nTracks = rl->GetHeader()->GetNtrack();
            
            AliStack* stack = rl->Stack();
            if (stack == NULL) {printf("stack\n");break;}
            TTree *TR = rl->TreeTR();
            if (TR == NULL) {printf("TR\n");break;}
            TBranch *branch = TR->GetBranch("TrackReferences");
            if (branch == NULL) {printf("branch\n");break;}

            int nPrimaries = stack->GetNprimary();
            
            std::vector<AliTrackReference*> trackRefs(nTracks, NULL);
            TClonesArray* tpcRefs = NULL;
            branch->SetAddress(&tpcRefs);
            int nr = TR->GetEntries();
            for (int r = 0;r < nr;r++)
            {
              TR->GetEvent(r);
              for (int i = 0;i < tpcRefs->GetEntriesFast();i++)
              {
                AliTrackReference* tpcRef = (AliTrackReference*) tpcRefs->UncheckedAt(i);
                if (tpcRef->DetectorId() != AliTrackReference::kTPC) continue;
                if (tpcRef->Label() < 0 || tpcRef->Label() >= nTracks)
                {
                  printf("Invalid reference %d / %d\n", tpcRef->Label(), nTracks);
                  continue;
                }
                if (trackRefs[tpcRef->Label()] != NULL) continue;
                trackRefs[tpcRef->Label()] = new AliTrackReference(*tpcRef);
              }
            }
            
            std::vector<AliHLTTPCCAMCInfo> mcInfo(nTracks);
            memset(mcInfo.data(), 0, nTracks * sizeof(mcInfo[0]));
            
            for (int i = 0;i < nTracks;i++)
            {
              mcInfo[i].fPID = -100;
              TParticle *particle = (TParticle*) stack->Particle(i);
              if (particle == NULL) continue;
              if (particle->GetPDG() == NULL) continue;
              
              int charge = (int) particle->GetPDG()->Charge();
              int prim = stack->IsPhysicalPrimary(i);
              int hasPrimDaughter = particle->GetFirstDaughter() != -1 && particle->GetFirstDaughter() < nPrimaries;
              
              mcInfo[i].fCharge = charge;
              mcInfo[i].fPrim = prim;
              mcInfo[i].fPrimDaughters = hasPrimDaughter;
              mcInfo[i].fGenRadius = sqrt(particle->Vx()*particle->Vx()+particle->Vy()*particle->Vy()+particle->Vz()*particle->Vz());
              
              Int_t pid = -1;
              if(TMath::Abs(particle->GetPdgCode()) == kElectron) pid = 0;
              if(TMath::Abs(particle->GetPdgCode()) == kMuonMinus) pid = 1;
              if(TMath::Abs(particle->GetPdgCode()) == kPiPlus) pid = 2;
              if(TMath::Abs(particle->GetPdgCode()) == kKPlus) pid = 3;
              if(TMath::Abs(particle->GetPdgCode()) == kProton) pid = 4;
              mcInfo[i].fPID = pid;
              
              AliTrackReference* ref = trackRefs[i];
              if (ref)
              {
                mcInfo[i].fX = ref->X();
                mcInfo[i].fY = ref->Y();
                mcInfo[i].fZ = ref->Z();
                mcInfo[i].fPx = ref->Px();
                mcInfo[i].fPy = ref->Py();
                mcInfo[i].fPz = ref->Pz();
              }
              
              //if (ref) printf("Particle %d: Charge %d, Prim %d, PrimDaughter %d, Pt %f %f ref %p\n", i, charge, prim, hasPrimDaughter, ref->Pt(), particle->Pt(), ref);
            }
            for (int i = 0;i < nTracks;i++) delete trackRefs[i];
            
            out.write((const char*) &nTracks, sizeof(nTracks));
            out.write((const char*) mcInfo.data(), nTracks * sizeof(mcInfo[0]));
            OK = true;
          } while (false);
            
          if (!OK)
          {
            printf("Error accessing MC data\n");
          }
        }
      }
      out.close();      
    }
  }

  if (nClustersTotal == 0)
  {
    //No input, skip processing
    fBenchmark.Stop(0);
    return(0);
  }
  
  //Prepare Output
  AliHLTTPCCASliceOutput::outputControlStruct outputControl;
  outputControl.fEndOfSpace = 0;
  outputControl.fOutputPtr =  (char*) outputPtr;
  outputControl.fOutputMaxSize = maxBufferSize;
  fTracker->SetOutputControl(&outputControl);

  memset(fSliceOutput, 0, fSliceCount * sizeof(AliHLTTPCCASliceOutput*));

  // reconstruct the event
  fBenchmark.Start(1);
  fTracker->ProcessSlices(fMinSlice, fSliceCount, fClusterData, fSliceOutput);
  fBenchmark.Stop(1);
  HLTInfo("Processed %d clusters", nClustersTotal);

  int ret = 0;
  size = 0;
 
  if( outputControl.fEndOfSpace ){
    HLTWarning( "Output buffer size exceeded buffer size %d, tracks are not stored", maxBufferSize );
    ret = -ENOSPC;
  } else {
    for (int islice = 0; islice < fSliceCount && fSliceOutput[islice]; islice++){
      int slice = fMinSlice + islice;
      HLTDebug( "%d tracks found for slice %d", fSliceOutput[islice]->NTracks(), slice );
      unsigned int blockSize = fSliceOutput[islice]->Size();
      if (blockSize > 0){
	AliHLTComponentBlockData bd;
	FillBlockData( bd );
	bd.fOffset = ((char*) fSliceOutput[islice] - (char*) outputPtr);
	bd.fSize = blockSize;
	bd.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( slice, slice, 0, fgkNPatches );
	bd.fDataType = AliHLTTPCCADefinitions::fgkTrackletsDataType;
	outputBlocks.push_back( bd );
	size += bd.fSize;
	fBenchmark.AddOutput(bd.fSize);
      }
    }
  }
  
  fBenchmark.Stop(0);
  HLTInfo(fBenchmark.GetStatistics());

  return((void*) (size_t) ret);
}
