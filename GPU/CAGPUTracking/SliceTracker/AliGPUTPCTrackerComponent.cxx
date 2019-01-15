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

#include "AliGPUTPCTrackerComponent.h"
#include "AliGPUReconstruction.h"
#include "AliGPUCAParam.h"

#include "AliHLTTPCRawCluster.h"
#include "AliHLTTPCClusterXYZ.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliGPUTPCGeometry.h"
#include "AliHLTTPCDefinitions.h"
#include "AliGPUTPCDefinitions.h"
#include "AliExternalTrackParam.h"
#include "TMath.h"
#include "AliCDBEntry.h"
#include "AliCDBManager.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "AliGPUTPCSliceOutput.h"
#include "AliGPUTPCClusterData.h"
#include "AliGPUTPCGMMergedTrackHit.h"
#if __GNUC__>= 3
using namespace std;
#endif

const AliHLTComponentDataType AliGPUTPCDefinitions::fgkTrackletsDataType = AliHLTComponentDataTypeInitializer( "CATRACKL", kAliHLTDataOriginTPC );

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp( AliGPUTPCTrackerComponent )

  AliGPUTPCTrackerComponent::AliGPUTPCTrackerComponent()
  :
  fClusterData( NULL ),
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
  fGPUType("CPU"),
  fGPUStuckProtection(0),
  fAsync(0),
  fSearchWindowDZDR(0.),
  fRec(0),
  fAsyncProcessor()
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt
}

AliGPUTPCTrackerComponent::AliGPUTPCTrackerComponent( const AliGPUTPCTrackerComponent& )
  :
  AliHLTProcessor(),
  fClusterData( NULL ),
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
  fGPUType("CPU"),
  fGPUStuckProtection(0),
  fAsync(0),
  fSearchWindowDZDR(0.),
  fRec(0),
  fAsyncProcessor()
{
  // see header file for class documentation
  HLTFatal( "copy constructor untested" );
}

AliGPUTPCTrackerComponent& AliGPUTPCTrackerComponent::operator=( const AliGPUTPCTrackerComponent& )
{
  // see header file for class documentation
  HLTFatal( "assignment operator untested" );
  return *this;
}

AliGPUTPCTrackerComponent::~AliGPUTPCTrackerComponent()
{
  // see header file for class documentation
  if (fClusterData) delete[] fClusterData;
  if (fRec) delete fRec;
}

//
// Public functions to implement AliHLTComponent's interface.
// These functions are required for the registration process
//

const char* AliGPUTPCTrackerComponent::GetComponentID()
{
  // see header file for class documentation
  return "TPCCATracker";
}

void AliGPUTPCTrackerComponent::GetInputDataTypes( vector<AliHLTComponentDataType>& list )
{
  // see header file for class documentation
  list.clear();
  list.push_back( AliHLTTPCDefinitions::RawClustersDataType() );
  list.push_back( AliHLTTPCDefinitions::ClustersXYZDataType() );
  list.push_back( AliHLTTPCDefinitions::AliHLTDataTypeClusterMCInfo() );
}

AliHLTComponentDataType AliGPUTPCTrackerComponent::GetOutputDataType()
{
  // see header file for class documentation
  return AliGPUTPCDefinitions::fgkTrackletsDataType;
}

void AliGPUTPCTrackerComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier )
{
  // define guess for the output data size
  constBase = 10000;       // minimum size
  inputMultiplier = 0.6; // size relative to input
}

AliHLTComponent* AliGPUTPCTrackerComponent::Spawn()
{
  // see header file for class documentation
  return new AliGPUTPCTrackerComponent;
}

void AliGPUTPCTrackerComponent::SetDefaultConfiguration()
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

int AliGPUTPCTrackerComponent::ReadConfigurationString(  const char* arguments )
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

    if ( argument.CompareTo( "-GPUType" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fGPUType = ( ( TObjString* )pTokens->At( i ) )->GetString();
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


int AliGPUTPCTrackerComponent::ReadCDBEntry( const char* cdbEntry, const char* chainId )
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


int AliGPUTPCTrackerComponent::Configure( const char* cdbEntry, const char* chainId, const char *commandLine )
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

  if (fRec) ConfigureSlices();

  return iResult1 ? iResult1 : ( iResult2 ? iResult2 :  iResult3  );
}

void AliGPUTPCTrackerComponent::ConfigureSlices()
{
  // Initialize the tracker slices
  AliGPUCASettingsRec rec;
  AliGPUCASettingsEvent ev;
  AliGPUCASettingsDeviceProcessing devProc;

  ev.solenoidBz = fSolenoidBz;
  ev.continuousMaxTimeBin = 0; //triggered events
  if( fNeighboursSearchArea>0 ) rec.NeighboursSearchArea = fNeighboursSearchArea;
  if( fClusterErrorCorrectionY>1.e-4 ) rec.ClusterError2CorrectionY = fClusterErrorCorrectionY*fClusterErrorCorrectionY;
  if( fClusterErrorCorrectionZ>1.e-4 ) rec.ClusterError2CorrectionZ = fClusterErrorCorrectionZ*fClusterErrorCorrectionZ;
  rec.MinNTrackClusters = fMinNTrackClusters;
  rec.SetMinTrackPt(fMinTrackPt);
  rec.SearchWindowDZDR = fSearchWindowDZDR;
  devProc.nDeviceHelperThreads = fGPUHelperThreads;
  rec.GlobalTracking = fGlobalTracking;
  devProc.stuckProtection = fGPUStuckProtection;
  
  fRec->SetSettings(&ev, &rec, &devProc);
  fRec->LoadClusterErrors();
  fRec->Init();
}

void* AliGPUTPCTrackerComponent::TrackerInit(void* par)
{
    //Create tracker instance and set parameters
    fRec = AliGPUReconstruction::CreateInstance(fAllowGPU ? fGPUType.Data() : "CPU", true);
    if (fRec == NULL) return((void*) -1);
    fClusterData = new AliGPUTPCClusterData[fgkNSlices];

    ConfigureSlices();
    return(NULL);
}

int AliGPUTPCTrackerComponent::DoInit( int argc, const char** argv )
{
  if ( fRec ) return EINPROGRESS;

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
      if (fAsyncProcessor.InitializeAsyncMemberTask(this, &AliGPUTPCTrackerComponent::TrackerInit, NULL, &initRetVal) != 0) return(-ENODEV);
      if (initRetVal) return(-ENODEV);
    }
    else
    {
      if (TrackerInit(NULL) != NULL) return(-ENODEV);
    }
  }

  return(retVal);
}

void* AliGPUTPCTrackerComponent::TrackerExit(void* par)
{
    if (fClusterData) delete[] fClusterData;
    fClusterData = NULL;
    if (fRec) delete fRec;
    fRec = NULL;
    return(NULL);
}

int AliGPUTPCTrackerComponent::DoDeinit()
{
  // see header file for class documentation
  if (fAsync)
  {
    void* initRetVal = NULL;
    fAsyncProcessor.InitializeAsyncMemberTask(this, &AliGPUTPCTrackerComponent::TrackerExit, NULL, &initRetVal);
    fAsyncProcessor.Deinitialize();
  }
  else
  {
    TrackerExit(NULL);
  }
  return 0;
}

int AliGPUTPCTrackerComponent::Reconfigure( const char* cdbEntry, const char* chainId )
{
  // Reconfigure the component from OCDB .
  return Configure( cdbEntry, chainId, NULL );
}

int AliGPUTPCTrackerComponent::DoEvent
  (
  const AliHLTComponentEventData& evtData,
  const AliHLTComponentBlockData* blocks,
  AliHLTComponentTriggerData& /*trigData*/,
  AliHLTUInt8_t* outputPtr,
  AliHLTUInt32_t& size,
  vector<AliHLTComponentBlockData>& outputBlocks )
{
  //* process event
  if (!fRec)
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
    if (fAsyncProcessor.InitializeAsyncMemberTask(this, &AliGPUTPCTrackerComponent::TrackerDoEvent, &tmpPar, &asyncRetVal, fAsync) != 0)
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
  
void* AliGPUTPCTrackerComponent::TrackerDoEvent(void* par)
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
  for (int slice = 0;slice < fgkNSlices;slice++)
  {
    int nClustersSliceTotal = 0;
    for (int patch = 0;patch < 6;patch++)
    {
      if (clustersXYZ[slice][patch]) nClustersSliceTotal += clustersXYZ[slice][patch]->fCount;
    }
    if (nClustersSliceTotal > 500000)
    {
      HLTWarning( "Too many clusters in tracker input: Slice %d, Number of Clusters %d, slice not included in tracking", slice, nClustersSliceTotal );
      fClusterData[slice].StartReading(slice, 0);
    }
    else if (nClustersSliceTotal == 0)
    {
      fClusterData[slice].StartReading(slice, 0);
    }
    else
    {
      fClusterData[slice].StartReading( slice, nClustersSliceTotal );
      AliGPUTPCClusterData::Data* pCluster = fClusterData[slice].Clusters();
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

          const int firstRow = AliGPUTPCGeometry::GetFirstRow(patch);
          for (int ic = 0;ic < clXYZ.fCount;ic++)
          {
            const AliHLTTPCClusterXYZ &c = clXYZ.fClusters[ic];
            const AliHLTTPCRawCluster &cRaw = clRaw.fClusters[ic];
            if ( c.GetZ() > fClusterZCut || c.GetZ() < -fClusterZCut) continue;
            if ( c.GetX() <1.f ) continue; // cluster xyz position was not calculated for whatever reason
            pCluster->fId = AliGPUTPCGeometry::CreateClusterID( slice, patch, ic );
            pCluster->fX = c.GetX();
            pCluster->fY = c.GetY();
            pCluster->fZ = c.GetZ();
            pCluster->fRow = firstRow + cRaw.GetPadRow();
            pCluster->fFlags = cRaw.GetFlags();
            if (cRaw.GetSigmaPad2() < kAlmost0 || cRaw.GetSigmaTime2() < kAlmost0) pCluster->fFlags |= AliGPUTPCGMMergedTrackHit::flagSingle;
            pCluster->fAmp = cRaw.GetCharge();
#ifdef GPUCA_FULL_CLUSTERDATA
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
      fClusterData[slice].SetNumberOfClusters(pCluster - fClusterData[slice].Clusters());
      nClustersTotal += fClusterData[slice].NumberOfClusters();
      HLTDebug("Read %d->%d hits for slice %d", nClustersSliceTotal, fClusterData[slice].NumberOfClusters(), slice );
    }
  }
  
  if (nClustersTotal == 0)
  {
    //No input, skip processing
    fBenchmark.Stop(0);
    return(0);
  }
  
  fRec->ClearIOPointers();
  for (int i = 0;i < 36;i++)
  {
      fRec->mIOPtrs.clusterData[i] = fClusterData[i].Clusters();
      fRec->mIOPtrs.nClusterData[i] = fClusterData[i].NumberOfClusters();
  }
  
  //Prepare Output
  fRec->SetOutputControl(outputPtr, maxBufferSize);

  // reconstruct the event
  fBenchmark.Start(1);
  fRec->RunTPCTrackingSlices();
  fBenchmark.Stop(1);
  HLTInfo("Processed %d clusters", nClustersTotal);

  int ret = 0;
  size = 0;
 
  if( fRec->OutputControl().EndOfSpace ){
    HLTWarning( "Output buffer size exceeded buffer size %d, tracks are not stored", maxBufferSize );
    ret = -ENOSPC;
  } else {
    for (int slice = 0; slice < fgkNSlices && fRec->SliceOutput()[slice]; slice++){
      HLTDebug( "%d tracks found for slice %d", fRec->SliceOutput()[slice]->NTracks(), slice );
      unsigned int blockSize = fRec->SliceOutput()[slice]->Size();
      if (blockSize > 0){
        AliHLTComponentBlockData bd;
        FillBlockData( bd );
        bd.fOffset = ((char*) fRec->SliceOutput()[slice] - (char*) outputPtr);
        bd.fSize = blockSize;
        bd.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( slice, slice, 0, fgkNPatches );
        bd.fDataType = AliGPUTPCDefinitions::fgkTrackletsDataType;
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
