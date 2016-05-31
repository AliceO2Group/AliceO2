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
#include "AliHLTTPCCATrackConvertor.h"
#include "AliHLTArray.h"

#include "AliHLTTPCRawCluster.h"
#include "AliHLTTPCClusterXYZ.h"
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

#if __GNUC__>= 3
using namespace std;
#endif

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
  fMinNTrackClusters( 30 ),
  fMinTrackPt(0.05),
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
  fGPULibrary("")
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
  fMinNTrackClusters( 30 ),
  fMinTrackPt( 0.05 ),
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
  fGPULibrary("")
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
}

AliHLTComponentDataType AliHLTTPCCATrackerComponent::GetOutputDataType()
{
  // see header file for class documentation
  return AliHLTTPCCADefinitions::fgkTrackletsDataType;
}

void AliHLTTPCCATrackerComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier )
{
  // define guess for the output data size
  constBase = 1200;       // minimum size
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
  fMinNTrackClusters = 30;
  fMinTrackPt = 0.05;
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

    param.Update();
    fTracker->InitializeSliceParam( slice, param );
    delete[] rowX;
  }
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
    fMinSlice = 0;
    fSliceCount = fgkNSlices;
    //Create tracker instance and set parameters
    fTracker = new AliHLTTPCCATrackerFramework(fAllowGPU, fGPULibrary, fGPUDeviceNum);
    if ( fAllowGPU && fTracker->GetGPUStatus() < 2 ) {
      HLTError("GPU Tracker requested but unavailable, aborting.");
      return -ENODEV;
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

    ConfigureSlices();
  }

  return(retVal);
}

int AliHLTTPCCATrackerComponent::DoDeinit()
{
  // see header file for class documentation
  if (fTracker) delete fTracker;
  fTracker = NULL;
  if (fClusterData) delete[] fClusterData;
  fClusterData = NULL;
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

  // min and max patch numbers and row numbers
  int sliceminPatch[fgkNSlices];
  int slicemaxPatch[fgkNSlices];
  for (int i = 0;i < fSliceCount;i++)
  {
    sliceminPatch[i] = 100;
    slicemaxPatch[i] = -1;
  }

  //Prepare everything for all slices

  for (int islice = 0;islice < fSliceCount;islice++)
  {
    int slice = fMinSlice + islice;

    // total n Hits
    int nClustersTotal = 0;

    // sort patches
    std::vector<unsigned long> patchIndicesRaw, patchIndicesXYZ;

    for ( unsigned long ndx = 0; ndx < evtData.fBlockCnt; ndx++ ) {
      const AliHLTComponentBlockData* iter = blocks + ndx;
      if ( slice != AliHLTTPCDefinitions::GetMinSliceNr( *iter ) ) continue;
      if ( iter->fDataType == AliHLTTPCDefinitions::ClustersXYZDataType() ){
        AliHLTTPCClusterXYZData* inPtrSP = ( AliHLTTPCClusterXYZData* )( iter->fPtr );
        nClustersTotal += inPtrSP->fCount;
        fBenchmark.AddInput(iter->fSize);
	int patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
	if ( sliceminPatch[islice] > patch ) {
	  sliceminPatch[islice] = patch;
	}
	if ( slicemaxPatch[islice] < patch ) {
	  slicemaxPatch[islice] = patch;
	}
	std::vector<unsigned long>::iterator pIter = patchIndicesXYZ.begin();
	while ( pIter != patchIndicesXYZ.end() && AliHLTTPCDefinitions::GetMinPatchNr( blocks[*pIter] ) < patch ) {
	  pIter++;
	}
	patchIndicesXYZ.insert( pIter, ndx );
      } else if ( iter->fDataType == AliHLTTPCDefinitions::RawClustersDataType() ){
 	int patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
	std::vector<unsigned long>::iterator pIter = patchIndicesRaw.begin();
	while ( pIter != patchIndicesRaw.end() && AliHLTTPCDefinitions::GetMinPatchNr( blocks[*pIter] ) < patch ) {
	  pIter++;
	}
	patchIndicesRaw.insert( pIter, ndx );
      }
    }


    // pass event to CA Tracker


    Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reading hits",
      "Total %d hits to read for slice %d", nClustersTotal, slice );

    if (nClustersTotal > 500000)
    {
      HLTWarning( "Too many clusters in tracker input: Slice %d, Number of Clusters %d, slice not included in tracking", slice, nClustersTotal );
      fClusterData[islice].StartReading( slice, 0 );
    }
    else
    {
      fClusterData[islice].StartReading( slice, nClustersTotal );

      for ( std::vector<unsigned long>::iterator pIter = patchIndicesXYZ.begin(); pIter != patchIndicesXYZ.end(); pIter++ ) {
        unsigned long ndx = *pIter;
        const AliHLTComponentBlockData* iter = blocks + ndx;

	if ( iter->fDataType != AliHLTTPCDefinitions::ClustersXYZDataType() ) continue;
	  
        int patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );

	int firstRow = AliHLTTPCGeometry::GetFirstRow(patch);
 
        int nPatchClust = 0;
	const AliHLTComponentBlockData* rawIter = NULL;

	for ( std::vector<unsigned long>::iterator pRawIter = patchIndicesRaw.begin(); pRawIter != patchIndicesRaw.end(); pRawIter++ ) {
	  const AliHLTComponentBlockData* iter1 = blocks + (*pRawIter);
 	  if( iter1->fDataType == AliHLTTPCDefinitions::RawClustersDataType() && AliHLTTPCDefinitions::GetMinPatchNr( *iter1 ) == patch ){
	    rawIter = iter1;
	    break;
	  }
	}
	if( rawIter==NULL ) continue;
	AliHLTTPCClusterXYZData* inPtrSP = ( AliHLTTPCClusterXYZData* )( iter->fPtr );
	AliHLTTPCRawClusterData* inPtrRaw = ( AliHLTTPCRawClusterData* )( rawIter->fPtr );

	AliHLTTPCCAClusterData::Data* pCluster = &fClusterData[islice].Clusters()[fClusterData[islice].NumberOfClusters()];

	nPatchClust = inPtrSP->fCount;
	const AliHLTTPCClusterXYZ* pLastHLTCluster = &inPtrSP->fClusters[inPtrSP->fCount];
	for ( const AliHLTTPCClusterXYZ* pHLTCluster = inPtrSP->fClusters; pHLTCluster < pLastHLTCluster; pHLTCluster++ ) {
	  if ( pHLTCluster->fZ > fClusterZCut || pHLTCluster->fZ < -fClusterZCut) continue;
	  pCluster->fId = pHLTCluster->GetRawClusterID();
	  pCluster->fX = pHLTCluster->GetX();
	  pCluster->fY = pHLTCluster->GetY();
	  pCluster->fZ = pHLTCluster->GetZ();
	  int ind = pHLTCluster->GetRawClusterIndex();
	  pCluster->fRow = firstRow + inPtrRaw->fClusters[ind].GetPadRow();
	  pCluster->fAmp = inPtrRaw->fClusters[ind].GetCharge();
	  pCluster++;
	}
	fClusterData[islice].SetNumberOfClusters(pCluster - fClusterData[islice].Clusters());
      
        Logging( kHLTLogInfo, "HLT::TPCCATracker::DoEvent", "Reading hits",
          "Read %d hits for slice %d - patch %d", nPatchClust, slice, patch );
      }
    }
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

  int ret = 0;
  unsigned int mySize = 0;
  int ntracks = 0;
  int error = 0;

  for (int islice = 0;islice < fSliceCount;islice++)
  {
    if (slicemaxPatch[islice] == -1) continue;
    int slice = fMinSlice + islice;

    if( outputControl.fEndOfSpace ){
      HLTWarning( "Output buffer size exceeded (buffer size %d, required size %d), tracks are not stored", maxBufferSize, mySize );
      ret = -ENOSPC;
      error = 1;
      break;     
    }

    if (fSliceOutput[islice])
    {
      // write reconstructed tracks
      Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reconstruct", "%d tracks found for slice %d", fSliceOutput[islice]->NTracks(), slice );

      mySize += fSliceOutput[islice]->Size();
      ntracks += fSliceOutput[islice]->NTracks();    
    }
    else
    {
      HLTWarning( "Error during Tracking, no tracks stored" );
      mySize = 0;
      ret = -ENOSPC;
      ntracks = 0;
      error = 1;
      break;
    }
  }

  size = 0;
  if (error == 0)
  {
    for (int islice = 0;islice < fSliceCount && fSliceOutput[islice];islice++)
    {
      int slice = fMinSlice + islice;

      mySize = fSliceOutput[islice]->Size();
      if (mySize > 0)
      {
        AliHLTComponentBlockData bd;
        FillBlockData( bd );
        bd.fOffset = ((char*) fSliceOutput[islice] - (char*) outputPtr);
        bd.fSize = mySize;
        bd.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( slice, slice, sliceminPatch[islice], slicemaxPatch[islice] );
        bd.fDataType = GetOutputDataType();
        outputBlocks.push_back( bd );
        size += mySize;
        fBenchmark.AddOutput(bd.fSize);
      }
    }
  }

  //No longer needed

  fBenchmark.Stop(0);

  // Set log level to "Warning" for on-line system monitoring

  //Min and Max Patch are taken for first slice processed...

  fBenchmark.SetName(Form("CATracker"));

  HLTInfo(fBenchmark.GetStatistics());
  //No longer needed

  return ret;
}


