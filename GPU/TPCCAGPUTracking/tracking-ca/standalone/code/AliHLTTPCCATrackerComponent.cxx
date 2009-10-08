// @(#) $Id: AliHLTTPCCATrackerComponent.cxx 35348 2009-10-08 12:04:48Z sgorbuno $
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

#if __GNUC__>= 3
using namespace std;
#endif

#include "AliHLTTPCCATrackerComponent.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackConvertor.h"
#include "AliHLTArray.h"

#include "AliHLTTPCSpacePointData.h"
#include "AliHLTTPCClusterDataFormat.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrackSegmentData.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCTrackletDataFormat.h"
#include "AliHLTTPCDefinitions.h"
#include "AliExternalTrackParam.h"
#include "TStopwatch.h"
#include "TMath.h"
#include "AliCDBEntry.h"
#include "AliCDBManager.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCAClusterData.h"

const AliHLTComponentDataType AliHLTTPCCADefinitions::fgkTrackletsDataType = AliHLTComponentDataTypeInitializer( "CATRACKL", kAliHLTDataOriginTPC );

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp( AliHLTTPCCATrackerComponent )

AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent()
    :
    fTracker( NULL ),
    fSolenoidBz( 0 ),
    fMinNTrackClusters( 0 ),
    fClusterZCut( 500. ),
    fNeighboursSearchArea( 0 ), 
    fClusterErrorCorrectionY(0), 
    fClusterErrorCorrectionZ(0),
    fFullTime( 0 ),
    fRecoTime( 0 ),
    fNEvents( 0 ),
    fOutputTRAKSEGS( 0 )
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt
}

AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent( const AliHLTTPCCATrackerComponent& )
    :
    AliHLTProcessor(),
    fTracker( NULL ),
    fSolenoidBz( 0 ),
    fMinNTrackClusters( 30 ),
    fClusterZCut( 500. ),
    fNeighboursSearchArea(0),
    fClusterErrorCorrectionY(0), 
    fClusterErrorCorrectionZ(0),
    fFullTime( 0 ),
    fRecoTime( 0 ),
    fNEvents( 0 ),
    fOutputTRAKSEGS( 0 )
{
  // see header file for class documentation
  HLTFatal( "copy constructor untested" );
}

AliHLTTPCCATrackerComponent& AliHLTTPCCATrackerComponent::operator=( const AliHLTTPCCATrackerComponent& )
{
  // see header file for class documentation
  HLTFatal( "assignment operator untested" );
  return *this;
}

AliHLTTPCCATrackerComponent::~AliHLTTPCCATrackerComponent()
{
  // see header file for class documentation
  delete fTracker;
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
  list.push_back( AliHLTTPCDefinitions::fgkClustersDataType );
}

AliHLTComponentDataType AliHLTTPCCATrackerComponent::GetOutputDataType()
{
  // see header file for class documentation
  if ( fOutputTRAKSEGS ) return AliHLTTPCDefinitions::fgkTrackSegmentsDataType;
  else return AliHLTTPCCADefinitions::fgkTrackletsDataType;
}

void AliHLTTPCCATrackerComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier )
{
  // define guess for the output data size
  constBase = 200;       // minimum size
  inputMultiplier = 0.5; // size relative to input
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

  fSolenoidBz = 5.;
  fMinNTrackClusters = 0;
  fClusterZCut = 500.;
  fNeighboursSearchArea = 0;
  fClusterErrorCorrectionY = 0;
  fClusterErrorCorrectionZ = 0;
  fOutputTRAKSEGS = 0;
  fFullTime = 0;
  fRecoTime = 0;
  fNEvents = 0;
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
      fSolenoidBz = ( ( TObjString* )pTokens->At( i ) )->GetString().Atof();
      HLTInfo( "Magnetic Field set to: %f", fSolenoidBz );
      continue;
    }

    if ( argument.CompareTo( "-minNClustersOnTrack" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fMinNTrackClusters = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      HLTInfo( "minNClustersOnTrack set to: %d", fMinNTrackClusters );
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

   if ( argument.CompareTo( "-outputTRAKSEGS" ) == 0 ) {
      fOutputTRAKSEGS = 1;
      HLTInfo( "The special output type \"TRAKSEGS\" is set" );
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

  int iResult2 = ReadCDBEntry( kAliHLTCDBSolenoidBz, chainId );

  //* read the actual CDB entry if required

  int iResult3 = ( cdbEntry ) ? ReadCDBEntry( cdbEntry, chainId ) : 0;

  //* read extra parameters from input (if they are)

  int iResult4 = 0;

  if ( commandLine && commandLine[0] != '\0' ) {
    HLTInfo( "received configuration string from HLT framework: \"%s\"", commandLine );
    iResult4 = ReadConfigurationString( commandLine );
  }

  // Initialise the tracker here

  return iResult1 ? iResult1 : ( iResult2 ? iResult2 : ( iResult3 ? iResult3 : iResult4 ) );
}



int AliHLTTPCCATrackerComponent::DoInit( int argc, const char** argv )
{
  // Configure the CA tracker component

  if ( fTracker ) return EINPROGRESS;


  fTracker = new AliHLTTPCCATrackerFramework();

  TString arguments = "";
  for ( int i = 0; i < argc; i++ ) {
    if ( !arguments.IsNull() ) arguments += " ";
    arguments += argv[i];
  }

  return Configure( NULL, NULL, arguments.Data() );
}


int AliHLTTPCCATrackerComponent::DoDeinit()
{
  // see header file for class documentation
  delete fTracker;
  fTracker = NULL;
  return 0;
}



int AliHLTTPCCATrackerComponent::Reconfigure( const char* cdbEntry, const char* chainId )
{
  // Reconfigure the component from OCDB .

  return Configure( cdbEntry, chainId, NULL );
}


bool AliHLTTPCCATrackerComponent::CompareClusters( AliHLTTPCSpacePointData *a, AliHLTTPCSpacePointData *b )
{
  //* Comparison function for sort clusters

  if ( a->fPadRow < b->fPadRow ) return 1;
  if ( a->fPadRow > b->fPadRow ) return 0;
  return ( a->fZ < b->fZ );
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

  AliHLTUInt32_t maxBufferSize = size;
  size = 0; // output size

  if ( GetFirstInputBlock( kAliHLTDataTypeSOR ) || GetFirstInputBlock( kAliHLTDataTypeEOR ) ) {
    return 0;
  }

  TStopwatch timer;

  // Event reconstruction in one TPC slice with CA Tracker

  //Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "CA::DoEvent()" );
  if ( evtData.fBlockCnt <= 0 ) {
    HLTWarning( "no blocks in event" );
    return 0;
  }

  const AliHLTComponentBlockData* iter = NULL;
  unsigned long ndx;
  AliHLTTPCClusterData* inPtrSP;

  // Determine the slice number

  //Find min and max slice number with now slice missing in between (for default output)
  int minslice = -1, maxslice = -1;
  int slice = -1;
  {
    std::vector<int> slices;
    std::vector<int>::iterator slIter;
    std::vector<unsigned> sliceCnts;
    std::vector<unsigned>::iterator slCntIter;

    for ( ndx = 0; ndx < evtData.fBlockCnt; ndx++ ) {
      iter = blocks + ndx;
      if ( iter->fDataType != AliHLTTPCDefinitions::fgkClustersDataType ) continue;

      slice = AliHLTTPCDefinitions::GetMinSliceNr( *iter );
	  if (slice < minslice || minslice == -1) minslice = slice;
	  if (slice > maxslice) maxslice = slice;

      bool found = 0;
      slCntIter = sliceCnts.begin();
      for ( slIter = slices.begin(); slIter != slices.end(); slIter++, slCntIter++ ) {
        if ( *slIter == slice ) {
          found = kTRUE;
          break;
        }
      }
      if ( !found ) {
        slices.push_back( slice );
        sliceCnts.push_back( 1 );
      } else *slCntIter++;
    }

	  if ( slices.size() == 0 ) {
		HLTWarning( "no slices found in event" );
		return 0;
	  }


    // Determine slice number to really use. (for obsolete output)
    if ( slices.size() > 1 ) {
		Logging( fOutputTRAKSEGS ? kHLTLogError : kHLTLogDebug, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
               "Multiple slice numbers found in event 0x%08lX (%lu). Determining maximum occuring slice number...",
               evtData.fEventID, evtData.fEventID );
      unsigned maxCntSlice = 0;
      slCntIter = sliceCnts.begin();
      for ( slIter = slices.begin(); slIter != slices.end(); slIter++, slCntIter++ ) {
        Logging( fOutputTRAKSEGS ? kHLTLogError : kHLTLogDebug, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
                 "Slice %lu found %lu times.", *slIter, *slCntIter );
        if ( fOutputTRAKSEGS && maxCntSlice < *slCntIter ) {
          maxCntSlice = *slCntIter;
          slice = *slIter;
        }
      }
      if (fOutputTRAKSEGS)
      {
	    Logging( kHLTLogError, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
               "Using slice %lu.", slice );
	}
    } else if ( slices.size() > 0 ) {
      slice = *( slices.begin() );
    }


	  if (fOutputTRAKSEGS)
	  {
		  minslice = maxslice = slice;
	  }
	  else
	  {
		  for (int islice = minslice;islice <= maxslice;islice++)
		  {
			  bool found = false;
			  for(slIter = slices.begin(); slIter != slices.end();slIter++)
			  {
				  if (*slIter == islice)
				  {
					  found = true;
					  break;
				  }
			  }
			  if (!found)
			  {
				  maxslice = islice - 1;
				  break;
			  }
		  }
	 }
  }

  if ( !fTracker ) fTracker = new AliHLTTPCCATrackerFramework;
  int slicecount = maxslice + 1 - minslice;
  if (slicecount > fTracker->MaxSliceCount())
  {
	maxslice = minslice + (slicecount = fTracker->MaxSliceCount());
  }
  int nClustersTotalSum = 0;
  AliHLTTPCCAClusterData* clusterData = new AliHLTTPCCAClusterData[slicecount];


  // min and max patch numbers and row numbers
  int* slicerow = new int[slicecount * 2];
  int* sliceminPatch = new int[slicecount];
  int* slicemaxPatch = new int[slicecount];
  memset(slicerow, 0, slicecount * 2 * sizeof(int));
  for (int i = 0;i < slicecount;i++)
  {
	  sliceminPatch[i] = 100;
	  slicemaxPatch[i] = -1;
  }

  //Prepare everything for all slices

  for (int islice = 0;islice < slicecount;islice++)
  {
	  slice = minslice + islice;

	  // Initialize the tracker slice
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
		int nRows = AliHLTTPCTransform::GetNRows();

		float padPitch = 0.4;
		float sigmaZ = 0.228808;

		float *rowX = new float [nRows];
		for ( int irow = 0; irow < nRows; irow++ ) {
		  rowX[irow] = AliHLTTPCTransform::Row2X( irow );
		}

		AliHLTTPCCAParam param;

		param.Initialize( iSec, nRows, rowX, alpha, dalpha,
						  inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, fSolenoidBz );
		param.SetHitPickUpFactor( 2 );
		if( fNeighboursSearchArea>0 ) param.SetNeighboursSearchArea( fNeighboursSearchArea );
		if( fClusterErrorCorrectionY>1.e-4 ) param.SetClusterError2CorrectionY( fClusterErrorCorrectionY*fClusterErrorCorrectionY );
		if( fClusterErrorCorrectionZ>1.e-4 ) param.SetClusterError2CorrectionZ( fClusterErrorCorrectionZ*fClusterErrorCorrectionZ );
		param.Update();
		fTracker->InitializeSliceParam( slice, param );
		delete[] rowX;
	  }

	  // total n Hits
	  int nClustersTotal = 0;

	  // sort patches
	  std::vector<unsigned long> patchIndices;

	  for ( ndx = 0; ndx < evtData.fBlockCnt; ndx++ ) {
		iter = blocks + ndx;
		if ( iter->fDataType != AliHLTTPCDefinitions::fgkClustersDataType ) continue;
		if ( slice != AliHLTTPCDefinitions::GetMinSliceNr( *iter ) ) continue;
		inPtrSP = ( AliHLTTPCClusterData* )( iter->fPtr );
		nClustersTotal += inPtrSP->fSpacePointCnt;
		int patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
		if ( sliceminPatch[islice] > patch ) {
		  sliceminPatch[islice] = patch;
		  slicerow[2 * islice + 0] = AliHLTTPCTransform::GetFirstRow( patch );
		}
		if ( slicemaxPatch[islice] < patch ) {
		  slicemaxPatch[islice] = patch;
		  slicerow[2 * islice + 1] = AliHLTTPCTransform::GetLastRow( patch );
		}
		std::vector<unsigned long>::iterator pIter = patchIndices.begin();
		while ( pIter != patchIndices.end() && AliHLTTPCDefinitions::GetMinPatchNr( blocks[*pIter] ) < patch ) {
		  pIter++;
		}
		patchIndices.insert( pIter, ndx );
	  }


	  // pass event to CA Tracker


	  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reading hits",
			   "Total %d hits to read for slice %d", nClustersTotal, slice );


	  clusterData[islice].StartReading( slice, nClustersTotal );

	  for ( std::vector<unsigned long>::iterator pIter = patchIndices.begin(); pIter != patchIndices.end(); pIter++ ) {
		ndx = *pIter;
		iter = blocks + ndx;

		int patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
		inPtrSP = ( AliHLTTPCClusterData* )( iter->fPtr );

		Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reading hits",
				 "Reading %d hits for slice %d - patch %d", inPtrSP->fSpacePointCnt, slice, patch );

		for ( unsigned int i = 0; i < inPtrSP->fSpacePointCnt; i++ ) {
		  AliHLTTPCSpacePointData *c = &( inPtrSP->fSpacePoints[i] );
		  if ( CAMath::Abs( c->fZ ) > fClusterZCut ) continue;
		  if ( c->fPadRow > 159 ) {
			HLTError( "Wrong TPC cluster with row number %d received", c->fPadRow );
			continue;
		  }
		  clusterData[islice].ReadCluster( c->fID, c->fPadRow, c->fX, c->fY, c->fZ, c->fCharge );
		}
	  }

	  clusterData[islice].FinishReading();
	  nClustersTotalSum += nClustersTotal;
  }

  //Prepare Output
  AliHLTTPCCASliceOutput::outputControlStruct outputControl;
  //Set tracker output so tracker does not have to output both formats!
  outputControl.fObsoleteOutput = fOutputTRAKSEGS;
  outputControl.fDefaultOutput = !fOutputTRAKSEGS;

  //For new output we can write directly to output buffer
  outputControl.fOutputPtr = fOutputTRAKSEGS ? NULL : (char*) outputPtr;
  outputControl.fOutputMaxSize = maxBufferSize;

  AliHLTTPCCASliceOutput** sliceOutput = new AliHLTTPCCASliceOutput*[slicecount];
  memset(sliceOutput, 0, slicecount * sizeof(AliHLTTPCCASliceOutput*));

  // reconstruct the event
  TStopwatch timerReco;
  fTracker->SetOutputControl(&outputControl);
  fTracker->ProcessSlices(minslice, slicecount, clusterData, sliceOutput);
  timerReco.Stop();
  
  int ret = 0;
  unsigned int mySize = 0;
  int ntracks;
  int error = 0;

  for (int islice = 0;islice < slicecount;islice++)
  {
	  slice = minslice + islice;

	  if (sliceOutput[islice])
	  {
		  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reconstruct",
				   "%d tracks found for slice %d", sliceOutput[islice]->NOutTracks(), slice );


		  // write reconstructed tracks

		  if ( fOutputTRAKSEGS ) {

			ntracks = sliceOutput[islice]->NOutTracks();

			AliHLTTPCTrackletData* outPtr = ( AliHLTTPCTrackletData* )( outputPtr );

			AliHLTTPCTrackSegmentData* currOutTracklet = outPtr->fTracklets;

			mySize =   ( ( AliHLTUInt8_t * )currOutTracklet ) -  ( ( AliHLTUInt8_t * )outputPtr );

			outPtr->fTrackletCnt = 0;

			for ( int itr = 0; itr < ntracks; itr++ ) {

			  AliHLTTPCCAOutTrack &t = sliceOutput[islice]->OutTracks()[itr];

			  //Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Wrtite output","track %d with %d hits", itr, t.NHits());

			  if ( t.NHits() < fMinNTrackClusters ) continue;

			  // calculate output track size

			  unsigned int dSize = sizeof( AliHLTTPCTrackSegmentData ) + t.NHits() * sizeof( unsigned int );

			  if ( mySize + dSize > maxBufferSize ) {
				HLTWarning( "Output buffer size exceed (buffer size %d, current size %d), %d tracks are not stored", maxBufferSize, mySize, ntracks - itr + 1 );
				ret = -ENOSPC;
				error = 1;
				break;
			  }

			  // convert CA track parameters to HLT Track Segment

			  int iFirstRow = 1000;
			  int iLastRow = -1;
			  int iFirstHit = sliceOutput[islice]->OutTrackHit(t.FirstHitRef());
			  int iLastHit = iFirstHit;
			  for ( int ih = 0; ih < t.NHits(); ih++ ) {
				int hitID = sliceOutput[islice]->OutTrackHit(t.FirstHitRef() + ih);
				int iRow = clusterData[islice].RowNumber( hitID );
				if ( iRow < iFirstRow ) {  iFirstRow = iRow; iFirstHit = hitID; }
				if ( iRow > iLastRow ) { iLastRow = iRow; iLastHit = hitID; }
			  }

			  AliHLTTPCCATrackParam par = t.StartPoint();

			  par.TransportToX( clusterData[islice].X( iFirstHit ), .99 );

			  AliExternalTrackParam tp;
			  AliHLTTPCCATrackConvertor::GetExtParam( par, tp, 0 );

			  currOutTracklet->fX = tp.GetX();
			  currOutTracklet->fY = tp.GetY();
			  currOutTracklet->fZ = tp.GetZ();
			  currOutTracklet->fCharge = ( int ) tp.GetSign();
			  currOutTracklet->fPt = TMath::Abs( tp.GetSignedPt() );
			  float snp =  tp.GetSnp() ;
			  if ( snp > .999 ) snp = .999;
			  if ( snp < -.999 ) snp = -.999;
			  currOutTracklet->fPsi = TMath::ASin( snp );
			  currOutTracklet->fTgl = tp.GetTgl();

			  currOutTracklet->fY0err = tp.GetSigmaY2();
			  currOutTracklet->fZ0err = tp.GetSigmaZ2();
			  float h = -currOutTracklet->fPt * currOutTracklet->fPt;
			  currOutTracklet->fPterr = h * h * tp.GetSigma1Pt2();
			  h = 1. / TMath::Sqrt( 1 - snp * snp );
			  currOutTracklet->fPsierr = h * h * tp.GetSigmaSnp2();
			  currOutTracklet->fTglerr = tp.GetSigmaTgl2();

			  if ( par.TransportToX( clusterData[islice].X( iLastHit ), .99 ) ) {
				currOutTracklet->fLastX = par.GetX();
				currOutTracklet->fLastY = par.GetY();
				currOutTracklet->fLastZ = par.GetZ();
			  } else {
				currOutTracklet->fLastX = clusterData[islice].X( iLastHit );
				currOutTracklet->fLastY = clusterData[islice].Y( iLastHit );
				currOutTracklet->fLastZ = clusterData[islice].Z( iLastHit );
			  }
			  //if( currOutTracklet->fLastX<10. ) {
			  //HLTError("CA last point: hitxyz=%f,%f,%f, track=%f,%f,%f, tracklet=%f,%f,%f, nhits=%d",clusterData[islice].X( iLastHit ),clusterData[islice].Y( iLastHit],clusterData[islice].Z( iLastHit],
			  //par.GetX(), par.GetY(),par.GetZ(),currOutTracklet->fLastX,currOutTracklet->fLastY ,currOutTracklet->fLastZ, t.NHits());
			  //}
		#ifdef INCLUDE_TPC_HOUGH
		#ifdef ROWHOUGHPARAMS
			  currOutTracklet->fTrackID = 0;
			  currOutTracklet->fRowRange1 = clusterData[islice].RowNumber( iFirstHit );
			  currOutTracklet->fRowRange2 = clusterData[islice].RowNumber( iLastHit );
			  currOutTracklet->fSector = slice;
			  currOutTracklet->fPID = 211;
		#endif
		#endif // INCLUDE_TPC_HOUGH


			  currOutTracklet->fNPoints = t.NHits();

			  for ( int i = 0; i < t.NHits(); i++ ) {
				currOutTracklet->fPointIDs[i] = clusterData[islice].Id( sliceOutput[islice]->OutTrackHit(t.FirstHitRef()+i) );
			  }

			  currOutTracklet = ( AliHLTTPCTrackSegmentData* )( ( Byte_t * )currOutTracklet + dSize );
			  mySize += dSize;
			  outPtr->fTrackletCnt++;
			}

		  } else { // default output type
			  mySize += sliceOutput[islice]->OutputMemorySize();
			  ntracks += sliceOutput[islice]->NTracks();
		  }
	  }
	  else
	  {
		  HLTWarning( "Output buffer size exceed (buffer size %d, current size %d), tracks are not stored", maxBufferSize, mySize );
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
	  for (int islice = 0;islice < slicecount;islice++)
	  {
		  slice = minslice + islice;
		  if (!fOutputTRAKSEGS) mySize = sliceOutput[islice]->OutputMemorySize();
		  if (mySize > 0)
		  {
			AliHLTComponentBlockData bd;
			FillBlockData( bd );
			bd.fOffset = fOutputTRAKSEGS ? 0 : ((char*) sliceOutput[islice] - (char*) outputPtr);
			bd.fSize = mySize;
			bd.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( slice, slice, sliceminPatch[islice], slicemaxPatch[islice] );
			bd.fDataType = GetOutputDataType();
			outputBlocks.push_back( bd );
			size += mySize;
		  }
	  }
  }

  //No longer needed
  delete[] clusterData;
  //These are only temporary pointers to the output and no longer needed
  delete[] sliceOutput;

  timer.Stop();

  fFullTime += timer.RealTime();
  fRecoTime += timerReco.RealTime();
  fNEvents++;

  // Set log level to "Warning" for on-line system monitoring
  int hz = ( int ) ( fFullTime > 1.e-10 ? fNEvents / fFullTime : 100000 );
  int hz1 = ( int ) ( fRecoTime > 1.e-10 ? fNEvents / fRecoTime : 100000 );
  //Min and Max Patch are taken for first slice processed...
  HLTInfo( "CATracker slices %d-%d: output %d tracks;  input %d clusters, patches %d..%d, rows %d..%d; time: full %d / reco %d Hz",
           minslice, maxslice, ntracks, nClustersTotalSum, sliceminPatch[0], slicemaxPatch[0], slicerow[0], slicerow[1], hz, hz1 );

  return ret;
}


