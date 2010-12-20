// @(#) $Id: AliHLTTPCCATrackerComponent.cxx 45450 2010-11-14 23:49:30Z sgorbuno $
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
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackConvertor.h"
#include "AliHLTArray.h"

#include "AliHLTTPCSpacePointData.h"
#include "AliHLTTPCClusterDataFormat.h"
#include "AliHLTTPCCACompressedInputData.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrackSegmentData.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCTrackletDataFormat.h"
#include "AliHLTTPCDefinitions.h"
#include "AliExternalTrackParam.h"
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
    fMinNTrackClusters( 30 ),
    fMinTrackPt(0.2),
    fClusterZCut( 500. ),
    fNeighboursSearchArea( 0 ), 
    fClusterErrorCorrectionY(0), 
    fClusterErrorCorrectionZ(0),
    fBenchmark("CATracker"), 
    fAllowGPU( 0)
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
    fMinTrackPt( 0.2 ),
    fClusterZCut( 500. ),
    fNeighboursSearchArea(0),
    fClusterErrorCorrectionY(0), 
    fClusterErrorCorrectionZ(0),
    fBenchmark("CATracker"),
    fAllowGPU( 0)
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
  if (fTracker) delete fTracker;
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
  list.push_back( AliHLTTPCCADefinitions::fgkCompressedInputDataType );
}

AliHLTComponentDataType AliHLTTPCCATrackerComponent::GetOutputDataType()
{
  // see header file for class documentation
  return AliHLTTPCCADefinitions::fgkTrackletsDataType;
}

void AliHLTTPCCATrackerComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier )
{
  // define guess for the output data size
  constBase = 200;       // minimum size
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
  fMinTrackPt = 0.2;
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
      HLTImportant( "Will try to run tracker on GPU" );
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

  int iResult2 = 0; //ReadCDBEntry( kAliHLTCDBSolenoidBz, chainId );
  fSolenoidBz = GetBz();

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

  TString arguments = "";
  for ( int i = 0; i < argc; i++ ) {
    if ( !arguments.IsNull() ) arguments += " ";
    arguments += argv[i];
  }

  int retVal = Configure( NULL, NULL, arguments.Data() );
  if (retVal == 0) fTracker = new AliHLTTPCCATrackerFramework(fAllowGPU);
  return(retVal);
}

int AliHLTTPCCATrackerComponent::DoDeinit()
{
  // see header file for class documentation
  if (fTracker) delete fTracker;
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

  fBenchmark.StartNewEvent();
  fBenchmark.Start(0);

  // Event reconstruction in one TPC slice with CA Tracker

  //Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "CA::DoEvent()" );
  if ( evtData.fBlockCnt <= 0 ) {
    HLTWarning( "no blocks in event" );
    return 0;
  }

  const AliHLTComponentBlockData* iter = NULL;
  unsigned long ndx;

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
      if ( iter->fDataType != AliHLTTPCDefinitions::fgkClustersDataType &&
	   iter->fDataType != AliHLTTPCCADefinitions::fgkCompressedInputDataType
	   ) continue;

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
      } else (*slCntIter)++;
    }

	  if ( slices.size() == 0 ) {
		HLTWarning( "no slices found in event" );
		return 0;
	  }


    // Determine slice number to really use. (for obsolete output)
    if ( slices.size() > 1 ) {
      Logging( kHLTLogDebug, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
               "Multiple slice numbers found in event 0x%08lX (%lu). Determining maximum occuring slice number...",
               evtData.fEventID, evtData.fEventID );
      slCntIter = sliceCnts.begin();
      for ( slIter = slices.begin(); slIter != slices.end(); slIter++, slCntIter++ ) {
        Logging( kHLTLogDebug, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
                 "Slice %lu found %lu times.", *slIter, *slCntIter );
      }
    } else if ( slices.size() > 0 ) {
      slice = *( slices.begin() );
    }


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

  if ( !fTracker ) fTracker = new AliHLTTPCCATrackerFramework(fAllowGPU);

  int slicecount = maxslice + 1 - minslice;
  if (slicecount > fTracker->MaxSliceCount())
  {
	maxslice = minslice + (slicecount = fTracker->MaxSliceCount()) - 1;
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
		param.SetMinNTrackClusters( fMinNTrackClusters );
		param.SetMinTrackPt( fMinTrackPt );

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
		if ( slice != AliHLTTPCDefinitions::GetMinSliceNr( *iter ) ) continue;
		if ( iter->fDataType == AliHLTTPCDefinitions::fgkClustersDataType ){
		  AliHLTTPCClusterData* inPtrSP = ( AliHLTTPCClusterData* )( iter->fPtr );
		  nClustersTotal += inPtrSP->fSpacePointCnt;
		  fBenchmark.AddInput(iter->fSize);
		} else 
		if ( iter->fDataType == AliHLTTPCCADefinitions::fgkCompressedInputDataType){
		  fBenchmark.AddInput(iter->fSize);
		  const AliHLTUInt8_t * inPtr =  (const AliHLTUInt8_t *)iter->fPtr;
		  while( inPtr< ((const AliHLTUInt8_t *) iter->fPtr) + iter->fSize ){
		    AliHLTTPCCACompressedClusterRow *row = (AliHLTTPCCACompressedClusterRow*)inPtr;
		    nClustersTotal+= row->fNClusters;	  
		    inPtr = (const AliHLTUInt8_t *)(row->fClusters+row->fNClusters);
		  }
		}
		else continue;

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
		int nPatchClust = 0;
	       
		if ( iter->fDataType == AliHLTTPCDefinitions::fgkClustersDataType ){
		  AliHLTTPCClusterData* inPtrSP = ( AliHLTTPCClusterData* )( iter->fPtr );
		  nPatchClust = inPtrSP->fSpacePointCnt;
		  for ( unsigned int i = 0; i < inPtrSP->fSpacePointCnt; i++ ) {
		    AliHLTTPCSpacePointData *c = &( inPtrSP->fSpacePoints[i] );
		    if ( CAMath::Abs( c->fZ ) > fClusterZCut ) continue;
		    if ( c->fPadRow > 159 ) {
		      HLTError( "Wrong TPC cluster with row number %d received", c->fPadRow );
		      continue;
		    }
		    clusterData[islice].ReadCluster( c->fID, c->fPadRow, c->fX, c->fY, c->fZ, c->fCharge );
		  }	      
		} else 	       
		if ( iter->fDataType == AliHLTTPCCADefinitions::fgkCompressedInputDataType){
		  const AliHLTUInt8_t * inPtr = (const AliHLTUInt8_t *)iter->fPtr;
		  nPatchClust=0;
		  while( inPtr< ((const AliHLTUInt8_t *)iter->fPtr) + iter->fSize ){
		    AliHLTTPCCACompressedClusterRow *row = (AliHLTTPCCACompressedClusterRow*)inPtr;
		    UInt_t id = row->fSlicePatchRowID;
		    UInt_t jslice = id>>10;	  
		    UInt_t jpatch = (id>>6) & 0x7;
		    UInt_t jrow   =  id     & 0x3F;     
		    jrow+= AliHLTTPCTransform::GetFirstRow( jpatch );
		    Double_t rowX = AliHLTTPCTransform::Row2X( jrow );
		    //cout<<"Read row: s "<<jslice<<" p "<<jpatch<<" r "<<jrow<<" x "<<row->fX<<" nclu "<<row->fNClusters<<" :"<<endl;
		    if( jrow > 159 ) {
		      HLTError( "Wrong TPC cluster with row number %d received", jrow );
		      continue;
		    }
		    for ( unsigned int i = 0; i < row->fNClusters; i++ ) {
		      AliHLTTPCCACompressedCluster *c = &( row->fClusters[i] );
		      
		      UInt_t ix0 = c->fP0 >>24;
		      UInt_t ix1 = c->fP1 >>24;
		      Double_t x = (ix1<<8) + ix0;
		      Double_t y = c->fP0 & 0x00FFFFFF;
		      Double_t z = c->fP1 & 0x00FFFFFF;
		      x = (x - 32768.)*1.e-4 + rowX;
		      y = (y - 8388608.)*1.e-4;
		      z = (z - 8388608.)*1.e-4;
		      
		      UInt_t cluId = AliHLTTPCSpacePointData::GetID( jslice, jpatch, nPatchClust );
		      //cout<<"clu "<<i<<": "<<x<<" "<<y<<" "<<z<<" "<<cluId<<endl;
		      if ( CAMath::Abs( z ) <= fClusterZCut ){
			clusterData[islice].ReadCluster( cluId, jrow, x, y, z, 0 );
		      }
		      nPatchClust++;		  
		    }
		    inPtr = (const AliHLTUInt8_t *)(row->fClusters+row->fNClusters);
		  }
		}
		Logging( kHLTLogInfo, "HLT::TPCCATracker::DoEvent", "Reading hits",
			 "Read %d hits for slice %d - patch %d", nPatchClust, slice, patch );
	  }

	  clusterData[islice].FinishReading();
	  nClustersTotalSum += nClustersTotal;
  }

  //Prepare Output
  AliHLTTPCCASliceOutput::outputControlStruct outputControl;
  //Set tracker output so tracker does not have to output both formats!
  outputControl.fEndOfSpace = 0;

  //For new output we can write directly to output buffer
  outputControl.fOutputPtr =  (char*) outputPtr;
  outputControl.fOutputMaxSize = maxBufferSize;

  AliHLTTPCCASliceOutput** sliceOutput = new AliHLTTPCCASliceOutput*[slicecount];
  memset(sliceOutput, 0, slicecount * sizeof(AliHLTTPCCASliceOutput*));

  // reconstruct the event

  fBenchmark.Start(1);
  fTracker->SetOutputControl(&outputControl);
  fTracker->ProcessSlices(minslice, slicecount, clusterData, sliceOutput);
  fBenchmark.Stop(1);
  
  int ret = 0;
  unsigned int mySize = 0;
  int ntracks = 0;
  int error = 0;

  for (int islice = 0;islice < slicecount;islice++)
  {
    if( outputControl.fEndOfSpace ){
      HLTWarning( "Output buffer size exceed, tracks are not stored" );
      ret = -ENOSPC;
      error = 1;
      break;     
    }
    slice = minslice + islice;
    
    if (sliceOutput[islice])
      {
	// write reconstructed tracks
	Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reconstruct",
		 "%d tracks found for slice %d", sliceOutput[islice]->NTracks(), slice );
	
	mySize += sliceOutput[islice]->Size();
	ntracks += sliceOutput[islice]->NTracks();	  
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
	mySize = sliceOutput[islice]->Size();
	if (mySize > 0)
	  {
	    AliHLTComponentBlockData bd;
	    FillBlockData( bd );
	    bd.fOffset = ((char*) sliceOutput[islice] - (char*) outputPtr);
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

  delete[] clusterData;
  delete[] sliceOutput;

  fBenchmark.Stop(0);

  // Set log level to "Warning" for on-line system monitoring

  //Min and Max Patch are taken for first slice processed...

  if( minslice==maxslice ) fBenchmark.SetName(Form("CATracker slice %d",minslice));
  else fBenchmark.SetName(Form("CATracker slices %d-%d",minslice,maxslice));

  HLTInfo(fBenchmark.GetStatistics());
  //No longer needed

  delete[] slicerow;
  delete[] sliceminPatch;
  delete[] slicemaxPatch;

  return ret;
}


