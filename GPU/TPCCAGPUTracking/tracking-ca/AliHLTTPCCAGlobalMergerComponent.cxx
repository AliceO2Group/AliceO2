// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  Matthias Kretz <kretz@kde.org>                          *
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


/** @file   AliHLTTPCCAGlobalMergerComponent.cxx
    @author Matthias Kretz
    @date
    @brief  HLT TPC CA global merger component.
*/

#if __GNUC__>= 3
using namespace std;
#endif

#include "AliHLTTPCCAGlobalMergerComponent.h"
#include "AliHLTTPCCASliceOutput.h"

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAMerger.h"
#include "AliHLTTPCCAMergerOutput.h"
#include "AliHLTTPCCATrackConvertor.h"

#include "AliHLTTPCGMMerger.h"
#include "AliHLTTPCGMMergedTrack.h"

#include "AliHLTTPCDefinitions.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrackSegmentData.h"
#include "AliHLTTPCTrack.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCTrackletDataFormat.h"

#include "AliCDBEntry.h"
#include "AliCDBManager.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "AliHLTExternalTrackParam.h"

#include <climits>
#include <cstdlib>
#include <cerrno>


// ROOT macro for the implementation of ROOT specific class methods
ClassImp( AliHLTTPCCAGlobalMergerComponent )


AliHLTTPCCAGlobalMergerComponent::AliHLTTPCCAGlobalMergerComponent()
: AliHLTProcessor(), fVersion(1), fGlobalMergerVersion0( 0 ), fGlobalMerger(0), fSolenoidBz( 0 ), fClusterErrorCorrectionY(0), fClusterErrorCorrectionZ(0), fBenchmark("GlobalMerger")
{
  // see header file for class documentation
}

AliHLTTPCCAGlobalMergerComponent::AliHLTTPCCAGlobalMergerComponent( const AliHLTTPCCAGlobalMergerComponent & ):AliHLTProcessor(), fVersion(1), fGlobalMergerVersion0( 0 ), fGlobalMerger(0), fSolenoidBz( 0 ), fClusterErrorCorrectionY(0), fClusterErrorCorrectionZ(0), fBenchmark("GlobalMerger")
{
// dummy
}

AliHLTTPCCAGlobalMergerComponent &AliHLTTPCCAGlobalMergerComponent::operator=( const AliHLTTPCCAGlobalMergerComponent & )
{
  // dummy
  return *this;
}

// Public functions to implement AliHLTComponent's interface.
// These functions are required for the registration process

const char *AliHLTTPCCAGlobalMergerComponent::GetComponentID()
{
  // see header file for class documentation
  return "TPCCAGlobalMerger";
}

void AliHLTTPCCAGlobalMergerComponent::GetInputDataTypes( AliHLTComponentDataTypeList &list )
{
  // see header file for class documentation
  list.clear();
  list.push_back( AliHLTTPCCADefinitions::fgkTrackletsDataType );
  //list.push_back( AliHLTTPCDefinitions::fgkTrackSegmentsDataType );
  //list.push_back( AliHLTTPCDefinitions::fgkVertexDataType );
}

AliHLTComponentDataType AliHLTTPCCAGlobalMergerComponent::GetOutputDataType()
{
  // see header file for class documentation
  //return AliHLTTPCDefinitions::fgkTracksDataType; // old
  return kAliHLTDataTypeTrack|kAliHLTDataOriginTPC;
}

void AliHLTTPCCAGlobalMergerComponent::GetOutputDataSize( unsigned long &constBase, double &inputMultiplier )
{
  // see header file for class documentation
  // XXX TODO: Find more realistic values.
  constBase = 0;
  inputMultiplier = 1.0;
}

AliHLTComponent *AliHLTTPCCAGlobalMergerComponent::Spawn()
{
  // see header file for class documentation
  return new AliHLTTPCCAGlobalMergerComponent;
}


void AliHLTTPCCAGlobalMergerComponent::SetDefaultConfiguration()
{
  // Set default configuration for the CA merger component
  // Some parameters can be later overwritten from the OCDB

  fVersion = 1;
  fSolenoidBz = -5.00668;
  fClusterErrorCorrectionY = 0;
  fClusterErrorCorrectionZ = 1.1;
  fBenchmark.Reset();
  fBenchmark.SetTimer(0,"total");
  fBenchmark.SetTimer(1,"reco");    
}

int AliHLTTPCCAGlobalMergerComponent::ReadConfigurationString(  const char* arguments )
{
  // Set configuration parameters for the CA merger component from the string

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

    if ( argument.CompareTo( "-version" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fVersion  = ( ( TObjString* )pTokens->At( i ) )->GetString().Atoi();
      HLTInfo( "Merger version set to: %d", fVersion );
      continue;
    }

    if ( argument.CompareTo( "-solenoidBz" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      HLTWarning("argument -solenoidBz is deprecated, magnetic field set up globally (%f)", GetBz());
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


int AliHLTTPCCAGlobalMergerComponent::ReadCDBEntry( const char* cdbEntry, const char* chainId )
{
  // see header file for class documentation

  const char* defaultNotify = "";

  if ( !cdbEntry ) {
    cdbEntry = "HLT/ConfigTPC/TPCCAGlobalMerger";
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



int AliHLTTPCCAGlobalMergerComponent::Configure( const char* cdbEntry, const char* chainId, const char *commandLine )
{
  // Configure the component
  // There are few levels of configuration,
  // parameters which are set on one step can be overwritten on the next step

  //* read hard-coded values

  SetDefaultConfiguration();

  //* read the default CDB entry

  int iResult1 = ReadCDBEntry( NULL, chainId );

  //* read magnetic field

  int iResult2 = 0;//ReadCDBEntry( kAliHLTCDBSolenoidBz, chainId );
  fSolenoidBz = GetBz();

  //* read the actual CDB entry if required

  int iResult3 = ( cdbEntry ) ? ReadCDBEntry( cdbEntry, chainId ) : 0;

  //* read extra parameters from input (if they are)

  int iResult4 = 0;

  if ( commandLine && commandLine[0] != '\0' ) {
    HLTInfo( "received configuration string from HLT framework: \"%s\"", commandLine );
    iResult4 = ReadConfigurationString( commandLine );
  }


  // Initialize the merger

  AliHLTTPCCAParam param;

  {
    // get gemetry
    int iSec = 0;
    float inRmin = 83.65;
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
    int nRows = AliHLTTPCTransform::GetNRows();
    float padPitch = 0.4;
    float sigmaZ = 0.228808;
    float *rowX = new float [nRows];
    for ( int irow = 0; irow < nRows; irow++ ) {
      rowX[irow] = AliHLTTPCTransform::Row2X( irow );
    }

    param.Initialize( iSec, nRows, rowX, alpha, dalpha,
                      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, fSolenoidBz );

    if( fClusterErrorCorrectionY>1.e-4 ) param.SetClusterError2CorrectionY( fClusterErrorCorrectionY*fClusterErrorCorrectionY );
    if( fClusterErrorCorrectionZ>1.e-4 ) param.SetClusterError2CorrectionZ( fClusterErrorCorrectionZ*fClusterErrorCorrectionZ );
    param.Update();

    delete[] rowX;
  }


  if( fVersion==0 ) fGlobalMergerVersion0->SetSliceParam( param );
  else fGlobalMerger->SetSliceParam( param );

  return iResult1 ? iResult1 : ( iResult2 ? iResult2 : ( iResult3 ? iResult3 : iResult4 ) );
}




int AliHLTTPCCAGlobalMergerComponent::DoInit( int argc, const char** argv )
{
  // see header file for class documentation

  if ( fGlobalMergerVersion0 || fGlobalMerger ) {
    return EINPROGRESS;
  }

  fGlobalMergerVersion0 = new AliHLTTPCCAMerger();
  fGlobalMerger         = new AliHLTTPCGMMerger();

  TString arguments = "";
  for ( int i = 0; i < argc; i++ ) {
    if ( !arguments.IsNull() ) arguments += " ";
    arguments += argv[i];
  }

  return Configure( NULL, NULL, arguments.Data()  );
}

int AliHLTTPCCAGlobalMergerComponent::Reconfigure( const char* cdbEntry, const char* chainId )
{
  // Reconfigure the component from OCDB

  return Configure( cdbEntry, chainId, NULL );
}



int AliHLTTPCCAGlobalMergerComponent::DoDeinit()
{
  // see header file for class documentation
  delete fGlobalMergerVersion0;
  fGlobalMergerVersion0 = 0;
  delete fGlobalMerger;
  fGlobalMerger = 0;

  return 0;
}

int AliHLTTPCCAGlobalMergerComponent::DoEvent( const AliHLTComponentEventData &evtData,
    const AliHLTComponentBlockData *blocks, AliHLTComponentTriggerData &/*trigData*/,
    AliHLTUInt8_t *outputPtr, AliHLTUInt32_t &size, AliHLTComponentBlockDataList &outputBlocks )
{
  // see header file for class documentation
  int iResult = 0;
  unsigned int maxBufferSize = size;

  size = 0;

  if ( !outputPtr ) {
    return -ENOSPC;
  }
  if ( !IsDataEvent() ) {
    return 0;
  }
  fBenchmark.StartNewEvent();
  fBenchmark.Start(0);

  if( fVersion==0 )  fGlobalMergerVersion0->Clear();
  else fGlobalMerger->Clear();

  const AliHLTComponentBlockData *const blocksEnd = blocks + evtData.fBlockCnt;
  for ( const AliHLTComponentBlockData *block = blocks; block < blocksEnd; ++block ) {
    if ( block->fDataType != AliHLTTPCCADefinitions::fgkTrackletsDataType ) {
      continue;
    }

    fBenchmark.AddInput(block->fSize);

    int slice = AliHLTTPCDefinitions::GetMinSliceNr( *block );
    if ( slice < 0 || slice >= AliHLTTPCTransform::GetNSlice() ) {
      HLTError( "invalid slice number %d extracted from specification 0x%08lx,  skipping block of type %s",
                slice, block->fSpecification, DataType2Text( block->fDataType ).c_str() );
      // just remember the error, if there are other valid blocks ignore the error, return code otherwise
      iResult = -EBADF;
      continue;
    }

    if ( slice != AliHLTTPCDefinitions::GetMaxSliceNr( *block ) ) {
      // the code was not written for/ never used with multiple slices in one data block/ specification
      HLTWarning( "specification 0x%08lx indicates multiple slices in data block %s: never used before, please audit the code",
                  block->fSpecification, DataType2Text( block->fDataType ).c_str() );
    }
    AliHLTTPCCASliceOutput *sliceOut =  reinterpret_cast<AliHLTTPCCASliceOutput *>( block->fPtr );
    //sliceOut->SetPointers();
    if( fVersion==0 ) fGlobalMergerVersion0->SetSliceData( slice, sliceOut );
    else fGlobalMerger->SetSliceData( slice, sliceOut );

	/*char filename[256];
	sprintf(filename, "debug%d.out", slice);
	FILE* fp = fopen(filename, "w+b");
	if (fp == NULL) printf("Error!!!\n");
	fwrite(sliceOut, 1, sliceOut->EstimateSize(sliceOut->NTracks(), sliceOut->NTrackClusters()), fp);
	fclose(fp);*/
  }
  fBenchmark.Start(1);
  if( fVersion==0 ) fGlobalMergerVersion0->Reconstruct();
  else fGlobalMerger->Reconstruct();
  fBenchmark.Stop(1);

  // Fill output 

  if( fVersion==0 ){

    const AliHLTTPCCAMergerOutput *mergerOutput = fGlobalMergerVersion0->Output();

    unsigned int mySize = 0;
    {
      AliHLTTracksData* outPtr = ( AliHLTTracksData* )( outputPtr );
      
      AliHLTExternalTrackParam* currOutTrack = outPtr->fTracklets;
      
      mySize =   ( ( AliHLTUInt8_t * )currOutTrack ) -  ( ( AliHLTUInt8_t * )outputPtr );
      
      outPtr->fCount = 0;
      
      int nTracks = mergerOutput->NTracks();
      
      for ( int itr = 0; itr < nTracks; itr++ ) {
	
	// convert AliHLTTPCCAMergedTrack to AliHLTTrack
	
	const AliHLTTPCCAMergedTrack &track = mergerOutput->Track( itr );
	
	unsigned int dSize = sizeof( AliHLTExternalTrackParam ) + track.NClusters() * sizeof( unsigned int );
	
	if ( mySize + dSize > maxBufferSize ) {
	  HLTWarning( "Output buffer size exceed (buffer size %d, current size %d), %d tracks are not stored", maxBufferSize, mySize, nTracks - itr + 1 );
	  iResult = -ENOSPC;
	  break;
	}
	
	// first convert to AliExternalTrackParam
	
	AliExternalTrackParam tp, tpEnd;
	AliHLTTPCCATrackConvertor::GetExtParam( track.InnerParam(), tp,  track.InnerAlpha() );
	AliHLTTPCCATrackConvertor::GetExtParam( track.OuterParam(), tpEnd, track.OuterAlpha() );
	
	// normalize the angle to +-Pi
	
	currOutTrack->fAlpha = tp.GetAlpha() - CAMath::Nint(tp.GetAlpha()/CAMath::TwoPi())*CAMath::TwoPi();      
	currOutTrack->fX = tp.GetX();
	currOutTrack->fY = tp.GetY();
	currOutTrack->fZ = tp.GetZ();      
	{
	  float sinA = TMath::Sin( track.OuterAlpha() - track.InnerAlpha());
	  float cosA = TMath::Cos( track.OuterAlpha() - track.InnerAlpha());
	  currOutTrack->fLastX = tpEnd.GetX()*cosA - tpEnd.GetY()*sinA;
	  currOutTrack->fLastY = tpEnd.GetX()*sinA + tpEnd.GetY()*cosA;
	  currOutTrack->fLastZ = tpEnd.GetZ();
	}
	currOutTrack->fq1Pt = tp.GetSigned1Pt();
	currOutTrack->fSinPsi = tp.GetSnp();
	currOutTrack->fTgl = tp.GetTgl();
	for( int i=0; i<15; i++ ) currOutTrack->fC[i] = tp.GetCovariance()[i];
	currOutTrack->fTrackID = itr;
	currOutTrack->fFlags = 0;
	currOutTrack->fNPoints = track.NClusters();    
	for ( int i = 0; i < track.NClusters(); i++ ) currOutTrack->fPointIDs[i] = mergerOutput->ClusterId( track.FirstClusterRef() + i );
	
	currOutTrack = ( AliHLTExternalTrackParam* )( (( Byte_t * )currOutTrack) + dSize );
	mySize += dSize;
	outPtr->fCount++;
      }    

      AliHLTComponentBlockData resultData;
      FillBlockData( resultData );
      resultData.fOffset = 0;
      resultData.fSize = mySize;
      resultData.fDataType = kAliHLTDataTypeTrack|kAliHLTDataOriginTPC;
      resultData.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( 0, 35, 0, 5 );
      outputBlocks.push_back( resultData );
      fBenchmark.AddOutput(resultData.fSize);
      
      size = resultData.fSize;
    }
    
    HLTInfo( "CAGlobalMerger:: output %d tracks", mergerOutput->NTracks() );
    fGlobalMergerVersion0->Clear();

  } else { // new merger 

    unsigned int mySize = 0;
    {
      AliHLTTracksData* outPtr = ( AliHLTTracksData* )( outputPtr );
      AliHLTExternalTrackParam* currOutTrack = outPtr->fTracklets;
      mySize =   ( ( AliHLTUInt8_t * )currOutTrack ) -  ( ( AliHLTUInt8_t * )outputPtr );
      outPtr->fCount = 0;   
      int nTracks = fGlobalMerger->NOutputTracks();

      for ( int itr = 0; itr < nTracks; itr++ ) {

	// convert AliHLTTPCGMMergedTrack to AliHLTTrack
	
	const AliHLTTPCGMMergedTrack &track = fGlobalMerger->OutputTracks()[ itr ];
	if( !track.OK() ) continue;
	unsigned int dSize = sizeof( AliHLTExternalTrackParam ) + track.NClusters() * sizeof( unsigned int );
	
	if ( mySize + dSize > maxBufferSize ) {
	  HLTWarning( "Output buffer size exceed (buffer size %d, current size %d), %d tracks are not stored", maxBufferSize, mySize, nTracks - itr + 1 );
	  iResult = -ENOSPC;
	  break;
	}

	// first convert to AliExternalTrackParam

	AliExternalTrackParam tp;
	track.GetParam().GetExtParam( tp,  track.GetAlpha() );
      
	// normalize the angle to +-Pi
	      
	currOutTrack->fAlpha = tp.GetAlpha() - CAMath::Nint(tp.GetAlpha()/CAMath::TwoPi())*CAMath::TwoPi();      
	currOutTrack->fX = tp.GetX();
	currOutTrack->fY = tp.GetY();
	currOutTrack->fZ = tp.GetZ();      
	currOutTrack->fLastX = track.LastX();
	currOutTrack->fLastY = track.LastY();
	currOutTrack->fLastZ = track.LastZ();
      
	currOutTrack->fq1Pt = tp.GetSigned1Pt();
	currOutTrack->fSinPsi = tp.GetSnp();
	currOutTrack->fTgl = tp.GetTgl();
	for( int i=0; i<15; i++ ) currOutTrack->fC[i] = tp.GetCovariance()[i];
	currOutTrack->fTrackID = itr;
	currOutTrack->fFlags = 0;
	currOutTrack->fNPoints = track.NClusters();    
	for ( int i = 0; i < track.NClusters(); i++ ) currOutTrack->fPointIDs[i] = fGlobalMerger->OutputClusterIds()[track.FirstClusterRef() + i ];
	
	currOutTrack = ( AliHLTExternalTrackParam* )( (( Byte_t * )currOutTrack) + dSize );
	mySize += dSize;
	outPtr->fCount++;
      }
  
      AliHLTComponentBlockData resultData;
      FillBlockData( resultData );
      resultData.fOffset = 0;
      resultData.fSize = mySize;
      resultData.fDataType = kAliHLTDataTypeTrack|kAliHLTDataOriginTPC;
      resultData.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( 0, 35, 0, 5 );
      outputBlocks.push_back( resultData );
      fBenchmark.AddOutput(resultData.fSize);
      
      size = resultData.fSize;
    }

    HLTInfo( "CAGlobalMerger:: output %d tracks", fGlobalMerger->NOutputTracks() );

    fGlobalMerger->Clear();
  }

  fBenchmark.Stop(0);
  HLTInfo( fBenchmark.GetStatistics() );
  return iResult;
}

