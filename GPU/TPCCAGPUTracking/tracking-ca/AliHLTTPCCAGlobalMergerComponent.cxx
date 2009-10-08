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
#include "AliHLTTPCCAMergerOutput.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCCAMerger.h"
#include "AliHLTTPCVertex.h"
#include "AliHLTTPCVertexData.h"
#include "AliHLTTPCTrackSegmentData.h"
#include "AliHLTTPCTrack.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCTrackletDataFormat.h"
#include "AliHLTTPCCADef.h"
#include "AliHLTTPCDefinitions.h"
#include "AliHLTTPCCATrackConvertor.h"
#include "AliHLTTPCCASliceOutput.h"

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
  : fGlobalMerger( 0 ), fSolenoidBz( 0 ), fClusterErrorCorrectionY(0), fClusterErrorCorrectionZ(0)
{
  // see header file for class documentation
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

  fSolenoidBz = 5.;
  fClusterErrorCorrectionY = 0;
  fClusterErrorCorrectionZ = 1.1;
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

    if ( argument.CompareTo( "-solenoidBz" ) == 0 ) {
      if ( ( bMissingParam = ( ++i >= pTokens->GetEntries() ) ) ) break;
      fSolenoidBz = ( ( TObjString* )pTokens->At( i ) )->GetString().Atof();
      HLTInfo( "Magnetic Field set to: %f", fSolenoidBz );
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

  int iResult2 = ReadCDBEntry( kAliHLTCDBSolenoidBz, chainId );

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


  fGlobalMerger->SetSliceParam( param );

  return iResult1 ? iResult1 : ( iResult2 ? iResult2 : ( iResult3 ? iResult3 : iResult4 ) );
}




int AliHLTTPCCAGlobalMergerComponent::DoInit( int argc, const char** argv )
{
  // see header file for class documentation

  if ( fGlobalMerger ) {
    return EINPROGRESS;
  }

  fGlobalMerger = new AliHLTTPCCAMerger();

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

  fGlobalMerger->Clear();

  const AliHLTComponentBlockData *const blocksEnd = blocks + evtData.fBlockCnt;
  for ( const AliHLTComponentBlockData *block = blocks; block < blocksEnd; ++block ) {
    if ( block->fDataType != AliHLTTPCCADefinitions::fgkTrackletsDataType ) {
      continue;
    }

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
    sliceOut->SetPointers();
    fGlobalMerger->SetSliceData( slice, sliceOut );

	/*char filename[256];
	sprintf(filename, "debug%d.out", slice);
	FILE* fp = fopen(filename, "w+b");
	if (fp == NULL) printf("Error!!!\n");
	fwrite(sliceOut, 1, sliceOut->EstimateSize(sliceOut->NTracks(), sliceOut->NTrackClusters()), fp);
	fclose(fp);*/
  }
  fGlobalMerger->Reconstruct();

  const AliHLTTPCCAMergerOutput *mergerOutput = fGlobalMerger->Output();


  // Fill output tracks

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
      
      currOutTrack->fAlpha = tp.GetAlpha();
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
    size = resultData.fSize;
  }

  HLTInfo( "CAGlobalMerger:: output %d tracks", mergerOutput->NTracks() );


  /* old format
  {
    // check if there was enough space in the output buffer

    int nTracks = mergerOutput->NTracks();

    AliHLTTPCTrackArray array( nTracks );

    int nClusters = 0;
    for ( int itr = 0; itr < nTracks; itr++ ) {

      // convert AliHLTTPCCAMergedTrack to AliHLTTPCTrack

      const AliHLTTPCCAMergedTrack &track = mergerOutput->Track( itr );

      AliHLTTPCTrack out;

      // first convert to AliExternalTrackParam

      AliExternalTrackParam tp, tpEnd;
      AliHLTTPCCATrackConvertor::GetExtParam( track.InnerParam(), tp, 0 );
      AliHLTTPCCATrackConvertor::GetExtParam( track.OuterParam(), tpEnd, 0 );

      // set parameters, with rotation to global coordinates

      out.SetCharge( ( int ) tp.GetSign() );
      out.SetPt( TMath::Abs( tp.GetSignedPt() ) );
      out.SetPsi( fmod( TMath::ASin( tp.GetSnp() ) + track.InnerAlpha() , 2*TMath::Pi() ) );
      out.SetTgl( tp.GetTgl() );
      {
        float sinA = TMath::Sin( track.InnerAlpha() );
        float cosA = TMath::Cos( track.InnerAlpha() );

        out.SetFirstPoint( tp.GetX()*cosA - tp.GetY()*sinA,
                           tp.GetX()*sinA + tp.GetY()*cosA,
                           tp.GetZ() );
      }

      {
        float sinA = TMath::Sin( track.OuterAlpha() );
        float cosA = TMath::Cos( track.OuterAlpha() );

        out.SetLastPoint( tpEnd.GetX()*cosA - tpEnd.GetY()*sinA,
                          tpEnd.GetX()*sinA + tpEnd.GetY()*cosA,
                          tpEnd.GetZ() );
      }

      // set parameter errors w/o rotation, as it is done in AliHLTTPCTrackArray

      out.SetY0err( tp.GetSigmaY2() );
      out.SetZ0err( tp.GetSigmaZ2() );
      out.SetPterr( tp.GetSigma1Pt2() );
      out.SetPsierr( tp.GetSigmaSnp2() );
      out.SetTglerr( tp.GetSigmaTgl2() );

      // set cluster ID's

      unsigned int hitID[1000];
      for ( int i = 0; i < track.NClusters(); i++ ) hitID[i] = mergerOutput->ClusterId( track.FirstClusterRef() + i );

      out.SetNHits( track.NClusters() );
      out.SetHits( track.NClusters(), hitID );

      out.SetSector( -1 );
      out.CalculateHelix();
      if ( !out.CheckConsistency() )  *( array.NextTrack() ) = out;
      nClusters += track.NClusters();
    }


    if ( sizeof( AliHLTTPCTrackletData ) + nTracks*sizeof( AliHLTTPCTrackSegmentData ) + nClusters*sizeof( unsigned int )
         > maxBufferSize ) {
      iResult = -ENOSPC;
    } else {
      AliHLTTPCTrackletData *outPtr = ( AliHLTTPCTrackletData* )( outputPtr );
      unsigned int nOutTracks = 0;
      mySize = array.WriteTracks( nOutTracks, outPtr->fTracklets );
      mySize += sizeof( AliHLTTPCTrackletData );
      outPtr->fTrackletCnt = nOutTracks;
    }
  }

  AliHLTComponentBlockData resultData;
  FillBlockData( resultData );
  resultData.fOffset = 0;
  resultData.fSize = mySize;
  resultData.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( 0, 35, 0, 5 );
  outputBlocks.push_back( resultData );
  size = resultData.fSize;

  HLTInfo( "CAGlobalMerger:: output %d tracks", mergerOutput->NTracks() );
  */
  return iResult;
}

