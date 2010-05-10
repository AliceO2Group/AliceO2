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


/** @file   AliHLTTPCCATrackerOutputConverter.cxx
    @author Matthias Kretz
    @date
    @brief  HLT TPC CA global merger component.
*/

#if __GNUC__>= 3
using namespace std;
#endif

#include "AliHLTTPCCATrackerOutputConverter.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrack.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCCADef.h"
#include "AliHLTTPCDefinitions.h"
#include "AliHLTTPCCATrackConvertor.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCAParam.h"

#include "AliCDBEntry.h"
#include "AliCDBManager.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "AliHLTExternalTrackParam.h"

#include <climits>
#include <cstdlib>
#include <cerrno>


// ROOT macro for the implementation of ROOT specific class methods
ClassImp( AliHLTTPCCATrackerOutputConverter )


AliHLTTPCCATrackerOutputConverter::AliHLTTPCCATrackerOutputConverter()
: fBenchmark("TPCCATrackerOutputConverter")
{
  // see header file for class documentation
}

// Public functions to implement AliHLTComponent's interface.
// These functions are required for the registration process

const char *AliHLTTPCCATrackerOutputConverter::GetComponentID()
{
  // see header file for class documentation
  return "TPCCATrackerOutputConverter";
}

void AliHLTTPCCATrackerOutputConverter::GetInputDataTypes( AliHLTComponentDataTypeList &list )
{
  // see header file for class documentation
  list.clear();
  list.push_back( AliHLTTPCCADefinitions::fgkTrackletsDataType );
}

AliHLTComponentDataType AliHLTTPCCATrackerOutputConverter::GetOutputDataType()
{
  // see header file for class documentation
  return kAliHLTDataTypeTrack|kAliHLTDataOriginTPC;
}

void AliHLTTPCCATrackerOutputConverter::GetOutputDataSize( unsigned long &constBase, double &inputMultiplier )
{
  // see header file for class documentation
  // XXX TODO: Find more realistic values.
  constBase = 0;
  inputMultiplier = 1.0;
}

AliHLTComponent *AliHLTTPCCATrackerOutputConverter::Spawn()
{
  // see header file for class documentation
  return new AliHLTTPCCATrackerOutputConverter;
}


void AliHLTTPCCATrackerOutputConverter::SetDefaultConfiguration()
{
  // Set default configuration for the CA merger component
  // Some parameters can be later overwritten from the OCDB

  fBenchmark.Reset();
  fBenchmark.SetTimer(0,"total");
}

int AliHLTTPCCATrackerOutputConverter::ReadConfigurationString(  const char* arguments )
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


int AliHLTTPCCATrackerOutputConverter::ReadCDBEntry( const char* cdbEntry, const char* chainId )
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



int AliHLTTPCCATrackerOutputConverter::Configure( const char* cdbEntry, const char* chainId, const char *commandLine )
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

  //* read the actual CDB entry if required

  int iResult3 = ( cdbEntry ) ? ReadCDBEntry( cdbEntry, chainId ) : 0;

  //* read extra parameters from input (if they are)

  int iResult4 = 0;

  if ( commandLine && commandLine[0] != '\0' ) {
    HLTInfo( "received configuration string from HLT framework: \"%s\"", commandLine );
    iResult4 = ReadConfigurationString( commandLine );
  }

  return iResult1 ? iResult1 : ( iResult2 ? iResult2 : ( iResult3 ? iResult3 : iResult4 ) );
}




int AliHLTTPCCATrackerOutputConverter::DoInit( int argc, const char** argv )
{
  // see header file for class documentation

  TString arguments = "";
  for ( int i = 0; i < argc; i++ ) {
    if ( !arguments.IsNull() ) arguments += " ";
    arguments += argv[i];
  }

  return Configure( NULL, NULL, arguments.Data()  );
}

int AliHLTTPCCATrackerOutputConverter::Reconfigure( const char* cdbEntry, const char* chainId )
{
  // Reconfigure the component from OCDB

  return Configure( cdbEntry, chainId, NULL );
}



int AliHLTTPCCATrackerOutputConverter::DoDeinit()
{
  // see header file for class documentation
  return 0;
}

int AliHLTTPCCATrackerOutputConverter::DoEvent( const AliHLTComponentEventData &evtData,
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

    
    const AliHLTTPCCASliceOutput &sliceOut =  *(reinterpret_cast<AliHLTTPCCASliceOutput *>( block->fPtr ));
    const AliHLTTPCCASliceOutTrack *sliceTr = sliceOut.GetFirstTrack();
   
    // Output block

    unsigned int blockSize = 0;
    AliHLTTracksData* outPtr = ( AliHLTTracksData* )( outputPtr + size );
    AliHLTExternalTrackParam* currOutTrack = outPtr->fTracklets;
    blockSize =   ( ( AliHLTUInt8_t * )currOutTrack ) -  ( ( AliHLTUInt8_t * )outputPtr );
    outPtr->fCount = 0;   
    AliHLTTPCCAParam sliceParam;

    for ( int itr = 0; itr < sliceOut.NTracks(); itr++ ) {
      
      int nClu = sliceTr->NClusters();

      unsigned int dSize = sizeof( AliHLTExternalTrackParam ) + nClu * sizeof( unsigned int );

      if ( size + blockSize + dSize > maxBufferSize ) {
        HLTWarning( "Output buffer size exceed (buffer size %d, current size %d), tracks are not stored", maxBufferSize, blockSize );
        iResult = -ENOSPC;
        break;
      }

     // first convert to AliExternalTrackParam

      AliHLTTPCCATrackParam t0;
      t0.InitParam();
      t0.SetParam(sliceTr->Param());

      AliExternalTrackParam tp;
      AliHLTTPCCATrackConvertor::GetExtParam( t0, tp, sliceParam.Alpha( slice ) );
      
      // normalize the angle to +-Pi
	      
      currOutTrack->fAlpha = tp.GetAlpha() - CAMath::Nint(tp.GetAlpha()/CAMath::TwoPi())*CAMath::TwoPi();      
      currOutTrack->fX = tp.GetX();
      currOutTrack->fY = tp.GetY();
      currOutTrack->fZ = tp.GetZ();      
      currOutTrack->fq1Pt = tp.GetSigned1Pt();
      currOutTrack->fSinPsi = tp.GetSnp();
      currOutTrack->fTgl = tp.GetTgl();
      for( int i=0; i<15; i++ ) currOutTrack->fC[i] = tp.GetCovariance()[i];
      currOutTrack->fTrackID = itr;
      currOutTrack->fFlags = 0;
      currOutTrack->fNPoints = nClu;    
      for( int i = 0; i< nClu; i++ ) {	
	int id, row;
	float x,y,z;
	sliceTr->Cluster( i ).Get(id,row,x,y,z);      
	currOutTrack->fPointIDs[i] = id;
	if( i == nClu-1 ){
	  currOutTrack->fLastX = x;
	  currOutTrack->fLastY = y;
	  currOutTrack->fLastZ = z;
	}
      }
      currOutTrack = ( AliHLTExternalTrackParam* )( (( Byte_t * )currOutTrack) + dSize );
      blockSize += dSize;
      outPtr->fCount++;
      sliceTr = sliceTr->GetNextTrack();
    }
 
    AliHLTComponentBlockData resultData;
    FillBlockData( resultData );
    resultData.fOffset = size;
    resultData.fSize = blockSize;
    resultData.fDataType = kAliHLTDataTypeTrack|kAliHLTDataOriginTPC;
    resultData.fSpecification = block->fSpecification;
    outputBlocks.push_back( resultData );
    fBenchmark.AddOutput(resultData.fSize);
    size += resultData.fSize;
  }

  fBenchmark.Stop(0);
  HLTInfo( fBenchmark.GetStatistics() );
  return iResult;
}

