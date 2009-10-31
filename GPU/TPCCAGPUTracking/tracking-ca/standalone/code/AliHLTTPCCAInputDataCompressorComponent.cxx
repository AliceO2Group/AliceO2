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

#include "AliHLTTPCCAInputDataCompressorComponent.h"
#include "AliHLTTPCCACompressedInputData.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCClusterDataFormat.h"
#include "AliHLTTPCSpacePointData.h"
#include "AliHLTTPCDefinitions.h"
#include "AliHLTTPCCADef.h"
#include "TStopwatch.h"


const AliHLTComponentDataType AliHLTTPCCADefinitions::fgkCompressedInputDataType = AliHLTComponentDataTypeInitializer( "CAINPACK", kAliHLTDataOriginTPC );

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp( AliHLTTPCCAInputDataCompressorComponent )

AliHLTTPCCAInputDataCompressorComponent::AliHLTTPCCAInputDataCompressorComponent()
  :
  AliHLTProcessor(),
  fTotalTime( 0 ),
  fTotalInputSize( 0 ),
  fTotalOutputSize( 0 ),
  fNEvents( 0 )
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt
}

AliHLTTPCCAInputDataCompressorComponent::AliHLTTPCCAInputDataCompressorComponent( const AliHLTTPCCAInputDataCompressorComponent& )
  :
  AliHLTProcessor(),
  fTotalTime( 0 ),
  fTotalInputSize( 0 ),
  fTotalOutputSize( 0 ),
  fNEvents( 0 )
{
  // see header file for class documentation
  HLTFatal( "copy constructor untested" );
}

AliHLTTPCCAInputDataCompressorComponent& AliHLTTPCCAInputDataCompressorComponent::operator=( const AliHLTTPCCAInputDataCompressorComponent& )
{
  // see header file for class documentation
  HLTFatal( "assignment operator untested" );
  return *this;
}

AliHLTTPCCAInputDataCompressorComponent::~AliHLTTPCCAInputDataCompressorComponent()
{
  // see header file for class documentation  
}

//
// Public functions to implement AliHLTComponent's interface.
// These functions are required for the registration process
//

const char* AliHLTTPCCAInputDataCompressorComponent::GetComponentID()
{
  // see header file for class documentation
  return "TPCCAInputDataCompressor";
}

void AliHLTTPCCAInputDataCompressorComponent::GetInputDataTypes( vector<AliHLTComponentDataType>& list )
{
  // see header file for class documentation
  list.clear();
  list.push_back( AliHLTTPCDefinitions::fgkClustersDataType );
}

AliHLTComponentDataType AliHLTTPCCAInputDataCompressorComponent::GetOutputDataType()
{
  // see header file for class documentation
  return AliHLTTPCCADefinitions::fgkCompressedInputDataType;
}

void AliHLTTPCCAInputDataCompressorComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier )
{
  // define guess for the output data size
  constBase = 200;       // minimum size
  inputMultiplier = 0.25; // size relative to input
}

AliHLTComponent* AliHLTTPCCAInputDataCompressorComponent::Spawn()
{
  // see header file for class documentation
  return new AliHLTTPCCAInputDataCompressorComponent;
}




int AliHLTTPCCAInputDataCompressorComponent::DoInit( int /*argc*/, const char** /*argv*/ )
{
  // Configure the CA tracker component
  fTotalTime = 0;
  fTotalInputSize = 0;
  fTotalOutputSize = 0; 
  fNEvents = 0;
  return 0;
}


int AliHLTTPCCAInputDataCompressorComponent::DoDeinit()
{
  // see header file for class documentation
  return 0;
}



int AliHLTTPCCAInputDataCompressorComponent::Reconfigure( const char* /*cdbEntry*/, const char* /*chainId*/ )
{
  // Reconfigure the component from OCDB .
  return 0;
}



int AliHLTTPCCAInputDataCompressorComponent::DoEvent
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

  // Preprocess the data for CA Slice Tracker

  if ( evtData.fBlockCnt <= 0 ) {
    HLTWarning( "no blocks in event" );
    return 0;
  }

  Int_t ret = 0;

  Int_t inTotalSize = 0;    
  Int_t outTotalSize = 0;
  Int_t minSlice = 100;
  Int_t maxSlice = -1;

  for ( unsigned long ndx = 0; ndx < evtData.fBlockCnt; ndx++ ) {
    const AliHLTComponentBlockData* iter = blocks + ndx;
    if ( iter->fDataType != AliHLTTPCDefinitions::fgkClustersDataType ) continue;    

    if( minSlice>AliHLTTPCDefinitions::GetMinSliceNr( *iter ) ) minSlice = AliHLTTPCDefinitions::GetMinSliceNr( *iter ) ;
    if( maxSlice<AliHLTTPCDefinitions::GetMaxSliceNr( *iter ) ) maxSlice = AliHLTTPCDefinitions::GetMaxSliceNr( *iter ) ;
   
   
    inTotalSize += iter->fSize;

    AliHLTTPCClusterData* inPtrSP = ( AliHLTTPCClusterData* )( iter->fPtr );    

    AliHLTTPCCACompressedCluster *outCluster = (AliHLTTPCCACompressedCluster*)( outputPtr+outTotalSize );
    AliHLTTPCCACompressedClusterRow *outRow = 0;

    Int_t dSize = 0;
    UShort_t oldId = 0;
    for ( unsigned int i = 0; i < inPtrSP->fSpacePointCnt; i++ ){ 
      AliHLTTPCSpacePointData *cluster = &( inPtrSP->fSpacePoints[i] );
      UInt_t origId = cluster->fID;
      UInt_t patch = (origId>>22)&0x7;
      UInt_t slice = origId>>25;
      UInt_t row = cluster->fPadRow;
      Double_t rowX = AliHLTTPCTransform::Row2X( row );
      row = row - AliHLTTPCTransform::GetFirstRow( patch );
      UShort_t id = (UShort_t)( (slice<<10) +(patch<<6) + row );
      if( i==0 || id!= oldId ){ 
	// fill new row header	
	outRow = (AliHLTTPCCACompressedClusterRow*) outCluster;
	outCluster = outRow->fClusters;  
	dSize+= ( ( AliHLTUInt8_t * )outCluster ) -  (( AliHLTUInt8_t * )outRow);
	if ( outTotalSize + dSize > (int) maxBufferSize ) break;
	outRow->fSlicePatchRowID = id;	
	outRow->fNClusters = 0;
	oldId = id;
	//cout<<"Fill row: s "<<slice<<" p "<<patch<<" r "<<row<<" x "<<outRow->fX<<":"<<endl;
      }
    
      // pack the cluster
      {
	// get coordinates in [um]
	
	Double_t x = (cluster->fX - rowX )*1.e4 + 32768.;
	Double_t y = (cluster->fY)*1.e4 + 8388608.;
	Double_t z = (cluster->fZ)*1.e4 + 8388608.;
	
	// truncate if necessary
	if( x<0 ) x = 0; else if( x > 0x0000FFFF ) x = 0x0000FFFF;
	if( y<0 ) y = 0; else if( y > 0x00FFFFFF ) y = 0x00FFFFFF;
	if( z<0 ) z = 0; else if( z > 0x00FFFFFF ) z = 0x00FFFFFF;
	
	UInt_t ix0 =  ( (UInt_t) x )&0x000000FF;
	UInt_t ix1 = (( (UInt_t) x )&0x0000FF00 )>>8;
	UInt_t iy = ( (UInt_t) y )&0x00FFFFFF;
	UInt_t iz = ( (UInt_t) z )&0x00FFFFFF;
	
	dSize+= sizeof( AliHLTTPCCACompressedCluster );
	if ( outTotalSize + dSize > (int) maxBufferSize ) break;      
	outCluster->fP0 = (ix0<<24) + iy;
	outCluster->fP1 = (ix1<<24) + iz;      
	outCluster++;
	outRow->fNClusters++;
	//cout<<"clu "<<outRow->fNClusters-1<<": "<<cluster->fX<<" "<<cluster->fY<<" "<<cluster->fZ<<" "<<cluster->fID<<endl;
      }
    }
    
    if ( outTotalSize + dSize > (int) maxBufferSize ) {
      HLTWarning( "Output buffer size exceed (buffer size %d, current size %d)", maxBufferSize, outTotalSize+ dSize );
      ret = -ENOSPC;
      break;
    }    
    AliHLTComponentBlockData bd;
    FillBlockData( bd );
    bd.fOffset = outTotalSize;
    bd.fSize = dSize;
    bd.fSpecification = iter->fSpecification;
    bd.fDataType = GetOutputDataType();
    outputBlocks.push_back( bd );
    outTotalSize+=dSize;    
  }

  size = outTotalSize;
  
  timer.Stop();
  
  fTotalTime += timer.RealTime();
  fTotalInputSize+= inTotalSize;
  fTotalOutputSize+= outTotalSize;
  fNEvents++;

  if( maxSlice<0 ) minSlice = -1;
  Int_t hz = ( int ) ( fTotalTime > 1.e-10 ? fNEvents / fTotalTime : 100000 );
  Float_t ratio = 0;
  if( fTotalOutputSize>0 ) ratio = (Float_t ) (fTotalInputSize/fTotalOutputSize);

  // Set log level to "Warning" for on-line system monitoring
  
  HLTInfo( "CAInputDataCompressor, slices %d-%d: speed %d Hz, ratio %f, %d events", minSlice, maxSlice, hz, ratio, fNEvents );

  return ret;
}


