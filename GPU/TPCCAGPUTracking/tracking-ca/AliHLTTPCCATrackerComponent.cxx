// @(#) $Id$
/**************************************************************************
 * This file is property of and copyright by the ALICE HLT Project        * 
 * ALICE Experiment at CERN, All rights reserved.                         *
 *                                                                        *
 * Primary Authors: Jochen Thaeder <thaeder@kip.uni-heidelberg.de>        *
 *                  Ivan Kisel <kisel@kip.uni-heidelberg.de>              *
 *                  for The ALICE HLT Project.                            *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/
 
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
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAHit.h"
#include "AliHLTTPCCAOutTrack.h"

#include "AliHLTTPCVertex.h"
#include "AliHLTTPCSpacePointData.h"
#include "AliHLTTPCVertexData.h"
#include "AliHLTTPCClusterDataFormat.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrackSegmentData.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCTrackletDataFormat.h"
#include "AliHLTTPCDefinitions.h"
#include "TMath.h"
#include "AliTPC.h"
#include "AliTPCParam.h"
#include "AliRun.h"
#include <stdlib.h>
#include <iostream>
#include <errno.h>


// this is a global object used for automatic component registration, do not use this
AliHLTTPCCATrackerComponent gAliHLTTPCCATrackerComponent;

ClassImp(AliHLTTPCCATrackerComponent)

AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent()
  :
  fTracker(NULL),
  fVertex(NULL),
  fBField(0)
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt
}

AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent(const AliHLTTPCCATrackerComponent&)
  :
  fTracker(NULL),
  fVertex(NULL),
  fBField(0)
{
  // see header file for class documentation
  HLTFatal("copy constructor untested");
}

AliHLTTPCCATrackerComponent& AliHLTTPCCATrackerComponent::operator=(const AliHLTTPCCATrackerComponent&)
{
  // see header file for class documentation
  HLTFatal("assignment operator untested");
  return *this;
}

AliHLTTPCCATrackerComponent::~AliHLTTPCCATrackerComponent()
    {
  // see header file for class documentation
    }

// Public functions to implement AliHLTComponent's interface.
// These functions are required for the registration process

const char* AliHLTTPCCATrackerComponent::GetComponentID() 
    {
  // see header file for class documentation
    return "TPCCATracker";
    }

void AliHLTTPCCATrackerComponent::GetInputDataTypes( vector<AliHLTComponentDataType>& list) 
    {
  // see header file for class documentation
    list.clear();
    list.push_back( AliHLTTPCDefinitions::fgkClustersDataType );
    list.push_back( AliHLTTPCDefinitions::fgkVertexDataType );
    }

AliHLTComponentDataType AliHLTTPCCATrackerComponent::GetOutputDataType() 
    {
  // see header file for class documentation
    return AliHLTTPCDefinitions::fgkTrackSegmentsDataType;
    }

void AliHLTTPCCATrackerComponent::GetOutputDataSize( unsigned long& constBase, double& inputMultiplier ) 
    {
    // see header file for class documentation
    // XXX TODO: Find more realistic values.
    constBase = 0;
    inputMultiplier = 0.2;
    }

AliHLTComponent* AliHLTTPCCATrackerComponent::Spawn() 
    {
  // see header file for class documentation
    return new AliHLTTPCCATrackerComponent;
    }

int AliHLTTPCCATrackerComponent::DoInit( int argc, const char** argv )
    {
  // see header file for class documentation

    if ( fTracker || fVertex )
	return EINPROGRESS;

    fTracker = new AliHLTTPCCATracker();
    fVertex = new AliHLTTPCVertex();


/* ---------------------------------------------------------------------------------
 * cmdline arguments not needed so far

    int i = 0;
    char* cpErr;

    while ( i < argc )
	{
	if ( !strcmp( argv[i], "bfield" ) )
	    {
	    if ( argc <= i+1 )
		{
		Logging( kHLTLogError, "HLT::TPCSliceTracker::DoInit", "Missing B-field", "Missing B-field specifier." );
		return ENOTSUP;
		}
	    fBField = strtod( argv[i+1], &cpErr );
	    if ( *cpErr )
		{
		Logging( kHLTLogError, "HLT::TPCSliceTracker::DoInit", "Missing multiplicity", "Cannot convert B-field specifier '%s'.", argv[i+1] );
		return EINVAL;
		}
	    i += 2;
	    continue;
	    }

	Logging(kHLTLogError, "HLT::TPCSliceTracker::DoInit", "Unknown Option", "Unknown option '%s'", argv[i] );
	return EINVAL;
	}
--------------------------------------------------------------------------------- */

    return 0;
    }

int AliHLTTPCCATrackerComponent::DoDeinit()
    {
  // see header file for class documentation
    if ( fTracker )
	delete fTracker;
    fTracker = NULL;
    if ( fVertex )
	delete fVertex;
    fVertex = NULL;
    return 0;
    }

int AliHLTTPCCATrackerComponent::DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, 
					      AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, 
					      AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks )
{

  // see header file for class documentation

  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "DoEvent", "DoEvent()" );
  if ( evtData.fBlockCnt<=0 )
    {
      Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "no blocks in event" );
      return 0;
    }

  const AliHLTComponentBlockData* iter = NULL;
  unsigned long ndx;
  AliHLTTPCClusterData* inPtrSP;
  AliHLTTPCVertexData* inPtrV = NULL;
  const AliHLTComponentBlockData* vertexIter=NULL;
 
 
  AliHLTUInt32_t vSize = 0;
  UInt_t offset=0, tSize = 0;

  // ------------------------------------------
  
  Int_t slice=-1, patch=-1, row[2];
  Int_t minPatch=INT_MAX, maxPatch = 0;
  offset = 0;
  std::vector<Int_t> slices;
  std::vector<Int_t>::iterator slIter, slEnd;
  std::vector<unsigned> sliceCnts;
  std::vector<unsigned>::iterator slCntIter;
  Int_t vertexSlice=-1;
  
  // Find min/max rows used in total and find and read out vertex if it is present
  // also determine correct slice number, if multiple slice numbers are present in event
  // (which should not happen in the first place) we use the one that occurs the most times
  row[0] = 0;
  row[1] = 0;
  bool found;
  for ( ndx = 0; ndx < evtData.fBlockCnt; ndx++ )
    {
      iter = blocks+ndx;
      slice = AliHLTTPCDefinitions::GetMinSliceNr( *iter );
      found = false;
      slIter = slices.begin();
      slEnd = slices.end();
      slCntIter = sliceCnts.begin();
      while ( slIter != slEnd )
	{
	  if ( *slIter == slice )
	    {
	      found = true;
	      break;
	    }
	  slIter++;
	  slCntIter++;
	}
      if ( !found )
	{
	  slices.insert( slices.end(), slice );
	  sliceCnts.insert( sliceCnts.end(), 1 );
	}
      else
	*slCntIter++;
      
      if ( iter->fDataType == AliHLTTPCDefinitions::fgkVertexDataType )
	{
	  inPtrV = (AliHLTTPCVertexData*)(iter->fPtr);
	  vertexIter = iter;
	  vSize = iter->fSize;
	  fVertex->Read( inPtrV );
	  vertexSlice = slice;
	}
      if ( iter->fDataType == AliHLTTPCDefinitions::fgkClustersDataType )
	{
	  patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
	  if ( minPatch>patch )
	    {
	      minPatch = patch;
	      row[0] = AliHLTTPCTransform::GetFirstRow( patch );
	    }
	  if ( maxPatch<patch )
	    {
	      maxPatch = patch;
	      row[1] = AliHLTTPCTransform::GetLastRow( patch );
	    }
	}
    }
  
  // Determine slice number to really use.
  if ( slices.size()>1 )
    {
      Logging( kHLTLogError, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
	       "Multiple slice numbers found in event 0x%08lX (%lu). Determining maximum occuring slice number...",
	       evtData.fEventID, evtData.fEventID );
      unsigned maxCntSlice=0;
      slIter = slices.begin();
      slEnd = slices.end();
      slCntIter = sliceCnts.begin();
      while ( slIter != slEnd )
	{
	  Logging( kHLTLogError, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
		   "Slice %lu found %lu times.", *slIter, *slCntIter );
	  if ( maxCntSlice<*slCntIter )
	    {
	      maxCntSlice = *slCntIter;
	      slice = *slIter;
	    }
	  slIter++;
	  slCntIter++;
	}
      Logging( kHLTLogError, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
	       "Using slice %lu.", slice );
    }
  else if ( slices.size()>0 )
    {
      slice = *(slices.begin());
    }
  else
    {
      slice = -1;
    }
  
    
  if ( vertexSlice != slice )
    {
      // multiple vertex blocks in event and we used the wrong one...
      found = false;
      for ( ndx = 0; ndx < evtData.fBlockCnt; ndx++ )
	{
	  iter = blocks+ndx;
	  if ( iter->fDataType == AliHLTTPCDefinitions::fgkVertexDataType && slice==AliHLTTPCDefinitions::GetMinSliceNr( *iter ) )
	    {
	      inPtrV = (AliHLTTPCVertexData*)(iter->fPtr);
	      vertexIter = iter;
	      vSize = iter->fSize;
	      fVertex->Read( inPtrV );
	      break;
	    }
	}
    }
    
  // read in all hits
  std::vector<unsigned long> patchIndices;
  std::vector<unsigned long>::iterator pIter, pEnd;
  for ( ndx = 0; ndx < evtData.fBlockCnt; ndx++ )
    {
      iter = blocks+ndx;
      
      if ( iter->fDataType == AliHLTTPCDefinitions::fgkClustersDataType && slice==AliHLTTPCDefinitions::GetMinSliceNr( *iter ) )
	{
	  patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
	  pIter = patchIndices.begin();
	  pEnd = patchIndices.end();
	  while ( pIter!=pEnd && AliHLTTPCDefinitions::GetMinSliceNr( blocks[*pIter] ) < patch )
	    pIter++;
	  patchIndices.insert( pIter, ndx );
	}
    }
       
  // Initialize tracker
  Double_t Bz = -5;

  {
    Int_t iSec = slice;
    Double_t inRmin = 83.65; 
    Double_t inRmax = 133.3;
    Double_t outRmin = 133.5; 
    Double_t outRmax = 247.7;
    Double_t plusZmin = 0.0529937; 
    Double_t plusZmax = 249.778; 
    Double_t minusZmin = -249.645; 
    Double_t minusZmax = -0.0799937; 
    Double_t dalpha = 0.349066;
    Double_t alpha = 0.174533 + dalpha*iSec;
    
    Bool_t zPlus = (iSec<18|| (iSec>=36&&iSec<54) );
    Bool_t rInner = (iSec<36);
    Double_t zMin =  zPlus ?plusZmin :minusZmin;
    Double_t zMax =  zPlus ?plusZmax :minusZmax;
    Double_t rMin =  rInner ?inRmin :outRmin;
    Double_t rMax =  rInner ?inRmax :outRmax;
    Int_t inNRows = 63;
    Int_t outNRows = 96;
    Double_t inRowXFirst = 85.225;
    Double_t outRowXFirst =135.1;
    Double_t inRowXStep = 0.75;
    Double_t outRowXStep = 1.;
    Int_t nRows = rInner ?inNRows :outNRows;
    Double_t rowXFirst = rInner ?inRowXFirst :outRowXFirst;
    Double_t rowXStep = rInner ?inRowXStep :outRowXStep;
    
    Int_t nSectors = 72/2;
    
    Double_t padPitch = 0.4;
    Double_t sigmaZ = 0.228808;
    
    //TPCZmin = -249.645, ZMax = 249.778
    
    if(0){
      if( !gAlice ) return 0;
      AliTPC *tpc = (AliTPC*) gAlice->GetDetector("TPC");      
      AliTPCParam *param = tpc->GetParam();
      cout<<" R inner = "<<param->GetInnerRadiusLow()<<" "<<param->GetInnerRadiusUp()<<endl;
      cout<<" R outer = "<<param->GetOuterRadiusLow()<<" "<<param->GetOuterRadiusUp()<<endl;
      cout<<" Pitch = "<<param->GetPadPitchWidth(0)<<endl;
      cout<<" Sigma Z = "<<param->GetZSigma()<<endl;
      nSectors = param->GetNSector();      
      if( iSec<0 || iSec >= nSectors ) return 0;      
      
      padPitch = param->GetPadPitchWidth(iSec);
      sigmaZ = param->GetZSigma();      
      alpha = param->GetAngle(iSec);      
      
      if( iSec<param->GetNInnerSector() ){
	dalpha = param->GetInnerAngle();
	rMin = param->GetInnerRadiusLow();
	rMax = param->GetInnerRadiusUp();
      } else {  
	dalpha = param->GetOuterAngle();
	rMin = param->GetOuterRadiusLow();
	rMax = param->GetOuterRadiusUp();
      }
      
      TGeoHMatrix  *mat = param->GetClusterMatrix(iSec);
      Double_t p0[3]={0, 0, 0 };
      Double_t p1[3]={0, 0, param->GetZLength(iSec) };
      Double_t p0C[3], p1C[3];
      mat->LocalToMaster(p0,p0C);
      mat->LocalToMaster(p1,p1C);
      Int_t iZ = (iSec%36)/18;
      if( iZ==0 ){
	zMin = p0C[2]; // plus Z
	zMax = p1C[2];
      } else {
	zMin = -p1C[2]; // minus Z
	zMax = p0C[2];      
      }
    }
     
    AliHLTTPCCAParam param;
    param.Initialize( iSec, inNRows+outNRows, inRowXFirst, inRowXStep,alpha, dalpha,
		      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, Bz );
      
    fTracker->Initialize( param );
 
    for( Int_t irow=0; irow<outNRows; irow++){
      fTracker->Rows()[inNRows+irow].X() = outRowXFirst + irow*outRowXStep;
    }
     
  }

  // pass event to CA Tracker
  
  fTracker->StartEvent();

  Int_t nHitsTotal = 0;
  pIter = patchIndices.begin();
  pEnd = patchIndices.end();
  while ( pIter!=pEnd ){
    ndx = *pIter;
    iter = blocks+ndx;	
    inPtrSP = (AliHLTTPCClusterData*)(iter->fPtr);
    nHitsTotal+=inPtrSP->fSpacePointCnt;
    pIter++;
  }
 
  AliHLTTPCCAHit *vHits = new AliHLTTPCCAHit[nHitsTotal];
  Double_t *vHitStore = new Double_t [nHitsTotal];
  Int_t nHits = 0;
    
  pIter = patchIndices.begin();
  while ( pIter!=pEnd )
    {
      ndx = *pIter;
      iter = blocks+ndx;
      
      patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
      inPtrSP = (AliHLTTPCClusterData*)(iter->fPtr);
      
      Logging( kHLTLogDebug, "HLT::TPCSliceTracker::DoEvent", "Reading hits",
	       "Reading hits for slice %d - patch %d", slice, patch );
      
      // Read patch hits

      Int_t oldRow = -1;
      Int_t nRowHits = 0;
      Int_t firstRowHit = 0;
      for (UInt_t i=0; i<inPtrSP->fSpacePointCnt; i++ )
	{	
	  AliHLTTPCSpacePointData* pSP = &(inPtrSP->fSpacePoints[i]);
	  //Logging( kHLTLogDebug, "HLT::TPCSliceTracker::DoEvent", "Reading hits", "hit pad %d, xyz=%f,%f,%f, sy=%f, sz=%f,", pSP->fPadRow, pSP->fX, pSP->fY, pSP->fZ, TMath::Sqrt(pSP->fSigmaY2), TMath::Sqrt(pSP->fSigmaZ2) );
	  if( pSP->fPadRow != oldRow ){
	    if( oldRow>=0 ) fTracker->ReadHitRow( oldRow, vHits+firstRowHit, nRowHits );
	    oldRow = pSP->fPadRow;
	    firstRowHit = nHits;
	    nRowHits = 0;
	  }
	  AliHLTTPCCAHit &h = vHits[nHits];
	  h.Y() = pSP->fY;
	  h.Z() = pSP->fZ;
	  h.ErrY() = TMath::Sqrt(pSP->fSigmaY2);
	  h.ErrZ() = TMath::Sqrt(pSP->fSigmaZ2);  
	  h.ID() = pSP->fID;
	  vHitStore[nHits] = pSP->fX;
	  nHits++;	
	  nRowHits++;
	}	
      if( oldRow>=0 ) fTracker->ReadHitRow( oldRow, vHits+firstRowHit, nRowHits );
      pIter++;
    }
  
  // reconstruct the event  

  fTracker->Reconstruct();


  // write reconstructed tracks

  AliHLTTPCTrackletData* outPtr = (AliHLTTPCTrackletData*)(outputPtr);

  AliHLTTPCTrackSegmentData* currOutTracklet = outPtr->fTracklets;

  Int_t ntracks = fTracker->NOutTracks();
  
  for( int itr=0; itr<ntracks; itr++ ){
    
    AliHLTTPCCAOutTrack &t = fTracker->OutTracks()[itr];    
    Int_t iFirstHit = fTracker->OutTrackHits()[t.FirstHitRef()];
    Int_t iLastHit = fTracker->OutTrackHits()[t.FirstHitRef()+t.NHits()-1];
    AliHLTTPCCAHit &firstHit = vHits[iFirstHit];
    AliHLTTPCCAHit &lastHit = vHits[iLastHit];
    
    t.Param().TransportBz(Bz, vHitStore[iFirstHit], firstHit.Y(), firstHit.Z() );
    currOutTracklet->fX = t.Param().Par()[0];
    currOutTracklet->fY = t.Param().Par()[1];
    currOutTracklet->fZ = t.Param().Par()[2];
    Double_t qp = t.Param().Par()[6];
    Double_t p = TMath::Abs(qp)>1.e-5 ?1./TMath::Abs(qp) :1.e5;
    Double_t ex = t.Param().Par()[3];
    Double_t ey = t.Param().Par()[4];
    Double_t ez = t.Param().Par()[5];
    Double_t et = TMath::Sqrt( ex*ex + ey*ey );
    currOutTracklet->fCharge = (qp>0) ?+1 :(qp<0 ?-1 :0);
    currOutTracklet->fPt = p*et;
    
    Double_t h3 =  TMath::Abs(ex) >1.e-5 ? p*ex/et :0;
    Double_t h4 =  TMath::Abs(ey) >1.e-5 ? p*ey/et :0;
    Double_t h5;
    Double_t h6 =  - currOutTracklet->fCharge * p * currOutTracklet->fPt;
    
    currOutTracklet->fPterr = ( h3*h3*t.Param().Cov()[9] + h4*h4*t.Param().Cov()[14] + h6*h6*t.Param().Cov()[27] 
				+ 2.*(h3*h4*t.Param().Cov()[13]+h3*h6*t.Param().Cov()[24]+h4*h6*t.Param().Cov()[25] )
				);
    
    currOutTracklet->fPsi = TMath::ATan2(ey, ex);
    
    h3 =  ex/(et*et);
    h4 = -ey/(et*et);
    currOutTracklet->fPsierr = h3*h3*t.Param().Cov()[9] + h4*h4*t.Param().Cov()[14] + 2.*h3*h4*t.Param().Cov()[13];
    
    currOutTracklet->fTgl = TMath::Abs(ex)>1.e-5  ? ez/ex :1.e5;
      
    h3 = (TMath::Abs(ex) >1.e-5) ? -ez/ex/ex :0;
    h5 = (TMath::Abs(ex) >1.e-5) ? 1./ex :0;
    currOutTracklet->fTglerr =  h3*h3*t.Param().Cov()[9] + h5*h5*t.Param().Cov()[20] + 2.*h3*h5*t.Param().Cov()[18]; 
    
    currOutTracklet->fCharge = -currOutTracklet->fCharge;
    t.Param().TransportBz(Bz, vHitStore[iLastHit], lastHit.Y(), lastHit.Z() );
    currOutTracklet->fLastX = t.Param().Par()[0];
    currOutTracklet->fLastY = t.Param().Par()[1];
    currOutTracklet->fLastZ = t.Param().Par()[2];

    currOutTracklet->fNPoints = t.NHits();

    for( Int_t i=0; i<t.NHits(); i++ ){
      currOutTracklet->fPointIDs[i] = fTracker->OutTrackHits()[t.FirstHitRef()+i];
    }
    
    Byte_t *tmpP = (Byte_t *)currOutTracklet;
    
    tmpP += sizeof(AliHLTTPCTrackSegmentData) + currOutTracklet->fNPoints*sizeof(UInt_t);
    currOutTracklet = (AliHLTTPCTrackSegmentData*)tmpP;
  }
  
  outPtr->fTrackletCnt = ntracks; 
  
  delete[] vHits;
  delete[] vHitStore;
  
  Logging( kHLTLogDebug, "HLT::TPCSliceTracker::DoEvent", "Tracks",
	   "Input: Number of tracks: %lu Slice/MinPatch/MaxPatch/RowMin/RowMax: %lu/%lu/%lu/%lu/%lu.", 
	   ntracks, slice, minPatch, maxPatch, row[0], row[1] );
  
  AliHLTUInt8_t *pbeg = (AliHLTUInt8_t *)outputPtr;
  AliHLTUInt8_t *pend = (AliHLTUInt8_t *)currOutTracklet;
  UInt_t mySize = pend - pbeg;
  
  AliHLTComponentBlockData bd;
  FillBlockData( bd );
  bd.fOffset = offset;
  bd.fSize = mySize;
  bd.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( slice, slice, minPatch, maxPatch );      
  outputBlocks.push_back( bd );
  
#ifdef FORWARD_VERTEX_BLOCK
  if ( vertexIter )
    {
      // Copy the descriptor block for the vertex information.
      //bd = *vertexIter;
      //outputBlocks.push_back( bd );
    }
#endif // FORWARD_VERTEX_BLOCK
  
  size = mySize;
  
  return 0;
}

	
