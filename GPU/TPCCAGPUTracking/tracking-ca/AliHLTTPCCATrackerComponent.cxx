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

#include <climits>
#include "AliHLTTPCCATrackerComponent.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAHit.h"
#include "AliHLTTPCCAOutTrack.h"

#include "AliHLTTPCSpacePointData.h"
#include "AliHLTTPCClusterDataFormat.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCTrackSegmentData.h"
#include "AliHLTTPCTrackArray.h"
#include "AliHLTTPCTrackletDataFormat.h"
#include "AliHLTTPCDefinitions.h"
#include "TStopwatch.h"
#include "TMath.h"

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp(AliHLTTPCCATrackerComponent)

AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent()
  :
  fTracker(NULL),
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
  AliHLTProcessor(),
  fTracker(NULL),
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

void AliHLTTPCCATrackerComponent::GetInputDataTypes( vector<AliHLTComponentDataType>& list) 
{
  // see header file for class documentation
  list.clear();
  list.push_back( AliHLTTPCDefinitions::fgkClustersDataType );
}

AliHLTComponentDataType AliHLTTPCCATrackerComponent::GetOutputDataType() 
{
  // see header file for class documentation
  return AliHLTTPCDefinitions::fgkTrackSegmentsDataType;
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

int AliHLTTPCCATrackerComponent::DoInit( int argc, const char** argv )
{
  // Initialize the CA tracker component 
  //
  // arguments could be:
  // bfield - the magnetic field value
  //

  if ( fTracker ) return EINPROGRESS;
  
  fTracker = new AliHLTTPCCATracker();
  
  // read command line

  int i = 0;
  char* cpErr;
  while ( i < argc ){
    if ( !strcmp( argv[i], "bfield" ) ){
      if ( i+1 >= argc )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing B-field", "Missing B-field specifier." );
	  return ENOTSUP;
	}
      fBField = strtod( argv[i+1], &cpErr );
      if ( *cpErr )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing multiplicity", "Cannot convert B-field specifier '%s'.", argv[i+1] );
	  return EINVAL;
	}

      Logging( kHLTLogDebug, "HLT::TPCCATracker::DoInit", "Reading command line",
	       "Magnetic field value is set to %f kG", fBField );

      i += 2;
      continue;
    }
    
    Logging(kHLTLogError, "HLT::TPCCATracker::DoInit", "Unknown Option", "Unknown option '%s'", argv[i] );
    return EINVAL;
  }
  
  return 0;
}

int AliHLTTPCCATrackerComponent::DoDeinit()
{
  // see header file for class documentation
  if ( fTracker ) delete fTracker;
  fTracker = NULL;  
  return 0;
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

  AliHLTUInt32_t MaxBufferSize = size;
  size = 0; // output size

  TStopwatch timer;

  // Event reconstruction in one TPC slice with CA Tracker

  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "DoEvent", "DoEvent()" );
  if ( evtData.fBlockCnt<=0 )
    {
      Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "no blocks in event" );
      return 0;
    }
  
  const AliHLTComponentBlockData* iter = NULL;
  unsigned long ndx;
  AliHLTTPCClusterData* inPtrSP; 
 
  // Determine the slice number 
  
  Int_t slice=-1;
  {
    std::vector<Int_t> slices;
    std::vector<Int_t>::iterator slIter;
    std::vector<unsigned> sliceCnts;
    std::vector<unsigned>::iterator slCntIter;
  
    for ( ndx = 0; ndx < evtData.fBlockCnt; ndx++ ){
      iter = blocks+ndx;
      if ( iter->fDataType != AliHLTTPCDefinitions::fgkClustersDataType ) continue;

      slice = AliHLTTPCDefinitions::GetMinSliceNr( *iter );

      Bool_t found = 0;
      slCntIter = sliceCnts.begin();
      for( slIter = slices.begin(); slIter!=slices.end(); slIter++, slCntIter++ ){
	if ( *slIter == slice ){
	  found = kTRUE;
	  break;
	}
      }
      if ( !found ){
	slices.push_back( slice );
	sliceCnts.push_back( 1 );
      } else *slCntIter++;     
    }
  
    
    // Determine slice number to really use.
    if ( slices.size()>1 )
      {
	Logging( kHLTLogError, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
		 "Multiple slice numbers found in event 0x%08lX (%lu). Determining maximum occuring slice number...",
		 evtData.fEventID, evtData.fEventID );
	unsigned maxCntSlice=0;
	slCntIter = sliceCnts.begin();
	for( slIter = slices.begin(); slIter != slices.end(); slIter++, slCntIter++ )
	  {
	    Logging( kHLTLogError, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
		     "Slice %lu found %lu times.", *slIter, *slCntIter );
	    if ( maxCntSlice<*slCntIter )
	      {
		maxCntSlice = *slCntIter;
		slice = *slIter;
	      }
	  }
	Logging( kHLTLogError, "HLT::TPCSliceTracker::DoEvent", "Multiple slices found in event",
		 "Using slice %lu.", slice );
      }
    else if ( slices.size()>0 )
      {
	slice = *(slices.begin());
      }
  }
  
  if( slice<0 ){
    Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "no slices found in event" );
    return 0;
  }


  // Initialize the tracker

  Double_t Bz = fBField;
  
  {
    if( !fTracker ) fTracker = new AliHLTTPCCATracker;
    Int_t iSec = slice;
    Double_t inRmin = 83.65; 
    //    Double_t inRmax = 133.3;
    //    Double_t outRmin = 133.5; 
    Double_t outRmax = 247.7;
    Double_t plusZmin = 0.0529937; 
    Double_t plusZmax = 249.778; 
    Double_t minusZmin = -249.645; 
    Double_t minusZmax = -0.0799937; 
    Double_t dalpha = 0.349066;
    Double_t alpha = 0.174533 + dalpha*iSec;
    
    Bool_t zPlus = (iSec<18 );
    Double_t zMin =  zPlus ?plusZmin :minusZmin;
    Double_t zMax =  zPlus ?plusZmax :minusZmax;
    //TPCZmin = -249.645, ZMax = 249.778    
    //    Double_t rMin =  inRmin;
    //    Double_t rMax =  outRmax;
    Int_t NRows = AliHLTTPCTransform::GetNRows();
        
    Double_t padPitch = 0.4;
    Double_t sigmaZ = 0.228808;
    
     Double_t rowX[NRows];
    for( Int_t irow=0; irow<NRows; irow++){
      rowX[irow] = AliHLTTPCTransform::Row2X( irow );
    }     
     
    AliHLTTPCCAParam param;
    param.Initialize( iSec, NRows, rowX, alpha, dalpha,
		      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, Bz );
    param.YErrorCorrection() = 1;
    param.ZErrorCorrection() = 2;

    fTracker->Initialize( param ); 

  }

    
  // min and max patch numbers and row numbers

  Int_t row[2] = {0,0};
  Int_t minPatch=INT_MAX, maxPatch = -1;

  // total n Hits

  Int_t nHitsTotal = 0;

  // sort patches

  std::vector<unsigned long> patchIndices;

  for ( ndx = 0; ndx < evtData.fBlockCnt; ndx++ ){
    iter = blocks+ndx;      
    if( iter->fDataType != AliHLTTPCDefinitions::fgkClustersDataType ) continue;
    if( slice!=AliHLTTPCDefinitions::GetMinSliceNr( *iter ) ) continue;
    inPtrSP = (AliHLTTPCClusterData*)(iter->fPtr);
    nHitsTotal+=inPtrSP->fSpacePointCnt;
    Int_t patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
    if ( minPatch>patch ){
      minPatch = patch;
      row[0] = AliHLTTPCTransform::GetFirstRow( patch );
    }
    if ( maxPatch<patch ){
      maxPatch = patch;
      row[1] = AliHLTTPCTransform::GetLastRow( patch );
    }
    std::vector<unsigned long>::iterator pIter = patchIndices.begin(); 
    while( pIter!=patchIndices.end() && AliHLTTPCDefinitions::GetMinPatchNr( blocks[*pIter] ) < patch ){
      pIter++;
    }
    patchIndices.insert( pIter, ndx );
  }
           

  // pass event to CA Tracker
  
  fTracker->StartEvent();

  AliHLTTPCCAHit vHits[nHitsTotal]; // CA hit array
  Double_t vHitStoreX[nHitsTotal];       // hit X coordinates
  Int_t vHitStoreID[nHitsTotal];            // hit ID's
  Int_t vHitRowID[nHitsTotal];            // hit ID's

  Int_t nHits = 0;
 
  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reading hits",
	   "Total %d hits to read for slice %d", nHitsTotal, slice );

  Int_t nClusters=0;

  for( std::vector<unsigned long>::iterator pIter = patchIndices.begin(); pIter!=patchIndices.end(); pIter++ ){
    ndx = *pIter;
    iter = blocks+ndx;
      
    Int_t patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
    inPtrSP = (AliHLTTPCClusterData*)(iter->fPtr);
      
    Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reading hits",
	     "Reading %d hits for slice %d - patch %d", inPtrSP->fSpacePointCnt, slice, patch );
      
    // Read patch hits, row by row

    Int_t oldRow = -1;
    Int_t nRowHits = 0;
    Int_t firstRowHit = 0;
    for (UInt_t i=0; i<inPtrSP->fSpacePointCnt; i++ ){	
      AliHLTTPCSpacePointData* pSP = &(inPtrSP->fSpacePoints[i]);

      if( pSP->fPadRow != oldRow ){
	if( oldRow>=0 ) fTracker->ReadHitRow( oldRow, vHits+firstRowHit, nRowHits );
	oldRow = pSP->fPadRow;
	firstRowHit = nHits;
	nRowHits = 0;
      }
      AliHLTTPCCAHit &h = vHits[nHits];
      if( TMath::Abs(pSP->fX- fTracker->Rows()[pSP->fPadRow].X() )>1.e-4 ) cout<<"row "<<(Int_t)pSP->fPadRow<<" "<<fTracker->Rows()[pSP->fPadRow].X()-pSP->fX <<endl;

      h.Y() = pSP->fY;
      h.Z() = pSP->fZ;
      h.ErrY() = TMath::Sqrt(TMath::Abs(pSP->fSigmaY2));
      h.ErrZ() = TMath::Sqrt(TMath::Abs(pSP->fSigmaZ2));  
      if( h.ErrY()<.1 ) h.ErrY() = .1;
      if( h.ErrZ()<.1 ) h.ErrZ() = .1;
      if( h.ErrY()>1. ) h.ErrY() = 1.;
      if( h.ErrZ()>1. ) h.ErrZ() = 1.;
      h.ID() = nHits;
      vHitStoreX[nHits] = pSP->fX;
      vHitStoreID[nHits] = pSP->fID;
      vHitRowID[nHits] = pSP->fPadRow;
      nHits++;	
      nRowHits++;
      nClusters++;
    }	
    if( oldRow>=0 ) fTracker->ReadHitRow( oldRow, vHits+firstRowHit, nRowHits );
  }

  // reconstruct the event  

  TStopwatch timerReco;

  fTracker->Reconstruct();

  timerReco.Stop();

  Int_t ret = 0;

  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reconstruct",
	   "%d tracks found for slice %d",fTracker->NOutTracks(), slice);

  // write reconstructed tracks

  AliHLTTPCTrackletData* outPtr = (AliHLTTPCTrackletData*)(outputPtr);

  AliHLTTPCTrackSegmentData* currOutTracklet = outPtr->fTracklets;

  Int_t ntracks = fTracker->NOutTracks();

  UInt_t mySize =   ((AliHLTUInt8_t *)currOutTracklet) -  ((AliHLTUInt8_t *)outputPtr);

  outPtr->fTrackletCnt = 0; 

  for( int itr=0; itr<ntracks; itr++ ){
    
    AliHLTTPCCAOutTrack &t = fTracker->OutTracks()[itr];    

    //Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Wrtite output","track %d with %d hits", itr, t.NHits());

    // calculate output track size

    UInt_t dSize = sizeof(AliHLTTPCTrackSegmentData) + t.NHits()*sizeof(UInt_t);
    
    if( mySize + dSize > MaxBufferSize ){
      Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "Wrtite output","Output buffer size exceed (buffer size %d, current size %d), %d tracks are not stored", MaxBufferSize, mySize, ntracks-itr+1);
      ret = -ENOSPC;
      break;
    }
    
    // convert CA track parameters to HLT Track Segment

    Int_t iFirstHit = fTracker->OutTrackHits()[t.FirstHitRef()];
    Int_t iLastHit = fTracker->OutTrackHits()[t.FirstHitRef()+t.NHits()-1];
    
    AliHLTTPCCAHit &firstHit = vHits[iFirstHit];
    AliHLTTPCCAHit &lastHit = vHits[iLastHit];
   
    t.Param().TransportBz(Bz, vHitStoreX[iFirstHit], firstHit.Y(), firstHit.Z() );

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
    
    currOutTracklet->fTgl = TMath::Abs(et)>1.e-5  ? ez/et :1.e5;
      
    if( TMath::Abs(et) >1.e-2 ){
      h3 = -ez*ex/et/et;
      h4 = -ez*ey/et/et;
      h5 = 1.;
      currOutTracklet->fTglerr =  ( h3*h3*t.Param().Cov()[9] + h4*h4*t.Param().Cov()[14] + h5*h5*t.Param().Cov()[20] 
				    + 2.*(h3*h4*t.Param().Cov()[13]+h3*h5*t.Param().Cov()[18]+h4*h5*t.Param().Cov()[19] )
				    )/et/et;
    }else{
      currOutTracklet->fTglerr =  1.e-2;
    }

    currOutTracklet->fCharge = -currOutTracklet->fCharge;
    
    t.Param().TransportBz(Bz, vHitStoreX[iLastHit], lastHit.Y(), lastHit.Z() );
     
    currOutTracklet->fLastX = t.Param().Par()[0];
    currOutTracklet->fLastY = t.Param().Par()[1];
    currOutTracklet->fLastZ = t.Param().Par()[2];

#ifdef INCLUDE_TPC_HOUGH
#ifdef ROWHOUGHPARAMS
    currOutTracklet->fTrackID = 0;
    currOutTracklet->fRowRange1 = vHitRowID[iFirstHit];
    currOutTracklet->fRowRange2 = vHitRowID[iLastHit];
    currOutTracklet->fSector = slice;
    currOutTracklet->fPID = 211;
#endif
#endif // INCLUDE_TPC_HOUGH


    currOutTracklet->fNPoints = t.NHits();

    for( Int_t i=0; i<t.NHits(); i++ ){
      currOutTracklet->fPointIDs[i] = vHitStoreID[fTracker->OutTrackHits()[t.FirstHitRef()+i]];
    }

    currOutTracklet = (AliHLTTPCTrackSegmentData*)( (Byte_t *)currOutTracklet + dSize );
    mySize+=dSize;
    outPtr->fTrackletCnt++; 
  }
  
  
  AliHLTComponentBlockData bd;
  FillBlockData( bd );
  bd.fOffset = 0;
  bd.fSize = mySize;
  bd.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( slice, slice, minPatch, maxPatch );      
  outputBlocks.push_back( bd );
  
  size = mySize;
  
  timerReco.Stop();

  // Set log level to "Warning" for on-line system monitoring

  Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "Tracks",
 	   "CATracker slice %d: output %d tracks;  input %d clusters, patches %d..%d, rows %d..%d; reco time %d/%d us", 
	   slice, ntracks, nClusters, minPatch, maxPatch, row[0], row[1], (Int_t) (timer.RealTime()*1.e6), (Int_t) (timerReco.RealTime()*1.e6) );

  return ret;

}

	
