// @(#) $Id$
//***************************************************************************
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
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackConvertor.h"

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

/** global object for registration 
 * Matthias 2009-01-13 temporarily using the global object approach again.
 * CA cade had to be disabled because of various compilation problems, so
 * the global object approach fits better for the moment.
 */
AliHLTTPCCATrackerComponent gAliHLTTPCCATrackerComponent;

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp(AliHLTTPCCATrackerComponent)

AliHLTTPCCATrackerComponent::AliHLTTPCCATrackerComponent()
  :
  fTracker(NULL),
  fSolenoidBz(0),
  fMinNTrackClusters(0),
  fCellConnectionAngleXY(45),
  fCellConnectionAngleXZ(45),
  fClusterZCut(500.),
  fFullTime(0),
  fRecoTime(0),
  fNEvents(0)
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
  fSolenoidBz(0),
  fMinNTrackClusters(30),
  fCellConnectionAngleXY(35),
  fCellConnectionAngleXZ(35),
  fClusterZCut(500.),
   fFullTime(0),
  fRecoTime(0),
  fNEvents(0)
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

Int_t AliHLTTPCCATrackerComponent::DoInit( Int_t argc, const char** argv )
{
  // Initialize the CA tracker component 
  //
  // arguments could be:
  // solenoidBz - the magnetic field value
  // minNTrackClusters - required minimum of clusters on the track
  //

  if ( fTracker ) return EINPROGRESS;

  fFullTime = 0;
  fRecoTime = 0;
  fNEvents = 0;

  fTracker = new AliHLTTPCCATracker();
  
  // read command line

  Int_t i = 0;
  char* cpErr;
  while ( i < argc ){
    if ( !strcmp( argv[i], "solenoidBz" ) || !strcmp( argv[i], "-solenoidBz" ) ){
      if ( i+1 >= argc )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing solenoidBz", "Missing solenoidBz specifier." );
	  return ENOTSUP;
	}
      fSolenoidBz = strtod( argv[i+1], &cpErr );
      if ( *cpErr )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing multiplicity", "Cannot convert solenoidBz specifier '%s'.", argv[i+1] );
	  return EINVAL;
	}

      Logging( kHLTLogInfo, "HLT::TPCCATracker::DoInit", "Reading command line",
	       "Magnetic field value is set to %f kG", fSolenoidBz );

      i += 2;
      continue;
    }

    if ( !strcmp( argv[i], "minNTrackClusters" ) || !strcmp( argv[i], "-minNTrackClusters" ) ){
      if ( i+1 >= argc )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing minNTrackClusters", "Missing minNTrackClusters specifier." );
	  return ENOTSUP;
	}
      fMinNTrackClusters = (Int_t ) strtod( argv[i+1], &cpErr );
      if ( *cpErr )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing multiplicity", "Cannot convert minNTrackClusters '%s'.", argv[i+1] );
	  return EINVAL;
	}

      Logging( kHLTLogInfo, "HLT::TPCCATracker::DoInit", "Reading command line",
	       "minNTrackClusters is set to %i ", fMinNTrackClusters );

      i += 2;
      continue;
    }

    if ( !strcmp( argv[i], "cellConnectionAngleXY" ) || !strcmp( argv[i], "-cellConnectionAngleXY" ) ){
      if ( i+1 >= argc )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing cellConnectionAngleXY", "Missing cellConnectionAngleXY specifier." );
	  return ENOTSUP;
	}
      fCellConnectionAngleXY = strtod( argv[i+1], &cpErr );
      if ( *cpErr )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing multiplicity", "Cannot convert cellConnectionAngleXY '%s'.", argv[i+1] );
	  return EINVAL;
	}

      Logging( kHLTLogInfo, "HLT::TPCCATracker::DoInit", "Reading command line",
	       "cellConnectionAngleXY is set to %f ", fCellConnectionAngleXY );

      i += 2;
      continue;
    }
     if ( !strcmp( argv[i], "cellConnectionAngleXZ" ) || !strcmp( argv[i], "-cellConnectionAngleXZ" ) ){
      if ( i+1 >= argc )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing cellConnectionAngleXZ", "Missing cellConnectionAngleXZ specifier." );
	  return ENOTSUP;
	}
      fCellConnectionAngleXZ = strtod( argv[i+1], &cpErr );
      if ( *cpErr )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing multiplicity", "Cannot convert cellConnectionAngleXZ '%s'.", argv[i+1] );
	  return EINVAL;
	}

      Logging( kHLTLogInfo, "HLT::TPCCATracker::DoInit", "Reading command line",
	       "cellConnectionAngleXZ is set to %f ", fCellConnectionAngleXZ );

      i += 2;
      continue;
    }
     if ( !strcmp( argv[i], "clusterZCut" ) || !strcmp( argv[i], "-clusterZCut" ) ){
      if ( i+1 >= argc )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing clusterZCut", "Missing clusterZCut specifier." );
	  return ENOTSUP;
	}
      fClusterZCut = TMath::Abs(strtod( argv[i+1], &cpErr ));
      if ( *cpErr )
	{
	  Logging( kHLTLogError, "HLT::TPCCATracker::DoInit", "Missing multiplicity", "Cannot convert clusterZCut '%s'.", argv[i+1] );
	  return EINVAL;
	}

      Logging( kHLTLogInfo, "HLT::TPCCATracker::DoInit", "Reading command line",
	       "clusterZCut is set to %f ", fClusterZCut );

      i += 2;
      continue;
    }
 
    Logging(kHLTLogError, "HLT::TPCCATracker::DoInit", "Unknown Option", "Unknown option '%s'", argv[i] );
    return EINVAL;
  }
  
  return 0;
}

Int_t AliHLTTPCCATrackerComponent::DoDeinit()
{
  // see header file for class documentation
  if ( fTracker ) delete fTracker;
  fTracker = NULL;  
  return 0;
}

Bool_t AliHLTTPCCATrackerComponent::CompareClusters(AliHLTTPCSpacePointData *a, AliHLTTPCSpacePointData *b)
{
  //* Comparison function for sorting clusters
  if( a->fPadRow<b->fPadRow ) return 1;
  if( a->fPadRow>b->fPadRow ) return 0;
  return (a->fZ < b->fZ);
}

Int_t AliHLTTPCCATrackerComponent::DoEvent
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

  if(GetFirstInputBlock( kAliHLTDataTypeSOR ) || GetFirstInputBlock( kAliHLTDataTypeEOR )){    
    return 0;
  }

  TStopwatch timer;

  // Event reconstruction in one TPC slice with CA Tracker
 
  //Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "CA::DoEvent()" );
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
    Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "DoEvent", "CA:: no slices found in event" );
    return 0;
  }


  // Initialize the tracker

  
  {
    if( !fTracker ) fTracker = new AliHLTTPCCATracker;
    Int_t iSec = slice;
    Float_t inRmin = 83.65; 
    //    Float_t inRmax = 133.3;
    //    Float_t outRmin = 133.5; 
    Float_t outRmax = 247.7;
    Float_t plusZmin = 0.0529937; 
    Float_t plusZmax = 249.778; 
    Float_t minusZmin = -249.645; 
    Float_t minusZmax = -0.0799937; 
    Float_t dalpha = 0.349066;
    Float_t alpha = 0.174533 + dalpha*iSec;
    
    Bool_t zPlus = (iSec<18 );
    Float_t zMin =  zPlus ?plusZmin :minusZmin;
    Float_t zMax =  zPlus ?plusZmax :minusZmax;
    //TPCZmin = -249.645, ZMax = 249.778    
    //    Float_t rMin =  inRmin;
    //    Float_t rMax =  outRmax;
    Int_t nRows = AliHLTTPCTransform::GetNRows();
        
    Float_t padPitch = 0.4;
    Float_t sigmaZ = 0.228808;
    
    Float_t *rowX = new Float_t [nRows];
    for( Int_t irow=0; irow<nRows; irow++){
      rowX[irow] = AliHLTTPCTransform::Row2X( irow );
    }     
     
    AliHLTTPCCAParam param;
    param.Initialize( iSec, nRows, rowX, alpha, dalpha,
		      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, fSolenoidBz );
    param.YErrorCorrection() = 1;
    param.ZErrorCorrection() = 2;
    param.CellConnectionAngleXY() = fCellConnectionAngleXY/180.*TMath::Pi();
    param.CellConnectionAngleXZ() = fCellConnectionAngleXZ/180.*TMath::Pi();
    param.Update();
    fTracker->Initialize( param ); 
    delete[] rowX;
  }

    
  // min and max patch numbers and row numbers

  Int_t row[2] = {0,0};
  Int_t minPatch=100, maxPatch = -1;

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

  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reading hits",
	   "Total %d hits to read for slice %d", nHitsTotal, slice );


  AliHLTTPCSpacePointData** vOrigClusters = new AliHLTTPCSpacePointData* [ nHitsTotal];

  Int_t nClusters=0;

  for( std::vector<unsigned long>::iterator pIter = patchIndices.begin(); pIter!=patchIndices.end(); pIter++ ){
    ndx = *pIter;
    iter = blocks+ndx;
      
    Int_t patch = AliHLTTPCDefinitions::GetMinPatchNr( *iter );
    inPtrSP = (AliHLTTPCClusterData*)(iter->fPtr);
      
    Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Reading hits",
	     "Reading %d hits for slice %d - patch %d", inPtrSP->fSpacePointCnt, slice, patch );
      
    for (UInt_t i=0; i<inPtrSP->fSpacePointCnt; i++ ){	
      vOrigClusters[nClusters++] = &(inPtrSP->fSpacePoints[i]);
    }
  }

  // sort clusters since they are not sorted fore some reason

  sort( vOrigClusters, vOrigClusters+nClusters, CompareClusters );

  Float_t *vHitStoreX = new Float_t [nHitsTotal];       // hit X coordinates
  Float_t *vHitStoreY = new Float_t [nHitsTotal];       // hit Y coordinates
  Float_t *vHitStoreZ = new Float_t [nHitsTotal];       // hit Z coordinates
  Int_t *vHitStoreIntID = new Int_t [nHitsTotal];            // hit ID's
  Int_t *vHitStoreID = new Int_t [nHitsTotal];            // hit ID's
  Int_t *vHitRowID = new Int_t [nHitsTotal];            // hit ID's

  Int_t nHits = 0;

  {
    Int_t rowNHits[200];
    Int_t rowFirstHits[200];
    for( Int_t ir=0; ir<200; ir++ ) rowNHits[ir] = 0;
    Int_t oldRow = -1;
    for (Int_t i=0; i<nClusters; i++ ){
      AliHLTTPCSpacePointData* pSP = vOrigClusters[i];
      if( oldRow>=0 && pSP->fPadRow < oldRow )
	HLTError("CA: clusters from row %d are readed twice",oldRow);      
      
      if( TMath::Abs(pSP->fZ)>fClusterZCut) continue;
      
      vHitStoreX[nHits] = pSP->fX;  
      vHitStoreY[nHits] = pSP->fY;
      vHitStoreZ[nHits] = pSP->fZ;
      vHitStoreIntID[nHits] = nHits;
      vHitStoreID[nHits] = pSP->fID;
      vHitRowID[nHits] = pSP->fPadRow;
      nHits++;	
      rowNHits[pSP->fPadRow]++;
    }	

    Int_t firstRowHit = 0;
    for( Int_t ir=0; ir<200; ir++ ){
      rowFirstHits[ir] = firstRowHit;
      firstRowHit+=rowNHits[ir];
    }

    fTracker->ReadEvent( rowFirstHits, rowNHits, vHitStoreY, vHitStoreZ, nHits );
  }

  if( vOrigClusters ) delete[] vOrigClusters;

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

  Int_t ntracks = *fTracker->NOutTracks();

  UInt_t mySize =   ((AliHLTUInt8_t *)currOutTracklet) -  ((AliHLTUInt8_t *)outputPtr);

  outPtr->fTrackletCnt = 0; 

  for( Int_t itr=0; itr<ntracks; itr++ ){
    
    AliHLTTPCCAOutTrack &t = fTracker->OutTracks()[itr];    

    //Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Wrtite output","track %d with %d hits", itr, t.NHits());

    if( t.NHits()<fMinNTrackClusters ) continue;

    // calculate output track size

    UInt_t dSize = sizeof(AliHLTTPCTrackSegmentData) + t.NHits()*sizeof(UInt_t);
    
    if( mySize + dSize > maxBufferSize ){
      Logging( kHLTLogWarning, "HLT::TPCCATracker::DoEvent", "Wrtite output","Output buffer size exceed (buffer size %d, current size %d), %d tracks are not stored", maxBufferSize, mySize, ntracks-itr+1);
      ret = -ENOSPC;
      break;
    }
    
    // convert CA track parameters to HLT Track Segment
 
    Int_t iFirstRow = 1000;
    Int_t iLastRow = -1;
    Int_t iFirstHit = fTracker->OutTrackHits()[t.FirstHitRef()];
    Int_t iLastHit = iFirstHit;
    for( Int_t ih=0; ih<t.NHits(); ih++ ){
      Int_t hitID = fTracker->OutTrackHits()[t.FirstHitRef() + ih ];
      Int_t iRow = vHitRowID[hitID];
      if( iRow<iFirstRow ){  iFirstRow = iRow; iFirstHit = hitID; }
      if( iRow>iLastRow ){ iLastRow = iRow; iLastHit = hitID; }
    }   

    AliHLTTPCCATrackParam par = t.StartPoint();

    par.TransportToX( vHitStoreX[iFirstHit], .99 );

    AliExternalTrackParam tp;
    AliHLTTPCCATrackConvertor::GetExtParam( par, tp, 0, fSolenoidBz );

    currOutTracklet->fX = tp.GetX();
    currOutTracklet->fY = tp.GetY();
    currOutTracklet->fZ = tp.GetZ();
    currOutTracklet->fCharge = (Int_t ) tp.GetSign();
    currOutTracklet->fPt = TMath::Abs(tp.GetSignedPt());
    Float_t snp =  tp.GetSnp() ;
    if( snp>.999 ) snp=.999;
    if( snp<-.999 ) snp=-.999;
    currOutTracklet->fPsi = TMath::ASin( snp );
    currOutTracklet->fTgl = tp.GetTgl();

    currOutTracklet->fY0err = tp.GetSigmaY2();
    currOutTracklet->fZ0err = tp.GetSigmaZ2();
    Float_t h = -currOutTracklet->fPt*currOutTracklet->fPt;
    currOutTracklet->fPterr = h*h*tp.GetSigma1Pt2();
    h = 1./TMath::Sqrt(1-snp*snp);
    currOutTracklet->fPsierr = h*h*tp.GetSigmaSnp2();
    currOutTracklet->fTglerr = tp.GetSigmaTgl2();
    
    if( par.TransportToX( vHitStoreX[iLastHit],.99 ) ){     
      currOutTracklet->fLastX = par.GetX();
      currOutTracklet->fLastY = par.GetY();
      currOutTracklet->fLastZ = par.GetZ();
    } else {
      currOutTracklet->fLastX = vHitStoreX[iLastHit];
      currOutTracklet->fLastY = vHitStoreY[iLastHit];
      currOutTracklet->fLastZ = vHitStoreZ[iLastHit];
    }
    //if( currOutTracklet->fLastX<10. ) {
    //HLTError("CA last point: hitxyz=%f,%f,%f, track=%f,%f,%f, tracklet=%f,%f,%f, nhits=%d",vHitStoreX[iLastHit],vHitStoreY[iLastHit],vHitStoreZ[iLastHit],
    //par.GetX(), par.GetY(),par.GetZ(),currOutTracklet->fLastX,currOutTracklet->fLastY ,currOutTracklet->fLastZ, t.NHits());
    //}
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

  if( vHitStoreX ) delete[] vHitStoreX;
  if( vHitStoreY ) delete[] vHitStoreY;
  if( vHitStoreZ ) delete[] vHitStoreZ;
  if( vHitStoreIntID ) delete[] vHitStoreIntID;
  if( vHitStoreID ) delete[] vHitStoreID;
  if( vHitRowID ) delete[] vHitRowID;
  
  AliHLTComponentBlockData bd;
  FillBlockData( bd );
  bd.fOffset = 0;
  bd.fSize = mySize;
  bd.fSpecification = AliHLTTPCDefinitions::EncodeDataSpecification( slice, slice, minPatch, maxPatch );      
  outputBlocks.push_back( bd );
  
  size = mySize;
  
  timer.Stop();

  fFullTime+= timer.RealTime();
  fRecoTime+= timerReco.RealTime();
  fNEvents++;

  // Set log level to "Warning" for on-line system monitoring
  Int_t hz = (Int_t) (fFullTime>1.e-10 ?fNEvents/fFullTime :100000);
  Int_t hz1 = (Int_t) (fRecoTime>1.e-10 ?fNEvents/fRecoTime :100000);
  Logging( kHLTLogDebug, "HLT::TPCCATracker::DoEvent", "Tracks",
 	   "CATracker slice %d: output %d tracks;  input %d clusters, patches %d..%d, rows %d..%d; reco time %d/%d Hz", 
	   slice, ntracks, nClusters, minPatch, maxPatch, row[0], row[1], hz, hz1 );

  return ret;
}

	
