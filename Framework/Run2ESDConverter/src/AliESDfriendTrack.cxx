/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

//-------------------------------------------------------------------------
//               Implementation of the AliESDfriendTrack class
//  This class keeps complementary to the AliESDtrack information 
//      Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch
//-------------------------------------------------------------------------
#include "AliTrackPointArray.h"
#include "AliESDfriendTrack.h"
#include "TObjArray.h"
#include "TClonesArray.h"
#include "AliKalmanTrack.h"
#include "AliVTPCseed.h"
#include "AliLog.h"

ClassImp(AliESDfriendTrack)

AliESDfriendTrack::AliESDfriendTrack(): 
AliVfriendTrack(), 
f1P(0), 
fnMaxITScluster(0),
fnMaxTPCcluster(0),
fnMaxTRDcluster(0),
fITSindex(0x0),
fTPCindex(0x0),
fTRDindex(0x0),
fPoints(0),
fCalibContainer(0),
fITStrack(0),
fTRDtrack(0),
fTPCOut(0),
fITSOut(0),
fTRDIn(0)
{
  //
  // Default constructor
  //
	//  Int_t i;
  //  fITSindex = new Int_t[fnMaxITScluster];
  //fTPCindex = new Int_t[fnMaxTPCcluster];
  //fTRDindex = new Int_t[fnMaxTRDcluster];
  //for (i=0; i<kMaxITScluster; i++) fITSindex[i]=-2;
  //for (i=0; i<kMaxTPCcluster; i++) fTPCindex[i]=-2;
  //for (i=0; i<kMaxTRDcluster; i++) fTRDindex[i]=-2;
  
  //fHmpPhotClus->SetOwner(kTRUE); 
  
}

AliESDfriendTrack::AliESDfriendTrack(const AliESDfriendTrack &t, Bool_t shallow): 
AliVfriendTrack(t),
f1P(t.f1P),
fnMaxITScluster(t.fnMaxITScluster),
fnMaxTPCcluster(t.fnMaxTPCcluster),
fnMaxTRDcluster(t.fnMaxTRDcluster),
fITSindex(0x0),
fTPCindex(0x0),
fTRDindex(0x0),
fPoints(0),
fCalibContainer(0),
fITStrack(0),
fTRDtrack(0),
fTPCOut(0),
fITSOut(0),
fTRDIn(0)
{
  //
  // Copy constructor
  //
  AliDebug(2,"Calling copy constructor");

  if (shallow) { // shallow copy for transfer to TClonesArray in the friends
    fITSindex = t.fITSindex;
    fTPCindex = t.fTPCindex;
    fTRDindex = t.fTRDindex; 
    fPoints   = t.fPoints;
    fCalibContainer = t.fCalibContainer;
    fTPCOut = t.fTPCOut;
    fITSOut = t.fITSOut;
    fTRDIn = t.fTRDIn;
    //
    return;
  }
  //
  Int_t i;
  if (fnMaxITScluster != 0){
	  fITSindex = new Int_t[fnMaxITScluster];
	  for (i=0; i<fnMaxITScluster; i++) fITSindex[i]=t.fITSindex[i];
  }
  if (fnMaxTPCcluster != 0){
	  fTPCindex = new Int_t[fnMaxTPCcluster];
	  for (i=0; i<fnMaxTPCcluster; i++) fTPCindex[i]=t.fTPCindex[i];
  }
  if (fnMaxTRDcluster != 0){
	  fTRDindex = new Int_t[fnMaxTRDcluster];
	  for (i=0; i<fnMaxTRDcluster; i++) fTRDindex[i]=t.fTRDindex[i]; 
  }
  AliDebug(2,Form("fnMaxITScluster = %d",fnMaxITScluster));
  AliDebug(2,Form("fnMaxTPCcluster = %d",fnMaxTPCcluster));
  AliDebug(2,Form("fnMaxTRDcluster = %d",fnMaxTRDcluster));
  if (t.fPoints) fPoints=new AliTrackPointArray(*t.fPoints);
  if (t.fCalibContainer) {
     fCalibContainer = new TObjArray(2);
     fCalibContainer->SetOwner();
     Int_t no=t.fCalibContainer->GetEntriesFast();
     for (i=0; i<no; i++) {
       TObject *o=t.fCalibContainer->At(i);
       if (o) fCalibContainer->AddLast(o->Clone());
     }
  }

  if (t.fTPCOut) fTPCOut = new AliExternalTrackParam(*(t.fTPCOut));
  if (t.fITSOut) fITSOut = new AliExternalTrackParam(*(t.fITSOut));
  if (t.fTRDIn)  fTRDIn = new AliExternalTrackParam(*(t.fTRDIn));
  
}

AliESDfriendTrack::~AliESDfriendTrack() {
  //
  // Simple destructor
  //
  if(fPoints)
    delete fPoints;
  fPoints=0;
  if (fCalibContainer){
    fCalibContainer->Delete();
    delete fCalibContainer;
    fCalibContainer=0;
  }
  if(fITStrack)
    delete fITStrack;
  fITStrack=0;
  if(fTRDtrack)
    delete fTRDtrack;
  fTRDtrack=0;
  if(fTPCOut)
    delete fTPCOut;
  fTPCOut=0;
  if(fITSOut)
    delete fITSOut;
  fITSOut=0;
  if(fTRDIn)
    delete fTRDIn;
  fTRDIn=0;
  if(fITSindex)
    delete[] fITSindex;
  fITSindex=0;
  if(fTPCindex)
    delete[] fTPCindex;
  fTPCindex=0;
  if(fTRDindex)
    delete[] fTRDindex;
  fTRDindex=0;
}

void AliESDfriendTrack::Clear(Option_t*)
{
  // clear pointers data, used for shallow copies
  fPoints = 0;
  fCalibContainer=0;
  fITStrack=0;
  fTRDtrack=0;
  fTPCOut=0;
  fITSOut=0;
  fTRDIn=0;
  fITSindex=0;
  fTPCindex=0;
  fTRDindex=0;
  //  
}

void AliESDfriendTrack::AddCalibObject(TObject * calibObject){
  //
  // add calibration object to array -
  // track is owner of the objects in the container 
  //
  if (!fCalibContainer) {
    fCalibContainer = new TObjArray(2);
    fCalibContainer->SetOwner();
  }
  fCalibContainer->AddLast(calibObject);
}

void AliESDfriendTrack::RemoveCalibObject(TObject * calibObject){
  //
  // remove calibration object from the array -
  //
  if (fCalibContainer) fCalibContainer->Remove(calibObject);
}

TObject * AliESDfriendTrack::GetCalibObject(Int_t index) const {
  //
  //
  //
  if (!fCalibContainer) return 0;
  if (index>=fCalibContainer->GetEntriesFast()) return 0;
  return fCalibContainer->At(index);
}

Int_t AliESDfriendTrack::GetTPCseed( AliTPCseed &seed) const {
  TObject* calibObject = NULL;
  AliVTPCseed* seedP = NULL;
  for (Int_t idx = 0; (calibObject = GetCalibObject(idx)); ++idx) {
    if ((seedP = dynamic_cast<AliVTPCseed*>(calibObject))) {
      seedP->CopyToTPCseed( seed );
      return 0;
    }
  }
  return -1;
}

const TObject* AliESDfriendTrack::GetTPCseed() const 
{
  TObject* calibObject = NULL;
  AliVTPCseed* seedP = NULL;
  for (Int_t idx = 0; (calibObject = GetCalibObject(idx)); ++idx) {
    if ((seedP = dynamic_cast<AliVTPCseed*>(calibObject))) return calibObject;
  }
  return 0;
}


void AliESDfriendTrack::ResetTPCseed( const AliTPCseed* seed )
{
  TObject* calibObject = NULL;
  AliVTPCseed* seedP = NULL;
  for (Int_t idx = 0; (calibObject = GetCalibObject(idx)); ++idx) {
    if ((seedP = dynamic_cast<AliVTPCseed*>(calibObject))) break;
  }
  if (seedP) seedP->SetFromTPCseed(seed);
}

void AliESDfriendTrack::SetTPCOut(const AliExternalTrackParam &param) {
  // 
  // backup TPC out track
  //
  if(fTPCOut)
    delete fTPCOut;
  fTPCOut=new AliExternalTrackParam(param);
} 
void AliESDfriendTrack::SetITSOut(const AliExternalTrackParam &param) {
  //
  // backup ITS out track
  //
  if(fITSOut)
    delete fITSOut;
  fITSOut=new AliExternalTrackParam(param);
} 
void AliESDfriendTrack::SetTRDIn(const AliExternalTrackParam  &param)  {
  //
  // backup TRD in track
  //
  if(fTRDIn)
    delete fTRDIn;
  fTRDIn=new AliExternalTrackParam(param);
} 

void AliESDfriendTrack::SetITSIndices(Int_t* indices, Int_t n){

	//
	// setting fITSindex
	// instantiating the pointer if still NULL
	//
        // TODO: what if the array was already set but
        // the old fnMaxITScluster and n differ!?
        if(fnMaxITScluster && fnMaxITScluster!=n){
	        AliError(Form("Array size does not fit %d/%d\n"
			      ,fnMaxITScluster,n));
	}
	
	fnMaxITScluster = n;
	AliDebug(2,Form("fnMaxITScluster = %d",fnMaxITScluster));
	if (fITSindex == 0x0){
		fITSindex = new Int_t[fnMaxITScluster];
	}
	for (Int_t i = 0; i < fnMaxITScluster; i++){
		fITSindex[i] = indices[i];
	}
}

void AliESDfriendTrack::SetTPCIndices(Int_t* indices, Int_t n){

	//
	// setting fTPCindex
	// instantiating the pointer if still NULL
	//
        // TODO: what if the array was already set but
        // the old fnMaxITScluster and n differ!?
        if(fnMaxTPCcluster && fnMaxTPCcluster!=n){
                AliError(Form("Array size does not fit %d/%d\n"
			      ,fnMaxTPCcluster,n));
	}
       
	fnMaxTPCcluster = n;
	AliDebug(2,Form("fnMaxTPCcluster = %d",fnMaxTPCcluster));
	if (fTPCindex == 0x0){
		fTPCindex = new Int_t[fnMaxTPCcluster];
	}
 	memcpy(fTPCindex,indices,sizeof(Int_t)*fnMaxTPCcluster); //RS
	//for (Int_t i = 0; i < fnMaxTPCcluster; i++){fTPCindex[i] = indices[i];}
}

void AliESDfriendTrack::SetTRDIndices(Int_t* indices, Int_t n){

	//
	// setting fTRDindex
	// instantiating the pointer if still NULL
	//
        // TODO: what if the array was already set but
        // the old fnMaxITScluster and n differ!?
        if(fnMaxTRDcluster && fnMaxTRDcluster!=n){
                AliError(Form("Array size does not fit %d/%d\n"
			      ,fnMaxTRDcluster,n));
	}
	
	fnMaxTRDcluster = n;
	AliDebug(2,Form("fnMaxTRDcluster = %d",fnMaxTRDcluster));
	if (fTRDindex == 0x0){
		fTRDindex = new Int_t[fnMaxTRDcluster];
	}
	memcpy(fTRDindex,indices,sizeof(Int_t)*fnMaxTRDcluster); //RS
	//for (Int_t i = 0; i < fnMaxTRDcluster; i++){fTRDindex[i] = indices[i];}
}

void AliESDfriendTrack::TagSuppressSharedObjectsBeforeDeletion()
{
  // before deletion of the track we need to suppress eventual shared objects (e.g. TPCclusters)
  //
  // at the moment take care of TPCseeds only
  
  TObject* calibObject = NULL;
  AliVTPCseed* seedP = 0;
  for (Int_t idx = 0; (calibObject = GetCalibObject(idx)); ++idx) {
    if ((seedP = dynamic_cast<AliVTPCseed*>(calibObject))) {
      seedP->TagSuppressSharedClusters();
      break;
    }
  }  
}
