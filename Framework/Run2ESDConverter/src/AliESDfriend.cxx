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
//               Implementation of the AliESDfriend class
//  This class contains some additional to the ESD information like
//  the clusters associated to tracks.
//      Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch
//-------------------------------------------------------------------------

#include "AliESDfriend.h"
#include "AliESDVZEROfriend.h"
#include "AliESDTZEROfriend.h"
#include "AliESDADfriend.h"
#include "AliESDCalofriend.h"

ClassImp(AliESDfriend)

AliESDfriend::AliESDfriend(): AliVfriendEvent(), 
  fESDIndicesStored(kFALSE),
  fTracks("AliESDfriendTrack",1),
  fESDVZEROfriend(NULL),
  fESDTZEROfriend(NULL),
  fESDADfriend(NULL),
  fESDCalofriend(NULL)
{
 //
 // Default constructor
 //
  fTracks.SetOwner(kTRUE);
  memset(fNclustersTPC,0,sizeof(fNclustersTPC));
  memset(fNclustersTPCused,0,sizeof(fNclustersTPCused));
}

AliESDfriend::AliESDfriend(const AliESDfriend &f) :
  AliVfriendEvent(f),
  fESDIndicesStored(f.fESDIndicesStored),
  fTracks(f.fTracks),
  fESDVZEROfriend(f.fESDVZEROfriend ? new AliESDVZEROfriend(*f.fESDVZEROfriend) : NULL),
  fESDTZEROfriend(f.fESDTZEROfriend ? new AliESDTZEROfriend(*f.fESDTZEROfriend) : NULL),
  fESDADfriend(f.fESDADfriend ? new AliESDADfriend(*f.fESDADfriend) : NULL),
  fESDCalofriend(f.fESDCalofriend ? new AliESDCalofriend(*f.fESDCalofriend) : NULL)
{
 //
 // Copy constructor
 //
 memcpy(fNclustersTPC,f.fNclustersTPC,sizeof(fNclustersTPC));
 memcpy(fNclustersTPCused,f.fNclustersTPCused,sizeof(fNclustersTPCused));

}

AliESDfriend& AliESDfriend::operator=(const AliESDfriend& esd)
{
    
    // Assignment operator
    if(&esd == this) return *this;
    TObject::operator=(esd);
    fESDIndicesStored = esd.fESDIndicesStored;
    // Clean up the old TClonesArray
    DeleteTracksSafe();
    //    fTracks.Delete();
    // Assign the new one
    fTracks = esd.fTracks;

    if(fESDVZEROfriend)
      delete fESDVZEROfriend;
    fESDVZEROfriend=0;
    if(esd.fESDVZEROfriend)
      fESDVZEROfriend = new AliESDVZEROfriend(*esd.fESDVZEROfriend);
    
    if(fESDTZEROfriend)
      delete fESDTZEROfriend;
    fESDTZEROfriend=0;
    if(esd.fESDTZEROfriend)
      fESDTZEROfriend = new AliESDTZEROfriend(*esd.fESDTZEROfriend);

    if(fESDADfriend)
      delete fESDADfriend;
    fESDADfriend=0;
    if(esd.fESDADfriend)
      fESDADfriend = new AliESDADfriend(*esd.fESDADfriend);

    if(fESDCalofriend)
      delete fESDCalofriend;
    fESDCalofriend=0;
    if(esd.fESDCalofriend)
      fESDCalofriend = new AliESDCalofriend(*esd.fESDCalofriend);
 
    memcpy(fNclustersTPC,esd.fNclustersTPC,sizeof(fNclustersTPC));
    memcpy(fNclustersTPCused,esd.fNclustersTPCused,sizeof(fNclustersTPCused));
 
 
    return *this;
}



AliESDfriend::~AliESDfriend() {
  //
  // Destructor
  //
  DeleteTracksSafe();
  //fTracks.Delete();
  if(fESDVZEROfriend)
    delete fESDVZEROfriend;
  fESDVZEROfriend=0;
  if(fESDTZEROfriend)
    delete fESDTZEROfriend;
  fESDTZEROfriend=0;
  if(fESDADfriend)
    delete fESDADfriend;
  fESDADfriend=0;
  if(fESDCalofriend)
    delete fESDCalofriend;
  fESDCalofriend=0;
}

void AliESDfriend::DeleteTracksSafe()
{
  // delete tracks taking care of eventual shared objects in the tracks (e.g. TPCclusters)
  int ntr=fTracks.GetEntriesFast();
  for (int i=0;i<ntr;i++) {
    AliESDfriendTrack* trc = (AliESDfriendTrack*)fTracks[i];
    trc->TagSuppressSharedObjectsBeforeDeletion();
  }
  fTracks.Delete();
}

void AliESDfriend::ResetSoft()
{
  // Reset friend information, used for the shalow copy
  for (int i=fTracks.GetEntriesFast();i--;) fTracks[i]->Clear();
  fTracks.Clear();    
  for (Int_t i=0;i<72;i++)
  {
    fNclustersTPC[i]=0;
    fNclustersTPCused[i]=0;
  }
  delete fESDVZEROfriend; fESDVZEROfriend=0;
  delete fESDTZEROfriend; fESDTZEROfriend=0;
  delete fESDADfriend; fESDADfriend=0;
  if (fESDCalofriend) {
    fESDCalofriend->DeAllocate(); 
    delete fESDCalofriend; fESDCalofriend=0;
  }
}


void AliESDfriend::Reset()
{
  //
  // Reset friend information
  //
  DeleteTracksSafe();
  //  fTracks.Delete();
  for (Int_t i=0;i<72;i++)
  {
    fNclustersTPC[i]=0;
    fNclustersTPCused[i]=0;
  }
  delete fESDVZEROfriend; fESDVZEROfriend=0;
  delete fESDTZEROfriend; fESDTZEROfriend=0;
  delete fESDADfriend; fESDADfriend=0;
  if (fESDCalofriend) {
    fESDCalofriend->DeAllocate(); 
    delete fESDCalofriend; fESDCalofriend=0;
  }
}  

void AliESDfriend::SetVZEROfriend(const AliESDVZEROfriend * obj)
{
  //
  // Set the VZERO friend data object
  // (complete raw data)
  if (!fESDVZEROfriend) fESDVZEROfriend = new AliESDVZEROfriend();
  if (obj) *fESDVZEROfriend = *obj;
}
void AliESDfriend::SetCalofriend(AliESDCalofriend * obj)
{
  //
  // Set the Calo friend data object
  // (complete raw data)
  if (!fESDCalofriend) fESDCalofriend = new AliESDCalofriend();
  if (obj) *fESDCalofriend = *obj;
}
void AliESDfriend::SetTZEROfriend(AliESDTZEROfriend * obj)
{
  //
  // Set the TZERO friend data object
  // (complete raw data)
  if (!fESDTZEROfriend) fESDTZEROfriend = new AliESDTZEROfriend();
  if (obj) *fESDTZEROfriend = *obj;
}
void AliESDfriend::SetADfriend(AliESDADfriend * obj)
{
  //
  // Set the AD friend data object
  // (complete raw data)
  if (!fESDADfriend) fESDADfriend = new AliESDADfriend();
  if (obj) *fESDADfriend = *obj;
}
