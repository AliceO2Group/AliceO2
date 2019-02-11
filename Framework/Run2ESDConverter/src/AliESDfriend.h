#ifndef ALIESDFRIEND_H
#define ALIESDFRIEND_H

//-------------------------------------------------------------------------
//                     Class AliESDfriend
//               This class contains ESD additions
//       Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch 
//-------------------------------------------------------------------------

#include <TObject.h>
#include <TClonesArray.h>

#include "AliESDfriendTrack.h"
#include "AliVfriendEvent.h"

#include "AliESDVZEROfriend.h"

class AliESDTZEROfriend;
class AliESDADfriend;
class AliESDCalofriend;

//_____________________________________________________________________________
class AliESDfriend : public AliVfriendEvent {
public:
  AliESDfriend();
  AliESDfriend(const AliESDfriend &);
  AliESDfriend& operator=(const AliESDfriend& esd);  
  virtual ~AliESDfriend();
  
  // This function will set the ownership
  // needed to read old ESDfriends
  void SetOwner(){
    fTracks.SetOwner();
    Int_t n=fTracks.GetEntriesFast();
    for(;n--;){
      AliESDfriendTrack *t=(AliESDfriendTrack *)fTracks.UncheckedAt(n);
      if(t)t->SetOwner();
    }
  }
  
  Int_t GetNumberOfTracks() const {return fTracks.GetEntriesFast();}
  AliESDfriendTrack *GetTrack(Int_t i) const {
     return (AliESDfriendTrack *)fTracks.At(i);
  }
  Int_t GetEntriesInTracks() const {return fTracks.GetEntries();}

  AliESDfriendTrack* AddTrack(const AliESDfriendTrack *t, Bool_t shallow=kFALSE) {
    return new(fTracks[fTracks.GetEntriesFast()]) AliESDfriendTrack(*t,shallow);
  }

  AliESDfriendTrack* AddTrackAt(const AliESDfriendTrack *t, Int_t i, Bool_t shallow=kFALSE) {
    return new(fTracks[i]) AliESDfriendTrack(*t,shallow);
  }

  void SetVZEROfriend(const AliESDVZEROfriend * obj);
  AliESDVZEROfriend *GetVZEROfriend(){ return fESDVZEROfriend; }
  const AliESDVZEROfriend *GetVZEROfriendConst() const { return fESDVZEROfriend; }
  AliVVZEROfriend *GetVVZEROfriend(){ return fESDVZEROfriend; }
  Int_t GetESDVZEROfriend( AliESDVZEROfriend &v ) const {
    if( fESDVZEROfriend ){ v=*fESDVZEROfriend; return 0; }
    return -1;
  }

  void SetTZEROfriend(AliESDTZEROfriend * obj);
  AliESDTZEROfriend *GetTZEROfriend(){ return fESDTZEROfriend; }
  void SetADfriend(AliESDADfriend * obj);
  AliESDADfriend *GetADfriend(){ return fESDADfriend; }
  void SetCalofriend(AliESDCalofriend * obj);
  AliESDCalofriend *GetCalofriend(){ return fESDCalofriend; }

  void Ls() const {
	  return fTracks.ls();
  }
  void Reset();
  void ResetSoft();
  // bit manipulation for filtering
  void SetSkipBit(Bool_t skip){SetBit(23,skip);}
  Bool_t TestSkipBit() const { return TestBit(23); }

  //TPC cluster occupancy
  Int_t GetNclustersTPC(UInt_t sector) const {return (sector<72)?fNclustersTPC[sector]:0;}
  Int_t GetNclustersTPCused(UInt_t sector) const {return (sector<72)?fNclustersTPCused[sector]:0;}
  void SetNclustersTPC(UInt_t sector, Int_t occupancy) {if (sector<72) fNclustersTPC[sector]=occupancy;}
  void SetNclustersTPCused(UInt_t sector, Int_t occupancy) {if (sector<72) fNclustersTPCused[sector]=occupancy;}
  //
  Bool_t GetESDIndicesStored() const {return fESDIndicesStored;}
  void   SetESDIndicesStored(Bool_t v) {fESDIndicesStored = v;}

 protected:
  void DeleteTracksSafe();

protected:
  Bool_t            fESDIndicesStored; // Flag new format of sparse friends
  TClonesArray      fTracks;    // ESD friend tracks
  AliESDVZEROfriend *fESDVZEROfriend; // VZERO object containing complete raw data
  AliESDTZEROfriend *fESDTZEROfriend; // TZERO calibration object
  AliESDADfriend *fESDADfriend; // AD object containing complete raw data
  AliESDCalofriend *fESDCalofriend; // Calo object containing complete raw data
  
  Int_t fNclustersTPC[72]; //cluster occupancy per sector per sector
  Int_t fNclustersTPCused[72]; //number of clusters used in tracking per sector

  ClassDef(AliESDfriend,7) // ESD friend
};

#endif


