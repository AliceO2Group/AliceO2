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

/* $Id:  $ */

//_________________________________________________________________________//
//_________________________________________________________________________//

#include "AliESDTOFCluster.h"
#include "TClonesArray.h"
#include "AliESDEvent.h"

ClassImp(AliESDTOFCluster)

//_________________________________________________________________________
AliESDTOFCluster::AliESDTOFCluster(Int_t clID) :
  fID(clID),
  fNTOFhits(0),
  fStatus(0),
  fNmatchableTracks(0)
{
  //
  // default ctor
  //

  for(Int_t i=kMaxHits;i--;)    fHitIndex[i] = -1;
  for(Int_t i=kMaxMatches;i--;) fMatchIndex[i] = -1;
}

//_________________________________________________________________________
AliESDTOFCluster::AliESDTOFCluster(Int_t ,Int_t ,Float_t ,Float_t ,Float_t ,Int_t*,Int_t ,Int_t ,
				   Bool_t ,Float_t ,Float_t ,Float_t ,
				   Int_t ,Float_t ,Float_t ,Float_t ,Float_t ,Double_t*) :
  fID(-1),
  fNTOFhits(0),
  fStatus(1),
  fNmatchableTracks(1)
{
  //
  // Constructor of AliESDTOFCluster object
  //

  // to be replaced with hit creation
  for(Int_t i=kMaxHits;i--;) fHitIndex[i] = -1;
  for(Int_t i=kMaxMatches;i--;) fMatchIndex[i] = -1;
  //
}

//_________________________________________________________________________
AliESDTOFCluster::AliESDTOFCluster(Int_t ,Int_t ,Float_t ,Float_t ,Float_t ,Int_t*,Int_t ,Int_t ,
				   Bool_t ,Float_t ,Float_t ,Float_t ) :
  fID(-1),
  fNTOFhits(0),
  fStatus(1),
  fNmatchableTracks(0)
{
  //
  // Constructor of AliESDTOFCluster object
  //

  // to be replaced with hit creation
  for(Int_t i=kMaxHits;i--;) fHitIndex[i] = -1;
  for(Int_t i=kMaxMatches;i--;) fMatchIndex[i] = -1;

}

//_________________________________________________________________________
AliESDTOFCluster::AliESDTOFCluster(const AliESDTOFCluster & source) :
  AliVTOFcluster(source),
  fID(source.fID),
  fNTOFhits(source.fNTOFhits),
  fStatus(source.fStatus),
  fNmatchableTracks(source.fNmatchableTracks)
{
  // 
  // copy ctor for AliESDTOFCluster object
  //

  SetEvent(source.GetEvent());

  for(Int_t i=kMaxHits;i--;)    fHitIndex[i] = source.fHitIndex[i];
  for(Int_t i=kMaxMatches;i--;) fMatchIndex[i] = source.fMatchIndex[i];

}

//_________________________________________________________________________
AliESDTOFCluster & AliESDTOFCluster::operator=(const AliESDTOFCluster & source)
{
  // 
  // assignment op-r
  //
  if (this == &source) return *this;
  AliVTOFcluster::operator=(source);
  fID = source.fID;
  fNTOFhits = source.fNTOFhits;
  fStatus = source.fStatus;
  fNmatchableTracks = source.fNmatchableTracks;

  for(Int_t i=kMaxHits;i--;)    fHitIndex[i] = source.fHitIndex[i];
  for(Int_t i=kMaxMatches;i--;) fMatchIndex[i] = source.fMatchIndex[i];

  return *this;
}

//_________________________________________________________________________
Int_t AliESDTOFCluster::Update(Int_t trackIndex,Float_t dX,Float_t dY,Float_t dZ,
			       Float_t length,Double_t expTimes[AliPID::kSPECIESC])
{
  // update cluster info by new track
  //  AliInfo(Form("TOF %d %d",fNmatchableTracks,fNTOFhits));

  if(fNmatchableTracks >= kMaxMatches) return 2; // max number reached
  //
  // check if the track is not already stored
  for (Int_t ii=fNmatchableTracks; ii--;) if (trackIndex==GetTrackIndex(ii)) return 1;
  //
  const AliESDEvent *event = (AliESDEvent *) GetEvent();
  TClonesArray *matchAr = event->GetESDTOFMatches();
  int ntr = matchAr->GetEntriesFast();
  new((*matchAr)[ntr]) AliESDTOFMatch(trackIndex,expTimes,dX,dY,dZ,length);
  int nmt = fNmatchableTracks++;
  fMatchIndex[nmt] = ntr;
  //
  return 0;
  //
}

//_________________________________________________________________________
void AliESDTOFCluster::AddESDTOFHitIndex(Int_t hitID)
{
  // register new hit in the cluster
  if(fNTOFhits >= kMaxHits) return;
  int nth = fNTOFhits++;
  fHitIndex[nth] = hitID; // add the hit to the array
}

//_________________________________________________________________________
void AliESDTOFCluster::AddTOFhit(AliESDTOFHit *hit)
{
  // add new hit
  if(fNTOFhits >= kMaxHits) return;
  // add the hit to the array
  const AliESDEvent *event = (AliESDEvent *) GetEvent();
  TClonesArray *hitAr = event->GetESDTOFHits();
  int nh = hitAr->GetEntriesFast();
  new((*hitAr)[nh]) AliESDTOFHit(*hit);
  //   hitN->SetIndex(nh); // RS: why do we need this
  int nth = fNTOFhits++;
  fHitIndex[nth] = nh;
  //
}

//_________________________________________________________________________
void AliESDTOFCluster::AddTOFhit(Int_t ,Int_t ,Float_t ,Float_t ,Float_t ,Int_t*,Int_t ,Int_t , Bool_t ,Float_t ,Float_t ,Float_t )
{
}

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetClusterIndex(int ihit) const
{
  AliESDTOFHit* hit = GetTOFHit(ihit);
  return hit ? hit->GetClusterIndex() : -1;
} 

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetTOFchannel(int ihit) const 
{
  AliESDTOFHit* hit = GetTOFHit(ihit);
  return hit ? hit->GetTOFchannel() : -1;
}

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetTime(int ihit) const 
{
  AliESDTOFHit* hit = GetTOFHit(ihit);
  return hit ? hit->GetTime() : 0;
}

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetTimeRaw(Int_t ihit) const 
{
 AliESDTOFHit* hit = GetTOFHit(ihit);
 return hit ? hit->GetTimeRaw() : 0;
} // TOF raw time

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetTOT(Int_t ihit) const 
{
  AliESDTOFHit* hit = GetTOFHit(ihit);
  return hit ? hit->GetTOT() : 0;
} // TOF tot

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetTOFsignalToT(Int_t ihit) const 
{
  AliESDTOFHit* hit = GetTOFHit(ihit);
  return hit ? hit->GetTOT() : 0; // RS: Why signalTot and TOT are the same?
} // TOF tot

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetLabel(Int_t i,Int_t ihit) const 
{
  AliESDTOFHit* hit = GetTOFHit(ihit);
  if (!hit || i>=3) return -999;
  return hit->GetTOFLabel(i);
}

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetDeltaBC(Int_t ihit) const 
{
  AliESDTOFHit* hit = GetTOFHit(ihit);
  return hit ? hit->GetDeltaBC() : 0;
}

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetL0L1Latency(Int_t ihit) const 
{
  AliESDTOFHit* hit = GetTOFHit(ihit);  
  return hit ? hit->GetL0L1Latency() : 0;
}

//_________________________________________________________________________
Bool_t AliESDTOFCluster::GetStatus() const 
{
  if(!fEvent){
    AliInfo("No AliESDEvent available here!");
    return 0;
  }
  return fStatus;
}

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetZ() const 
{
  AliESDTOFHit* hit = GetTOFHit(0);
  return hit ? hit->GetZ() : 0;
}

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetPhi() const 
{
  AliESDTOFHit* hit = GetTOFHit(0);
  return hit ? hit->GetPhi() : 0;
}

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetR() const 
{
  AliESDTOFHit* hit = GetTOFHit(0);
  return hit ? hit->GetR() : 0;
}

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetNMatchableTracks() const 
{
  return fNmatchableTracks;
}

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetNTOFhits() const 
{
  return fNTOFhits;
}

//_________________________________________________________________________
Int_t AliESDTOFCluster::GetTrackIndex(Int_t i) const 
{
  AliESDTOFMatch* match = 0;
  return (i<fNmatchableTracks && (match=GetTOFMatch(i))) ? match->GetTrackIndex() : -999;
}

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetDistanceInStripPlane(Int_t i) const 
{
  // distance
  AliESDTOFMatch* match = 0;
  if (i>=fNmatchableTracks || !(match=GetTOFMatch(i))) return -999;
  Double_t dz = match->GetDz();
  Double_t dx = match->GetDx();
  return TMath::Sqrt(dx*dx+dz*dz);
}

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetDx(Int_t i)  const 
{
  AliESDTOFMatch* match = 0;  
  return (i<fNmatchableTracks && (match=GetTOFMatch(i))) ? match->GetDx() : -999;
} // distance, X component

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetDy(Int_t i)  const 
{
  AliESDTOFMatch* match = 0;  
  return (i<fNmatchableTracks && (match=GetTOFMatch(i))) ? match->GetDy() : -999;
} // distance, Y component

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetDz(Int_t i)  const 
{
  AliESDTOFMatch* match = 0;  
  return (i<fNmatchableTracks && (match=GetTOFMatch(i))) ? match->GetDz() : -999;
} // distance, Z component

//_________________________________________________________________________
Float_t AliESDTOFCluster::GetLength(Int_t i) const 
{
  AliESDTOFMatch* match = 0;  
  return (i<fNmatchableTracks && (match=GetTOFMatch(i))) ? match->GetTrackLength() : -999.;
} // reconstructed track length at TOF

//_________________________________________________________________________
Double_t AliESDTOFCluster::GetIntegratedTime(Int_t iPart,Int_t i) const 
{
  AliESDTOFMatch* match = 0;  
  return (i<fNmatchableTracks && (match=GetTOFMatch(i))) ? match->GetIntegratedTimes(iPart) : -999.;
} // reconstructed track length at TOF

//_________________________________________________________________________
void AliESDTOFCluster::Print(const Option_t*) const
{
  // print full chain
  printf("#%5d TOF Cluster %5d of %d Hits, %d Matchable Tracks\n",GetESDID(),fID, fNTOFhits, fNmatchableTracks);
  for (int ih=0;ih<fNTOFhits;ih++) {
    AliESDTOFHit* ht = GetTOFHit(ih);
    if (ht) {
      printf("%d: ",ih);
      ht->Print();
    }
  }
  //
  for (int it=0;it<fNmatchableTracks;it++) {
    AliESDTOFMatch* mt = GetTOFMatch(it);
    if (mt) {
      printf("%d: ",it);
      mt->Print();
    }
  }
  //
}

//_________________________________________________________________________
AliESDTOFHit* AliESDTOFCluster::GetTOFHit(Int_t i) const
{
  return fEvent ? ((AliESDTOFHit *) ((AliESDEvent *) GetEvent())->GetESDTOFHits()->At(fHitIndex[i])):0;
}

//_________________________________________________________________________
AliESDTOFMatch* AliESDTOFCluster::GetTOFMatch(Int_t i) const 
{
  return fEvent ? ((AliESDTOFMatch *) ((AliESDEvent *) GetEvent())->GetESDTOFMatches()->At(fMatchIndex[i])) : 0;
}

//_________________________________________________________________________
void AliESDTOFCluster::SuppressMatchedTrack(Int_t id)
{
  // suppress the reference to track id
  for (int it=fNmatchableTracks;it--;) {
    AliESDTOFMatch* mtc = GetTOFMatch(it);
    if (mtc->GetTrackIndex()!=id) continue;
    // need to suprress the match: simply remove reference to it
    int rmID = fMatchIndex[it];
    for (int jt=it+1;jt<fNmatchableTracks;jt++) fMatchIndex[jt-1] = fMatchIndex[jt];
    fNmatchableTracks--;
    // remove match rmID
    TClonesArray* arrMatch = ((AliESDEvent *)GetEvent())->GetESDTOFMatches();
    int last = arrMatch->GetEntriesFast()-1;
    AliESDTOFMatch* mtcL = (AliESDTOFMatch*)arrMatch->At(last);
    if (rmID!=last) {
      *mtc = *mtcL; // assign the last match to removed slot
      int trID = mtc->GetTrackIndex();
      AliESDtrack* trc = ((AliESDEvent *)GetEvent())->GetTrack(trID);
      trc->ReplaceTOFMatchID(last,rmID); // fix the reference to reassigned match
      // fix the 
    }
    arrMatch->RemoveAt(last);
    break;
  }
  //
  if (!fNmatchableTracks) { // no matches left, clear all hits: prepare for selfelimination
    // first remove associated hits
    TClonesArray* arrHits = ((AliESDEvent *)GetEvent())->GetESDTOFHits();
    TClonesArray* arrClus = ((AliESDEvent *)GetEvent())->GetESDTOFClusters();
    int last = arrHits->GetEntriesFast()-1;
    for (;fNTOFhits--;) { // remove hits
      int hID = fHitIndex[int(fNTOFhits)];
      AliESDTOFHit* hit = (AliESDTOFHit*)arrHits->At(hID);
      AliESDTOFHit* hitL = (AliESDTOFHit*)arrHits->At(last);
      if (hID!=last) {
	*hit = *hitL; // assign the last match to removed slot
	// fix reference on hitL in the owner cluster
	int clID = hit->GetESDTOFClusterIndex();
	AliESDTOFCluster* clusL = (AliESDTOFCluster*)arrClus->At(clID);
	clusL->ReplaceHitID(last,hID);
      }
      arrHits->RemoveAt(last--);
    }
  }

}

//_________________________________________________________________________
void AliESDTOFCluster::ReplaceHitID(int oldID, int newID)
{
  // replace the reference to hit from oldID by newID
  for (int it=fNTOFhits;it--;) {
    if (fHitIndex[it]==oldID) {
      fHitIndex[it]=newID;
      return;
    }
  }
}

//_________________________________________________________________________
void AliESDTOFCluster::ReplaceMatchID(int oldID, int newID)
{
  // replace the reference to match from oldID by newID
  for (int it=fNmatchableTracks;it--;) {
    if (fMatchIndex[it]==oldID) {
      fMatchIndex[it]=newID;
      return;
    }
  }
}

//_________________________________________________________________________
void AliESDTOFCluster::ReplaceMatchedTrackID(int oldID, int newID)
{
  // replace the reference to track oldID by newID
  for (int it=fNmatchableTracks;it--;) {
    AliESDTOFMatch* mtc = GetTOFMatch(it);
    if (mtc->GetTrackIndex()!=oldID) continue;
    mtc->SetTrackIndex(newID);
    break;
  }
  //
}

//_________________________________________________________________________
void AliESDTOFCluster::FixSelfReferences(int oldID, int newID)
{
  // replace the references (in tracks and hist) to this cluster from oldID by newID  
  for (int it=fNmatchableTracks;it--;) {
    int trID = GetTOFMatch(it)->GetTrackIndex();
    AliESDtrack* trc = ((AliESDEvent *)GetEvent())->GetTrack(trID);
    trc->ReplaceTOFClusterID(oldID,newID);
  }
  for (int it=fNTOFhits;it--;) {
    AliESDTOFHit* hit = GetTOFHit(it);
    if (hit) hit->SetESDTOFClusterIndex(newID);
  }
  //
}
