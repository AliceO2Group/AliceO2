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

// $Id$

//-----------------------------------------------------------------------------
/// \class AliESDMuonCluster
///
/// Class to describe the MUON clusters in the Event Summary Data
///
/// \author Philippe Pillot, Subatech
//-----------------------------------------------------------------------------

#include "AliESDEvent.h"
#include "AliESDMuonCluster.h"
#include "AliESDMuonPad.h"

#include "AliLog.h"

#include <TClonesArray.h>
#include <Riostream.h>

using std::endl;
using std::cout;
/// \cond CLASSIMP
ClassImp(AliESDMuonCluster)
/// \endcond

//_____________________________________________________________________________
AliESDMuonCluster::AliESDMuonCluster()
: TObject(),
  fCharge(0.),
  fChi2(0.),
  fPads(0x0),
  fNPads(0),
  fPadsId(0x0),
  fLabel(-1)
{
  /// default constructor
  fXYZ[0] = fXYZ[1] = fXYZ[2] = 0.;
  fErrXY[0] = fErrXY[1] = 0.;
}

//_____________________________________________________________________________
AliESDMuonCluster::AliESDMuonCluster (const AliESDMuonCluster& cluster)
: TObject(cluster),
  fCharge(cluster.fCharge),
  fChi2(cluster.fChi2),
  fPads(0x0),
  fNPads(cluster.fNPads),
  fPadsId(0x0),
  fLabel(cluster.fLabel)
{
  /// Copy constructor
  fXYZ[0] = cluster.fXYZ[0];
  fXYZ[1] = cluster.fXYZ[1];
  fXYZ[2] = cluster.fXYZ[2];
  fErrXY[0] = cluster.fErrXY[0];
  fErrXY[1] = cluster.fErrXY[1];
  
  if (cluster.fPads) {
    fPads = new TClonesArray("AliESDMuonPad",cluster.fPads->GetEntriesFast());
    AliESDMuonPad *pad = (AliESDMuonPad*) cluster.fPads->First();
    while (pad) {
      new ((*fPads)[fPads->GetEntriesFast()]) AliESDMuonPad(*pad);
      pad = (AliESDMuonPad*) cluster.fPads->After(pad);
    }
  }
  
  if (cluster.fPadsId) fPadsId = new TArrayI(*(cluster.fPadsId));
}

//_____________________________________________________________________________
AliESDMuonCluster& AliESDMuonCluster::operator=(const AliESDMuonCluster& cluster)
{
  /// Equal operator
  if (this == &cluster) return *this;
  
  TObject::operator=(cluster); // don't forget to invoke the base class' assignment operator
  
  fXYZ[0] = cluster.fXYZ[0];
  fXYZ[1] = cluster.fXYZ[1];
  fXYZ[2] = cluster.fXYZ[2];
  fErrXY[0] = cluster.fErrXY[0];
  fErrXY[1] = cluster.fErrXY[1];
  
  fCharge = cluster.fCharge;
  fChi2 = cluster.fChi2;
  fLabel = cluster.fLabel;
  
  delete fPads;
  if (cluster.fPads) {
    fPads = new TClonesArray("AliESDMuonPad",cluster.fPads->GetEntriesFast());
    AliESDMuonPad *pad = (AliESDMuonPad*) cluster.fPads->First();
    while (pad) {
      new ((*fPads)[fPads->GetEntriesFast()]) AliESDMuonPad(*pad);
      pad = (AliESDMuonPad*) cluster.fPads->After(pad);
    }
  } else fPads = 0x0;
  
  SetPadsId(cluster.fNPads, cluster.GetPadsId());
  
  return *this;
}

//_____________________________________________________________________________
void AliESDMuonCluster::Copy(TObject &obj) const {
  
  /// This overwrites the virtual TOBject::Copy()
  /// to allow run time copying without casting
  /// in AliESDEvent

  if(this==&obj)return;
  AliESDMuonCluster *robj = dynamic_cast<AliESDMuonCluster*>(&obj);
  if(!robj)return; // not an AliESDMuonCluster
  *robj = *this;

}

//__________________________________________________________________________
AliESDMuonCluster::~AliESDMuonCluster()
{
  /// Destructor
  delete fPads;
  delete fPadsId;
}

//__________________________________________________________________________
void AliESDMuonCluster::Clear(Option_t* opt)
{
  /// Clear arrays
  if (opt && opt[0] == 'C') {
    if (fPads) fPads->Clear("C");
  } else {
    delete fPads; fPads = 0x0;
  }
  delete fPadsId; fPadsId = 0x0;
  fNPads = 0;
}

//_____________________________________________________________________________
void AliESDMuonCluster::AddPadId(UInt_t padId)
{
  /// Add the given pad Id to the list associated to the cluster
  if (!fPadsId) fPadsId = new TArrayI(10);
  if (fPadsId->GetSize() <= fNPads) fPadsId->Set(fNPads+10);
  fPadsId->AddAt(static_cast<Int_t>(padId), fNPads++);
}

//_____________________________________________________________________________
void AliESDMuonCluster::SetPadsId(Int_t nPads, const UInt_t *padsId)
{
  /// Fill the list pads'Id associated to the cluster with the given list
  
  if (nPads <= 0 || !padsId) {
    delete fPadsId;
    fPadsId = 0x0;
    fNPads = 0;
    return;
  }
  
  if (!fPadsId) fPadsId = new TArrayI(nPads, reinterpret_cast<const Int_t*>(padsId));
  else fPadsId->Set(nPads, reinterpret_cast<const Int_t*>(padsId));
  fNPads = nPads;
  
}

//_____________________________________________________________________________
void AliESDMuonCluster::MovePadsToESD(AliESDEvent &esd)
{
  /// move the pads to the new ESD structure
  if (!fPads) return;
  for (Int_t i = 0; i < fPads->GetEntriesFast(); i++) {
    AliESDMuonPad *pad = static_cast<AliESDMuonPad*>(fPads->UncheckedAt(i));
    AliESDMuonPad *newPad = esd.NewMuonPad();
    *newPad = *pad;
    AddPadId(newPad->GetUniqueID());
  }
  delete fPads;
  fPads = 0x0;
}

//_____________________________________________________________________________
void AliESDMuonCluster::Print(Option_t */*option*/) const
{
  /// print cluster content
  UInt_t cId = GetUniqueID();
  
  cout<<Form("clusterID=%u (ch=%d, det=%d, index=%d)",
	     cId,GetChamberId(),GetDetElemId(),GetClusterIndex())<<endl;
  
  cout<<Form("  position=(%5.2f, %5.2f, %5.2f), sigma=(%5.2f, %5.2f, 0.0)",
	     GetX(),GetY(),GetZ(),GetErrX(),GetErrY())<<endl;
  
  cout<<Form("  charge=%5.2f, chi2=%5.2f, MClabel=%d", GetCharge(), GetChi2(), GetLabel())<<endl;
  
  if (PadsStored()) {
    cout<<"  pad infos:"<<endl;
    for (Int_t iPad=0; iPad<GetNPads(); iPad++) cout<<"  "<<GetPadId(iPad)<<endl;
  }
}

