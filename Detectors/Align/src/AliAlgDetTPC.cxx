// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetTPC.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TPC detector wrapper

#include "Align/AliAlgDetTPC.h"
#include "Align/AliAlgVol.h"
#include "Align/AliAlgSensTPC.h"
#include "Align/AliAlgSteer.h"
//#include "AliGeomManager.h"
//#include "AliESDtrack.h"
#include "Framework/Logger.h"
#include <TGeoManager.h>

ClassImp(o2::align::AliAlgDetTPC);

namespace o2
{
namespace align
{

//____________________________________________
AliAlgDetTPC::AliAlgDetTPC(const char* title)
{
  // default c-tor
  SetNameTitle(AliAlgSteer::GetDetNameByDetID(AliAlgSteer::kTPC), title);
  SetDetID(AliAlgSteer::kTPC);
}

//____________________________________________
AliAlgDetTPC::~AliAlgDetTPC()
{
  // d-tor
}

//____________________________________________
void AliAlgDetTPC::DefineVolumes()
{
  // define TPC volumes
  //
  const int kNSect = 18, kAC = 2, kIOROC = 2;
  const char* kSide[kAC] = {"A", "C"};
  const char* kROC[kIOROC] = {"Inner", "Outer"};
  //  AliAlgSensTPC *chamb=0;
  //
  int labDet = GetDetLabel();
  AliAlgVol* volTPC = new AliAlgVol("ALIC_1/TPC_M_1", labDet);
  AddVolume(volTPC);
  //

  for (int roc = 0; roc < kIOROC; roc++) {     // inner/outer
    for (int side = 0; side < kAC; side++) {   // A/C
      for (int isc = 0; isc < kNSect; isc++) { // sector ID
        const char* symname = Form("TPC/Endcap%s/Sector%d/%sChamber", kSide[side], isc + 1, kROC[roc]);
        if (!gGeoManager->GetAlignableEntry(symname)) {
          AliErrorF("Did not find alignable %s", symname);
          continue;
        }
        Int_t iid = side * kNSect + isc;
        UShort_t vid = AliGeomManager::LayerToVolUID(AliGeomManager::kTPC1 + roc, iid);
        iid = labDet + (1 + side) * 10000 + (1 + isc) * 100 + (1 + roc);
        AliAlgSensTPC* sens = new AliAlgSensTPC(symname, vid, iid, isc);
        sens->SetParent(volTPC);
        AddVolume(sens);
      } // sector ID
    }   // A/C
  }     // inner/outer
  //
}

//____________________________________________
Bool_t AliAlgDetTPC::AcceptTrack(const AliESDtrack* trc, Int_t trtype) const
{
  // test if detector had seed this track
  if (!CheckFlags(trc, trtype))
    return kFALSE;
  if (trc->GetNcls(1) < fNPointsSel[trtype])
    return kFALSE;
  return kTRUE;
}

} // namespace align
} // namespace o2
