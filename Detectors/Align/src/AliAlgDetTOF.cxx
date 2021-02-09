// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetTOF.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Wrapper for TOF detector

#include "AliAlgDetTOF.h"
#include "AliAlgVol.h"
#include "AliAlgSensTOF.h"
#include "AliAlgSteer.h"
#include "AliGeomManager.h"
#include "AliTOFGeometry.h"
#include "AliESDtrack.h"
#include <TGeoManager.h>

ClassImp(o2::align::AliAlgDetTOF);

namespace o2
{
namespace align
{

//____________________________________________
AliAlgDetTOF::AliAlgDetTOF(const char* title)
{
  // default c-tor
  SetNameTitle(AliAlgSteer::GetDetNameByDetID(AliAlgSteer::kTOF), title);
  SetDetID(AliAlgSteer::kTOF);
}

//____________________________________________
AliAlgDetTOF::~AliAlgDetTOF()
{
  // d-tor
}

//____________________________________________
void AliAlgDetTOF::DefineVolumes()
{
  // define TOF volumes
  //
  const int kNSect = 18, kNStrips = AliTOFGeometry::NStripA() + 2 * AliTOFGeometry::NStripB() + 2 * AliTOFGeometry::NStripC();
  int labDet = GetDetLabel();
  AliAlgSensTOF* strip = 0;
  //
  //  AddVolume( volTOF = new AliAlgVol("TOF") ); // no main volume, why?
  AliAlgVol* sect[kNSect] = {0};
  //
  for (int isc = 0; isc < kNSect; isc++) {
    int iid = labDet + (1 + isc) * 100;
    AddVolume(sect[isc] = new AliAlgVol(Form("TOF/sm%02d", isc), iid));
  }
  //
  int cnt = 0;
  for (int isc = 0; isc < kNSect; isc++) {
    for (int istr = 1; istr <= kNStrips; istr++) { // strip
      int iid = labDet + (1 + isc) * 100 + (1 + istr);
      int vid = AliGeomManager::LayerToVolUID(AliGeomManager::kTOF, cnt++);
      const char* symname = Form("TOF/sm%02d/strip%02d", isc, istr);
      if (!gGeoManager->GetAlignableEntry(symname))
        continue;
      AddVolume(strip = new AliAlgSensTOF(symname, vid, iid, isc));
      strip->SetParent(sect[isc]);
    } // strip
  }   // layer
  //
}

//____________________________________________
Bool_t AliAlgDetTOF::AcceptTrack(const AliESDtrack* trc, Int_t trtype) const
{
  // test if detector had seed this track
  return CheckFlags(trc, trtype);
}

} // namespace align
} // namespace o2
