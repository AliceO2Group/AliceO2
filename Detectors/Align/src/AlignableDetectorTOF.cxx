// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableDetectorTOF.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Wrapper for TOF detector

#include "Align/AlignableDetectorTOF.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableSensorTOF.h"
#include "Align/Controller.h"
//#include "AliGeomManager.h"
//#include "AliTOFGeometry.h"
//#include "AliESDtrack.h"
#include <TGeoManager.h>

ClassImp(o2::align::AlignableDetectorTOF);

namespace o2
{
namespace align
{

//____________________________________________
AlignableDetectorTOF::AlignableDetectorTOF(const char* title)
{
  // default c-tor
  SetNameTitle(Controller::getDetNameByDetID(Controller::kTOF), title);
  setDetID(Controller::kTOF);
}

//____________________________________________
AlignableDetectorTOF::~AlignableDetectorTOF()
{
  // d-tor
}

//____________________________________________
void AlignableDetectorTOF::defineVolumes()
{
  // define TOF volumes
  //
  const int kNSect = 18, kNStrips = AliTOFGeometry::NStripA() + 2 * AliTOFGeometry::NStripB() + 2 * AliTOFGeometry::NStripC();
  int labDet = getDetLabel();
  AlignableSensorTOF* strip = 0;
  //
  //  AddVolume( volTOF = new AlignableVolume("TOF") ); // no main volume, why?
  AlignableVolume* sect[kNSect] = {0};
  //
  for (int isc = 0; isc < kNSect; isc++) {
    int iid = labDet + (1 + isc) * 100;
    addVolume(sect[isc] = new AlignableVolume(Form("TOF/sm%02d", isc), iid));
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
      addVolume(strip = new AlignableSensorTOF(symname, vid, iid, isc));
      strip->setParent(sect[isc]);
    } // strip
  }   // layer
  //
}

//____________________________________________
bool AlignableDetectorTOF::AcceptTrack(const AliESDtrack* trc, int trtype) const
{
  // test if detector had seed this track
  return CheckFlags(trc, trtype);
}

} // namespace align
} // namespace o2
