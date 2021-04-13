// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetHMPID.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  HMPID detector wrapper

#include "Align/AliAlgDetHMPID.h"
//#include "AliHMPIDParam.h"
#include "Align/AliAlgVol.h"
#include "Align/AliAlgSensHMPID.h"
#include "Align/AliAlgSteer.h"
//#include "AliGeomManager.h"
//#include "AliESDtrack.h"
#include "Framework/Logger.h"
#include <TGeoManager.h>

ClassImp(o2::align::AliAlgDetHMPID);

namespace o2
{
namespace align
{

//____________________________________________
AliAlgDetHMPID::AliAlgDetHMPID(const char* title)
{
  // default c-tor
  SetNameTitle(AliAlgSteer::GetDetNameByDetID(AliAlgSteer::kHMPID), title);
  SetDetID(AliAlgSteer::kHMPID);
}

//____________________________________________
AliAlgDetHMPID::~AliAlgDetHMPID()
{
  // d-tor
}

//____________________________________________
void AliAlgDetHMPID::DefineVolumes()
{
  // define HMPID volumes
  //
  int labDet = GetDetLabel();
  AliGeomManager::ELayerID idHMPID = AliGeomManager::kHMPID;
  for (int iCh = AliHMPIDParam::kMinCh; iCh <= AliHMPIDParam::kMaxCh; iCh++) {
    const char* symname = Form("/HMPID/Chamber%i", iCh);
    if (!gGeoManager->GetAlignableEntry(symname)) {
      AliErrorF("Did not find alignable %s", symname);
      continue;
    }
    uint16_t vid = AliGeomManager::LayerToVolUID(idHMPID, iCh);
    int iid = labDet + (1 + iCh) * 10000;
    AliAlgSensHMPID* sens = new AliAlgSensHMPID(symname, vid, iid);
    AddVolume(sens);
  } //iCh loop
  //
}

//____________________________________________
bool AliAlgDetHMPID::AcceptTrack(const AliESDtrack* trc, int trtype) const
{
  // test if detector had seed this track
  if (!CheckFlags(trc, trtype))
    return false;
  if (trc->GetNcls(1) < fNPointsSel[trtype])
    return false;
  return true;
}

} // namespace align
} // namespace o2
