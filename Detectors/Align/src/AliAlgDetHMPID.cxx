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

#include "AliAlgDetHMPID.h"
#include "AliHMPIDParam.h"
#include "AliAlgVol.h"
#include "AliAlgSensHMPID.h"
#include "AliAlgSteer.h"
#include "AliGeomManager.h"
#include "AliESDtrack.h"
#include "AliLog.h"
#include <TGeoManager.h>

ClassImp(AliAlgDetHMPID);

//____________________________________________
AliAlgDetHMPID::AliAlgDetHMPID(const char* title)
{
  // default c-tor
  SetNameTitle(AliAlgSteer::GetDetNameByDetID(AliAlgSteer::kHMPID),title);
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
  for(Int_t iCh=AliHMPIDParam::kMinCh;iCh<=AliHMPIDParam::kMaxCh;iCh++) {
    const char *symname = Form("/HMPID/Chamber%i",iCh);
    if (!gGeoManager->GetAlignableEntry(symname)) {
      AliErrorF("Did not find alignable %s",symname);
      continue;
    }
    UShort_t vid = AliGeomManager::LayerToVolUID(idHMPID,iCh);
    int iid = labDet + (1+iCh)*10000;
    AliAlgSensHMPID* sens = new AliAlgSensHMPID(symname,vid,iid);
    AddVolume(sens);
  }//iCh loop
  //
}

//____________________________________________
Bool_t AliAlgDetHMPID::AcceptTrack(const AliESDtrack* trc, Int_t trtype) const 
{
  // test if detector had seed this track
  if (!CheckFlags(trc,trtype)) return kFALSE;
  if (trc->GetNcls(1)<fNPointsSel[trtype]) return kFALSE;
  return kTRUE;
}
