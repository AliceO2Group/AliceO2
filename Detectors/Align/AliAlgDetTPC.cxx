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

#include "AliAlgDetTPC.h"
#include "AliAlgVol.h"
#include "AliAlgSensTPC.h"
#include "AliAlgSteer.h"
#include "AliGeomManager.h"
#include "AliESDtrack.h"
#include "AliLog.h"
#include <TGeoManager.h>

ClassImp(AliAlgDetTPC);

//____________________________________________
AliAlgDetTPC::AliAlgDetTPC(const char* title)
{
  // default c-tor
  SetNameTitle(AliAlgSteer::GetDetNameByDetID(AliAlgSteer::kTPC),title);
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
  const int kNSect = 18, kAC=2, kIOROC=2;
  const char* kSide[kAC] = {"A","C"};
  const char* kROC[kIOROC] = {"Inner","Outer"};
  //  AliAlgSensTPC *chamb=0;
  //
  int labDet = GetDetLabel();
  AliAlgVol* volTPC = new AliAlgVol("ALIC_1/TPC_M_1",labDet);
  AddVolume( volTPC ); 
  //
  
  for (int roc=0;roc<kIOROC;roc++) { // inner/outer
    for (int side=0;side<kAC;side++) { // A/C
      for (int isc=0;isc<kNSect;isc++) { // sector ID
	const char *symname = Form("TPC/Endcap%s/Sector%d/%sChamber",kSide[side],isc+1,kROC[roc]);
	if (!gGeoManager->GetAlignableEntry(symname)) {
	  AliErrorF("Did not find alignable %s",symname);
	  continue;
	}
	Int_t iid = side*kNSect+isc;
	UShort_t vid = AliGeomManager::LayerToVolUID(AliGeomManager::kTPC1+roc,iid);
	iid = labDet + (1+side)*10000 + (1+isc)*100+(1+roc);
	AliAlgSensTPC* sens = new AliAlgSensTPC(symname,vid,iid,isc);
	sens->SetParent(volTPC);
	AddVolume(sens);
      } // sector ID
    } // A/C
  } // inner/outer
  //
}

//____________________________________________
Bool_t AliAlgDetTPC::AcceptTrack(const AliESDtrack* trc, Int_t trtype) const 
{
  // test if detector had seed this track
  if (!CheckFlags(trc,trtype)) return kFALSE;
  if (trc->GetNcls(1)<fNPointsSel[trtype]) return kFALSE;
  return kTRUE;
}
