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

#include "AliAlgDOFStat.h"
#include "AliAlgSteer.h"
#include "AliLog.h"
#include <TMath.h>
#include <TCollection.h>

using namespace TMath;

ClassImp(AliAlgDOFStat)

//_________________________________________________________
AliAlgDOFStat::AliAlgDOFStat(Int_t n)
  : TNamed("DOFstat","DOF statistics")
  ,fNDOFs(n)
  ,fNMerges(1)
  ,fStat(0)
{
  // def c-tor
  if (fNDOFs) {
    fStat = new Int_t[n];
    memset(fStat,0,fNDOFs*sizeof(Int_t));
  }
  //
}

//_________________________________________________________
AliAlgDOFStat::~AliAlgDOFStat()
{
  // d-r
  delete[] fStat;
}


//____________________________________________
void AliAlgDOFStat::Print(Option_t*) const
{
  // print info
  printf("NDOFs: %d, NMerges: %d\n",fNDOFs,fNMerges);
  //
}

//____________________________________________
TH1F* AliAlgDOFStat::CreateHisto(AliAlgSteer* st) const
{
  // create histo with stat. If steer object is supplied, build labels
  if (!fNDOFs) return 0;
  TH1F* h = new TH1F("DOFstat","statistics per DOF",fNDOFs,0,fNDOFs);
  for (int i=fNDOFs;i--;) {
    h->SetBinContent(i+1,fStat[i]);
    if (st) h->GetXaxis()->SetBinLabel(i+1,st->GetDOFLabelTxt(i));
  }
  return h;
}

//______________________________________________________________________________
Long64_t AliAlgDOFStat::Merge(TCollection *list) 
{
  // merge statistics
  int nmerged = 0;
  TIter next(list);
  TObject *obj;
  while((obj=next())) {
    AliAlgDOFStat* stAdd = dynamic_cast<AliAlgDOFStat*>(obj);
    if (!stAdd) continue;
    if (fNDOFs != stAdd->fNDOFs) {
      AliErrorF("Different NDOF: %d vs %d",fNDOFs,stAdd->fNDOFs);
      return 0;
    } 
    for (int i=fNDOFs;i--;) fStat[i]+=stAdd->fStat[i];
    fNMerges += stAdd->fNMerges;
    nmerged++;
  }
  return nmerged;
}
