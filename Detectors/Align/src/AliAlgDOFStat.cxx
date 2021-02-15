// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDOFStat.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Mergable bbject for statistics of points used by each DOF

#include "Align/AliAlgDOFStat.h"
#include "Align/AliAlgSteer.h"
#include "Framework/Logger.h"
#include <TMath.h>
#include <TCollection.h>

using namespace TMath;

ClassImp(o2::align::AliAlgDOFStat);

namespace o2
{
namespace align
{

//_________________________________________________________
AliAlgDOFStat::AliAlgDOFStat(Int_t n)
  : TNamed("DOFstat", "DOF statistics"), fNDOFs(n), fNMerges(1), fStat(0)
{
  // def c-tor
  if (fNDOFs) {
    fStat = new Int_t[n];
    memset(fStat, 0, fNDOFs * sizeof(Int_t));
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
  printf("NDOFs: %d, NMerges: %d\n", fNDOFs, fNMerges);
  //
}

//____________________________________________
TH1F* AliAlgDOFStat::CreateHisto(AliAlgSteer* st) const
{
  // create histo with stat. If steer object is supplied, build labels
  if (!fNDOFs)
    return 0;
  TH1F* h = new TH1F("DOFstat", "statistics per DOF", fNDOFs, 0, fNDOFs);
  for (int i = fNDOFs; i--;) {
    h->SetBinContent(i + 1, fStat[i]);
    if (st)
      h->GetXaxis()->SetBinLabel(i + 1, st->GetDOFLabelTxt(i));
  }
  return h;
}

//______________________________________________________________________________
Long64_t AliAlgDOFStat::Merge(TCollection* list)
{
  // merge statistics
  int nmerged = 0;
  TIter next(list);
  TObject* obj;
  while ((obj = next())) {
    AliAlgDOFStat* stAdd = dynamic_cast<AliAlgDOFStat*>(obj);
    if (!stAdd)
      continue;
    if (fNDOFs != stAdd->fNDOFs) {
      LOG(ERROR) << "Different NDOF: " << fNDOFs << " vs " << stAdd->fNDOFs << ".";
      return 0;
    }
    for (int i = fNDOFs; i--;)
      fStat[i] += stAdd->fStat[i];
    fNMerges += stAdd->fNMerges;
    nmerged++;
  }
  return nmerged;
}

} // namespace align
} // namespace o2
