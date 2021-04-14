// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DOFStatistics.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Mergable bbject for statistics of points used by each DOF

#include "Align/DOFStatistics.h"
#include "Align/Controller.h"
#include "Framework/Logger.h"
#include <TMath.h>
#include <TCollection.h>

using namespace TMath;

ClassImp(o2::align::DOFStatistics);

namespace o2
{
namespace align
{

//_________________________________________________________
DOFStatistics::DOFStatistics(int n)
  : TNamed("DOFstat", "DOF statistics"), mNDOFs(n), mNMerges(1), mStat(0)
{
  // def c-tor
  if (mNDOFs) {
    mStat = new int[n];
    memset(mStat, 0, mNDOFs * sizeof(int));
  }
  //
}

//_________________________________________________________
DOFStatistics::~DOFStatistics()
{
  // d-r
  delete[] mStat;
}

//____________________________________________
void DOFStatistics::Print(Option_t*) const
{
  // print info
  printf("NDOFs: %d, NMerges: %d\n", mNDOFs, mNMerges);
  //
}

//____________________________________________
TH1F* DOFStatistics::createHisto(Controller* st) const
{
  // create histo with stat. If steer object is supplied, build labels
  if (!mNDOFs)
    return 0;
  TH1F* h = new TH1F("DOFstat", "statistics per DOF", mNDOFs, 0, mNDOFs);
  for (int i = mNDOFs; i--;) {
    h->SetBinContent(i + 1, mStat[i]);
    if (st)
      h->GetXaxis()->SetBinLabel(i + 1, st->getDOFLabelTxt(i));
  }
  return h;
}

//______________________________________________________________________________
int64_t DOFStatistics::merge(TCollection* list)
{
  // merge statistics
  int nmerged = 0;
  TIter next(list);
  TObject* obj;
  while ((obj = next())) {
    DOFStatistics* stAdd = dynamic_cast<DOFStatistics*>(obj);
    if (!stAdd)
      continue;
    if (mNDOFs != stAdd->mNDOFs) {
      LOG(ERROR) << "Different NDOF: " << mNDOFs << " vs " << stAdd->mNDOFs << ".";
      return 0;
    }
    for (int i = mNDOFs; i--;)
      mStat[i] += stAdd->mStat[i];
    mNMerges += stAdd->mNMerges;
    nmerged++;
  }
  return nmerged;
}

} // namespace align
} // namespace o2
