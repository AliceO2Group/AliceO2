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
#include <TCollection.h>

ClassImp(o2::align::DOFStatistics);

namespace o2
{
namespace align
{

//____________________________________________
std::unique_ptr<TH1F> DOFStatistics::buildHistogram(Controller* controller) const
{
  // create histo with stat. If steer object is supplied, build labels
  auto histogram = std::make_unique<TH1F>("DOFstat", "statistics per DOF", getNDOFs(), 0, getNDOFs());
  for (size_t i = 0; i < getNDOFs(); ++i) {
    // Bin 0 is underflow bin
    histogram->SetBinContent(i + 1, mStat[i]);
    if (controller != nullptr) {
      histogram->GetXaxis()->SetBinLabel(i + 1, controller->getDOFLabelTxt(i));
    }
  }
  return histogram;
}

//______________________________________________________________________________
int64_t DOFStatistics::merge(TCollection* list)
{
  // merge statistics
  int nMerged = 0;
  TIter next{list};
  TObject* obj = nullptr;
  while ((obj = next()) != nullptr) {
    DOFStatistics* otherStats = dynamic_cast<DOFStatistics*>(obj);
    if (!otherStats) {
      continue;
    }
    assert(getNDOFs() == otherStats->getNDOFs());
    std::transform(std::begin(otherStats->mStat), std::end(otherStats->mStat), std::begin(mStat), std::begin(mStat), std::plus<>{});
    mNMerges += otherStats->mNMerges;
    nMerged++;
  }
  return nMerged;
}

} // namespace align
} // namespace o2
