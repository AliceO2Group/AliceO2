// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/HistogramRegistry.h"

namespace o2::framework
{

// define histogram callbacks for runtime histogram creation
#define CALLB(HType)                                            \
  {                                                             \
    k##HType,                                                   \
      [](HistogramSpec const& histSpec) {                       \
        return HistFactory::createHistVariant<HType>(histSpec); \
      }                                                         \
  }

const std::map<HistType, std::function<HistPtr(const HistogramSpec&)>> HistFactory::HistogramCreationCallbacks{
  CALLB(TH1D), CALLB(TH1F), CALLB(TH1I), CALLB(TH1C), CALLB(TH1S),
  CALLB(TH2D), CALLB(TH2F), CALLB(TH2I), CALLB(TH2C), CALLB(TH2S),
  CALLB(TH3D), CALLB(TH3F), CALLB(TH3I), CALLB(TH3C), CALLB(TH3S),
  CALLB(THnD), CALLB(THnF), CALLB(THnI), CALLB(THnC), CALLB(THnS), CALLB(THnL),
  CALLB(THnSparseD), CALLB(THnSparseF), CALLB(THnSparseI), CALLB(THnSparseC), CALLB(THnSparseS), CALLB(THnSparseL),
  CALLB(TProfile), CALLB(TProfile2D), CALLB(TProfile3D),
  //CALLB(StepTHnF), CALLB(StepTHnD)
};

#undef CALLB

} // namespace o2::framework
