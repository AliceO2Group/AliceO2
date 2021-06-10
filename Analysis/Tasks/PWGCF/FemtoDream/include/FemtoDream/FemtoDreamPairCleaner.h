// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamPairCleaner.h
/// \brief FemtoDreamPairCleaner - Makes sure only proper candidates are paired
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPAIRCLEANER_H_
#define ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPAIRCLEANER_H_

#include "Framework/HistogramRegistry.h"
#include <Rtypes.h>

using namespace o2::framework;

namespace o2::femtoDream
{

namespace femtoDreamPairCleaner
{
enum CleanConf { kStrict,
                 kLoose,
                 kDeltaEtaDeltaPhiStar };
}

class FemtoDreamPairCleaner
{
 public:
  virtual ~FemtoDreamPairCleaner() = default;

  void init(femtoDreamPairCleaner::CleanConf conf, HistogramRegistry* registry)
  {
    if (registry) {
      mHistogramRegistry = registry;
    }
  }

 private:
  HistogramRegistry* mHistogramRegistry; ///< For QA output

  ClassDefNV(FemtoDreamPairCleaner, 1);
};
} // namespace o2::femtoDream

#endif /* ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPAIRCLEANER_H_ */
