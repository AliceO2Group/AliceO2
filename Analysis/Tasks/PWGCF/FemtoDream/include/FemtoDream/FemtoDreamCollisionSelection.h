// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamCollisionSelection.h
/// \brief FemtoDreamCollisionSelection - event selection within the o2femtodream framework
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCOLLISIONSELECTION_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCOLLISIONSELECTION_H_

#include "AnalysisCore/TriggerAliases.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/Logger.h"
#include <Rtypes.h>

#include <string>
#include <iostream>

using namespace o2::framework;

namespace o2::analysis
{
namespace femtoDream
{

/// \class FemtoDreamCollisionSelection
/// \brief Cut class to contain and execute all cuts applied to events
class FemtoDreamCollisionSelection
{
 public:
  void setCuts(float zvtxMax, bool checkTrigger, int trig, bool checkOffline)
  {
    mCutsSet = true;
    mZvtxMax = zvtxMax;
    mCheckTrigger = checkTrigger;
    mTrigger = static_cast<triggerAliases>(trig);
    mCheckOffline = checkOffline;
  }

  /// Initializes histograms for the task
  void init(HistogramRegistry* registry)
  {
    if (!mCutsSet) {
      LOGF(error, "Event selection not set - quitting!");
    }
    mHistogramRegistry = registry;
    mHistogramRegistry->add("Event/zvtxhist", "; vtx_{z} (cm); Entries", kTH1F, {{1000, -15, 15}});
    mHistogramRegistry->add("Event/MultV0M", "; vMultV0M; Entries", kTH1F, {{1000, 0, 1000}});
  }

  void printCuts()
  {
    std::cout << "Debug information for FemtoDreamCollisionSelection \n Max. z-vertex: " << mZvtxMax << "\n Check trigger: " << mCheckTrigger << "\n Trigger: " << mTrigger << "\n Check offline: " << mCheckOffline << "\n";
  }

  template <typename T>
  bool isSelected(T const& col);

  template <typename T>
  void fillQA(T const& col);

  /// \todo to be implemented!
  template <typename T1, typename T2>
  float computeSphericity(T1 const& col, T2 const& tracks);

 private:
  bool mCutsSet = false;           ///< Protection against running without cuts
  float mZvtxMax = 999.f;          ///< Maximal deviation from nominal z-vertex (cm)
  bool mCheckTrigger = false;      ///< Check for trigger
  triggerAliases mTrigger = kINT7; ///< Trigger to check for
  bool mCheckOffline = false;      ///< Check for offline criteria (might change)

  HistogramRegistry* mHistogramRegistry = nullptr; ///< For QA output

  ClassDefNV(FemtoDreamCollisionSelection, 1);
};

template <typename T>
void FemtoDreamCollisionSelection::fillQA(T const& col)
{
  if (mHistogramRegistry) {
    mHistogramRegistry->fill(HIST("Event/zvtxhist"), col.posZ());
    mHistogramRegistry->fill(HIST("Event/MultV0M"), col.multV0M());
  }
}

template <typename T>
bool FemtoDreamCollisionSelection::isSelected(T const& col)
{
  if (std::abs(col.posZ()) > mZvtxMax) {
    return false;
  }

  if (mCheckTrigger && col.alias()[mTrigger] != 1) {
    return false;
  }

  if (mCheckOffline && col.sel7() != 1) {
    return false;
  }

  return true;
}

template <typename T1, typename T2>
float FemtoDreamCollisionSelection::computeSphericity(T1 const& col, T2 const& tracks)
{
  return 2.f;
}

} /* namespace femtoDream */
} /* namespace o2::analysis */

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCOLLISIONSELECTION_H_ */
