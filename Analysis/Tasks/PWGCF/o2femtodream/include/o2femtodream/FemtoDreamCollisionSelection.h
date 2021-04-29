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
/// \brief Definition of the FemtoDreamCollisionSelection
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCOLLISIONSELECTION_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCOLLISIONSELECTION_H_

#include "AnalysisCore/TriggerAliases.h"
//#include "Framework/HistogramRegistry.h"
//#include "Framework/Expressions.h"

#include <Rtypes.h>

#include <string>
#include <iostream>

//using namespace o2::framework;
//using namespace o2::framework::expressions;

namespace o2::analysis
{
namespace femtoDream
{

/// \class FemtoDreamCollisionSelection
/// \brief Cut class to contain and execute all cuts applied to events
class FemtoDreamCollisionSelection
{
 public:
  FemtoDreamCollisionSelection();
  FemtoDreamCollisionSelection(float zvtxMax, bool checkTrigger, triggerAliases trig, bool checkOffline);
  virtual ~FemtoDreamCollisionSelection() = default;

  /// Initialized histograms for the task
  void init(); //HistogramRegistry* registry);

  //const Filter AODFilter()
  //{
  //  return (nabs(o2::aod::collision::posZ) < mZvtxMax);
  //}

  template <typename T>
  bool isSelected(T const& col);

  template <typename T>
  void fillQA(T const& col);

  /// \todo to be implemented!
  template <typename T1, typename T2>
  float computeSphericity(T1 const& col, T2 const& tracks);

  static std::string getCutHelp();

  void printCuts();

 private:
  float mZvtxMax;          ///< Maximal deviation from nominal z-vertex (cm)
  bool mCheckTrigger;      ///< Check for trigger
  triggerAliases mTrigger; ///< Trigger to check for
  bool mCheckOffline;      ///< Check for offline criteria (might change)

  //HistogramRegistry* mHistogramRegistry; ///< For QA output
  bool mDoQA;                            ///< Switch for protection

  ClassDefNV(FemtoDreamCollisionSelection, 1);
};

template <typename T>
inline bool FemtoDreamCollisionSelection::isSelected(T const& col)
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

template <typename T>
inline void FemtoDreamCollisionSelection::fillQA(T const& col)
{
  if (mDoQA) {
    //mHistogramRegistry->fill(HIST("Event/zvtxhist"), col.posZ());
    //mHistogramRegistry->fill(HIST("Event/MultV0M"), col.multV0M());
  }
}

template <typename T1, typename T2>
inline float FemtoDreamCollisionSelection::computeSphericity(T1 const& col, T2 const& tracks)
{
  return 2.f;
}

} /* namespace femtoDream */
} /* namespace o2::analysis */

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCOLLISIONSELECTION_H_ */
