// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamSelection.h
/// \brief FemtoDreamSelection - small generic class to do selections
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMSELECTION_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMSELECTION_H_

#include <Rtypes.h>

namespace o2::analysis
{
namespace femtoDream
{

namespace femtoDreamSelection
{
enum SelectionType { kUpperLimit,
                     kAbsUpperLimit,
                     kLowerLimit,
                     kAbsLowerLimit,
                     kEqual };

}

template <class T1, class T2>
class FemtoDreamSelection
{
 public:
  FemtoDreamSelection();
  FemtoDreamSelection(T1 selVal, T2 selVar, femtoDreamSelection::SelectionType selType)
    : mSelVal(selVal),
      mSelVar(selVar),
      mSelType(selType)
  {
  }

  T1 getSelectionValue() { return mSelVal; }
  T2 getSelectionVariable() { return mSelVar; }
  femtoDreamSelection::SelectionType getSelectionType() { return mSelType; }

  bool isSelected(T1 observable)
  {
    switch (mSelType) {
      case (femtoDreamSelection::SelectionType::kUpperLimit):
        return (observable < mSelVal);
      case (femtoDreamSelection::SelectionType::kAbsUpperLimit):
        return (std::abs(observable) < mSelVal);
        break;
      case (femtoDreamSelection::SelectionType::kLowerLimit):
        return (observable > mSelVal);
      case (femtoDreamSelection::SelectionType::kAbsLowerLimit):
        return (std::abs(observable) > mSelVal);
        break;
      case (femtoDreamSelection::SelectionType::kEqual):
        /// \todo can the comparison be done a bit nicer?
        return (std::abs(observable - mSelVal) < std::abs(mSelVal * 1e-6));
        break;
    }
    return false;
  }

  template <typename T>
  void checkSelectionSetBit(T1 observable, T& cutContainer, size_t& counter)
  {
    if (isSelected(observable)) {
      cutContainer |= 1UL << counter;
    }
    ++counter;
  }

 private:
  T1 mSelVal{0.f};
  T2 mSelVar;
  femtoDreamSelection::SelectionType mSelType;

  ClassDefNV(FemtoDreamSelection, 1);
};

} /* namespace femtoDream */
} /* namespace o2::analysis */

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMSELECTION_H_ */
