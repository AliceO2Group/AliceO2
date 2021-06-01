// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamObjectSelection.h
/// \brief FemtoDreamObjectSelection - Partent class of all selections
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMOBJECTSELECTION_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMOBJECTSELECTION_H_

#include "FemtoDreamSelection.h"

#include "ReconstructionDataFormats/PID.h"
#include "Framework/HistogramRegistry.h"
#include <Rtypes.h>

#include <string>
#include <cmath>

using namespace o2::framework;

namespace o2::analysis
{
namespace femtoDream
{

/// \class FemtoDreamObjectSelection
/// \brief Cut class to contain and execute all cuts applied to tracks

template <class T1, class T2>
class FemtoDreamObjectSelection
{
 public:
  virtual ~FemtoDreamObjectSelection() = default;

  /// Initializes histograms for the task
  void init(HistogramRegistry* registry)
  {
    mHistogramRegistry = registry;
  }

  void fillSelectionHistogram(const char* name)
  {
    int nBins = mSelections.size();
    mHistogramRegistry->add(name, "; Cut; Value", kTH1F, {{nBins, 0, static_cast<double>(nBins)}});
    auto hist = mHistogramRegistry->get<TH1>(HIST("TrackCuts/cuthist"));
    for (size_t i = 0; i < mSelections.size(); ++i) {
      hist->GetXaxis()->SetBinLabel(i + 1, Form("%u", mSelections.at(i).getSelectionVariable()));
      hist->SetBinContent(i + 1, mSelections.at(i).getSelectionValue());
    }
  }

  template <typename T>
  void setSelection(T& selVals, T2 selVar, femtoDreamSelection::SelectionType selType)
  {
    std::vector<T1> tmpSelVals = selVals; // necessary due to some features of the Configurable
    std::vector<FemtoDreamSelection<T1, T2>> tempVec;
    for (const T1 selVal : tmpSelVals) {
      tempVec.push_back(FemtoDreamSelection<T1, T2>(selVal, selVar, selType));
    }
    setSelection(tempVec);
  }

  void setSelection(std::vector<FemtoDreamSelection<T1, T2>>& sels)
  {
    switch (sels.at(0).getSelectionType()) {
      case (femtoDreamSelection::SelectionType::kUpperLimit):
      case (femtoDreamSelection::SelectionType::kAbsUpperLimit):
        std::sort(sels.begin(), sels.end(), [](FemtoDreamSelection<T1, T2> a, FemtoDreamSelection<T1, T2> b) {
          return a.getSelectionValue() > b.getSelectionValue();
        });
        break;
      case (femtoDreamSelection::SelectionType::kLowerLimit):
      case (femtoDreamSelection::SelectionType::kAbsLowerLimit):
      case (femtoDreamSelection::SelectionType::kEqual):
        std::sort(sels.begin(), sels.end(), [](FemtoDreamSelection<T1, T2> a, FemtoDreamSelection<T1, T2> b) {
          return a.getSelectionValue() < b.getSelectionValue();
        });
        break;
    }
    for (const auto sel : sels) {
      mSelections.push_back(sel);
    }
  }

  T1 getMinimalSelection(T2 selVar, femtoDreamSelection::SelectionType selType)
  {
    T1 minimalSel;
    switch (selType) {
      case (femtoDreamSelection::SelectionType::kUpperLimit):
      case (femtoDreamSelection::SelectionType::kAbsUpperLimit):
        minimalSel = -999.e9;
        break;
      case (femtoDreamSelection::SelectionType::kLowerLimit):
      case (femtoDreamSelection::SelectionType::kAbsLowerLimit):
      case (femtoDreamSelection::SelectionType::kEqual):
        minimalSel = 999.e9;
        break;
    }

    for (auto sel : mSelections) {
      if (sel.getSelectionVariable() == selVar) {
        switch (sel.getSelectionType()) {
          case (femtoDreamSelection::SelectionType::kUpperLimit):
          case (femtoDreamSelection::SelectionType::kAbsUpperLimit):
            if (minimalSel < sel.getSelectionValue()) {
              minimalSel = sel.getSelectionValue();
            }
            break;
          case (femtoDreamSelection::SelectionType::kLowerLimit):
          case (femtoDreamSelection::SelectionType::kAbsLowerLimit):
          case (femtoDreamSelection::SelectionType::kEqual):
            if (minimalSel > sel.getSelectionValue()) {
              minimalSel = sel.getSelectionValue();
            }
            break;
        }
      }
    }
    return minimalSel;
  }

  size_t getNSelections()
  {
    return mSelections.size();
  }

  size_t getNSelections(T2 selVar)
  {
    size_t counter = 0;
    for (auto it : mSelections) {
      if (it.getSelectionVariable() == selVar) {
        ++counter;
      }
    }
    return counter;
  }

 protected:
  std::vector<FemtoDreamSelection<T1, T2>> mSelections;
  HistogramRegistry* mHistogramRegistry; ///< For QA output

  ClassDefNV(FemtoDreamObjectSelection, 1);
};

} // namespace femtoDream
} // namespace o2::analysis

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMOBJECTSELECTION_H_ */
