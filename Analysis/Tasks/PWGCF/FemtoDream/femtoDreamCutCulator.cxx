// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file femtoDreamCutCulator.cxx
/// \brief Executable that encodes physical selection criteria in a bit-wise selection
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "include/FemtoDream/FemtoDerived.h"
#include "include/FemtoDream/FemtoDreamSelection.h"
#include "include/FemtoDream/FemtoDreamTrackSelection.h"
#include "include/FemtoDream/FemtoDreamCutculator.h"
#include <iostream>

using namespace o2::analysis::femtoDream;

/// The function takes the path to the dpl-config.json as a argument and the does a Q&A session for the user to find the appropriate selection criteria for the analysis task
int main(int argc, char* argv[])
{
  FemtoDreamCutculator cut;
  cut.init(argv[1]);

  cut.setTrackSelection(femtoDreamTrackSelection::kSign, femtoDreamSelection::kEqual, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kpTMin, femtoDreamSelection::kLowerLimit, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kpTMax, femtoDreamSelection::kUpperLimit, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kEtaMax, femtoDreamSelection::kAbsUpperLimit, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kTPCnClsMin, femtoDreamSelection::kLowerLimit, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kTPCfClsMin, femtoDreamSelection::kLowerLimit, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kTPCsClsMax, femtoDreamSelection::kUpperLimit, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kDCAxyMax, femtoDreamSelection::kAbsUpperLimit, "ConfTrk");
  cut.setTrackSelection(femtoDreamTrackSelection::kDCAzMax, femtoDreamSelection::kAbsUpperLimit, "ConfTrk");

  /// \todo factor out the pid here
  // cut.setTrackSelection(femtoDreamTrackSelection::kPIDnSigmaMax, femtoDreamSelection::kAbsUpperLimit, "ConfTrk");

  cut.analyseCuts();
}
