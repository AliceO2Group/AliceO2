// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSelectorPID.h
/// \brief PID track selector class
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#ifndef O2_ANALYSIS_TRACKSELECTORPID_H_
#define O2_ANALYSIS_TRACKSELECTORPID_H_

#include <tuple>
#include <vector>
#include <array>
#include <cmath>

#include <TDatabasePDG.h>
#include <TPDGCode.h>

#include "CommonConstants/MathConstants.h"
#include "Framework/Logger.h"

using std::array;
using namespace o2::constants::math;

/// Base class for calculating properties of reconstructed decays.

class TrackSelectorPID
{
 public:
  /// Default constructor
  TrackSelectorPID() = default;

  /// Constructor with PDG code initialisation
  explicit TrackSelectorPID(int pdg);

  /// Default destructor
  ~TrackSelectorPID() = default;

  /// PDG code setter
  void setPDG(int pdg) { mPdg = std::abs(pdg); }

 private:
  uint mPdg = kPiPlus; // PDG code of the expected particle

  ClassDef(TrackSelectorPID, 1);
};

#endif // O2_ANALYSIS_TRACKSELECTORPID_H_
