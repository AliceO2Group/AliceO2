// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_CALIBVDRIFT_H_
#define ALICEO2_TRD_CALIBVDRIFT_H_

/// \file   CalibVDrift.h
/// \author Ole Schmidt, ole.schmidt@cern.ch

#include "DataFormatsTRD/Constants.h"
#include <array>

namespace o2
{
namespace trd
{

/// \brief VDrift calibration class
///
/// This class is used to determine chamber-wise vDrift values
///
/// origin: TRD
/// \author Ole Schmidt, ole.schmidt@cern.ch

class CalibVDrift
{
 public:
  /// default constructor
  CalibVDrift() = default;

  /// default destructor
  ~CalibVDrift() = default;

  /// set input angular difference sums
  void setAngleDiffSums(float* input)
  {
    for (int i = 0; i < constants::MAXCHAMBER * constants::NBINSANGLEDIFF; ++i) {
      mAngleDiffSums[i] = input[i];
    }
  }

  /// set input angular difference bin counters
  void setAngleDiffCounters(short* input)
  {
    for (int i = 0; i < constants::MAXCHAMBER * constants::NBINSANGLEDIFF; ++i) {
      mAngleDiffCounters[i] = input[i];
    }
  }

  /// main processing function
  void process();

 private:
  std::array<float, constants::MAXCHAMBER * constants::NBINSANGLEDIFF> mAngleDiffSums{};     ///< input TRD track to tracklet angular difference sums per bin
  std::array<short, constants::MAXCHAMBER * constants::NBINSANGLEDIFF> mAngleDiffCounters{}; ///< input bin counters
};

} // namespace trd

} // namespace o2
#endif
