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

/// \file
/// \brief
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDX_H_
#define ALICEO2_TPC_CALIBDEDX_H_

#include <array>
#include <cstddef>

// o2 includes
#include "TPCCalibration/CalibdEdxBase.h"

namespace o2::tpc
{

// forward declarations
class CalibdEdxHistos;

/// Container used to store the dE/dx calibration output
struct CalibdEdx : public CalibdEdxBase<float> {
 public:
  using MIPArray = std::array<float, totalStacks>;

  CalibdEdx() = default;
  CalibdEdx(const CalibdEdxHistos& histos) { process(histos); };

  /// \brief Compute the center of gravity for every GEM stack histogram.
  void process(const CalibdEdxHistos&);
};

} // namespace o2::tpc
#endif
