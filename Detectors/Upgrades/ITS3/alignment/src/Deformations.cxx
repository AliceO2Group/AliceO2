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

#include "ITS3Align/Deformations.h"
#include "ITS3Align/MisalignmentParameters.h"

#include "Framework/Logger.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace o2::its3::align
{

void Deformations::init(const fs::path& path)
{
  if (!fs::exists(path)) {
    LOGP(fatal, "File {} does not exists!", path.c_str());
  }

  auto params = MisalignmentParameters::load(path.string());
  LOGP(info, "Loaded Parameters");

  // Set the legendre pols
  for (int iSensor{0}; iSensor < 6; ++iSensor) {
    mLegX[iSensor] = o2::math_utils::Legendre2DPolynominal(params->getLegendreCoeffX(iSensor));
    mLegY[iSensor] = o2::math_utils::Legendre2DPolynominal(params->getLegendreCoeffY(iSensor));
    mLegZ[iSensor] = o2::math_utils::Legendre2DPolynominal(params->getLegendreCoeffZ(iSensor));
  }
}

} // namespace o2::its3::align
