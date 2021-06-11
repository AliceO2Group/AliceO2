// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoParam.cxx
/// \brief Error parameterizations and helper functions for TRD reconstruction
/// \author Ole Schmidt

#include "TRDBase/RecoParam.h"
#include <fairlogger/Logger.h>
#include <cmath>

using namespace o2::trd;

// error parameterizations taken from http://cds.cern.ch/record/2724259 Appendix A
void RecoParam::setBfield(float bz)
{
  if (std::fabs(std::fabs(bz) - 2) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.2 T
      mA2 = 1.6e-3f;
      mB = -1.43e-2f;
      mC2 = 4.55e-2f;
    } else {
      // magnetic field -0.2 T
      mA2 = 1.6e-3f;
      mB = 1.43e-2f;
      mC2 = 4.55e-2f;
    }
  } else if (std::fabs(std::fabs(bz) - 5) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.5 T
      mA2 = 1.6e-3f;
      mB = 0.125f;
      mC2 = 0.0961f;
    } else {
      // magnetic field -0.5 T
      mA2 = 1.6e-3f;
      mB = -0.14f;
      mC2 = 0.1156f;
    }
  } else {
    LOG(WARNING) << "No error parameterization available for Bz= " << bz << ". Keeping default value (sigma_y = const. = 1cm)";
  }
}

void RecoParam::recalcTrkltCov(const float tilt, const float snp, const float rowSize, std::array<float, 3>& cov) const
{
  float t2 = tilt * tilt;      // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = getRPhiRes(snp);
  float sz2 = rowSize * rowSize / 12.f;
  cov[0] = c2 * (sy2 + t2 * sz2);
  cov[1] = c2 * tilt * (sz2 - sy2);
  cov[2] = c2 * (t2 * sy2 + sz2);
}
