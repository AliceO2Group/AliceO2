// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoParam.h
/// \brief Error parameterizations and helper functions for TRD reconstruction
/// \author Ole Schmidt

#ifndef O2_TRD_RECOPARAM_H
#define O2_TRD_RECOPARAM_H

#include <array>
#include "Rtypes.h"

namespace o2
{
namespace trd
{

class RecoParam
{
 public:
  RecoParam() = default;
  RecoParam(const RecoParam&) = default;
  ~RecoParam() = default;

  /// Load parameterization for given magnetic field
  void setBfield(float bz);

  /// Recalculate tracklet covariance based on phi angle of related track
  void recalcTrkltCov(const float tilt, const float snp, const float rowSize, std::array<float, 3>& cov) const;

  /// Get tracklet r-phi resolution for given phi angle
  /// Resolution depends on the track angle sin(phi) = snp and is approximated by the formula
  /// sigma_y(snp) = sqrt(a^2 + c^2 * (snp - b^2)^2)
  /// more details are given in http://cds.cern.ch/record/2724259 in section 5.3.3
  /// \param phi angle of related track
  /// \return sigma_y^2 of tracklet
  float getRPhiRes(float snp) const { return (mA2 + mC2 * (snp - mB) * (snp - mB)); }

  /// Get tracklet z correction coefficient for track-eta based corraction
  float getZCorrCoeffNRC() const { return mZCorrCoefNRC; }

 private:
  // tracklet error parameterization depends on the magnetic field
  float mA2{1.f};            ///< parameterization for tracklet position resolution
  float mB{0.f};             ///< parameterization for tracklet position resolution
  float mC2{0.f};            ///< parameterization for tracklet position resolution
  float mZCorrCoefNRC{1.4f}; ///< tracklet z-position depends linearly on track dip angle

  ClassDefNV(RecoParam, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_RECOPARAM_H
