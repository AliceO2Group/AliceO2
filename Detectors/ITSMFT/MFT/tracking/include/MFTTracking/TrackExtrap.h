// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackExtrap.h
/// \brief Definition of tools for track extrapolation
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#ifndef ALICEO2_MFT_TRACKEXTRAP_H_
#define ALICEO2_MFT_TRACKEXTRAP_H_

#include <cstddef>

#include <TMatrixD.h>
#include "MFTTracking/MFTTrackingParam.h"

namespace o2
{
namespace mft
{

class TrackParamMFT;

/// Class holding tools for track extrapolation
class TrackExtrap
{
 public:
  // static class
  TrackExtrap() = default;
  ~TrackExtrap() = default;

  TrackExtrap(const TrackExtrap&) = delete;
  TrackExtrap& operator=(const TrackExtrap&) = delete;
  TrackExtrap(TrackExtrap&&) = delete;
  TrackExtrap& operator=(TrackExtrap&&) = delete;

  void setBz(float bZ) { mBZField = bZ; } /// Set the magnetic field for the MFT
  const float getBz() const { return mBZField; }
  const int getSignBz() const { return std::copysign(1, mBZField); }
  /// Return true if the field is switched ON
  const bool isFieldON() { return mIsFieldON; }

  bool extrapToZ(TrackParamMFT* TrackParamMFT, double zEnd, bool isFieldON = true);
  void extrapToZCov(TrackParamMFT* TrackParamMFT, double zEnd, bool updatePropagator = false, bool isFieldON = true);
  void linearExtrapToZ(TrackParamMFT* TrackParamMFT, double zEnd);
  void linearExtrapToZCov(TrackParamMFT* TrackParamMFT, double zEnd, bool updatePropagator);
  void quadraticExtrapToZ(TrackParamMFT* TrackParamMFT, double zEnd);
  void quadraticExtrapToZCov(TrackParamMFT* TrackParamMFT, double zEnd, bool updatePropagator);
  void helixExtrapToZ(TrackParamMFT* TrackParamMFT, double zEnd);
  void helixExtrapToZCov(TrackParamMFT* TrackParamMFT, double zEnd, bool updatePropagator);
  void addMCSEffect(TrackParamMFT* TrackParamMFT, double dZ, double x0);

 private:
  Float_t mBZField;        // kiloGauss.
  bool mIsFieldON = false; ///< true if the field is switched ON
};

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TRACKEXTRAP_H_
