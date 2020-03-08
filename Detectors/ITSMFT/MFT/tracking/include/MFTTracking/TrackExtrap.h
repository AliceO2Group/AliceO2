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

namespace o2
{
namespace mft
{

class TrackParam;

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

  void setBz(float bZ) { bFieldZ = bZ; } /// Set the magnetic field for the MFT
  float getBz() { return bFieldZ; }

  /// Return true if the field is switched ON
  static bool isFieldON() { return sFieldON; }

  static bool extrapToZ(TrackParam* trackParam, double zEnd, bool isFieldON = true);
  static bool extrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator = false, bool isFieldON = true);
  static void linearExtrapToZ(TrackParam* trackParam, double zEnd);
  static void linearExtrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator);
  static void helixExtrapToZ(TrackParam* trackParam, double zEnd);
  static void helixExtrapToZCov(TrackParam* trackParam, double zEnd, bool updatePropagator);
  static void addMCSEffect(TrackParam* trackParam, double dZ, double x0, bool isFieldON = true);

 private:
  static Float_t bFieldZ; // Tesla.
  static bool sFieldON;   ///< true if the field is switched ON
};

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TRACKEXTRAP_H_
