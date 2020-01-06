// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/ChamberHV.h
/// \brief  HV values for MID RPCs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 April 2018
#ifndef O2_MID_CHAMBERHV_H
#define O2_MID_CHAMBERHV_H

#include <array>
#include "MIDBase/DetectorParameters.h"

namespace o2
{
namespace mid
{
class ChamberHV
{
 public:
  /// Get HV for detection element
  double getHV(int deId) const { return mHV[deId]; }

  /// sets the HV for detection element
  void setHV(int deId, double hv) { mHV[deId] = hv; }

 private:
  std::array<double, detparams::NDetectionElements> mHV; ///< High voltage values
};

ChamberHV createDefaultChamberHV();

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHAMBERHV_H */
