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

#ifndef O2_TRD_PADCALIBALIASES_H
#define O2_TRD_PADCALIBALIASES_H

///////////////////////////////////////////////////////////////////////////////
// Store for the pad status across the entire TRD.
// PadCalibrations stores the roc, and PadParameters the actual content of
//  statuses.
//
// Note to run2 correlation:
// This then matches similarly to the old AliTRDCalPadStatus which held an
//  array of AliTRDCalSingleChamberStatus.
// Only real change to the interface is that one now needs to explicitly
//  state the roc in question when querying a pads status, as the method is
//  higher up in the heirachy now.
///////////////////////////////////////////////////////////////////////////////

/// \file PadCalibrationsAliases.h
/// \brief Aliases for calibration values stored on a per-pad basis

#include "TRDBase/PadCalibrations.h"

namespace o2
{
namespace trd
{

class PadStatus : public PadCalibrations<char>
{
 public:
  enum { kMasked = 2,
         kPadBridgedLeft = 4,
         kPadBridgedRight = 8,
         kReadSecond = 16,
         kNotConnected = 32 };
  bool isMasked(int roc, int col, int row) const { return ((getValue(roc, col, row) & kMasked) ? true : false); };
  bool isBridgedLeft(int roc, int col, int row) const { return ((getValue(roc, col, row) & kPadBridgedLeft) ? true : false); };
  bool isBridgedRight(int roc, int col, int row) const { return ((getValue(roc, col, row) & kPadBridgedRight) ? true : false); };
  bool isNotConnected(int roc, int col, int row) const { return ((getValue(roc, col, row) & kNotConnected) ? true : false); };
};

using LocalT0 = PadCalibrations<float>;
using LocalVDrift = PadCalibrations<float>;
using LocalGainFactor = PadCalibrations<float>;
using PadNoise = PadCalibrations<float>;

} // namespace trd
} // namespace o2

#endif // O2_TRD_PADCALIBALIASES_H
