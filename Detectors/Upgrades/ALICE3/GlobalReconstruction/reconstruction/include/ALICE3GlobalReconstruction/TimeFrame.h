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
///
/// \file TimeFrame.h
/// \brief CPU TRK TimeFrame wrapper.
///

#ifndef ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAME_H
#define ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAME_H

#include "ALICE3GlobalReconstruction/TimeFrameMixin.h"
#include "ITStracking/TimeFrame.h"

namespace o2::trk
{

template <int nLayers = 11>
class TimeFrame : public TimeFrameMixin<nLayers, o2::its::TimeFrame<nLayers>>
{
 public:
  TimeFrame() = default;
  ~TimeFrame() override = default;
};

} // namespace o2::trk

#endif // ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAME_H
