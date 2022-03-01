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

/// \file MCHChannelCalibrator.h
/// \brief Transitional compatibility header
///
/// \author Andrea Ferrero

#ifndef O2_MCH_CALIBRATION_CHANNEL_CALIBRATOR_H_
#define O2_MCH_CALIBRATION_CHANNEL_CALIBRATOR_H_

#include "MCHCalibration/BadChannelCalibrator.h"

namespace o2::mch::calibration
{
using MCHChannelCalibrator [[deprecated("Use BadChannelCalibrator instead")]] = BadChannelCalibrator;
using PedestalProcessor [[deprecated("Use PedestalData instead")]] = PedestalData;
using ChannelPedestal [[deprecated("Use PedestalChannel instead")]] = PedestalChannel;
} // namespace o2::mch::calibration
#endif
