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

/// \file   MIDWorkflow/ChannelCalibratorSpec.h
/// \brief  Noise and dead channels calibrator spec for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   21 February 2022

#ifndef O2_MID_ChannelCalibratorSPEC_H
#define O2_MID_ChannelCalibratorSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/FEEIdConfig.h"

namespace o2
{
namespace mid
{
framework::DataProcessorSpec getChannelCalibratorSpec(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks);
}
} // namespace o2

#endif // O2_MID_ChannelCalibratorSPEC_H
