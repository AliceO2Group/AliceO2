// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDWorkflow/RawDecoderSpec.h
/// \brief  Data processor spec for MID raw decoder device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 February 2020

#ifndef O2_MID_RAWDECODERSPEC_H
#define O2_MID_RAWDECODERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"

namespace o2
{
namespace mid
{
framework::DataProcessorSpec getRawDecoderSpec(bool isDebugMode = false);
framework::DataProcessorSpec getRawDecoderSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay);
framework::DataProcessorSpec getRawDecoderSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, header::DataHeader::SubSpecificationType subSpec);
} // namespace mid
} // namespace o2

#endif //O2_MID_RAWDECODERSPEC_H
