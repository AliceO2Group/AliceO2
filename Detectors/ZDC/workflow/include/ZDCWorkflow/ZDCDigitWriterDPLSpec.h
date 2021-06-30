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

/// @file   ZDCDigitWriterDPLSpec.h

#ifndef O2_ZDCDIGITWRITERDPLSPEC_H
#define O2_ZDCDIGITWRITERDPLSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsZDC/MCLabel.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

/// create a processor spec
framework::DataProcessorSpec getZDCDigitWriterDPLSpec(bool mctruth, bool simVersion);

} // namespace zdc
} // namespace o2

#endif /* O2_ZDCDIGITWRITERDPLSPEC_H */
