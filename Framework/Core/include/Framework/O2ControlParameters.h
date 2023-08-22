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

#ifndef O2_FRAMEWORK_O2CONTROLPARAMETERS_H
#define O2_FRAMEWORK_O2CONTROLPARAMETERS_H

#include "Framework/DataProcessorSpec.h"
// TODO: merge the header below with this one once downstream includes <Framework/O2ControlParameters.h>
#include "Framework/O2ControlLabels.h"

namespace o2::framework
{

/// DataProcessorMetadata which are recognized by the --o2-control dump tool
/// and influence its output.
namespace ecs
{

// This key will demand AliECS to kill the task if it uses more CPU than the specified number in the value.
// The value should be a string with a floating point number, where "1.0" corresponds to 100% usage of one CPU.
const extern decltype(DataProcessorMetadata::key) cpuKillThreshold;

// This key will demand AliECS to kill the task if it uses more private memory than the specified number in the value.
// The value should be a string with a positive floating-point number or an integer (e.g. "128")
const extern decltype(DataProcessorMetadata::key) privateMemoryKillThresholdMB;

} // namespace ecs

} // namespace o2::framework

#endif // O2_FRAMEWORK_O2CONTROLPARAMETERS_H
