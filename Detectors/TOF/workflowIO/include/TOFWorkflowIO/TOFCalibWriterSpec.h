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

/// @file   TOFCalibWriterSpec.h

#ifndef TOFWORKFLOW_TOFCALIBWRITER_H_
#define TOFWORKFLOW_TOFCALIBWRITER_H_

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

/// create a processor spec
/// write TOF calbi info in a root file
o2::framework::DataProcessorSpec getTOFCalibWriterSpec(const char* outdef = "o2calib_tof.root", bool toftpc = false, bool addDia = false, bool onlyDia = false);

} // namespace tof
} // namespace o2

#endif /* TOFWORKFLOW_TOFCALIBWRITER_H_ */
