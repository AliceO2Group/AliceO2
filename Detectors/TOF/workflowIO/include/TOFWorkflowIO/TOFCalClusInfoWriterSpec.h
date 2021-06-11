// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFCalClusInfoWriterSpec.h

#ifndef STEER_DIGITIZERWORKFLOW_TOFCALCLUSINFOWRITER_H_
#define STEER_DIGITIZERWORKFLOW_TOFCALCLUSINFOWRITER_H_

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

/// create a processor spec
/// write ITS tracks a root file
o2::framework::DataProcessorSpec getTOFCalClusInfoWriterSpec(bool isCosmics = 0);

} // namespace tof
} // namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_TOFCALCLUSINFOWRITER_H_ */
