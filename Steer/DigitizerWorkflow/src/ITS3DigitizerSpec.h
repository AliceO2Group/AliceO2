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

#ifndef STEER_DIGITIZERWORKFLOW_ITS3DIGITIZER_H_
#define STEER_DIGITIZERWORKFLOW_ITS3DIGITIZER_H_

#include "Framework/DataProcessorSpec.h"

namespace o2::its3
{

o2::framework::DataProcessorSpec getITS3DigitizerSpec(int channel, bool mctruth = true);

} // namespace o2::its3
  // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_ITS3DIGITIZER_H_ */
