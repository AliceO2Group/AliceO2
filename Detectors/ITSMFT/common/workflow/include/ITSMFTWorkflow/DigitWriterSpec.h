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

#ifndef STEER_ITSMFTDIGITWRITER_H_
#define STEER_ITSMFTDIGITWRITER_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace itsmft
{

o2::framework::DataProcessorSpec getITSDigitWriterSpec(bool mctruth = true, bool dec = false, bool calib = false);
o2::framework::DataProcessorSpec getMFTDigitWriterSpec(bool mctruth = true, bool dec = false, bool calib = false);

} // end namespace itsmft
} // end namespace o2

#endif /* STEER_ITSMFTDIGITWRITER_H_ */
