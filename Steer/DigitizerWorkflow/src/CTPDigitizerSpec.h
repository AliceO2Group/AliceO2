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

#ifndef STEER_DIGITIZERWORKFLOW_CTPDIGITIZER_H_
#define STEER_DIGITIZERWORKFLOW_CTPDIGITIZER_H_

#include "Framework/DataProcessorSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace ctp
{

o2::framework::DataProcessorSpec getCTPDigitizerSpec(int channel, std::vector<o2::detectors::DetID>& detList, bool mctruth = true);

} // namespace ctp
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_CTPDIGITIZER_H_ */
