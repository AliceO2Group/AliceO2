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

/// @file  FV0DCSDataProcessor.h
/// @brief Task for processing FV0 DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FV0_DATAPROCESSOR_H
#define O2_FV0_DATAPROCESSOR_H

#include "DetectorsDCS/DataPointIdentifier.h"
#include "FITDCSMonitoring/FITDCSDataProcessor.h"

#include <string>
#include <vector>

namespace o2
{
namespace fv0
{

class FV0DCSDataProcessor : public o2::fit::FITDCSDataProcessor
{
 public:
  FV0DCSDataProcessor(const std::string detectorName, const o2::header::DataDescription& dataDescription)
    : o2::fit::FITDCSDataProcessor(detectorName, dataDescription) {}

 protected:
  std::vector<o2::dcs::DataPointIdentifier> getHardCodedDPIDs() override;
}; // end class

} // namespace fv0
} // namespace o2

#endif // O2_FV0_DATAPROCESSOR_H
