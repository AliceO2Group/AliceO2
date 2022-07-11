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

/// @file  FDDDCSDataProcessor.h
/// @brief Task for processing FDD DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FDD_DATAPROCESSOR_H
#define O2_FDD_DATAPROCESSOR_H

#include "DetectorsDCS/DataPointIdentifier.h"
#include "FITDCSMonitoring/FITDCSDataProcessor.h"

#include <string>
#include <vector>

namespace o2
{
namespace fdd
{

class FDDDCSDataProcessor : public o2::fit::FITDCSDataProcessor
{
 public:
  FDDDCSDataProcessor(const std::string detectorName, const o2::header::DataDescription& dataDescription)
    : o2::fit::FITDCSDataProcessor(detectorName, dataDescription) {}

 protected:
  std::vector<o2::dcs::DataPointIdentifier> getHardCodedDPIDs() override;
}; // end class

} // namespace fdd
} // namespace o2

#endif // O2_FDD_DATAPROCESSOR_H
