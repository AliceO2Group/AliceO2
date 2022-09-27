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

/// @file  FITDCSDataProcessor.h
/// @brief Task for processing FIT DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FIT_DCSDATAPROCESSOR_H
#define O2_FIT_DCSDATAPROCESSOR_H

#include "DetectorsDCS/DataPointIdentifier.h"
#include "FITDCSMonitoring/FITDCSDataReader.h"
#include "Framework/DataAllocator.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace o2
{
namespace fit
{

/// Virtual task for processing FIT DCS data
///
/// Virtual task for processing FIT DCS data. Each subdetector implements
/// a sub-class with detector specific functionality, e.g. the hard coded
/// DP IDs.
class FITDCSDataProcessor : public o2::framework::Task
{
 public:
  FITDCSDataProcessor(const std::string& detectorName, const o2::header::DataDescription& dataDescription)
    : mDetectorName(detectorName),
      mDataDescription(dataDescription) {}
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  const std::string& getDetectorName() const;
  bool getVerboseMode() const;
  void setVerboseMode(bool verboseMode = true);

 protected:
  /// Gets the DP IDs. They are hard coded in the function implementations
  /// as an alternative to fetch them from CCDB.
  virtual std::vector<o2::dcs::DataPointIdentifier> getHardCodedDPIDs() = 0;

  std::string mDetectorName; ///< Detector name
  bool mVerbose = false;     ///< Verbose mode

 private:
  /// Send the DP output
  void sendDPsOutput(o2::framework::DataAllocator& output);

  std::unique_ptr<o2::fit::FITDCSDataReader> mDataReader;
  std::chrono::high_resolution_clock::time_point mTimer;
  int64_t mDPsUpdateInterval;
  o2::header::DataDescription mDataDescription; ///< DataDescription for the DCS DPs

}; // end class

} // namespace fit
} // namespace o2

#endif // O2_FIT_DCSDATAPROCESSOR_H
