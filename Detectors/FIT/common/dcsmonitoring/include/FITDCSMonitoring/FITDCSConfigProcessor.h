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

/// \file FITDCSConfigProcessorSpec.cxx
/// \brief FIT processor spec for DCS configurations
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FIT_DCSCONFIGPROCESSORSPEC_H
#define O2_FIT_DCSCONFIGPROCESSORSPEC_H

#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "FITDCSMonitoring/FITDeadChannelMapReader.h"
#include "FITDCSMonitoring/FITFEEConfigurationReader.h"
#include "FITDCSMonitoring/FITHvConfigurationReader.h"
#include "FITDCSMonitoring/DcsCcdbInfo.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include <chrono>
#include <gsl/span>
#include <memory>
#include <string>
#include <vector>

namespace o2
{
namespace fit
{

class FITDCSConfigProcessor : public o2::framework::Task
{
 public:
  FITDCSConfigProcessor(const std::string& detectorName, const o2::header::DataDescription& dataDescriptionDChM, const o2::header::DataDescription& dataDescriptionFeeConfig, const o2::header::DataDescription& dataDescriptionHvConfig)
    : mDetectorName(detectorName),
      mDataDescriptionDChM(dataDescriptionDChM),
      mDataDescriptionFeeConfig(dataDescriptionFeeConfig),
      mDataDescriptionHvConfig(dataDescriptionHvConfig) {} // TODO AM: how to pass dd

 protected:
  /// Initializes the DCS config reader.
  /// Can be overriden in case another reader (subclass of o2::fit::FITDeadChannelMapReader) is needed.
  virtual void initDeadChannelMapReader();

  long getValidityTime(o2::framework::ProcessingContext& pc);

  void setupDeadChannelMapReader(o2::framework::InitContext& ic);

  template <typename ConfigurationReaderType>
  void setupFeeConfigurationReader(o2::framework::InitContext& ic, FITFEEConfigurationReader<ConfigurationReaderType>& feeConfig)
  {
    feeConfig.setFilename(ic.options().get<std::string>("filename-fee-config"));
    mFeeConfigCcdbInfo.setCcdbPath(mDetectorName + "/Config/FeeConfiguration");
    mFeeConfigCcdbInfo.setValidityPeriodInDays(ic.options().get<uint32_t>("valid-days-fee-config"));
  }

  void setupHvConfigurationReader(o2::framework::InitContext& ic, FITHvConfigurationReader& hvReader);

  void handleDeadChannelMapUpdate(o2::framework::ProcessingContext& pc, long dataTime, gsl::span<const char> dataBuffer);
  void processDeadChannelMap(const long& dataTime, gsl::span<const char> dataBuffer);

  template <typename ConfigObjectType>
  void sendObject(o2::framework::DataAllocator& output, const ConfigObjectType& object, o2::ccdb::CcdbObjectInfo& info, const o2::header::DataDescription& descriptor)
  {
    auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBPayload, descriptor, 0}, *image.get());
    output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBWrapper, descriptor, 0}, info);
  }

  std::unique_ptr<FITDeadChannelMapReader> mDeadChannelMapReader;

  const o2::header::DataDescription& getFeeConfigDescription() const
  {
    return mDataDescriptionFeeConfig;
  }

  const o2::header::DataDescription& getHvConfigDescription() const
  {
    return mDataDescriptionHvConfig;
  }

  const o2::header::DataDescription& getDeadChannelMapDescription() const
  {
    return mDataDescriptionDChM;
  }

 protected:
  DcsCcdbInfo mFeeConfigCcdbInfo{180u};
  DcsCcdbInfo mHvConfigCcdbInfo{180u};

 private:
  std::string mDetectorName;                        ///< Detector name
  o2::header::DataDescription mDataDescriptionDChM; ///< DataDescription for the dead channel map
  o2::header::DataDescription mDataDescriptionFeeConfig;
  o2::header::DataDescription mDataDescriptionHvConfig;
};

} // namespace fit
} // namespace o2

#endif // O2_FIT_DCSCONFIGPROCESSORSPEC_H
