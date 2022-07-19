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
#include "FITDCSMonitoring/FITDCSConfigReader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include <chrono>
#include <gsl/span>
#include <memory>
#include <string>
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace fit
{

class FITDCSConfigProcessor : public o2::framework::Task
{
 public:
  FITDCSConfigProcessor(const std::string& detectorName, const o2::header::DataDescription& dataDescriptionBChM)
    : mDetectorName(detectorName),
      mDataDescriptionBChM(dataDescriptionBChM) {} // TODO AM: how to pass dd

  void init(o2::framework::InitContext& ic) final
  {
    initDCSConfigReader();
    mDCSConfigReader->setFileNameBChM(ic.options().get<string>("filename-bchm"));
    mDCSConfigReader->setCcdbPathBChM(mDetectorName + "/Calib/BadChannelMap");
    mVerbose = ic.options().get<bool>("use-verbose-mode");
    mDCSConfigReader->setVerboseMode(mVerbose);
    LOG(info) << "Verbose mode: " << mVerbose;
    LOG(info) << "Expected bad channel map file name: " << mDCSConfigReader->getFileNameBChM();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // Get the time of the data
    auto timeNow = std::chrono::high_resolution_clock::now();
    long dataTime = (long)(pc.services().get<o2::framework::TimingInfo>().creation);
    if (dataTime == 0xffffffffffffffff) {                                                                   // means it is not set
      dataTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }

    // Get the input file
    gsl::span<const char> configBuf = pc.inputs().get<gsl::span<char>>("inputConfig");
    std::string configFileName = pc.inputs().get<std::string>("inputConfigFileName");
    LOG(info) << "Got input file " << configFileName << " of size " << configBuf.size();

    if (!configFileName.compare(mDCSConfigReader->getFileNameBChM())) {
      // Got bad channel map
      processBChM(dataTime, configBuf);
      sendBChMOutput(pc.outputs());
      mDCSConfigReader->resetStartValidityBChM();
      mDCSConfigReader->resetBChM();
    } else {
      LOG(error) << "Unknown input file: " << configFileName;
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
  }

 protected:
  /// Initializes the DCS config reader.
  /// Can be overriden in case another reader (subclass of o2::fit::FITDCSConfigReader) is needed.
  virtual void initDCSConfigReader()
  {
    mDCSConfigReader = std::make_unique<FITDCSConfigReader>(FITDCSConfigReader());
  }

  std::unique_ptr<FITDCSConfigReader> mDCSConfigReader; ///< Reader for the DCS configurations

 private:
  /// Processing the bad channel map
  void processBChM(const long& dataTime, gsl::span<const char> configBuf)
  {
    if (!mDCSConfigReader->isStartValidityBChMSet()) {
      if (mVerbose) {
        LOG(info) << "Start validity for DCS data set to = " << dataTime;
      }
      mDCSConfigReader->setStartValidityBChM(dataTime);
    }
    mDCSConfigReader->processBChM(configBuf);
    mDCSConfigReader->updateBChMCcdbObjectInfo();
  }

  /// Sending the bad channeel map output to CCDB
  void sendBChMOutput(o2::framework::DataAllocator& output)
  {
    const auto& payload = mDCSConfigReader->getBChM();
    auto& info = mDCSConfigReader->getObjectInfoBChM();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, mDataDescriptionBChM, 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, mDataDescriptionBChM, 0}, info);
  }

  std::string mDetectorName;                        ///< Detector name
  o2::header::DataDescription mDataDescriptionBChM; ///< DataDescription for the bad channel map
  bool mVerbose = false;                            ///< Verbose mode
};

} // namespace fit
} // namespace o2

#endif // O2_FIT_DCSCONFIGPROCESSORSPEC_H
