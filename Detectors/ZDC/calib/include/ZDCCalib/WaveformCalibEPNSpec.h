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

/// @file   InterCalibEPNSpec.h
/// @brief  ZDC intercalibration pre-processing on EPN
/// @author pietro.cortese@cern.ch

#ifndef O2_ZDC_WAVEFORMCALIBEPN_SPEC
#define O2_ZDC_WAVEFORMCALIBEPN_SPEC

#include <TStopwatch.h>
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "CommonUtils/NameConf.h"
#include "ZDCCalib/WaveformCalibData.h"
#include "ZDCCalib/WaveformCalibEPN.h"
#include "ZDCCalib/WaveformCalibConfig.h"

namespace o2
{
namespace zdc
{

class WaveformCalibEPNSpec : public o2::framework::Task
{
 public:
  WaveformCalibEPNSpec();
  WaveformCalibEPNSpec(const int verbosity);
  ~WaveformCalibEPNSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  int mVerbosity = 0;        // Verbosity level
  bool mInitialized = false; // Connect once to CCDB during initialization
  WaveformCalibEPN mWorker;  // Waveform calibration object
  TStopwatch mTimer;
};

framework::DataProcessorSpec getWaveformCalibEPNSpec();

} // namespace zdc
} // namespace o2

#endif
