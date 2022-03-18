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

/// @file   InterCalibSpec.h
/// @brief  Convert ZDC data to CTF (EncodedBlocks)
/// @author pietro.cortese@cern.ch

#ifndef O2_ZDC_ENERGYCALIB_SPEC
#define O2_ZDC_ENERGYCALIB_SPEC

#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <TStopwatch.h>
#include "CommonUtils/NameConf.h"

namespace o2
{
namespace zdc
{

class EnergyCalibSpec : public o2::framework::Task
{
 public:
  EnergyCalibSpec();
  EnergyCalibSpec(const int verbosity);
  ~EnergyCalibSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  std::string mccdbHost{o2::base::NameConf::getCCDBServer()}; // Alternative ccdb server
  int mVerbosity = 0;                                         // Verbosity level during recostruction
  bool mInitialized = false;                                  // Connect once to CCDB during initialization
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getEnergyCalibSpec(const int verbosity);

} // namespace zdc
} // namespace o2

#endif
