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

/// @file   StandaloneAODProducerSpec.h
/// @brief  Convert CTF (EncodedBlocks) to AO2D PHOS standalone

#ifndef O2_PHOS_STANDALONEAODPRODUCERSPEC_SPEC
#define O2_PHOS_STANDALONEAODPRODUCERSPEC_SPEC

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <TStopwatch.h>

namespace o2
{
namespace phos
{

class StandaloneAODProducerSpec : public o2::framework::Task
{
 public:
  StandaloneAODProducerSpec();
  ~StandaloneAODProducerSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  int64_t mTFNumber = -1;          // Timeframe ID
  int mRunNumber = -1;             // Run number
  uint32_t mCaloAmp = 0xFFFFFF00;  // 15 bits
  uint32_t mCaloTime = 0xFFFFFF00; // 15 bits
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getPHOSStandaloneAODProducerSpec();

} // namespace phos
} // namespace o2

#endif