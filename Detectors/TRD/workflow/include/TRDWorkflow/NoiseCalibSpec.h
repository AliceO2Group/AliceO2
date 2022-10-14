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

#ifndef O2_TRD_NOISECALIBSPEC_H
#define O2_TRD_NOISECALIBSPEC_H

/// \file   NoiseCalibSpec.h
/// \brief Extract mean ADC values per pad from digits and send them to the aggregator

#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsTRD/NoiseCalibration.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

class TRDNoiseCalibSpec : public o2::framework::Task
{
 public:
  TRDNoiseCalibSpec() = default;
  void init(o2::framework::InitContext& ic) final
  {
    // Do we need to initialize something?
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto digits = pc.inputs().get<gsl::span<Digit>>("trddigits");
    auto trigRecs = pc.inputs().get<gsl::span<TriggerRecord>>("trdtriggerrec");

    // TODO:
    // Here we have the digits and trigger records available for a given time frame.
    // We should extract the relevant information (ADC sum per pad and number of ADC values added for each pad)
    // and store that in padAdcInfo which is then send to the aggregator were another device needs to be prepared
    // which combines the information from all different EPNs and creates the CCDB object

    PadAdcInfo padAdcInfo;

    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "PADADCS", 0, Lifetime::Timeframe}, padAdcInfo);
  }

 private:
  // Do we need any private members here, some settings for example?
};

o2::framework::DataProcessorSpec getTRDNoiseCalibSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trddigits", o2::header::gDataOriginTRD, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "trd-noise-calib",
    inputs,
    Outputs{{o2::header::gDataOriginTRD, "PADADCS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TRDNoiseCalibSpec>()},
    Options{}};
}

} // namespace trd
} // namespace o2

#endif // O2_TRD_NOISECALIBSPEC_H
