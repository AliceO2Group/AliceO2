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

/// \file   MID/Workflow/src/RawCheckerSpec.cxx
/// \brief  Data processor spec for MID raw bare checker device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   06 April 2020

#include "MIDWorkflow/RawCheckerSpec.h"

#include <fstream>
#include <chrono>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ParallelContext.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDQC/RawDataChecker.h"

namespace o2
{
namespace mid
{

template <typename RAWCHECKER>
std::string getSummary(const RAWCHECKER& checker, size_t maxErrors)
{
  std::stringstream ss;
  if (checker.getNEventsFaulty() >= maxErrors) {
    ss << "Too many errors found (" << checker.getNEventsFaulty() << "): abort check!\n";
  }
  ss << "Number of busy raised: " << checker.getNBusyRaised() << "\n";
  ss << "Fraction of faulty events: " << checker.getNEventsFaulty() << " / " << checker.getNEventsProcessed() << " = " << static_cast<double>(checker.getNEventsFaulty()) / ((checker.getNEventsProcessed() > 0) ? static_cast<double>(checker.getNEventsProcessed()) : 1.);
  return ss.str();
}

template <typename RAWCHECKER>
class RawCheckerDeviceDPL
{
 public:
  RawCheckerDeviceDPL<RAWCHECKER>(const std::vector<uint16_t>& feeIds, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay) : mFeeIds(feeIds), mCrateMasks(crateMasks), mElectronicsDelay(electronicsDelay) {}

  void init(o2::framework::InitContext& ic)
  {

    auto syncTrigger = ic.options().get<int>("mid-checker-sync-trigger");
    mChecker.setSyncTrigger(syncTrigger);

    auto outFilename = ic.options().get<std::string>("mid-checker-outfile");
    mChecker.setElectronicsDelay(mElectronicsDelay);
    if constexpr (std::is_same_v<RAWCHECKER, RawDataChecker>) {
      mChecker.init(mCrateMasks);
      if (outFilename.empty()) {
        outFilename = "raw_checker_out.txt";
      }
    } else {
      auto idx = ic.services().get<o2::framework::ParallelContext>().index1D();
      auto feeId = mFeeIds[idx];
      mChecker.init(feeId, mCrateMasks.getMask(feeId));
      if (outFilename.empty()) {
        std::stringstream ss;
        ss << "raw_checker_out_GBT_" << feeId << ".txt";
        outFilename = ss.str();
      }
    }

    mOutFile.open(outFilename.c_str());

    auto stop = [this]() {
      if constexpr (std::is_same_v<RAWCHECKER, RawDataChecker>) {
        if (!mChecker.checkMissingLinks()) {
          mOutFile << mChecker.getDebugMessage() << "\n";
        }
      }
      bool hasProcessed = (mChecker.getNEventsProcessed() > 0);
      double scaleFactor = (mChecker.getNEventsProcessed() > 0) ? 1.e6 / static_cast<double>(mChecker.getNEventsProcessed()) : 0.;
      LOG(info) << "Processing time / " << mChecker.getNEventsProcessed() << " BCs: full: " << mTimer.count() * scaleFactor << " us  checker: " << mTimerAlgo.count() * scaleFactor << " us";
      std::string summary = getSummary(mChecker, mMaxErrors);
      mOutFile << summary << "\n";
      LOG(info) << summary;
    };
    ic.services().get<o2::framework::CallbackService>().set(o2::framework::CallbackService::Id::Stop, stop);

    mMaxErrors = ic.options().get<int>("mid-checker-max-errors");
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    if (mChecker.getNEventsFaulty() >= mMaxErrors) {
      // Abort checking: too many errors found
      return;
    }

    auto tStart = std::chrono::high_resolution_clock::now();

    auto msg = pc.inputs().getByPos(0);
    auto data = o2::framework::DataRefUtils::as<const ROBoard>(msg);

    auto msgROF = pc.inputs().getByPos(1);
    auto inROFRecords = o2::framework::DataRefUtils::as<const ROFRecord>(msgROF);

    std::vector<ROFRecord> dummy;
    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    if (!mChecker.process(data, inROFRecords, dummy)) {
      mOutFile << mChecker.getDebugMessage() << "\n";
    }
    mTimerAlgo += std::chrono::high_resolution_clock::now() - tAlgoStart;
    mTimer += std::chrono::high_resolution_clock::now() - tStart;
  }

 private:
  RAWCHECKER mChecker{};                       ///< Raw data checker
  std::vector<uint16_t> mFeeIds{};             ///< Vector of FEE IDs
  CrateMasks mCrateMasks{};                    ///< Crate masks file
  ElectronicsDelay mElectronicsDelay{};        ///< Delay in the electronics
  size_t mMaxErrors{0};                        ///< Maximum number of errors
  std::ofstream mOutFile{};                    ///< Output file
  std::chrono::duration<double> mTimer{0};     ///< full timer
  std::chrono::duration<double> mTimerAlgo{0}; ///< algorithm timer
};

framework::DataProcessorSpec getRawCheckerSpec(const std::vector<uint16_t>& feeIds, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, bool perGBT)
{
  std::vector<o2::framework::InputSpec> inputSpecs{o2::framework::InputSpec{"mid_decoded", header::gDataOriginMID, "DECODED", 0, o2::framework::Lifetime::Timeframe}, o2::framework::InputSpec{"mid_decoded_rof", header::gDataOriginMID, "DECODEDROF", 0, o2::framework::Lifetime::Timeframe}};

  return o2::framework::DataProcessorSpec{
    "MIDRawDataChecker",
    {inputSpecs},
    {o2::framework::Outputs{}},
    perGBT ? o2::framework::AlgorithmSpec{
               o2::framework::adaptFromTask<RawCheckerDeviceDPL<GBTRawDataChecker>>(feeIds, crateMasks, electronicsDelay)}
           : o2::framework::adaptFromTask<RawCheckerDeviceDPL<RawDataChecker>>(feeIds, crateMasks, electronicsDelay),
    o2::framework::Options{{"mid-checker-sync-trigger", o2::framework::VariantType::Int, 0x1, {"Trigger used for synchronisation (default is orbit 0x1)"}}, {"mid-checker-max-errors", o2::framework::VariantType::Int, 10000, {"Maximum number of errors"}}, {"mid-checker-outfile", o2::framework::VariantType::String, "", {"Checker output file"}}}};
}

} // namespace mid
} // namespace o2
