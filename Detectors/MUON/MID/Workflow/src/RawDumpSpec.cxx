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

/// \file   MID/Workflow/src/RawDumpSpec.cxx
/// \brief  MID raw data dumper device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 April 2022

#include "MIDWorkflow/RawDumpSpec.h"

#include <fstream>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DPLUtils/DPLRawParser.h"
#include "Headers/RDHAny.h"
#include "MIDRaw/Decoder.h"
#include "MIDWorkflow/RawInputSpecHandler.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class RawDumpDeviceDPL
{
 public:
  RawDumpDeviceDPL(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, header::DataHeader::SubSpecificationType subSpec) : mIsDebugMode(isDebugMode), mFeeIdConfig(feeIdConfig), mCrateMasks(crateMasks), mElectronicsDelay(electronicsDelay), mSubSpec(subSpec) {}

  void init(of::InitContext& ic)
  {
    auto rdhOnly = ic.options().get<bool>("rdh-only");
    auto dumpDecoded = ic.options().get<bool>("decode");
    auto dumpAll = ic.options().get<bool>("payload-and-decode");

    if (rdhOnly) {
      mDumpDecoded = false;
      mDumpPayload = false;
    } else if (dumpDecoded) {
      mDumpDecoded = true;
      mDumpPayload = false;
    } else if (dumpAll) {
      mDumpDecoded = true;
      mDumpPayload = false;
    }

    printf("rdhOnly: %i  decoded: %i  all: %i  DD: %i  DP: %i\n", rdhOnly, dumpDecoded, dumpAll, mDumpDecoded, mDumpPayload);

    // if (!outFilename.empty()) {
    //   mOutFile.open(outFilename.c_str());
    // }
  }

  void printPayload(gsl::span<const uint8_t> payload, bool isBare)
  {
    std::stringstream ss;
    size_t wordLength = isBare ? 16 : 32;
    ss << "\n";
    bool show = false;
    for (size_t iword = 0; iword < payload.size(); iword += wordLength) {
      auto word = payload.subspan(iword, wordLength);
      if (isBare) {
        for (auto it = word.rbegin(); it != word.rend(); ++it) {
          auto ibInWord = word.rend() - it;
          if (ibInWord == 4 || ibInWord == 9) {
            ss << " ";
          }
          if (ibInWord == 5 || ibInWord == 10) {
            ss << "  ";
          }
          ss << fmt::format("{:02x}", static_cast<int>(*it));
        }
      } else {
        for (auto it = word.begin(); it != word.end(); ++it) {
          if (iword == 0 && std::distance(word.begin(), it) == 1 && static_cast<int>(*it) != 1) {
            show = true;
          }
          ss << fmt::format("{:02x}", static_cast<int>(*it));
        }
      }
      ss << "\n";
    }
    // show = o2::raw::RDHUtils::getTriggerType(rdhPtr) & 0xA00;
    if (show) {
      LOG(info) << ss.str();
    }
  }

  void run(of::ProcessingContext& pc)
  {
    if (isDroppedTF(pc, header::gDataOriginMID)) {
      std::vector<ROBoard> data;
      std::vector<ROFRecord> rofs;
      pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODED", mSubSpec}, data);
      pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODEDROF", mSubSpec}, rofs);
      return;
    }

    std::vector<of::InputSpec> filter{of::InputSpec{"filter", of::ConcreteDataTypeMatcher{header::gDataOriginMID, header::gDataDescriptionRawData}, of::Lifetime::Timeframe}};

    of::DPLRawParser parser(pc.inputs(), filter);

    if (!mDecoder) {
      auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(parser.begin().raw());
      mDecoder = createDecoder(*rdhPtr, mIsDebugMode, mElectronicsDelay, mCrateMasks, mFeeIdConfig);
    }

    mDecoder->clear();
    size_t firstRof = 0;
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      gsl::span<const uint8_t> payload(it.data(), it.size());
      bool isBare = o2::mid::raw::isBare(*rdhPtr);
      o2::raw::RDHUtils::printRDH(rdhPtr);
      if (mDumpPayload) {
        printPayload(payload, isBare);
      }

      if (mDumpDecoded) {
        mDecoder->process(payload, *rdhPtr);
        std::stringstream ss;
        for (auto rofIt = mDecoder->getROFRecords().begin() + firstRof, end = mDecoder->getROFRecords().end(); rofIt != end; ++rofIt) {
          ss << fmt::format("BCid: 0x{:x} Orbit: 0x{:x}  EvtType: {:d}", rofIt->interactionRecord.bc, rofIt->interactionRecord.orbit, static_cast<int>(rofIt->eventType)) << std::endl;
          for (auto colIt = mDecoder->getData().begin() + rofIt->firstEntry, end = mDecoder->getData().begin() + rofIt->getEndIndex(); colIt != end; ++colIt) {
            ss << *colIt << std::endl;
          }
        }
        LOG(info) << ss.str();
        firstRof = mDecoder->getROFRecords().size();
      }
    }
  }

 private:
  std::ofstream mOutFile; /// Output file
  std::unique_ptr<Decoder> mDecoder{nullptr};
  bool mIsDebugMode{false};
  FEEIdConfig mFeeIdConfig{};
  CrateMasks mCrateMasks{};
  ElectronicsDelay mElectronicsDelay{};
  header::DataHeader::SubSpecificationType mSubSpec{0};
  bool mDumpPayload = true;
  bool mDumpDecoded = false;
};

of::DataProcessorSpec getRawDumpSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, std::vector<of::InputSpec> inputSpecs, bool askDISTSTF, o2::header::DataHeader::SubSpecificationType subSpecType)
{
  if (askDISTSTF) {
    inputSpecs.emplace_back(getDiSTSTFSpec());
  }

  return of::DataProcessorSpec{
    "MIDRawDump",
    {inputSpecs},
    {},
    of::adaptFromTask<o2::mid::RawDumpDeviceDPL>(isDebugMode, feeIdConfig, crateMasks, electronicsDelay, subSpecType),
    o2::framework::Options{{"decode", o2::framework::VariantType::Bool, false, {"Dump decoded raw data"}},
                           {"rdh-only", o2::framework::VariantType::Bool, false, {"Only dump RDH"}},
                           {"payload-and-decode", o2::framework::VariantType::Bool, false, {"Dump payload and decoded data"}}}};
}

of::DataProcessorSpec getRawDumpSpec(bool isDebugMode)
{
  return getRawDumpSpec(isDebugMode, FEEIdConfig(), CrateMasks(), ElectronicsDelay(), true);
}

of::DataProcessorSpec getRawDumpSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, bool askDISTSTF)
{
  std::vector<of::InputSpec> inputSpecs{{"mid_raw", of::ConcreteDataTypeMatcher{header::gDataOriginMID, header::gDataDescriptionRawData}, of::Lifetime::Optional}};
  header::DataHeader::SubSpecificationType subSpec{0};
  return getRawDumpSpec(isDebugMode, feeIdConfig, crateMasks, electronicsDelay, inputSpecs, askDISTSTF, subSpec);
}

of::DataProcessorSpec getRawDumpSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, bool askDISTSTF, header::DataHeader::SubSpecificationType subSpec)
{
  std::vector<of::InputSpec> inputSpecs{{"mid_raw", header::gDataOriginMID, header::gDataDescriptionRawData, subSpec, o2::framework::Lifetime::Optional}};

  return getRawDumpSpec(isDebugMode, feeIdConfig, crateMasks, electronicsDelay, inputSpecs, askDISTSTF, subSpec);
}
} // namespace mid
} // namespace o2
