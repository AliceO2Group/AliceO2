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

#ifndef O2_CALIBRATION_TPCCALIBPEDESTALSPEC_H
#define O2_CALIBRATION_TPCCALIBPEDESTALSPEC_H

/// @file   TPCCalibPadRawSpec.h
/// @brief  TPC Pad-wise raw data calibration processor
/// @author Jens Wiechula
/// @author David Silvermyr

#include <vector>
#include <string>
#include <chrono>
#include <fmt/format.h>

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"

#include "CommonUtils/MemFileHelper.h"
#include "Headers/DataHeader.h"

#include "DetectorsBase/TFIDInfoHelper.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/CalibPedestal.h"
#include "TPCCalibration/CalibPulser.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/CalibProcessingHelper.h"
#include "TPCWorkflow/CalibRawPartInfo.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

enum class CalibRawType {
  Pedestal,
  Pulser,
  CE,
};

const std::unordered_map<std::string, CDBType> CalibRawTypeMap{
  {"pedestal", CDBType::CalPedestalNoise},
  {"pulser", CDBType::CalPulser},
  {"ce", CDBType::CalCE},
};

template <class T>
class TPCCalibPedestalDevice : public o2::framework::Task
{
 public:
  TPCCalibPedestalDevice(CDBType calibType, uint32_t lane, const std::vector<int>& sectors, uint32_t publishAfterTFs, bool useDigitsAsInput) : mCalibType{calibType}, mLane{lane}, mSectors(sectors), mPublishAfter(publishAfterTFs), mUseDigits(useDigitsAsInput) {}

  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    // TODO: clean up to not go via RawReaderCRUManager
    mCalibration.init(); // initialize configuration via configKeyValues
    mRawReader.createReader("");

    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mCalibration.update(padROCPos, cru, data);
      mCalibration.setNumberOfProcessedTimeBins(std::max(mCalibration.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mRawReader.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
      CRU cruID(cru);
      mCalibration.updateROC(cruID.roc(), rowInSector - (rowInSector > 62) * 63, padInRow, timeBin, adcValue);
      return true;
    });

    mMaxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    mUseOldSubspec = ic.options().get<bool>("use-old-subspec");
    mForceQuit = ic.options().get<bool>("force-quit");
    mResetAfterPublish = ic.options().get<bool>("reset-after-publish");
    mDirectFileDump = ic.options().get<bool>("direct-file-dump");
    mSyncOffsetReference = ic.options().get<uint32_t>("sync-offset-reference");
    mDecoderType = ic.options().get<uint32_t>("decoder-type");
    if (mUseOldSubspec) {
      LOGP(info, "Using old subspecification (CruId << 16) | ((LinkId + 1) << (CruEndPoint == 1 ? 8 : 0))");
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // in case the maximum number of events was reached don't do further processing
    if (mReadyToQuit) {
      return;
    }

    if (mUseDigits) {
      std::array<std::vector<Digit>, Sector::MAXSECTOR> digits;
      copyDigits(pc.inputs(), digits);
      mCalibration.setDigits(&digits);
      mCalibration.processEvent();
    } else {
      auto& reader = mRawReader.getReaders()[0];
      calib_processing_helper::processRawData(pc.inputs(), reader, mUseOldSubspec, mSectors, nullptr, mSyncOffsetReference, mDecoderType);
      mCalibration.endEvent();
      mCalibration.endReader();
    }

    if (mCalibInfo.tfIDInfo.isDummy()) {
      base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibInfo.tfIDInfo);
    }

    mCalibration.incrementNEvents();
    const auto nTFs = mCalibration.getNumberOfProcessedEvents();
    LOGP(info, "Number of processed TFs: {} ({})", nTFs, mMaxEvents);

    if ((mPublishAfter && (nTFs % mPublishAfter) == 0)) {
      LOGP(info, "Publishing after {} TFs", nTFs);
      sendOutput(pc.outputs());
      mCalibSent = false;
      if (mResetAfterPublish) {
        mCalibration.resetData();
        mCalibInfo.tfIDInfo.tfCounter = -1U;
      }
    }

    if (mMaxEvents && (nTFs >= mMaxEvents) && !mCalibSent) {
      LOGP(info, "Maximm number of TFs reached ({}), no more processing will be done", mMaxEvents);
      mReadyToQuit = true;
      sendOutput(pc.outputs());
      if (mForceQuit) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
      } else {
        // pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    if (!mCalibSent) {
      sendOutput(ec.outputs());
      ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }

 private:
  T mCalibration;
  rawreader::RawReaderCRUManager mRawReader;
  CDBType mCalibType;                 ///< calibration type
  CalibRawPartInfo mCalibInfo{};      ///< TF id of first TF in interval
  uint32_t mMaxEvents{0};             ///< maximum number of events to process
  uint32_t mPublishAfter{0};          ///< number of events after which to dump the calibration
  uint32_t mLane{0};                  ///< lane number of processor
  uint32_t mSyncOffsetReference{144}; ///< reference sync offset for decoding
  uint32_t mDecoderType{0};           ///< decoder type: 0 - TPC, 1 - GPU
  std::vector<int> mSectors{};        ///< sectors to process in this instance
  bool mReadyToQuit{false};           ///< if processor is ready to quit
  bool mCalibSent{false};             ///< if calibration object already sent / dumped
  bool mUseOldSubspec{false};         ///< use the old subspec definition
  bool mUseDigits{false};             ///< use digits as input
  bool mForceQuit{false};             ///< for quit after processing finished
  bool mDirectFileDump{false};        ///< directly dump the calibration data to file
  bool mResetAfterPublish{false};     ///< reset calibration after it was published

  //____________________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    mCalibration.analyse();
    const int type = int(mCalibType);
    mCalibInfo.calibType = type;
    auto& calibObject = mCalibration.getCalDets();
    const auto& cdbType = CDBTypeMap.at(mCalibType);
    const auto name = cdbType.substr(cdbType.rfind("/") + 1);
    auto image = o2::utils::MemFileHelper::createFileImage(&calibObject, typeid(calibObject), name.data(), "data");
    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)((mLane << 4))};
    output.snapshot(Output{gDataOriginTPC, "CLBPART", subSpec}, *image.get());
    output.snapshot(Output{gDataOriginTPC, "CLBPARTINFO", subSpec}, mCalibInfo);
    dumpCalibData();
    ++mCalibInfo.publishCycle;
    mCalibSent = true;
  }

  //____________________________________________________________________________
  void dumpCalibData()
  {
    if (mDirectFileDump && !mCalibSent) {
      mCalibration.setDebugLevel();
      const auto& cdbType = CDBTypeMap.at(mCalibType);
      const auto name = cdbType.substr(cdbType.rfind("/") + 1);
      LOGP(info, "Dumping output {} lane: {:02}, firsTF: {} cycle: {}.root", name, mLane, mCalibInfo.tfIDInfo.tfCounter, mCalibInfo.publishCycle);
      mCalibration.dumpToFile(fmt::format("{}_{:02}_{}_{}.root", name, mLane, mCalibInfo.tfIDInfo.tfCounter, mCalibInfo.publishCycle));
    }
  }

  void copyDigits(InputRecord& inputs, std::array<std::vector<Digit>, Sector::MAXSECTOR>& digits)
  {
    std::vector<InputSpec> filter = {
      {"check", ConcreteDataTypeMatcher{"TPC", "DIGITS"}, Lifetime::Timeframe},
    };
    for (auto const& inputRef : InputRecordWalker(inputs)) {
      auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(inputRef);
      if (sectorHeader == nullptr) {
        LOG(error) << "sector header missing on header stack for input on " << inputRef.spec->binding;
        continue;
      }
      const int sector = sectorHeader->sector();
      digits[sector] = inputs.get<std::vector<o2::tpc::Digit>>(inputRef);
    }
  }
};

template <typename... Args>
AlgorithmSpec getRawDevice(CDBType calibType, Args... args)
{
  if (calibType == CDBType::CalPedestalNoise) {
    return adaptFromTask<TPCCalibPedestalDevice<CalibPedestal>>(calibType, args...);
  } else if (calibType == CDBType::CalPulser) {
    return adaptFromTask<TPCCalibPedestalDevice<CalibPulser>>(calibType, args...);
  } else if (calibType == CDBType::CalCE) {
    return adaptFromTask<TPCCalibPedestalDevice<CalibPulser>>(calibType, args...);
  } else {
    return AlgorithmSpec{};
  }
};

DataProcessorSpec getTPCCalibPadRawSpec(const std::string inputSpec, uint32_t ilane = 0, std::vector<int> sectors = {}, uint32_t publishAfterTFs = 0, CDBType rawType = CDBType::CalPedestalNoise)
{
  const bool useDigitsAsInput = inputSpec.find("DIGITS") != std::string::npos;

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, "CLBPART"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, "CLBPARTINFO"}, Lifetime::Sporadic);

  AlgorithmSpec spec;

  const auto id = fmt::format("calib-tpc-raw-{:02}", ilane);
  return DataProcessorSpec{
    id.data(),
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{getRawDevice(rawType, ilane, sectors, publishAfterTFs, useDigitsAsInput)},
    Options{
      {"max-events", VariantType::Int, 0, {"maximum number of events to process"}},
      {"use-old-subspec", VariantType::Bool, false, {"use old subsecifiation definition"}},
      {"reset-after-publish", VariantType::Bool, false, {"reset calibration after publishing"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}},
      {"sync-offset-reference", VariantType::UInt32, 144u, {"Reference BCs used for the global sync offset in the CRUs"}},
      {"decoder-type", VariantType::UInt32, 1u, {"Decoder to use: 0 - TPC, 1 - GPU"}},
    } // end Options
  };  // end DataProcessorSpec
}

} // namespace o2::tpc

#endif
