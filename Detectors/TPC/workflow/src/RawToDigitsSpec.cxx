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

#include <string_view>
#include <unordered_map>
#include <vector>
#include <string>
#include "fmt/format.h"

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"

#include "DataFormatsTPC/TPCSectorHeader.h"
#include "Headers/DataHeader.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"

#include "TPCQC/Clusters.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/DigitDump.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/CalibProcessingHelper.h"
#include "TPCReconstruction/IonTailCorrection.h"
using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

class TPCDigitDumpDevice : public o2::framework::Task
{
 public:
  TPCDigitDumpDevice(const std::vector<int>& sectors, bool sendCEdigits) : mSectors(sectors), mSendCEdigits(sendCEdigits) {}

  void init(o2::framework::InitContext& ic) final
  {
    // parse command line arguments
    mMaxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    mSyncOffsetReference = ic.options().get<uint32_t>("sync-offset-reference");
    mDecoderType = ic.options().get<uint32_t>("decoder-type");
    mUseTrigger = !ic.options().get<bool>("ignore-trigger");
    mUseOldSubspec = ic.options().get<bool>("use-old-subspec");
    const bool createOccupancyMaps = ic.options().get<bool>("create-occupancy-maps");
    mForceQuit = ic.options().get<bool>("force-quit");
    mCheckDuplicates = ic.options().get<bool>("check-for-duplicates");
    mApplyTailCancellation = ic.options().get<bool>("apply-ion-tail-cancellation");
    mRemoveDuplicates = ic.options().get<bool>("remove-duplicates");
    if (mSendCEdigits) {
      mRemoveCEdigits = true;
    } else {
      mRemoveCEdigits = ic.options().get<bool>("remove-ce-digits");
    }

    if (mUseOldSubspec) {
      LOGP(info, "Using old subspecification (CruId << 16) | ((LinkId + 1) << (CruEndPoint == 1 ? 8 : 0))");
    }

    // set up ADC value filling
    mRawReader.createReader("");

    if (!ic.options().get<bool>("ignore-grp")) {
      const auto inputGRP = o2::base::NameConf::getGRPFileName();
      const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
      if (grp) {
        const auto nhbf = (int)grp->getNHBFPerTF();
        const int lastTimeBin = nhbf * 891 / 2;
        mDigitDump.setTimeBinRange(0, lastTimeBin);
        LOGP(info, "Using GRP NHBF = {} to set last time bin to {}, might be overwritte via --configKeyValues", nhbf, lastTimeBin);
      }
    }

    mDigitDump.init();
    mDigitDump.setInMemoryOnly();
    const auto pedestalFile = ic.options().get<std::string>("pedestal-url");
    if (pedestalFile.length()) {
      if (pedestalFile.find("ccdb") != std::string::npos) {
        LOGP(info, "Loading pedestals from ccdb: {}", pedestalFile);
        auto& cdb = o2::ccdb::BasicCCDBManager::instance();
        cdb.setURL(pedestalFile);
        if (cdb.isHostReachable()) {
          auto pedestalNoise = cdb.get<std::unordered_map<std::string, CalPad>>("TPC/Calib/PedestalNoise");
          CalPad* pedestal = nullptr;
          if (pedestalNoise) {
            pedestal = &pedestalNoise->at("Pedestals");
          }
          if (pedestal) {
            mDigitDump.setPedestals(pedestal);
          } else {
            LOGP(error, "could not load pedestals from {}", pedestalFile);
          }
        } else {
          LOGP(error, "ccdb access to {} requested, but host is not reachable. Cannot load Pedestals", pedestalFile);
        }
      } else {
        LOGP(info, "Setting pedestal file: {}", pedestalFile);
        mDigitDump.setPedestalAndNoiseFile(pedestalFile);
      }
    }

    // set up cluster qc if requested
    if (createOccupancyMaps) {
      mClusterQC = std::make_unique<qc::Clusters>();
    }

    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mDigitDump.update(padROCPos, cru, data);
      mDigitDump.setNumberOfProcessedTimeBins(std::max(mDigitDump.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mRawReader.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
      CRU cruID(cru);
      const PadRegionInfo& regionInfo = Mapper::instance().getPadRegionInfo(cruID.region());
      mDigitDump.updateCRU(cruID, rowInSector - regionInfo.getGlobalRowOffset(), padInRow, timeBin, adcValue);
      if (mClusterQC) {
        mClusterQC->fillADCValue(cru, rowInSector, padInRow, timeBin, adcValue);
      }
      return true;
    });
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // in case the maximum number of events was reached don't do further processing
    if (mReadyToQuit) {
      return;
    }

    auto& reader = mRawReader.getReaders()[0];
    mActiveSectors = calib_processing_helper::processRawData(pc.inputs(), reader, mUseOldSubspec, std::vector<int>(), nullptr, mSyncOffsetReference, mDecoderType, mUseTrigger);

    mDigitDump.incrementNEvents();
    if (mClusterQC) {
      mClusterQC->endTF();
    }
    LOGP(info, "Number of processed events: {} ({})", mDigitDump.getNumberOfProcessedEvents(), mMaxEvents);

    snapshotDigits(pc.outputs());

    if (mMaxEvents && (mDigitDump.getNumberOfProcessedEvents() >= mMaxEvents)) {
      LOGP(info, "Maximm number of events reached ({}), no more processing will be done", mMaxEvents);
      mReadyToQuit = true;
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
    if (mActiveSectors) {
      snapshotDigits(ec.outputs());
    }
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    if (mClusterQC) {
      dumpClusterQC();
    }
  }

 private:
  DigitDump mDigitDump;
  std::unique_ptr<qc::Clusters> mClusterQC;
  rawreader::RawReaderCRUManager mRawReader;
  uint32_t mMaxEvents{0};
  uint32_t mSyncOffsetReference{144};
  uint32_t mDecoderType{0};
  bool mReadyToQuit{false};
  bool mCalibDumped{false};
  bool mUseOldSubspec{false};
  bool mForceQuit{false};
  bool mCheckDuplicates{false};
  bool mApplyTailCancellation{false};
  bool mRemoveDuplicates{false};
  bool mRemoveCEdigits{false};
  bool mSendCEdigits{false};
  bool mUseTrigger{false};
  uint64_t mActiveSectors{0};  ///< bit mask of active sectors
  std::vector<int> mSectors{}; ///< tpc sector configuration

  //____________________________________________________________________________
  void snapshotDigits(DataAllocator& output)
  {
    if (mCheckDuplicates || mRemoveDuplicates) {
      // iplicityly sorts
      mDigitDump.checkDuplicates(mRemoveDuplicates);
    } else {
      mDigitDump.sortDigits();
    }

    std::array<std::vector<Digit>, Sector::MAXSECTOR> ceDigits;
    if (mRemoveCEdigits) {
      mDigitDump.removeCEdigits(10, 100, &ceDigits);
    }

    if (mApplyTailCancellation) {
      auto& digits = mDigitDump.getDigits();
      IonTailCorrection itCorr;
      for (auto isector : mSectors) {
        itCorr.filterDigitsDirect(digits[isector]);
      }
    }

    for (auto isector : mSectors) {
      o2::tpc::TPCSectorHeader header{isector};
      header.activeSectors = mActiveSectors;
      // digit for now are transported per sector, not per lane
      output.snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe, header},
                      mDigitDump.getDigits(isector));
      if (mSendCEdigits) {
        output.snapshot(Output{"TPC", "CEDIGITS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe, header},
                        ceDigits[isector]);
      }
    }
    mDigitDump.clearDigits();
    mActiveSectors = 0;
  }

  //____________________________________________________________________________
  void dumpClusterQC()
  {
    mClusterQC->normalize();
    mClusterQC->dumpToFile("ClusterQC.root", 2);
  }
};

DataProcessorSpec getRawToDigitsSpec(int channel, const std::string inputSpec, bool ignoreDistStf, std::vector<int> const& tpcSectors, bool sendCEdigits)
{
  using device = o2::tpc::TPCDigitDumpDevice;

  std::vector<OutputSpec> outputs;
  for (auto isector : tpcSectors) {
    outputs.emplace_back("TPC", "DIGITS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe);
    if (sendCEdigits) {
      outputs.emplace_back("TPC", "CEDIGITS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe);
    }
  }

  std::vector<InputSpec> inputs;
  if (inputSpec != "") {
    inputs = select(inputSpec.data());
  } else {
    inputs.emplace_back(InputSpec{"zsraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Timeframe});
    if (!ignoreDistStf) {
      inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
    }
  }

  return DataProcessorSpec{
    fmt::format("tpc-raw-to-digits-{}", channel),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(tpcSectors, sendCEdigits)},
    Options{
      {"max-events", VariantType::Int, 0, {"maximum number of events to process"}},
      {"use-old-subspec", VariantType::Bool, false, {"use old subsecifiation definition"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"pedestal-url", VariantType::String, "", {"file with pedestals and noise or ccdb url for zero suppression"}},
      {"create-occupancy-maps", VariantType::Bool, false, {"create occupancy maps and store them to local root file for debugging"}},
      {"check-for-duplicates", VariantType::Bool, false, {"check if duplicate digits exist and only report them"}},
      {"apply-ion-tail-cancellation", VariantType::Bool, false, {"Apply ion tail cancellation"}},
      {"remove-duplicates", VariantType::Bool, false, {"check if duplicate digits exist and remove them"}},
      {"remove-ce-digits", VariantType::Bool, false, {"find CE position and remove digits around it"}},
      {"ignore-grp", VariantType::Bool, false, {"ignore GRP file"}},
      {"sync-offset-reference", VariantType::UInt32, 144u, {"Reference BCs used for the global sync offset in the CRUs"}},
      {"decoder-type", VariantType::UInt32, 1u, {"Decoder to use: 0 - TPC, 1 - GPU"}},
      {"ignore-trigger", VariantType::Bool, false, {"Ignore the trigger information"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace tpc
} // namespace o2
