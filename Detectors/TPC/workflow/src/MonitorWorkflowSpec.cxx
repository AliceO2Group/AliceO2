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

#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "fmt/format.h"

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/InputRecordWalker.h"

#include "DataFormatsTPC/TPCSectorHeader.h"
#include "Headers/DataHeader.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"

#include "TPCQC/Clusters.h"
#include "TPCBase/Mapper.h"
#include "TPCMonitor/SimpleEventDisplayGUI.h"
#include "TPCCalibration/DigitDump.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/CalibProcessingHelper.h"
using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2::tpc
{

class TPCMonitorDevice : public o2::framework::Task
{
 public:
  TPCMonitorDevice(bool useDigits) : mUseDigits(useDigits) {}

  void init(o2::framework::InitContext& ic) final
  {
    mBlocking = ic.options().get<bool>("blocking");
    const int maxTimeBins = ic.options().get<int>("max-time-bins");

    // set up ADC value filling
    mRawReader.createReader("");
    mDigitDump.init();
    mDigitDump.setInMemoryOnly();
    mDigitDump.setTimeBinRange(0, maxTimeBins);
    const auto pedestalFile = ic.options().get<std::string>("pedestal-file");
    if (pedestalFile.length()) {
      LOGP(info, "Setting pedestal file: {}", pedestalFile);
      mDigitDump.setPedestalAndNoiseFile(pedestalFile);
    }

    // TODO: Get rid of digit dump and use mDigits?
    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mDigitDump.update(padROCPos, cru, data);
      mDigitDump.setNumberOfProcessedTimeBins(std::max(mDigitDump.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mRawReader.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
      CRU cruID(cru);
      const PadRegionInfo& regionInfo = Mapper::instance().getPadRegionInfo(cruID.region());
      mDigitDump.updateCRU(cruID, rowInSector - regionInfo.getGlobalRowOffset(), padInRow, timeBin, adcValue);
      return true;
    });

    if (mUseDigits) {
      mEventDisplayGUI.getEventDisplay().setDigits(&mDigits);
    } else {
      mEventDisplayGUI.getEventDisplay().setDigits(&mDigitDump.getDigits());
    }

    mGUIThread = std::make_unique<std::thread>(&SimpleEventDisplayGUI::startGUI, &mEventDisplayGUI, maxTimeBins);

    auto finishFunction = [this]() {
      if (mGUIThread) {
        mGUIThread->join();
      }
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(finishFunction);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto validInputs = pc.inputs().countValidInputs();
    mEventDisplayGUI.setDataAvailable(validInputs);

    while (mBlocking && validInputs && !mEventDisplayGUI.isNextEventRequested() && !mEventDisplayGUI.isStopRequested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      LOGP(info, "wait for next event stop {}", mEventDisplayGUI.isStopRequested());
    }

    if (mEventDisplayGUI.isStopRequested()) {
      LOGP(info, "call end processing");
      pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
      return;
    }

    LOGP(info, "next event requested next {}, processing {}, updating {}", mEventDisplayGUI.isNextEventRequested(), mEventDisplayGUI.isProcessingEvent(), mEventDisplayGUI.isWaitingForDigitUpdate());
    if (!mEventDisplayGUI.isNextEventRequested() && !mEventDisplayGUI.isProcessingEvent()) {
      return;
    }
    mEventDisplayGUI.resetNextEventReqested();

    const auto tf = pc.services().get<o2::framework::TimingInfo>().tfCounter;
    mEventDisplayGUI.getEventDisplay().setPresentEventNumber(size_t(tf));
    LOGP(info, "processing tF {}", tf);

    if (!mUseDigits) {
      mDigitDump.clearDigits();
      auto& reader = mRawReader.getReaders()[0];
      calib_processing_helper::processRawData(pc.inputs(), reader);

      mDigitDump.incrementNEvents();
    } else {
      // clear digits
      std::for_each(mDigits.begin(), mDigits.end(), [](auto& vec) { vec.clear(); });

      copyDigits(pc.inputs());
    }

    mEventDisplayGUI.resetUpdatingDigits();
    mEventDisplayGUI.setDataAvailable(false);
  }

 private:
  SimpleEventDisplayGUI mEventDisplayGUI;                    ///< Event display
  DigitDump mDigitDump;                                      ///< used to covert
  std::array<std::vector<Digit>, Sector::MAXSECTOR> mDigits; ///< TPC digits
  std::unique_ptr<std::thread> mGUIThread;                   ///< thread running the GUI

  rawreader::RawReaderCRUManager mRawReader;
  bool mUseDigits{false};
  bool mBlocking{false};

  void copyDigits(InputRecord& inputs)
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
      mDigits[sector] = inputs.get<std::vector<o2::tpc::Digit>>(inputRef);
    }
  }
};

DataProcessorSpec getMonitorWorkflowSpec(std::string inputSpec)
{
  const bool useDigitsAsInput = inputSpec.find("DIGITS") != std::string::npos;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tpc-monitor-workflow",
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<TPCMonitorDevice>(useDigitsAsInput)},
    Options{
      {"pedestal-file", VariantType::String, "", {"file with pedestals and noise for zero suppression"}},
      {"max-time-bins", VariantType::Int, 114048, {"maximum number of time bins to show"}},
      {"blocking", VariantType::Bool, false, {"block processing until next event is received"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
