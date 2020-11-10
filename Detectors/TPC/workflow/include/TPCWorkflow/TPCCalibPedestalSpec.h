// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_TPCCALIBPEDESTALSPEC_H
#define O2_CALIBRATION_TPCCALIBPEDESTALSPEC_H

/// @file   TPCCalibPedestalSpec.h
/// @brief  TPC Pedestal calibration processor

#include "DetectorsCalibration/Utils.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"

#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "DetectorsCalibration/Utils.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCCalibration/CalibPedestal.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include <vector>
#include <string>
#include "DetectorsRaw/RDHUtils.h"
#include "TPCBase/RDHUtils.h"
#include "TPCBase/CDBInterface.h"
#include "CCDB/CcdbApi.h"

using namespace o2::framework;
using namespace o2::tpc;
using RDHUtils = o2::raw::RDHUtils;
using clbUtils = o2::calibration::Utils;

namespace o2
{
namespace calibration
{

class TPCCalibPedestalDevice : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    mCalibPedestal.init(); // initialize configuration via configKeyValues
    mRawReader.createReader("");
    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mCalibPedestal.update(padROCPos, cru, data);
      mCalibPedestal.setNumberOfProcessedTimeBins(std::max(mCalibPedestal.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mMaxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    mUseOldSubspec = ic.options().get<bool>("use-old-subspec");
    mForceQuit = ic.options().get<bool>("force-quit");
    mDirectFileDump = ic.options().get<bool>("direct-file-dump");
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

    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};

    for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);

      // ---| extract hardware information to do the processing |---
      const auto subSpecification = dh->subSpecification;
      rdh_utils::FEEIDType feeID = (rdh_utils::FEEIDType)dh->subSpecification;
      rdh_utils::FEEIDType cruID, linkID, endPoint;

      if (mUseOldSubspec) {
        //---| old definition by Gvozden |---
        cruID = (rdh_utils::FEEIDType)(subSpecification >> 16);
        linkID = (rdh_utils::FEEIDType)((subSpecification + (subSpecification >> 8)) & 0xFF) - 1;
        endPoint = (rdh_utils::FEEIDType)((subSpecification >> 8) & 0xFF) > 0;
      } else {
        //---| new definition by David |---
        rdh_utils::getMapping(feeID, cruID, endPoint, linkID);
      }

      const auto globalLinkID = linkID + endPoint * 12;

      // ---| update hardware information in the reader |---
      auto& reader = mRawReader.getReaders()[0];
      reader->forceCRU(cruID);
      reader->setLink(globalLinkID);

      LOGP(info, "Specifier: {}/{}/{}", dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>(), subSpecification);
      LOGP(info, "Payload size: {}", dh->payloadSize);
      LOGP(info, "CRU: {}; linkID: {}; endPoint: {}; globalLinkID: {}", cruID, linkID, endPoint, globalLinkID);

      // TODO: exception handling needed?
      const gsl::span<const char> raw = pc.inputs().get<gsl::span<char>>(ref);
      o2::framework::RawParser parser(raw.data(), raw.size());

      // TODO: it would be better to have external event handling and then moving the event processing functionality to CalibRawBase and RawReader to not repeat it in other places
      rawreader::ADCRawData rawData;
      rawreader::GBTFrame gFrame;

      for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
        // debugging stuff
        /*
        auto* rdhPtr = it.get_if<o2::header::RAWDataHeader>();
        if (!rdhPtr) {
          break;
        }
        const auto& rdh = *rdhPtr;
        auto feeID = RDHUtils::getFEEID(rdh);
        rdh_utils::FEEIDType cru, link, endPoint;
        rdh_utils::getMapping(feeID, cru, endPoint, link);
        LOGP(info, "feeID: {} -- CRU: {}; linkID: {}; endPoint: {}", feeID, cru, link, endPoint);
        */

        const auto size = it.size();
        auto data = it.data();
        //LOGP(info, "Data size: {}", size);

        int iFrame = 0;
        for (int i = 0; i < size; i += 16) {
          gFrame.setFrameNumber(iFrame);
          gFrame.setPacketNumber(iFrame / 508);
          gFrame.readFromMemory(gsl::span<const o2::byte>(data + i, 16));

          // extract the half words from the 4 32-bit words
          gFrame.getFrameHalfWords();

          gFrame.getAdcValues(rawData);
          gFrame.updateSyncCheck(false);

          ++iFrame;
        }
      }

      reader->runADCDataCallback(rawData);
    }

    mCalibPedestal.incrementNEvents();
    LOGP(info, "Number of processed events: {} ({})", mCalibPedestal.getNumberOfProcessedEvents(), mMaxEvents);

    if ((mCalibPedestal.getNumberOfProcessedEvents() >= mMaxEvents) && !mCalibDumped) {
      LOGP(info, "Maximm number of events reached ({}), no more processing will be done", mMaxEvents);
      mReadyToQuit = true;
      dumpCalibData();
      if (mForceQuit) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
      } else {
        //pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    dumpCalibData();
    sendOutput(ec.outputs());
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  CalibPedestal mCalibPedestal;
  rawreader::RawReaderCRUManager mRawReader;
  uint32_t mMaxEvents{100};
  bool mReadyToQuit{false};
  bool mCalibDumped{false};
  bool mUseOldSubspec{false};
  bool mForceQuit{false};
  bool mDirectFileDump{false};

  //____________________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    CDBStorage::MetaData_t md;
    long timeStart = 0;
    long timeEnd = 99999999999999;

    std::array<const CalDet<float>*, 2> data = {&mCalibPedestal.getPedestal(), &mCalibPedestal.getNoise()};
    std::array<CDBType, 2> dataType = {CDBType::CalPedestal, CDBType::CalNoise};

    for (size_t i = 0; i < data.size(); ++i) {
      auto cal = data[i];
      o2::ccdb::CcdbObjectInfo w;
      auto image = o2::ccdb::CcdbApi::createObjectImage(cal, &w);

      w.setPath(CDBTypeMap.at(dataType[i]));
      w.setStartValidityTimestamp(timeStart);
      w.setEndValidityTimestamp(timeEnd);

      LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)i};
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, subSpec}, *image.get());
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, subSpec}, w);
    }
  }

  //____________________________________________________________________________
  void dumpCalibData()
  {
    if (mDirectFileDump && !mCalibDumped) {
      LOGP(info, "Dumping output");
      mCalibPedestal.analyse();
      mCalibPedestal.dumpToFile("pedestals.root");
      mCalibDumped = true;
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTPCCalibPedestalSpec(const std::string inputSpec)
{
  using device = o2::calibration::TPCCalibPedestalDevice;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});

  return DataProcessorSpec{
    "calib-tpc-pedestal",
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"max-events", VariantType::Int, 100, {"maximum number of events to process"}},
      {"use-old-subspec", VariantType::Bool, false, {"use old subsecifiation definition"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}},
    } // end Options
  };  // end DataProcessorSpec
}

} // namespace framework
} // namespace o2

#endif
