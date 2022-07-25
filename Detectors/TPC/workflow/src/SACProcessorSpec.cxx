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

/// @file SACProcessorSpec.cxx
/// @brief TPC Integrated Analogue Current processing
/// @author Jens Wiechula

#include <vector>
#include <chrono>

#include "Framework/Task.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/RawParser.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/CCDBParamSpec.h"
#include "TPCBase/CDBInterface.h"

#include "DataFormatsTPC/RawDataTypes.h"
#include "TPCBase/RDHUtils.h"
#include "TPCCalibration/SACDecoder.h"
#include "TPCWorkflow/SACProcessorSpec.h"

using HighResClock = std::chrono::high_resolution_clock;
using namespace o2::framework;

namespace o2::tpc
{

class SACProcessorDevice : public Task
{
 public:
  enum class DebugFlags {
    HBFInfo = 0x10, ///< Show data for each HBF
  };
  using FEEIDType = rdh_utils::FEEIDType;

  SACProcessorDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  void init(InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mDebugLevel = ic.options().get<uint32_t>("debug-level");
    mDecoder.setDebugLevel(mDebugLevel);
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    const auto startTime = HighResClock::now();

    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}}; // TODO: Change to SAC when changed in DD
    for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      // ---| extract hardware information to do the processing |---
      const auto feeId = (FEEIDType)dh->subSpecification;
      const auto link = rdh_utils::getLink(feeId);
      const uint32_t cruID = rdh_utils::getCRU(feeId);
      const auto endPoint = rdh_utils::getEndPoint(feeId);

      // only select SACs
      // ToDo: cleanup once SACs will be propagated not as RAWDATA, but SAC.
      if (link != rdh_utils::SACLinkID) {
        continue;
      }
      if (mDebugLevel & (uint32_t)DebugFlags::HBFInfo) {
        LOGP(info, "SAC Processing firstTForbit {:9}, tfCounter {:5}, run {:6}, feeId {:6} ({:3}/{}/{:2})", tinfo.firstTForbit, tinfo.tfCounter, tinfo.runNumber, feeId, cruID, endPoint, link);
      }

      // ---| data loop |---
      const gsl::span<const char> raw = pc.inputs().get<gsl::span<char>>(ref);
      RawParser parser(raw.data(), raw.size());
      for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
        const auto size = it.size();
        if (size == 0) {
          auto* rdhPtr = it.get_if<o2::header::RAWDataHeaderV6>();
          if (!rdhPtr) {
            throw std::runtime_error("could not get RDH from packet");
          }
          // TODO: should only be done once for the first packet
          if ((mDecoder.getReferenceTime() < 0) && (rdhPtr->packetCounter == 0)) {
            const double referenceTime = o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS() + tinfo.firstTForbit * o2::constants::lhc::LHCOrbitMUS * 0.001;
            LOGP(info, "setting time stamp reset reference to: {}, at tfCounter: {}, firstTForbit: {}", referenceTime, tinfo.tfCounter, tinfo.firstTForbit);
            mDecoder.setReferenceTime(referenceTime); // TODO set proper time
          }
          continue;
        }
        auto data = (const char*)it.data();
        mDecoder.process(data, size);
      }
    }

    mDecoder.runDecoding();
    sendData(pc.outputs());

    if (mDebugLevel & (uint32_t)sac::Decoder::DebugFlags::TimingInfo) {
      auto endTime = HighResClock::now();
      auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
      LOGP(info, "Time spent for TF {}, firstTForbit {}: {} s", tinfo.tfCounter, tinfo.firstTForbit, elapsed_seconds.count());
    }
  }

  void sendData(DataAllocator& output)
  {
    output.snapshot(Output{"TPC", "REFTIMESAC", 0, Lifetime::Timeframe}, mDecoder.getDecodedData().referenceTime);
    output.snapshot(Output{"TPC", "DECODEDSAC", 0, Lifetime::Timeframe}, mDecoder.getDecodedData().getGoodData());
    mDecoder.clearDecodedData();
  }

  void endOfStream(EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    mDecoder.finalize();
    sendData(ec.outputs());
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 private:
  uint32_t mDebugLevel{0}; ///< Debug level
  sac::Decoder mDecoder;   ///< Decoder for SAC data
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
};

o2::framework::DataProcessorSpec getSACProcessorSpec()
{
  using device = o2::tpc::SACProcessorDevice;

  std::vector<InputSpec> inputs;
  inputs.emplace_back(InputSpec{"tpcraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Optional});
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TPC", "DECODEDSAC", 0, Lifetime::Timeframe);
  outputs.emplace_back("TPC", "REFTIMESAC", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-sac-processor",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest)},
    Options{
      {"debug-level", VariantType::UInt32, 0u, {"amount of debug to show"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
