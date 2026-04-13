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

/// @file   CMVToVectorSpec.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Processor to convert CMVs to a vector in a CRU

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <fmt/format.h>
#include <fmt/chrono.h>

#include "TFile.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "CommonUtils/TreeStreamRedirector.h"

#include "DataFormatsTPC/CMV.h"
#include "DataFormatsTPC/RawDataTypes.h"
#include "TPCBase/RDHUtils.h"
#include "TPCBase/Mapper.h"
#include "TPCWorkflow/ProcessingHelpers.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using RDHUtils = o2::raw::RDHUtils;
using RawDataType = o2::tpc::raw_data_types::Type;

namespace o2::tpc
{

class CMVToVectorDevice : public o2::framework::Task
{
 public:
  using FEEIDType = rdh_utils::FEEIDType;
  CMVToVectorDevice(const std::vector<uint32_t>& crus) : mCRUs(crus) {}

  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    mWriteDebug = ic.options().get<bool>("write-debug");
    mWriteDebugOnError = ic.options().get<bool>("write-debug-on-error");
    mWriteRawDataOnError = ic.options().get<bool>("write-raw-data-on-error");
    mRawDataType = ic.options().get<int>("raw-data-type");
    o2::framework::RawParser<>::setCheckIncompleteHBF(ic.options().get<bool>("check-incomplete-hbf"));

    mDebugStreamFileName = ic.options().get<std::string>("debug-file-name").data();
    mRawOutputFileName = ic.options().get<std::string>("raw-file-name").data();

    initCMV();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto runNumber = processing_helpers::getRunNumber(pc);
    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
    const auto& mapper = Mapper::instance();

    // open files if necessary
    if ((mWriteDebug || mWriteDebugOnError) && !mDebugStream) {
      const auto debugFileName = fmt::format(fmt::runtime(mDebugStreamFileName), fmt::arg("run", runNumber));
      LOGP(info, "Creating debug stream {}", debugFileName);
      mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>(debugFileName.data(), "recreate");
    }

    if (mWriteRawDataOnError && !mRawOutputFile.is_open()) {
      std::string_view rawType = (mRawDataType < 2) ? "tf" : "raw";
      if (mRawDataType == 5) {
        rawType = "cmv.raw";
      }
      const auto rawFileName = fmt::format(fmt::runtime(mRawOutputFileName), fmt::arg("run", runNumber), fmt::arg("raw_type", rawType));
      LOGP(info, "Creating raw debug file {}", rawFileName);
      mRawOutputFile.open(rawFileName, std::ios::binary);
    }

    uint32_t heartbeatOrbit = 0;
    uint16_t heartbeatBC = 0;
    uint32_t tfCounter = 0;
    bool first = true;
    bool hasErrors = false;

    for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      tfCounter = dh->tfCounter;
      const auto subSpecification = dh->subSpecification;
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      LOGP(debug, "Processing TF {}, subSpecification {}, payloadSize {}", tfCounter, subSpecification, payloadSize);

      // ---| data loop |---
      const gsl::span<const char> raw = pc.inputs().get<gsl::span<char>>(ref);
      try {
        o2::framework::RawParser parser(raw.data(), raw.size());
        size_t lastErrorCount = 0;

        for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
          const auto size = it.size();

          if (parser.getNErrors() > lastErrorCount) {
            lastErrorCount = parser.getNErrors();
            hasErrors = true;
          }

          // skip empty packages (HBF open)
          if (size == 0) {
            continue;
          }

          auto rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
          const auto rdhVersion = RDHUtils::getVersion(rdhPtr);
          if (!rdhPtr || rdhVersion < 6) {
            throw std::runtime_error(fmt::format("could not get RDH from packet, or version {} < 6", rdhVersion).data());
          }

          // ---| extract hardware information to do the processing |---
          const auto feeId = (FEEIDType)RDHUtils::getFEEID(*rdhPtr);
          const auto link = rdh_utils::getLink(feeId);
          const uint32_t cruID = rdh_utils::getCRU(feeId);
          const auto detField = RDHUtils::getDetectorField(*rdhPtr);

          LOGP(debug, "Detected CMV packet: CRU {}, link {}, feeId {}", cruID, link, feeId);

          if ((detField != (decltype(detField))RawDataType::CMV) || (link != rdh_utils::CMVLinkID)) {
            LOGP(debug, "Skipping packet: detField {}, (expected RawDataType {}), link {}, (expected CMVLinkID {})", detField, (decltype(detField))RawDataType::CMV, link, rdh_utils::CMVLinkID);
            continue;
          }

          LOGP(debug, "Processing firstTForbit {:9}, tfCounter {:5}, run {:6}, feeId {:6}, cruID {:3}, link {:2}", dh->firstTForbit, dh->tfCounter, dh->runNumber, feeId, cruID, link);

          if (std::find(mCRUs.begin(), mCRUs.end(), cruID) == mCRUs.end()) {
            LOGP(warning, "CMV CRU {:3} not configured in CRUs, skipping", cruID);
            continue;
          }

          auto& cmvVec = mCMVvectors[cruID];
          auto& infoVec = mCMVInfos[cruID];

          if (size != sizeof(cmv::Container)) {
            LOGP(warning, "CMV packet size mismatch: got {} bytes, expected {} bytes (sizeof cmv::Container). Skipping package.", size, sizeof(cmv::Container));
            hasErrors = true;
            continue;
          }
          auto data = it.data();
          auto& cmvs = *((cmv::Container*)(data));
          const uint32_t orbit = cmvs.header.heartbeatOrbit;
          const uint16_t bc = cmvs.header.heartbeatBC;

          // record packet meta and append its CMV vector (3564 TB)
          infoVec.emplace_back(orbit, bc);
          cmvVec.reserve(cmvVec.size() + cmv::NTimeBinsPerPacket);
          for (uint32_t tb = 0; tb < cmv::NTimeBinsPerPacket; ++tb) {
            cmvVec.push_back(cmvs.getCMV(tb));
            // LOGP(debug, "Appended CMV {} for timebin {}, CRU {}, orbit {}, bc {}", cmvs.getCMV(tb), tb, cruID, orbit, bc);
          }
        }
      } catch (const std::exception& e) {
        // error message throtteling
        using namespace std::literals::chrono_literals;
        static std::unordered_map<uint32_t, size_t> nErrorPerSubspec;
        static std::chrono::time_point<std::chrono::steady_clock> lastReport = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        static size_t reportedErrors = 0;
        const size_t MAXERRORS = 10;
        const auto sleepTime = 10min;
        ++nErrorPerSubspec[subSpecification];

        if ((now - lastReport) < sleepTime) {
          if (reportedErrors < MAXERRORS) {
            ++reportedErrors;
            std::string sleepInfo;
            if (reportedErrors == MAXERRORS) {
              sleepInfo = fmt::format(", maximum error count ({}) reached, not reporting for the next {}", MAXERRORS, sleepTime);
            }
            LOGP(alarm, "EXCEPTION in processRawData: {} -> skipping part:{}/{} of spec:{}/{}/{}, size:{}, error count for subspec: {}{}", e.what(), dh->splitPayloadIndex, dh->splitPayloadParts,
                 dh->dataOrigin, dh->dataDescription, subSpecification, payloadSize, nErrorPerSubspec.at(subSpecification), sleepInfo);
            lastReport = now;
          }
        } else {
          lastReport = now;
          reportedErrors = 0;
        }
        continue;
      }
    }

    hasErrors |= snapshotCMVs(pc.outputs(), tfCounter);

    if (mWriteDebug || (mWriteDebugOnError && hasErrors)) {
      writeDebugOutput(tfCounter);
    }

    if (mWriteRawDataOnError && hasErrors) {
      writeRawData(pc.inputs());
    }

    // clear output
    initCMV();
  }

  void closeFiles()
  {
    LOGP(info, "closeFiles");

    if (mDebugStream) {
      // set some default aliases
      auto& stream = (*mDebugStream) << "cmvs";
      auto& tree = stream.getTree();
      tree.SetAlias("sector", "int(cru/10)");
      mDebugStream->Close();
      mDebugStream.reset(nullptr);
      mRawOutputFile.close();
    }
  }

  void stop() final
  {
    LOGP(info, "stop");
    closeFiles();
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    // ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    closeFiles();
  }

 private:
  /// CMV information for each cru
  struct CMVInfo {
    CMVInfo() = default;
    CMVInfo(const CMVInfo&) = default;
    CMVInfo(uint32_t orbit, uint16_t bc) : heartbeatOrbit(orbit), heartbeatBC(bc) {}

    uint32_t heartbeatOrbit{0};
    uint16_t heartbeatBC{0};

    bool operator==(const uint32_t orbit) const { return (heartbeatOrbit == orbit); }
    bool operator==(const CMVInfo& inf) const { return (inf.heartbeatOrbit == heartbeatOrbit) && (inf.heartbeatBC == heartbeatBC); }
    bool matches(uint32_t orbit, int16_t bc) const { return ((heartbeatOrbit == orbit) && (heartbeatBC == bc)); }
  };

  int mRawDataType{0};                                             ///< type of raw data to dump in case of errors
  bool mWriteDebug{false};                                         ///< write a debug output
  bool mWriteDebugOnError{false};                                  ///< write a debug output in case of errors
  bool mWriteRawDataOnError{false};                                ///< write raw data in case of errors
  std::vector<uint32_t> mCRUs;                                     ///< CRUs expected for this device
  std::unordered_map<uint32_t, std::vector<uint16_t>> mCMVvectors; ///< raw 16-bit CMV values per cru over all CMV packets in the TF
  std::unordered_map<uint32_t, std::vector<CMVInfo>> mCMVInfos;    ///< CMV packet information within the TF
  std::string mDebugStreamFileName;                                ///< name of the debug stream output file
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream;   ///< debug output streamer
  std::ofstream mRawOutputFile;                                    ///< raw output file
  std::string mRawOutputFileName;                                  ///< name of the raw output file

  //____________________________________________________________________________
  bool snapshotCMVs(DataAllocator& output, uint32_t tfCounter)
  {
    bool hasErrors = false;

    // send data per CRU with its own orbit/BC vector
    for (auto& [cru, cmvVec] : mCMVvectors) {
      const header::DataHeader::SubSpecificationType subSpec{cru << 7};
      const auto& infVec = mCMVInfos[cru];

      if (infVec.size() != 4) {
        // LOGP(error, "CRU {:3}: expected 4 packets per TF, got {}", cru, infVec.size());
        hasErrors = true;
      }
      if (cmvVec.size() != cmv::NTimeBinsPerPacket * infVec.size()) {
        // LOGP(error, "CRU {:3}: vector size {} does not match expected {}", cru, cmvVec.size(), cmv::NTimeBinsPerPacket * infVec.size());
        hasErrors = true;
      }

      std::vector<uint64_t> orbitBCInfo;
      orbitBCInfo.reserve(infVec.size());
      for (const auto& inf : infVec) {
        orbitBCInfo.emplace_back((uint64_t(inf.heartbeatOrbit) << 32) + uint64_t(inf.heartbeatBC));
      }

      LOGP(debug, "Sending CMVs for CRU {} of size {} ({} packets)", cru, cmvVec.size(), infVec.size());
      output.snapshot(Output{gDataOriginTPC, "CMVVECTOR", subSpec}, cmvVec);
      output.snapshot(Output{gDataOriginTPC, "CMVORBITS", subSpec}, orbitBCInfo);
    }

    return hasErrors;
  }

  //____________________________________________________________________________
  void initCMV()
  {
    for (const auto cruID : mCRUs) {
      auto& cmvVec = mCMVvectors[cruID];
      cmvVec.clear();

      auto& infosCRU = mCMVInfos[cruID];
      infosCRU.clear();
    }
  }

  //____________________________________________________________________________
  void writeDebugOutput(uint32_t tfCounter)
  {
    const auto& mapper = Mapper::instance();

    mDebugStream->GetFile()->cd();
    auto& stream = (*mDebugStream) << "cmvs";
    uint32_t seen = 0;
    static uint32_t firstOrbit = std::numeric_limits<uint32_t>::max();

    for (auto cru : mCRUs) {
      if (mCMVInfos.find(cru) == mCMVInfos.end()) {
        continue;
      }

      auto& infos = mCMVInfos[cru];
      auto& cmvVec = mCMVvectors[cru];

      stream << "cru=" << cru
             << "tfCounter=" << tfCounter
             << "nCMVs=" << cmvVec.size()
             << "cmvs=" << cmvVec
             << "\n";
    }
  }

  void writeRawData(InputRecord& inputs)
  {
    if (!mRawOutputFile.is_open()) {
      return;
    }

    using DataHeader = o2::header::DataHeader;

    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
    for (auto const& ref : InputRecordWalker(inputs, filter)) {
      auto dh = DataRefUtils::getHeader<header::DataHeader*>(ref);
      // LOGP(info, "write header: {}/{}/{}, payload size: {} / {}", dh->dataOrigin, dh->dataDescription, dh->subSpecification, dh->payloadSize, ref.payloadSize);
      if (((mRawDataType == 1) || (mRawDataType == 3)) && (dh->payloadSize == 2 * sizeof(o2::header::RAWDataHeader))) {
        continue;
      }

      if (mRawDataType < 2) {
        mRawOutputFile.write(ref.header, sizeof(DataHeader));
      }
      if (mRawDataType < 5) {
        mRawOutputFile.write(ref.payload, ref.payloadSize);
      }

      if (mRawDataType == 5) {
        const gsl::span<const char> raw = inputs.get<gsl::span<char>>(ref);
        try {
          o2::framework::RawParser parser(raw.data(), raw.size());
          for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
            const auto size = it.size();
            // skip empty packages (HBF open)
            if (size == 0) {
              continue;
            }

            auto rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
            const auto rdhVersion = RDHUtils::getVersion(rdhPtr);
            if (!rdhPtr || rdhVersion < 6) {
              throw std::runtime_error(fmt::format("could not get RDH from packet, or version {} < 6", rdhVersion).data());
            }

            // ---| extract hardware information to do the processing |---
            const auto feeId = (FEEIDType)RDHUtils::getFEEID(*rdhPtr);
            const auto link = rdh_utils::getLink(feeId);
            const auto detField = RDHUtils::getDetectorField(*rdhPtr);

            // only select CMVs
            if ((detField != (decltype(detField))RawDataType::CMV) || (link != rdh_utils::CMVLinkID)) {
              continue;
            }

            // write out raw data
            mRawOutputFile.write((const char*)it.raw(), RDHUtils::getMemorySize(rdhPtr));
          }
        } catch (...) {
        }
      }
    }
  }
};

o2::framework::DataProcessorSpec getCMVToVectorSpec(const std::string inputSpec, std::vector<uint32_t> const& crus)
{
  using device = o2::tpc::CMVToVectorDevice;

  std::vector<OutputSpec> outputs;
  for (const uint32_t cru : crus) {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    outputs.emplace_back(gDataOriginTPC, "CMVVECTOR", subSpec, Lifetime::Timeframe);
    outputs.emplace_back(gDataOriginTPC, "CMVORBITS", subSpec, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    fmt::format("tpc-cmv-to-vector"),
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<device>(crus)},
    Options{
      {"write-debug", VariantType::Bool, false, {"write a debug output tree"}},
      {"write-debug-on-error", VariantType::Bool, false, {"write a debug output tree in case errors occurred"}},
      {"debug-file-name", VariantType::String, "/tmp/cmv_vector_debug.{run}.root", {"name of the debug output file"}},
      {"write-raw-data-on-error", VariantType::Bool, false, {"dump raw data in case errors occurred"}},
      {"raw-file-name", VariantType::String, "/tmp/cmv_debug.{run}.{raw_type}", {"name of the raw output file"}},
      {"raw-data-type", VariantType::Int, 0, {"Which raw data to dump: 0-full TPC with DH, 1-full TPC with DH skip empty, 2-full TPC no DH, 3-full TPC no DH skip empty, 4-IDC raw only 5-CMV raw only"}},
      {"check-incomplete-hbf", VariantType::Bool, false, {"false: don't check; true: check and report"}},
    } // end Options
  }; // end DataProcessorSpec
}
} // namespace o2::tpc