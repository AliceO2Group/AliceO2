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

#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/format.h>
#include <fmt/chrono.h>

#include "TFile.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
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
#include "CommonUtils/NameConf.h"
#include "DataFormatsTPC/Constants.h"
#include "CommonConstants/LHCConstants.h"
#include "CCDB/BasicCCDBManager.h"

#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/IDC.h"
#include "DataFormatsTPC/RawDataTypes.h"
#include "TPCBase/Utils.h"
#include "TPCBase/RDHUtils.h"
#include "TPCBase/Mapper.h"
#include "TPCWorkflow/ProcessingHelpers.h"

using namespace o2::framework;
using o2::constants::lhc::LHCMaxBunches;
using o2::header::gDataOriginTPC;
using o2::tpc::constants::LHCBCPERTIMEBIN;
using RDHUtils = o2::raw::RDHUtils;
using RawDataType = o2::tpc::raw_data_types::Type;

namespace o2::tpc
{

class IDCToVectorDevice : public o2::framework::Task
{
 public:
  using FEEIDType = rdh_utils::FEEIDType;
  IDCToVectorDevice(const std::vector<uint32_t>& crus) : mCRUs(crus) {}

  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    mWriteDebug = ic.options().get<bool>("write-debug");
    mWriteDebugOnError = ic.options().get<bool>("write-debug-on-error");
    mWriteRawDataOnError = ic.options().get<bool>("write-raw-data-on-error");
    mRawDataType = ic.options().get<int>("raw-data-type");

    mDebugStreamFileName = ic.options().get<std::string>("debug-file-name").data();
    mRawOutputFileName = ic.options().get<std::string>("raw-file-name").data();

    mSwapLinks = ic.options().get<bool>("swap-links");

    auto pedestalFile = ic.options().get<std::string>("pedestal-url");
    if (pedestalFile.length()) {
      if (pedestalFile.find("ccdb") != std::string::npos) {
        if (pedestalFile.find("-default") != std::string::npos) {
          pedestalFile = o2::base::NameConf::getCCDBServer();
        }
        LOGP(info, "Loading pedestals from ccdb: {}", pedestalFile);
        auto& cdb = o2::ccdb::BasicCCDBManager::instance();
        cdb.setURL(pedestalFile);
        if (cdb.isHostReachable()) {
          auto pedestalNoise = cdb.get<std::unordered_map<std::string, CalPad>>("TPC/Calib/PedestalNoise");
          try {
            if (!pedestalNoise) {
              throw std::runtime_error("Couldn't retrieve PedestaNoise map");
            }
            mPedestal = std::make_unique<CalPad>(pedestalNoise->at("Pedestals"));
          } catch (const std::exception& e) {
            LOGP(fatal, "could not load pedestals from {} ({}), required for IDC processing", pedestalFile, e.what());
          }
        } else {
          LOGP(fatal, "ccdb access to {} requested, but host is not reachable. Cannot load pedestals, required for IDC processing", pedestalFile);
        }
      } else {
        LOGP(info, "Loading pedestals from file: {}", pedestalFile);
        auto calPads = utils::readCalPads(pedestalFile, "Pedestals");
        if (calPads.size() != 1) {
          LOGP(fatal, "Pedestal could not be loaded from file {}, required for IDC processing", pedestalFile);
        } else {
          mPedestal.reset(calPads[0]);
        }
      }
    } else {
      LOGP(error, "No pedestal file set, IDCs will be without pedestal subtraction!");
    }

    initIDC();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto runNumber = processing_helpers::getRunNumber(pc);
    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}}; // TODO: Change to IDC when changed in DD
    const auto& mapper = Mapper::instance();

    // open files if necessary
    if ((mWriteDebug || mWriteDebugOnError) && !mDebugStream) {
      const auto debugFileName = fmt::format(mDebugStreamFileName, fmt::arg("run", runNumber));
      LOGP(info, "creating debug stream {}", debugFileName);
      mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>(debugFileName.data(), "recreate");
    }

    if (mWriteRawDataOnError && !mRawOutputFile.is_open()) {
      std::string_view rawType = (mRawDataType < 2) ? "tf" : "raw";
      if (mRawDataType == 4) {
        rawType = "idc.raw";
      }
      const auto rawFileName = fmt::format(mRawOutputFileName, fmt::arg("run", runNumber), fmt::arg("raw_type", rawType));
      LOGP(info, "creating raw debug file {}", rawFileName);
      mRawOutputFile.open(rawFileName, std::ios::binary);
    }

    uint32_t heartbeatOrbit = 0;
    uint32_t heartbeatBC = 0;
    uint32_t tfCounter = 0;
    bool first = true;
    bool hasErrors = false;

    CalPad* pedestals = mPedestal.get();

    for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      tfCounter = dh->tfCounter;
      const auto subSpecification = dh->subSpecification;
      auto payloadSize = DataRefUtils::getPayloadSize(ref);

      // ---| data loop |---
      const gsl::span<const char> raw = pc.inputs().get<gsl::span<char>>(ref);
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
          const uint32_t cruID = rdh_utils::getCRU(feeId);
          const auto endPoint = rdh_utils::getEndPoint(feeId);
          const auto detField = RDHUtils::getDetectorField(*rdhPtr);

          // only select IDCs
          // ToDo: cleanup once IDCs will be propagated not as RAWDATA, but IDC.
          if ((detField != (decltype(detField))RawDataType::IDC) || (link != rdh_utils::IDCLinkID)) {
            continue;
          }
          LOGP(debug, "IDC Processing firstTForbit {:9}, tfCounter {:5}, run {:6}, feeId {:6} ({:3}/{}/{:2})", dh->firstTForbit, dh->tfCounter, dh->runNumber, feeId, cruID, endPoint, link);

          if (std::find(mCRUs.begin(), mCRUs.end(), cruID) == mCRUs.end()) {
            LOGP(error, "IDC CRU {:3} not configured in CRUs, skipping", cruID);
            continue;
          }

          const CRU cru(cruID);
          const int sector = cru.sector();
          const auto& partInfo = mapper.getPartitionInfo(cru.partition());
          const int fecLinkOffsetCRU = (partInfo.getNumberOfFECs() + 1) / 2;
          const int fecSectorOffset = partInfo.getSectorFECOffset();
          const GlobalPadNumber regionPadOffset = Mapper::GLOBALPADOFFSET[cru.region()];
          const GlobalPadNumber numberPads = Mapper::PADSPERREGION[cru.region()];
          int sampaOnFEC{}, channelOnSAMPA{};
          auto& idcVec = mIDCvectors[cruID];
          auto& infoVec = mIDCInfos[cruID];

          assert(size == sizeof(idc::Container));
          auto data = it.data();
          auto& idcs = *((idc::Container*)(data));
          const uint32_t orbit = idcs.header.heartbeatOrbit;
          const uint32_t bc = idcs.header.heartbeatBC;
          // LOGP(info, "IDC Procssing orbit/BC: {:9}/{:4}", orbit, bc);

          auto infoIt = std::find(infoVec.begin(), infoVec.end(), orbit);
          if (!infoVec.size()) {
            infoVec.emplace_back(orbit, bc);
            infoIt = infoVec.end() - 1;
          } else if (infoIt == infoVec.end()) {
            auto& lastInfo = infoVec.back();
            if ((orbit - lastInfo.heartbeatOrbit) != mNOrbitsIDC) {
              LOGP(error, "received packet with invalid jump in idc orbit ({} - {} == {} != {})", orbit, lastInfo.heartbeatOrbit, int(orbit) - int(lastInfo.heartbeatOrbit), mNOrbitsIDC);
              hasErrors = true;
            }
            infoVec.emplace_back(orbit, bc);
            infoIt = infoVec.end() - 1;
          }

          // check if end poit was already processed
          auto& lastInfo = *infoIt;
          if (lastInfo.wasEPseen(endPoint)) {
            LOGP(debug, "Already received another data packet for CRU {}, ep {}, orbit {}, bc {}", cruID, endPoint, orbit, bc);
            continue;
          }

          lastInfo.setEPseen(endPoint);
          // idc value offset in present time frame
          const size_t idcOffset = std::distance(infoVec.begin(), infoIt);

          // TODO: for debugging, remove later
          // LOGP(info, "processing IDCs for CRU {}, ep {}, feeId {:6} ({:3}/{}/{:2}), detField: {}, orbit {}, bc {}, idcOffset {}, idcVec size {}, epSeen {:02b}", cruID, endPoint, feeId, cruID, endPoint, link, detField, orbit, bc, idcOffset, idcVec.size(), lastInfo.epSeen);

          const float norm = 1. / float(mTimeStampsPerIntegrationInterval);
          for (uint32_t iLinkTmp = 0; iLinkTmp < Mapper::LinksPerRegionPerEndpoint[cru.region()][endPoint]; ++iLinkTmp) {
            const uint32_t iLink = mSwapLinks ? (idc::Links - iLinkTmp - 1) : iLinkTmp;

            if (!idcs.hasLink(iLink)) {
              continue;
            }

            const int fecInSector = iLinkTmp + endPoint * fecLinkOffsetCRU + fecSectorOffset;

            for (uint32_t iChannel = 0; iChannel < idc::Channels; ++iChannel) {
              auto val = idcs.getChannelValueFloat(iLink, iChannel);
              Mapper::getSampaAndChannelOnFEC(cruID, iChannel, sampaOnFEC, channelOnSAMPA);
              const GlobalPadNumber padInSector = mapper.globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
              if (pedestals) {
                val -= pedestals->getValue(sector, padInSector) * mTimeStampsPerIntegrationInterval;
                val *= norm;
              }
              const GlobalPadNumber padInRegion = padInSector - regionPadOffset;
              const GlobalPadNumber vectorPosition = padInRegion + idcOffset * numberPads;
              // TODO: for debugging, remove later
              // auto rawVal = idcs.getChannelValue(iLink, iChannel);
              // auto rawValF = idcs.getChannelValueFloat(iLink, iChannel);
              // LOGP(info, "filling channel {}, link {}, fecLinkOffsetCRU {:2}, fecSectorOffset {:3}, fecInSector {:3}, idcVec[{} ({})] = {} ({} / {})", iChannel, iLink, fecLinkOffsetCRU, fecSectorOffset, fecInSector, vectorPosition, padInRegion, val, rawVal, rawValF);
              idcVec[vectorPosition] = val;
            }
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
            LOGP(alarm, "EXCEPTIION in processRawData: {} -> skipping part:{}/{} of spec:{}/{}/{}, size:{}, error count for subspec: {}{}", e.what(), dh->splitPayloadIndex, dh->splitPayloadParts,
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

    hasErrors |= snapshotIDCs(pc.outputs(), tfCounter);

    if (mWriteDebug || (mWriteDebugOnError && hasErrors)) {
      writeDebugOutput(tfCounter);
    }

    if (mWriteRawDataOnError && hasErrors) {
      writeRawData(pc.inputs());
    }

    // clear output
    initIDC();
  }

  void closeFiles()
  {
    LOGP(info, "closeFiles");

    if (mDebugStream) {
      // set some default aliases
      auto& stream = (*mDebugStream) << "idcs";
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
  /// IDC information for each cru
  struct IDCInfo {
    IDCInfo() = default;
    IDCInfo(const IDCInfo&) = default;
    IDCInfo(uint32_t orbit, uint16_t bc) : heartbeatOrbit(orbit), heartbeatBC(bc) {}

    uint32_t heartbeatOrbit{0};
    uint16_t heartbeatBC{0};
    uint16_t epSeen{0};

    bool operator==(const uint32_t orbit) const { return (heartbeatOrbit == orbit); }
    bool operator==(const IDCInfo& inf) const { return (inf.heartbeatOrbit == heartbeatOrbit) && (inf.heartbeatBC == heartbeatBC) && (inf.epSeen == epSeen); }
    void setEPseen(uint32_t ep) { epSeen |= uint16_t(1 << ep); }
    bool wasEPseen(uint32_t ep) const { return epSeen & uint16_t(1 << ep); }
    bool matches(uint32_t orbit, int16_t bc) const { return ((heartbeatOrbit == orbit) && (heartbeatBC == bc)); }
    bool hasBothEPs() const { return epSeen == 3; }
  };

  const int mNOrbitsIDC{12};                                                                    ///< number of orbits over which IDCs are integrated, TODO: take from IDC header
  const int mTimeStampsPerIntegrationInterval{(LHCMaxBunches * mNOrbitsIDC) / LHCBCPERTIMEBIN}; ///< number of time stamps for each integration interval (5346)
  const uint32_t mMaxIDCPerTF{uint32_t(std::ceil(256.f / mNOrbitsIDC))};                        ///< maximum number of IDCs expected per TF, TODO: better way to get max number of orbits
  int mRawDataType{0};                                                                          ///< type of raw data to dump in case of errors
  bool mSwapLinks{false};                                                                       ///< swap links to circumvent bug in FW
  bool mWriteDebug{false};                                                                      ///< write a debug output
  bool mWriteDebugOnError{false};                                                               ///< write a debug output in case of errors
  bool mWriteRawDataOnError{false};                                                             ///< write raw data in case of errors
  std::vector<uint32_t> mCRUs;                                                                  ///< CRUs expected for this device
  std::unordered_map<uint32_t, std::vector<float>> mIDCvectors;                                 ///< decoded IDCs per cru for each pad in the region over all IDC packets in the TF
  std::unordered_map<uint32_t, std::vector<IDCInfo>> mIDCInfos;                                 ///< IDC packet information within the TF
  std::string mDebugStreamFileName;                                                             ///< name of the debug stream output file
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream;                                ///< debug output streamer
  std::unique_ptr<CalPad> mPedestal{};                                                          ///< noise and pedestal values
  std::ofstream mRawOutputFile;                                                                 ///< raw output file
  std::string mRawOutputFileName;                                                               ///< name of the raw output file

  //____________________________________________________________________________
  bool snapshotIDCs(DataAllocator& output, uint32_t tfCounter)
  {
    LOGP(debug, "snapshotIDCs");

    // check integrety of data between CRUs
    size_t packetsInTF = 0;
    std::vector<IDCInfo> const* infVecComp = nullptr;
    std::vector<uint64_t> orbitBCInfo;
    bool hasErrors = false;

    for (const auto& [cru, infVec] : mIDCInfos) {
      packetsInTF = std::max(infVec.size(), packetsInTF);
    }

    for (const auto& [cru, infVec] : mIDCInfos) {

      for (const auto& inf : infVec) {
        if (!inf.hasBothEPs()) {
          LOGP(error, "IDC CRU {:3}: data missing at ({:8}, {:4}) for one or both end points {:02b} in TF {}", cru, inf.heartbeatOrbit, inf.heartbeatBC, inf.epSeen, tfCounter);
          hasErrors = true;
        }
      }

      if (!infVecComp) {
        infVecComp = &infVec;
        std::for_each(infVec.begin(), infVec.end(), [&orbitBCInfo](const auto& inf) { orbitBCInfo.emplace_back((uint64_t(inf.heartbeatOrbit) << 32) + uint64_t(inf.heartbeatBC)); });
        continue;
      }

      if (packetsInTF != infVec.size()) {
        LOGP(error, "IDC CRU {:3}: number of IDC packets {} does not match max over all CRUs {} in TF {}", cru, packetsInTF, infVec.size(), tfCounter);
        hasErrors = true;
      }

      if (!std::equal(infVecComp->begin(), infVecComp->end(), infVec.begin())) {
        LOGP(error, "IDC CRU {:3}: mismatch in orbit numbers", cru);
        hasErrors = true;
      }
    }

    // send data
    for (auto& [cru, idcVec] : mIDCvectors) {
      idcVec.resize(Mapper::PADSPERREGION[CRU(cru).region()] * packetsInTF);
      const header::DataHeader::SubSpecificationType subSpec{cru << 7};
      LOGP(debug, "Sending IDCs for CRU {} of size {}", cru, idcVec.size());
      output.snapshot(Output{gDataOriginTPC, "IDCVECTOR", subSpec}, idcVec);
      output.snapshot(Output{gDataOriginTPC, "IDCORBITS", subSpec}, orbitBCInfo);
    }

    return hasErrors;
  }

  //____________________________________________________________________________
  void initIDC()
  {
    for (const auto cruID : mCRUs) {
      const CRU cru(cruID);
      const GlobalPadNumber numberPads = Mapper::PADSPERREGION[cru.region()] * mMaxIDCPerTF;
      auto& idcVec = mIDCvectors[cruID];
      idcVec.resize(numberPads);
      std::fill(idcVec.begin(), idcVec.end(), -1.f);

      auto& infosCRU = mIDCInfos[cruID];
      infosCRU.clear();
    }
  }

  //____________________________________________________________________________
  void writeDebugOutput(uint32_t tfCounter)
  {
    const auto& mapper = Mapper::instance();

    mDebugStream->GetFile()->cd();
    auto& stream = (*mDebugStream) << "idcs";
    uint32_t seen = 0;
    static uint32_t firstOrbit = std::numeric_limits<uint32_t>::max();

    for (auto cru : mCRUs) {
      if (mIDCInfos.find(cru) == mIDCInfos.end()) {
        continue;
      }
      auto& infos = mIDCInfos[cru];
      auto& idcVec = mIDCvectors[cru];

      for (int i = 0; i < infos.size(); ++i) {
        auto& info = infos[i];

        if (firstOrbit == std::numeric_limits<uint32_t>::max()) {
          firstOrbit = info.heartbeatOrbit;
        }
        auto idcFirst = idcVec.begin() + i * Mapper::PADSPERREGION[cru % Mapper::NREGIONS];
        auto idcLast = idcFirst + Mapper::PADSPERREGION[cru % Mapper::NREGIONS];
        std::vector<float> idcs(idcFirst, idcLast);
        std::vector<short> cpad(idcs.size());
        std::vector<short> row(idcs.size());
        for (int ipad = 0; ipad < idcs.size(); ++ipad) {
          const auto& padPos = mapper.padPos(ipad + Mapper::GLOBALPADOFFSET[cru % Mapper::NREGIONS]);
          row[ipad] = (short)padPos.getRow();
          const short pads = (short)mapper.getNumberOfPadsInRowSector(row[ipad]);
          cpad[ipad] = (short)padPos.getPad() - pads / 2;
        }
        auto idcSort = idcs;
        std::sort(idcSort.begin(), idcSort.end());
        const auto idcSize = idcSort.size();
        float median = idcSize % 2 ? idcSort[idcSize / 2] : (idcSort[idcSize / 2] + idcSort[idcSize / 2 - 1]) / 2.f;
        // outlier removal
        auto itEnd = idcSort.end();
        while (std::abs(*(itEnd - 1) - median) > 40) {
          --itEnd;
        }

        float mean = 0;
        const auto nForMean = std::distance(idcSort.begin(), itEnd);
        if (nForMean > 0) {
          mean = std::accumulate(idcSort.begin(), itEnd, 0.f) / float(nForMean);
        }
        uint32_t outliers = uint32_t(idcSort.size() - nForMean);

        stream << "cru=" << cru
               << "entry=" << i
               << "epSeen=" << info.epSeen
               << "tfCounter=" << tfCounter
               << "firstOrbit=" << firstOrbit
               << "orbit=" << info.heartbeatOrbit
               << "bc=" << info.heartbeatBC
               << "idcs=" << idcs
               << "cpad=" << cpad
               << "row=" << row
               << "outliers=" << outliers
               << "idc_mean=" << mean
               << "idc_median=" << median
               << "\n";
      }
    }
  }

  void writeRawData(InputRecord& inputs)
  {
    if (!mRawOutputFile.is_open()) {
      return;
    }

    using DataHeader = o2::header::DataHeader;

    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}}; // TODO: Change to IDC when changed in DD
    for (auto const& ref : InputRecordWalker(inputs, filter)) {
      auto dh = DataRefUtils::getHeader<header::DataHeader*>(ref);
      // LOGP(info, "write header: {}/{}/{}, payload size: {} / {}", dh->dataOrigin, dh->dataDescription, dh->subSpecification, dh->payloadSize, ref.payloadSize);
      if (((mRawDataType == 1) || (mRawDataType == 3)) && (dh->payloadSize == 2 * sizeof(o2::header::RAWDataHeader))) {
        continue;
      }

      if (mRawDataType < 2) {
        mRawOutputFile.write(ref.header, sizeof(DataHeader));
      }
      if (mRawDataType < 4) {
        mRawOutputFile.write(ref.payload, ref.payloadSize);
      }

      if (mRawDataType == 4) {
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

            // only select IDCs
            if ((detField != (decltype(detField))RawDataType::IDC) || (link != rdh_utils::IDCLinkID)) {
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

o2::framework::DataProcessorSpec getIDCToVectorSpec(const std::string inputSpec, std::vector<uint32_t> const& crus)
{
  using device = o2::tpc::IDCToVectorDevice;

  std::vector<OutputSpec> outputs;
  for (const uint32_t cru : crus) {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    outputs.emplace_back(gDataOriginTPC, "IDCVECTOR", subSpec, Lifetime::Timeframe);
    outputs.emplace_back(gDataOriginTPC, "IDCORBITS", subSpec, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    fmt::format("tpc-idc-to-vector"),
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<device>(crus)},
    Options{
      {"write-debug", VariantType::Bool, false, {"write a debug output tree"}},
      {"write-debug-on-error", VariantType::Bool, false, {"write a debug output tree in case errors occurred"}},
      {"debug-file-name", VariantType::String, "/tmp/idc_vector_debug.{run}.root", {"name of the debug output file"}},
      {"write-raw-data-on-error", VariantType::Bool, false, {"dump raw data in case errors occurred"}},
      {"raw-file-name", VariantType::String, "/tmp/idc_debug.{run}.{raw_type}", {"name of the raw output file"}},
      {"raw-data-type", VariantType::Int, 0, {"Which raw data to dump: 0-full TPC with DH, 1-full TPC with DH skip empty, 2-full TPC no DH, 3-full TPC no DH skip empty, 4-IDC raw only"}},
      {"pedestal-url", VariantType::String, "ccdb-default", {"ccdb-default: load from NameConf::getCCDBServer() OR ccdb url (must contain 'ccdb' OR pedestal file name"}},
      {"swap-links", VariantType::Bool, false, {"swap links to circumvent bug in FW"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
