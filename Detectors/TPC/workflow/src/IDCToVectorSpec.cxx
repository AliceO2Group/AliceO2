// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <cmath>
#include "fmt/format.h"

#include "TFile.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/InputRecordWalker.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsTPC/Constants.h"
#include "CommonConstants/LHCConstants.h"

#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/IDC.h"
#include "TPCBase/Utils.h"
#include "TPCBase/RDHUtils.h"
#include "TPCBase/Mapper.h"

using namespace o2::framework;
using o2::constants::lhc::LHCMaxBunches;
using o2::header::gDataOriginTPC;
using o2::tpc::constants::LHCBCPERTIMEBIN;
using RDHUtils = o2::raw::RDHUtils;

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
    if (ic.options().get<bool>("write-debug")) {
      mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>("idc_vector_debug.root", "recreate");
    }
    const auto pedestalFile = ic.options().get<std::string>("pedestal-file");
    if (pedestalFile.length()) {
      LOGP(info, "Setting pedestal file: {}", pedestalFile);
      auto calPads = utils::readCalPads(pedestalFile, "Pedestals,Noise");
      if (calPads.size() != 2) {
        LOGP(error, "Pedestals and noise not loaded correctly, size {} != 2", calPads.size());
      } else {
        for (auto p : calPads) {
          mNoisePedestal.emplace_back(p);
        }
      }
    }

    initIDC();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}}; // TODO: Change to IDC when changed in DD
    const auto& mapper = Mapper::instance();

    uint32_t heartbeatOrbit = 0;
    uint32_t heartbeatBC = 0;
    uint32_t tfCounter = 0;
    bool first = true;

    for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      // ---| extract hardware information to do the processing |---
      const auto feeId = (FEEIDType)dh->subSpecification;
      const auto link = rdh_utils::getLink(feeId);
      const uint32_t cruID = rdh_utils::getCRU(feeId);
      const auto endPoint = rdh_utils::getEndPoint(feeId);
      tfCounter = dh->tfCounter;

      // only select IDCs
      // ToDo: cleanup once IDCs will be propagated not as RAWDATA, but IDC.
      if (link != rdh_utils::IDCLinkID) {
        continue;
      }
      LOGP(info, "IDC Processing firstTForbit {:9}, tfCounter {:5}, run {:6}, feeId {:6} ({:3}/{}/{:2})", dh->firstTForbit, dh->tfCounter, dh->runNumber, feeId, cruID, endPoint, link);

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

      CalPad* pedestals = nullptr;
      if (mNoisePedestal.size() && mNoisePedestal[0]) {
        pedestals = mNoisePedestal[0].get();
      }

      // ---| data loop |---
      const gsl::span<const char> raw = pc.inputs().get<gsl::span<char>>(ref);
      o2::framework::RawParser parser(raw.data(), raw.size());
      for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
        const auto size = it.size();
        assert(size == sizeof(idc::Container));
        auto data = it.data();
        auto& idcs = *((idc::Container*)(data));
        const uint32_t orbit = idcs.header.heartbeatOrbit;
        const uint32_t bc = idcs.header.heartbeatBC;
        //LOGP(info, "IDC Procssing orbit/BC: {:9}/{:4}", orbit, bc);

        auto infoIt = std::find(infoVec.begin(), infoVec.end(), orbit);
        if (!infoVec.size()) {
          infoVec.emplace_back(orbit, bc);
          infoIt = infoVec.end() - 1;
          //} else if (!infoVec.back().matches(orbit, bc)) {
        } else if (infoIt == infoVec.end()) {
          auto& lastInfo = infoVec.back();
          if ((orbit - lastInfo.heartbeatOrbit) != mNOrbitsIDC) {
            LOGP(error, "received packet with invalid jump in idc orbit ({} - {} == {} != {})", orbit, lastInfo.heartbeatOrbit, orbit - lastInfo.heartbeatOrbit, mNOrbitsIDC);
          }
          infoVec.emplace_back(orbit, bc);
          infoIt = infoVec.end() - 1;
        }

        // check if end poit was already processed
        auto& lastInfo = *infoIt;
        if (lastInfo.wasEPseen(endPoint)) {
          LOGP(info, "Already received another data packet for CRU {}, ep {}, orbit {}, bc {}", cruID, endPoint, orbit, bc);
          continue;
        }

        lastInfo.setEPseen(endPoint);
        // idc value offset in present time frame
        const size_t idcOffset = std::distance(infoVec.begin(), infoIt);

        // TODO: for debugging, remove later
        /*
        auto* rdhPtr = it.get_if<o2::header::RAWDataHeaderV6>();
        const auto feeId2 = (FEEIDType)RDHUtils::getFEEID(*rdhPtr);
        const auto link2 = rdh_utils::getLink(feeId2);
        const uint32_t cruID2 = rdh_utils::getCRU(feeId2);
        const auto endPoint2 = rdh_utils::getEndPoint(feeId2);
        const auto detField = RDHUtils::getDetectorField(*rdhPtr);
        LOGP(info, "processing IDCs for CRU {}, ep {}, feeId {:6} ({:3}/{}/{:2}), detField: {}, orbit {}, bc {}, idcOffset {}, idcVec size {}, epSeen {:02b}", cruID, endPoint, feeId2, cruID2, endPoint2, link2, detField, orbit, bc, idcOffset, idcVec.size(), lastInfo.epSeen);
        */

        const float norm = 1. / float(mTimeStampsPerIntegrationInterval);
        for (uint32_t iLink = 0; iLink < idc::Links; ++iLink) {
          if (!idcs.hasLink(iLink)) {
            continue;
          }
          const int fecInSector = iLink + endPoint * fecLinkOffsetCRU + fecSectorOffset;

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
            //auto rawVal = idcs.getChannelValue(iLink, iChannel);
            //auto rawValF = idcs.getChannelValueFloat(iLink, iChannel);
            //LOGP(info, "filling channel {}, link {}, fecLinkOffsetCRU {:2}, fecSectorOffset {:3}, fecInSector {:3}, idcVec[{} ({})] = {} ({} / {})", iChannel, iLink, fecLinkOffsetCRU, fecSectorOffset, fecInSector, vectorPosition, padInRegion, val, rawVal, rawValF);
            idcVec[vectorPosition] = val;
          }
        }
      }
    }

    if (mDebugStream) {
      writeDebugOutput(tfCounter);
    }
    snapshotIDCs(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    if (mDebugStream) {
      mDebugStream->Close();
    }
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
  std::vector<uint32_t> mCRUs;                                                                  ///< CRUs expected for this device
  std::unordered_map<uint32_t, std::vector<float>> mIDCvectors;                                 ///< decoded IDCs per cru for each pad in the region over all IDC packets in the TF
  std::unordered_map<uint32_t, std::vector<IDCInfo>> mIDCInfos;                                 ///< IDC packet information within the TF
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream;                                ///< debug output streamer
  std::vector<std::unique_ptr<CalPad>> mNoisePedestal{};                                        ///< noise and pedestal values

  //____________________________________________________________________________
  void snapshotIDCs(DataAllocator& output)
  {
    LOGP(info, "snapshotIDCs");

    // check integrety of data between CRUs
    size_t orbitsInTF = 0;
    std::vector<IDCInfo> const* infVecComp = nullptr;
    std::vector<uint64_t> orbitBCInfo;

    for (const auto& [cru, infVec] : mIDCInfos) {

      for (const auto& inf : infVec) {
        if (!inf.hasBothEPs()) {
          LOGP(fatal, "IDC CRU {:3}: data missing at ({:8}, {:4}) for one or both end points {:02b}", cru, inf.heartbeatOrbit, inf.heartbeatBC, inf.epSeen);
        }
      }

      if (!infVecComp) {
        infVecComp = &infVec;
        orbitsInTF = infVec.size();
        std::for_each(infVec.begin(), infVec.end(), [&orbitBCInfo](const auto& inf) { orbitBCInfo.emplace_back((uint64_t(inf.heartbeatOrbit) << 32) + uint64_t(inf.heartbeatBC)); });
        continue;
      }

      if (orbitsInTF != infVec.size()) {
        LOGP(fatal, "IDC CRU {:3}: unequal number of IDC values {} != {}", cru, orbitsInTF, infVec.size());
      }

      if (!std::equal(infVecComp->begin(), infVecComp->end(), infVec.begin())) {
        LOGP(fatal, "IDC CRU {:3}: mismatch in orbits");
      }
    }

    // send data
    for (auto& [cru, idcVec] : mIDCvectors) {
      idcVec.resize(Mapper::PADSPERREGION[CRU(cru).region()] * orbitsInTF);
      const header::DataHeader::SubSpecificationType subSpec{cru << 7};
      output.snapshot(Output{gDataOriginTPC, "IDCVECTOR", subSpec}, idcVec);
      output.snapshot(Output{gDataOriginTPC, "IDCORBITS", subSpec}, orbitBCInfo);
    }

    // clear output
    initIDC();
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
    for (auto cru : mCRUs) {
      if (mIDCInfos.find(cru) == mIDCInfos.end()) {
        continue;
      }
      auto& infos = mIDCInfos[cru];
      auto& idcVec = mIDCvectors[cru];

      for (int i = 0; i < infos.size(); ++i) {
        auto& info = infos[i];

        auto idcFirst = idcVec.begin() + i * Mapper::PADSPERREGION[cru];
        auto idcLast = idcFirst + Mapper::PADSPERREGION[cru];
        std::vector<float> idcs(idcFirst, idcLast);
        std::vector<short> cpad(idcs.size());
        std::vector<short> row(idcs.size());
        for (int ipad = 0; ipad < idcs.size(); ++ipad) {
          const auto& padPos = mapper.padPos(ipad + Mapper::GLOBALPADOFFSET[cru % 10]);
          row[ipad] = (short)padPos.getRow();
          const short pads = (short)mapper.getNumberOfPadsInRowSector(row[ipad]);
          cpad[ipad] = (short)padPos.getPad() - pads / 2;
        }
        float mean = std::accumulate(idcs.begin(), idcs.end(), 0.f) / float(idcs.size());

        stream << "cru=" << cru
               << "epSeen=" << info.epSeen
               << "tfCounter=" << tfCounter
               << "orbit=" << info.heartbeatOrbit
               << "bc=" << info.heartbeatBC
               << "idcs=" << idcs
               << "cpad=" << cpad
               << "row=" << row
               << "idc_mean=" << mean
               << "\n";
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
      {"write-debug", VariantType::Bool, false, {"write a debug output tree."}},
      {"pedestal-file", VariantType::String, "", {"file with pedestals and noise for zero suppression"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
