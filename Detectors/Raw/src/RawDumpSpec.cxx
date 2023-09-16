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

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "DetectorsRaw/RawDumpSpec.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"
#include "CommonUtils/StringUtils.h"
#include "CommonConstants/Triggers.h"
#include "Algorithm/RangeTokenizer.h"
#include <cstdio>
#include <unordered_map>
#include <filesystem>

namespace o2::raw
{
namespace o2h = o2::header;
using namespace o2::framework;
using RDHUtils = o2::raw::RDHUtils;
using DetID = o2::detectors::DetID;

class RawDump : public Task
{
 public:
  static constexpr o2h::DataDescription DESCRaw{"RAWDATA"}, DESCCRaw{"CRAWDATA"};

  struct LinkInfo {
    FILE* fileHandler = nullptr;
    o2::header::RDHAny rdhSOX;
    o2::InteractionRecord firstIR{};
    bool hasSOX{false};
    DetID detID{};
  };

  RawDump(bool TOFUncompressed = false);
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  static std::string getReadoutType(DetID id);

 private:
  LinkInfo& getLinkInfo(o2h::DataOrigin detOr, const header::RDHAny* rdh);
  std::string getFileName(DetID detID, const header::RDHAny* rdh);
  std::string getBaseFileNameITS(const header::RDHAny* rdh);
  std::string getBaseFileNameTPC(const header::RDHAny* rdh);
  std::string getBaseFileNameTRD(const header::RDHAny* rdh);
  std::string getBaseFileNameTOF(const header::RDHAny* rdh);
  std::string getBaseFileNameEMC(const header::RDHAny* rdh);
  std::string getBaseFileNamePHS(const header::RDHAny* rdh);
  std::string getBaseFileNameCPV(const header::RDHAny* rdh);
  std::string getBaseFileNameMFT(const header::RDHAny* rdh);
  std::string getBaseFileNameMID(const header::RDHAny* rdh);
  std::string getBaseFileNameMCH(const header::RDHAny* rdh);
  std::string getBaseFileNameCTP(const header::RDHAny* rdh);
  std::string getBaseFileNameFT0(const header::RDHAny* rdh);
  std::string getBaseFileNameFV0(const header::RDHAny* rdh);
  std::string getBaseFileNameFDD(const header::RDHAny* rdh);
  std::string getBaseFileNameZDC(const header::RDHAny* rdh);
  std::string getBaseFileNameHMP(const header::RDHAny* rdh);

  bool mFatalOnDeadBeef{false};
  bool mSkipDump{false};
  bool mTOFUncompressed{false};
  bool mImposeSOX{false};
  int mVerbosity{0};
  int mTFCount{0};
  uint64_t mTPCLinkRej{0}; // pattern of TPC links to reject
  o2::InteractionRecord mFirstIR{};
  std::string mOutDir{};
  std::unordered_map<uint64_t, LinkInfo> mLinksInfo{};
  std::unordered_map<std::string, FILE*> mName2File{};
  std::unordered_map<int, DetID> mOrigin2DetID{};
  std::array<std::string, o2::detectors::DetID::getNDetectors()> mConfigEntries{};
  std::array<int, o2::detectors::DetID::getNDetectors()> mFilesPerDet{};
};

//________________________________________
RawDump::RawDump(bool TOFUncompressed) : mTOFUncompressed{TOFUncompressed}
{
  for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
    mOrigin2DetID[DetID::getDataOrigin(id)] = id;
  }
}

//________________________________________
void RawDump::init(InitContext& ic)
{
  mFatalOnDeadBeef = ic.options().get<bool>("fatal-on-deadbeef");
  mVerbosity = ic.options().get<int>("dump-verbosity");
  mOutDir = ic.options().get<std::string>("output-directory");
  mSkipDump = ic.options().get<bool>("skip-dump");
  mImposeSOX = !ic.options().get<bool>("skip-impose-sox");
  auto vrej = o2::RangeTokenizer::tokenize<int>(ic.options().get<std::string>("reject-tpc-links"));
  for (auto i : vrej) {
    if (i < 63) {
      mTPCLinkRej |= 0x1UL << i;
      LOGP(info, "Will reject TPC link {}", i);
    } else {
      LOGP(error, "LinkID cannot exceed 63, asked {}", i);
    }
  }
  if (mOutDir.size()) {
    if (!std::filesystem::exists(mOutDir)) {
#if defined(__clang__)
      // clang `create_directories` implementation is misbehaving and can
      // return false even if the directory is actually successfully created
      // so we work around that "feature" by not checking the
      // return value at all but using a second call to `exists`
      std::filesystem::create_directories(mOutDir);
      if (!std::filesystem::exists(mOutDir)) {
        LOG(fatal) << "could not create output directory " << mOutDir;
      }
#else
      if (!std::filesystem::create_directories(mOutDir)) {
        LOG(fatal) << "could not create output directory " << mOutDir;
      }
#endif
      LOGP(info, "Created output directory {}", mOutDir);
    }
  } else {
    mOutDir = "./";
  }
}

//________________________________________
void RawDump::run(ProcessingContext& pc)
{

  DPLRawParser parser(pc.inputs());

  auto procDEADBEEF = [this](const o2::header::DataHeader* dh) {
    static DetID::mask_t repDeadBeef{};
    if (dh->subSpecification == 0xdeadbeef && dh->payloadSize == 0) {
      if (this->mFatalOnDeadBeef) {
        LOGP(fatal, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{}", dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit);
      } else {
        if (!repDeadBeef[DetID(dh->dataOrigin.str)] || this->mVerbosity > 0) {
          LOGP(warn, "Skipping input [{}/{}/{:#x}] TF#{} 1st_orbit:{}", dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit);
          repDeadBeef |= DetID::getMask(dh->dataOrigin.str);
        }
        return false;
      }
    }
    return true;
  };

  auto isRORC = [](DetID id) {
    return id == DetID::PHS || id == DetID::EMC || id == DetID::HMP;
  };

  if (mTFCount == 0 && mImposeSOX) { // make sure all links payload starts with SOX
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* dh = it.o2DataHeader();
      if (!procDEADBEEF(dh)) {
        continue;
      }
      const auto rdh = reinterpret_cast<const header::RDHAny*>(it.raw());
      if (!RDHUtils::checkRDH(rdh, true)) {
        o2::raw::RDHUtils::printRDH(rdh);
        continue;
      }
      auto& lInfo = getLinkInfo(dh->dataOrigin, rdh);
      if (!lInfo.firstIR.isDummy()) { // already processed
        continue;
      }
      lInfo.firstIR = {0, o2::raw::RDHUtils::getTriggerOrbit(rdh)};
      if (lInfo.firstIR < mFirstIR) {
        mFirstIR = lInfo.firstIR;
      }
      auto trig = o2::raw::RDHUtils::getTriggerType(rdh);
      lInfo.hasSOX = trig & (o2::trigger::SOT | o2::trigger::SOC);
      lInfo.rdhSOX = *rdh;
      lInfo.detID = mOrigin2DetID[dh->dataOrigin];
    }
    // now write RDH with SOX (if needed)
    for (auto& lit : mLinksInfo) {
      auto& lInfo = lit.second;
      if (!lInfo.hasSOX && lInfo.fileHandler) {
        auto trig = o2::raw::RDHUtils::getTriggerType(lInfo.rdhSOX);
        if (o2::raw::RDHUtils::getTriggerIR(lInfo.rdhSOX) != mFirstIR) { // need to write cooked header
          o2::raw::RDHUtils::setTriggerOrbit(lInfo.rdhSOX, mFirstIR.orbit);
          o2::raw::RDHUtils::setOffsetToNext(lInfo.rdhSOX, sizeof(o2::header::RDHAny));
          o2::raw::RDHUtils::setMemorySize(lInfo.rdhSOX, sizeof(o2::header::RDHAny));
          o2::raw::RDHUtils::setPacketCounter(lInfo.rdhSOX, o2::raw::RDHUtils::getPacketCounter(lInfo.rdhSOX) - 1);
          trig = isRORC(lInfo.detID) ? o2::trigger::SOT : (o2::trigger::SOC | o2::trigger::ORBIT | o2::trigger::HB | o2::trigger::TF);
          o2::raw::RDHUtils::setTriggerType(lInfo.rdhSOX, trig);
          o2::raw::RDHUtils::setStop(lInfo.rdhSOX, 0x1);
          if (mVerbosity > 0) {
            LOGP(info, "Writing cooked up RDH with SOX");
            o2::raw::RDHUtils::printRDH(lInfo.rdhSOX);
          }
          auto ws = std::fwrite(&lInfo.rdhSOX, 1, sizeof(o2::header::RDHAny), lInfo.fileHandler);
          if (ws != sizeof(o2::header::RDHAny)) {
            LOGP(fatal, "Failed to write cooked up RDH with SOX");
          }
          lInfo.hasSOX = true; // flag that it is already written
        }                      // otherwhise data has RDH with orbit matching to SOX, we will simply set a flag while writing the data
      } else if (lInfo.fileHandler && o2::raw::RDHUtils::getTriggerOrbit(lInfo.rdhSOX) != mFirstIR.orbit) {
        o2::raw::RDHUtils::printRDH(lInfo.rdhSOX);
        LOGP(error, "Original data had SOX set but the orbit differs from the smallest seen {}, keep original one", mFirstIR.orbit);
      }
    }
  }

  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    auto const* dh = it.o2DataHeader();
    if (!procDEADBEEF(dh)) {
      continue;
    }
    const auto rdh = reinterpret_cast<const header::RDHAny*>(it.raw());
    if (!RDHUtils::checkRDH(rdh, true)) {
      o2::raw::RDHUtils::printRDH(rdh);
      continue;
    }
    if (mVerbosity > 1) {
      o2::raw::RDHUtils::printRDH(rdh);
    }
    FILE* fh = nullptr;
    auto& lInfo = getLinkInfo(dh->dataOrigin, rdh);
    if (!mSkipDump && lInfo.fileHandler) {
      auto sz = o2::raw::RDHUtils::getOffsetToNext(rdh);
      auto raw = it.raw();
      if (mTFCount == 0 && !lInfo.hasSOX && o2::raw::RDHUtils::getTriggerIR(rdh) == mFirstIR) { // need to add SOX bit to existing RDH
        auto trig = o2::raw::RDHUtils::getTriggerType(rdh);
        trig |= isRORC(lInfo.detID) ? o2::trigger::SOT : (o2::trigger::SOC | o2::trigger::ORBIT | o2::trigger::HB | o2::trigger::TF);
        auto rdhC = *rdh;
        o2::raw::RDHUtils::setTriggerType(rdhC, trig);
        if (mVerbosity > 0) {
          LOGP(info, "Write existing RDH with SOX added");
          o2::raw::RDHUtils::printRDH(rdhC);
        }
        auto ws = std::fwrite(&rdhC, 1, sizeof(o2::header::RDHAny), lInfo.fileHandler);
        if (ws != sizeof(o2::header::RDHAny)) {
          LOGP(fatal, "Failed to write RDH with SOX added");
        }
        raw += sizeof(o2::header::RDHAny);
        sz -= sizeof(o2::header::RDHAny);
        if (o2::raw::RDHUtils::getStop(rdhC)) {
          lInfo.hasSOX = true;
        }
      }
      auto ws = std::fwrite(raw, 1, sz, lInfo.fileHandler);
      if (ws != sz) {
        LOGP(fatal, "Failed to write payload of {} bytes", sz);
      }
    }
  }
  mTFCount++;
}

//____________________________________________________________
void RawDump::endOfStream(EndOfStreamContext& ec)
{
  LOGP(info, "closing {} output files", mName2File.size());
  for (auto h : mName2File) {
    std::fclose(h.second);
  }
  for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
    if (mConfigEntries[id].empty()) {
      continue;
    }
    auto fnm = fmt::format("{}{}{}raw.cfg", mOutDir, mOutDir.back() == '/' ? "" : "/", DetID::getName(id));
    auto fh = std::fopen(fnm.c_str(), "w");
    if (!fh) {
      LOGP(fatal, "Failed to create configuration file {}");
    }
    auto ws = std::fwrite(mConfigEntries[id].data(), 1, mConfigEntries[id].size(), fh);
    if (ws != mConfigEntries[id].size()) {
      LOGP(fatal, "Failed to write configuration to {}", fnm);
    }
    fclose(fh);
    LOGP(info, "Wrote configuration for {} {} raw files to {}", DetID::getName(id), mFilesPerDet[id], o2::utils::Str::getFullPath(fnm));
  }
}

//_____________________________________________________________________
RawDump::LinkInfo& RawDump::getLinkInfo(o2h::DataOrigin detOr, const header::RDHAny* rdh)
{
  uint32_t feeid = RDHUtils::getFEEID(rdh);
  uint64_t id = (uint64_t(detOr) << 32) + feeid;
  auto& linkInfo = mLinksInfo[id];
  if (!linkInfo.fileHandler) {
    DetID detID = mOrigin2DetID[detOr];
    auto name = getFileName(detID, rdh);
    if (name.empty()) {
      return linkInfo; // reject data of this RDH
    }
    linkInfo.fileHandler = mName2File[name];
    if (!linkInfo.fileHandler) {
      linkInfo.fileHandler = std::fopen(name.c_str(), "w");
      if (!linkInfo.fileHandler) {
        LOGP(fatal, "Failed to create file {} for Det={} / FeeID=0x{:05x}", name, detOr.str, feeid);
      }
      mName2File[name] = linkInfo.fileHandler;
      mConfigEntries[detID] += fmt::format(
        "[input-{}-{}]\n"
        "dataOrigin = {}\n"
        "dataDescription = {}\n"
        "readoutCard = {}\n"
        "filePath = {}\n\n",
        detOr.str, mFilesPerDet[detID]++, detOr.str, (detID != DetID::TOF || mTOFUncompressed) ? DESCRaw.str : DESCCRaw.str, getReadoutType(detID), o2::utils::Str::getFullPath(name));
    }
    if (mVerbosity > 0) {
      RDHUtils::printRDH(rdh);
      LOGP(info, "Write Det={}/0x{:05x} to {}", detOr.str, feeid, o2::utils::Str::getFullPath(name));
    }
  }
  return linkInfo;
}

//_____________________________________________________________________
std::string RawDump::getFileName(DetID detID, const header::RDHAny* rdh)
{
  // TODO
  std::string baseName{};
  switch (detID) {
    case DetID::ITS:
      baseName = getBaseFileNameITS(rdh);
      break;
    case DetID::MFT:
      baseName = getBaseFileNameMFT(rdh);
      break;
    case DetID::TPC:
      baseName = getBaseFileNameTPC(rdh);
      break;
    case DetID::TRD:
      baseName = getBaseFileNameTRD(rdh);
      break;
    case DetID::TOF:
      baseName = getBaseFileNameTOF(rdh);
      break;
    case DetID::EMC:
      baseName = getBaseFileNameEMC(rdh);
      break;
    case DetID::PHS:
      baseName = getBaseFileNamePHS(rdh);
      break;
    case DetID::CPV:
      baseName = getBaseFileNameCPV(rdh);
      break;
    case DetID::CTP:
      baseName = getBaseFileNameCTP(rdh);
      break;
    case DetID::MID:
      baseName = getBaseFileNameMID(rdh);
      break;
    case DetID::MCH:
      baseName = getBaseFileNameMCH(rdh);
      break;
    case DetID::FT0:
      baseName = getBaseFileNameFT0(rdh);
      break;
    case DetID::FV0:
      baseName = getBaseFileNameFV0(rdh);
      break;
    case DetID::FDD:
      baseName = getBaseFileNameFDD(rdh);
      break;
    case DetID::ZDC:
      baseName = getBaseFileNameZDC(rdh);
      break;
    case DetID::HMP:
      baseName = getBaseFileNameHMP(rdh);
      break;

    default:
      baseName = fmt::format("feeID0x{:05x}", RDHUtils::getFEEID(rdh));
      break;
  }
  return baseName.empty() ? std::string{} : fmt::format("{}{}{}_{}.raw", mOutDir, mOutDir.back() == '/' ? "" : "/", detID.getName(), baseName);
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameCTP(const header::RDHAny* rdh)
{
  return "alio2-cr1-flp163_cru1111_0";
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameFT0(const header::RDHAny* rdh)
{
  return fmt::format("alio2-cr1-flp200_cru{}_{}", RDHUtils::getCRUID(rdh), RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameFV0(const header::RDHAny* rdh)
{
  return fmt::format("alio2-cr1-flp180_cru{}_{}", RDHUtils::getCRUID(rdh), RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameFDD(const header::RDHAny* rdh)
{
  return fmt::format("alio2-cr1-flp201_cru{}_{}", RDHUtils::getCRUID(rdh), RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameHMP(const header::RDHAny* rdh)
{
  uint16_t cruID = RDHUtils::getCRUID(rdh);
  if (cruID < 120 || cruID > 123) {
    auto flpname = fmt::format("flp-unknown_crorc{}_{}_ddlID{}", cruID, RDHUtils::getLinkID(rdh), RDHUtils::getFEEID(rdh));
    LOGP(error, "Unrecognized HMP flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("alio2-cr1-flp{}_crorc{}_{}", cruID < 122 ? 160 : 161, cruID, RDHUtils::getLinkID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameEMC(const header::RDHAny* rdh)
{
  static const std::array<int, 20> CRORCID = {110, 112, 110, 112, 110, 112, 111, 113, 111, 113, 111, 113, 114, 116, 114, 116, 115, 117, 115, 117};                     // CRORC ID w.r.t SM
  static const std::array<int, 40> CRORCLink = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 1, 0, 1, 2, 3, 2, 3, 4, -1, 4, 5, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 1, 3, 2}; // CRORC link w.r.t FEE ID

  uint16_t ddlID = RDHUtils::getFEEID(rdh), crorc = -1, link = -1, flpID = -1;
  if (ddlID < 40) {
    crorc = CRORCID[ddlID / 2];
    link = CRORCLink[ddlID];
    flpID = ddlID <= 23 ? 146 : 147;
  } else if (ddlID == 44) {
    crorc = 111;
    link = 5;
    flpID = 146;
  } else if (ddlID == 45) {
    crorc = 117;
    link = 3;
    flpID = 147;
  } else {
    auto flpname = fmt::format("flp-unknown_crorc{}_{}_ddlID{}", RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), ddlID);
    LOGP(error, "Unrecognized EMC flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("alio2-cr1-flp{}_crorc{}_{}", flpID, crorc, link);
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNamePHS(const header::RDHAny* rdh)
{
  uint16_t ddlID = RDHUtils::getFEEID(rdh), crorc = -1, link = -1, flpID = -1;
  if (ddlID < 6) {
    flpID = 164;
    if (ddlID < 2) {
      crorc = 304;
      link = ddlID;
    } else {
      crorc = 243;
      link = ddlID - 2;
    }
  } else if (ddlID < 14) {
    flpID = 165;
    if (ddlID < 10) {
      crorc = 75;
    } else {
      crorc = 106;
    }
    link = (ddlID - 6) % 4;
  }
  if (crorc < 0) {
    auto flpname = fmt::format("flp-unknown_crorc{}_{}_ddlID{}", RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), ddlID);
    LOGP(error, "Unrecognized PHS flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("alio2-cr1-flp{}_crorc{}_{}", flpID, crorc, link);
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameTOF(const header::RDHAny* rdh)
{
  constexpr int NLinks = 72;
  static constexpr int CRUFROMLINK[NLinks] = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  static constexpr Int_t CRUID[4] = {225, 226, 227, 228};
  static constexpr Int_t FLPFROMCRU[4] = {179, 179, 178, 178};
  uint16_t feeID = RDHUtils::getFEEID(rdh);
  if (feeID > 71) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)), feeID);
    LOGP(error, "Unrecognized TOF flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("alio2-cr1-flp{}_cru{}_{}", FLPFROMCRU[CRUFROMLINK[feeID]], CRUID[CRUFROMLINK[feeID]], RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameMID(const header::RDHAny* rdh)
{
  uint16_t feeID = RDHUtils::getFEEID(rdh);
  if (feeID > 3) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)), feeID);
    LOGP(error, "Unrecognized MID flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("alio2-cr1-flp159_cru{}_{}", feeID / 2, RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameMCH(const header::RDHAny* rdh)
{
  uint16_t cruID = RDHUtils::getCRUID(rdh) & 0xff;
  if (cruID > 0x1f) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)), RDHUtils::getFEEID(rdh));
    LOGP(error, "Unrecognized MCH flp, setting to {}", flpname);
    return flpname;
  }

  return fmt::format("alio2-cr1-flp{}_cru{}_{}", 148 + cruID / 3, cruID, RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameZDC(const header::RDHAny* rdh)
{
  return fmt::format("alio2-cr1-flp181_cru{}_{}", RDHUtils::getCRUID(rdh), RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameCPV(const header::RDHAny* rdh)
{
  uint16_t feeid = RDHUtils::getFEEID(rdh);
  if (feeid > 2) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)), RDHUtils::getFEEID(rdh));
    LOGP(error, "Unrecognized CPV flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("alio2-cr1-flp162_cru{}_{}", RDHUtils::getCRUID(rdh), RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameITS(const header::RDHAny* rdh)
{
  static const std::unordered_map<int, std::string> cru2name = {
    {183, "alio2-cr1-flp187"},
    {172, "alio2-cr1-flp198"},
    {181, "alio2-cr1-flp188"},
    {196, "alio2-cr1-flp203"},
    {184, "alio2-cr1-flp189"},
    {191, "alio2-cr1-flp189"},
    {179, "alio2-cr1-flp190"},
    {192, "alio2-cr1-flp190"},
    {175, "alio2-cr1-flp191"},
    {182, "alio2-cr1-flp191"},
    {187, "alio2-cr1-flp192"},
    {176, "alio2-cr1-flp192"},
    {177, "alio2-cr1-flp193"},
    {178, "alio2-cr1-flp193"},
    {194, "alio2-cr1-flp194"},
    {174, "alio2-cr1-flp194"},
    {180, "alio2-cr1-flp195"},
    {193, "alio2-cr1-flp195"},
    {185, "alio2-cr1-flp196"},
    {189, "alio2-cr1-flp196"},
    {186, "alio2-cr1-flp197"},
    {195, "alio2-cr1-flp197"},
  };
  auto ent = cru2name.find(RDHUtils::getCRUID(rdh));
  if (ent == cru2name.end()) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)), RDHUtils::getFEEID(rdh));
    LOGP(error, "Unrecognized ITS flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("{}_cru{}_{}", ent->second, RDHUtils::getCRUID(rdh), RDHUtils::getEndPointID(rdh));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameMFT(const header::RDHAny* rdh)
{
  static const std::unordered_map<int, std::pair<int, std::string>> cru2name = {
    {0x800, {570, "alio2-cr1-flp182"}},
    {0x801, {542, "alio2-cr1-flp186"}},
    {0x802, {548, "alio2-cr1-flp183"}},
    {0x803, {211, "alio2-cr1-flp185"}},
    {0x804, {569, "alio2-cr1-flp184"}},
    {0x805, {543, "alio2-cr1-flp184"}},
    {0x806, {552, "alio2-cr1-flp185"}},
    {0x807, {554, "alio2-cr1-flp183"}},
    {0x808, {547, "alio2-cr1-flp186"}},
    {0x809, {567, "alio2-cr1-flp182"}},
  };
  auto ent = cru2name.find(RDHUtils::getCRUID(rdh));
  if (ent == cru2name.end()) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)), RDHUtils::getFEEID(rdh));
    LOGP(error, "Unrecognized MFT flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("{}_cru{}_{}", ent->second.second, RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameTPC(const header::RDHAny* rdh)
{
  constexpr int NFLP = 361;
  constexpr const char* CRU_FLPS[NFLP] = {
    "alio2-cr1-flp070", "alio2-cr1-flp069", "alio2-cr1-flp070", "alio2-cr1-flp069", "alio2-cr1-flp072", "alio2-cr1-flp071", "alio2-cr1-flp072", "alio2-cr1-flp071", "alio2-cr1-flp072", "alio2-cr1-flp071", "alio2-cr1-flp002", "alio2-cr1-flp001", "alio2-cr1-flp002", "alio2-cr1-flp001", "alio2-cr1-flp004", "alio2-cr1-flp003", "alio2-cr1-flp004", "alio2-cr1-flp003",
    "alio2-cr1-flp004", "alio2-cr1-flp003", "alio2-cr1-flp006", "alio2-cr1-flp005", "alio2-cr1-flp006", "alio2-cr1-flp005", "alio2-cr1-flp008", "alio2-cr1-flp007", "alio2-cr1-flp008", "alio2-cr1-flp007", "alio2-cr1-flp008", "alio2-cr1-flp007", "alio2-cr1-flp010", "alio2-cr1-flp009", "alio2-cr1-flp010", "alio2-cr1-flp009", "alio2-cr1-flp012", "alio2-cr1-flp011",
    "alio2-cr1-flp012", "alio2-cr1-flp011", "alio2-cr1-flp012", "alio2-cr1-flp011", "alio2-cr1-flp014", "alio2-cr1-flp013", "alio2-cr1-flp014", "alio2-cr1-flp013", "alio2-cr1-flp016", "alio2-cr1-flp015", "alio2-cr1-flp016", "alio2-cr1-flp015", "alio2-cr1-flp016", "alio2-cr1-flp015", "alio2-cr1-flp018", "alio2-cr1-flp017", "alio2-cr1-flp018", "alio2-cr1-flp017",
    "alio2-cr1-flp020", "alio2-cr1-flp019", "alio2-cr1-flp020", "alio2-cr1-flp019", "alio2-cr1-flp020", "alio2-cr1-flp019", "alio2-cr1-flp022", "alio2-cr1-flp021", "alio2-cr1-flp022", "alio2-cr1-flp021", "alio2-cr1-flp024", "alio2-cr1-flp023", "alio2-cr1-flp024", "alio2-cr1-flp023", "alio2-cr1-flp024", "alio2-cr1-flp023", "alio2-cr1-flp026", "alio2-cr1-flp025",
    "alio2-cr1-flp026", "alio2-cr1-flp025", "alio2-cr1-flp028", "alio2-cr1-flp027", "alio2-cr1-flp028", "alio2-cr1-flp027", "alio2-cr1-flp028", "alio2-cr1-flp027", "alio2-cr1-flp030", "alio2-cr1-flp029", "alio2-cr1-flp030", "alio2-cr1-flp029", "alio2-cr1-flp032", "alio2-cr1-flp031", "alio2-cr1-flp032", "alio2-cr1-flp031", "alio2-cr1-flp032", "alio2-cr1-flp031",
    "alio2-cr1-flp034", "alio2-cr1-flp033", "alio2-cr1-flp034", "alio2-cr1-flp033", "alio2-cr1-flp036", "alio2-cr1-flp035", "alio2-cr1-flp036", "alio2-cr1-flp035", "alio2-cr1-flp036", "alio2-cr1-flp035", "alio2-cr1-flp038", "alio2-cr1-flp037", "alio2-cr1-flp038", "alio2-cr1-flp037", "alio2-cr1-flp040", "alio2-cr1-flp039", "alio2-cr1-flp040", "alio2-cr1-flp039",
    "alio2-cr1-flp040", "alio2-cr1-flp039", "alio2-cr1-flp042", "alio2-cr1-flp041", "alio2-cr1-flp042", "alio2-cr1-flp041", "alio2-cr1-flp044", "alio2-cr1-flp043", "alio2-cr1-flp044", "alio2-cr1-flp043", "alio2-cr1-flp044", "alio2-cr1-flp043", "alio2-cr1-flp046", "alio2-cr1-flp045", "alio2-cr1-flp046", "alio2-cr1-flp045", "alio2-cr1-flp048", "alio2-cr1-flp047",
    "alio2-cr1-flp048", "alio2-cr1-flp047", "alio2-cr1-flp048", "alio2-cr1-flp047", "alio2-cr1-flp050", "alio2-cr1-flp049", "alio2-cr1-flp050", "alio2-cr1-flp049", "alio2-cr1-flp052", "alio2-cr1-flp051", "alio2-cr1-flp052", "alio2-cr1-flp051", "alio2-cr1-flp052", "alio2-cr1-flp051", "alio2-cr1-flp054", "alio2-cr1-flp053", "alio2-cr1-flp054", "alio2-cr1-flp053",
    "alio2-cr1-flp056", "alio2-cr1-flp055", "alio2-cr1-flp056", "alio2-cr1-flp055", "alio2-cr1-flp056", "alio2-cr1-flp055", "alio2-cr1-flp058", "alio2-cr1-flp057", "alio2-cr1-flp058", "alio2-cr1-flp057", "alio2-cr1-flp060", "alio2-cr1-flp059", "alio2-cr1-flp060", "alio2-cr1-flp059", "alio2-cr1-flp060", "alio2-cr1-flp059", "alio2-cr1-flp062", "alio2-cr1-flp061",
    "alio2-cr1-flp062", "alio2-cr1-flp061", "alio2-cr1-flp064", "alio2-cr1-flp063", "alio2-cr1-flp064", "alio2-cr1-flp063", "alio2-cr1-flp064", "alio2-cr1-flp063", "alio2-cr1-flp066", "alio2-cr1-flp065", "alio2-cr1-flp066", "alio2-cr1-flp065", "alio2-cr1-flp068", "alio2-cr1-flp067", "alio2-cr1-flp068", "alio2-cr1-flp067", "alio2-cr1-flp068", "alio2-cr1-flp067",
    "alio2-cr1-flp074", "alio2-cr1-flp073", "alio2-cr1-flp074", "alio2-cr1-flp073", "alio2-cr1-flp076", "alio2-cr1-flp075", "alio2-cr1-flp076", "alio2-cr1-flp075", "alio2-cr1-flp076", "alio2-cr1-flp075", "alio2-cr1-flp078", "alio2-cr1-flp077", "alio2-cr1-flp078", "alio2-cr1-flp077", "alio2-cr1-flp080", "alio2-cr1-flp079", "alio2-cr1-flp080", "alio2-cr1-flp079",
    "alio2-cr1-flp080", "alio2-cr1-flp079", "alio2-cr1-flp082", "alio2-cr1-flp081", "alio2-cr1-flp082", "alio2-cr1-flp081", "alio2-cr1-flp084", "alio2-cr1-flp083", "alio2-cr1-flp084", "alio2-cr1-flp083", "alio2-cr1-flp084", "alio2-cr1-flp083", "alio2-cr1-flp086", "alio2-cr1-flp085", "alio2-cr1-flp086", "alio2-cr1-flp085", "alio2-cr1-flp088", "alio2-cr1-flp087",
    "alio2-cr1-flp088", "alio2-cr1-flp087", "alio2-cr1-flp088", "alio2-cr1-flp087", "alio2-cr1-flp090", "alio2-cr1-flp089", "alio2-cr1-flp090", "alio2-cr1-flp089", "alio2-cr1-flp092", "alio2-cr1-flp091", "alio2-cr1-flp092", "alio2-cr1-flp091", "alio2-cr1-flp092", "alio2-cr1-flp091", "alio2-cr1-flp094", "alio2-cr1-flp093", "alio2-cr1-flp094", "alio2-cr1-flp093",
    "alio2-cr1-flp096", "alio2-cr1-flp095", "alio2-cr1-flp096", "alio2-cr1-flp095", "alio2-cr1-flp096", "alio2-cr1-flp095", "alio2-cr1-flp098", "alio2-cr1-flp097", "alio2-cr1-flp098", "alio2-cr1-flp097", "alio2-cr1-flp100", "alio2-cr1-flp099", "alio2-cr1-flp100", "alio2-cr1-flp099", "alio2-cr1-flp100", "alio2-cr1-flp099", "alio2-cr1-flp102", "alio2-cr1-flp101",
    "alio2-cr1-flp102", "alio2-cr1-flp101", "alio2-cr1-flp104", "alio2-cr1-flp103", "alio2-cr1-flp104", "alio2-cr1-flp103", "alio2-cr1-flp104", "alio2-cr1-flp103", "alio2-cr1-flp106", "alio2-cr1-flp105", "alio2-cr1-flp106", "alio2-cr1-flp105", "alio2-cr1-flp108", "alio2-cr1-flp107", "alio2-cr1-flp108", "alio2-cr1-flp107", "alio2-cr1-flp108", "alio2-cr1-flp107",
    "alio2-cr1-flp110", "alio2-cr1-flp109", "alio2-cr1-flp110", "alio2-cr1-flp109", "alio2-cr1-flp112", "alio2-cr1-flp111", "alio2-cr1-flp112", "alio2-cr1-flp111", "alio2-cr1-flp112", "alio2-cr1-flp111", "alio2-cr1-flp114", "alio2-cr1-flp113", "alio2-cr1-flp114", "alio2-cr1-flp113", "alio2-cr1-flp116", "alio2-cr1-flp115", "alio2-cr1-flp116", "alio2-cr1-flp115",
    "alio2-cr1-flp116", "alio2-cr1-flp115", "alio2-cr1-flp118", "alio2-cr1-flp117", "alio2-cr1-flp118", "alio2-cr1-flp117", "alio2-cr1-flp120", "alio2-cr1-flp119", "alio2-cr1-flp120", "alio2-cr1-flp119", "alio2-cr1-flp120", "alio2-cr1-flp119", "alio2-cr1-flp122", "alio2-cr1-flp121", "alio2-cr1-flp122", "alio2-cr1-flp121", "alio2-cr1-flp124", "alio2-cr1-flp123",
    "alio2-cr1-flp124", "alio2-cr1-flp123", "alio2-cr1-flp124", "alio2-cr1-flp123", "alio2-cr1-flp126", "alio2-cr1-flp125", "alio2-cr1-flp126", "alio2-cr1-flp125", "alio2-cr1-flp128", "alio2-cr1-flp127", "alio2-cr1-flp128", "alio2-cr1-flp127", "alio2-cr1-flp128", "alio2-cr1-flp127", "alio2-cr1-flp130", "alio2-cr1-flp129", "alio2-cr1-flp130", "alio2-cr1-flp129",
    "alio2-cr1-flp132", "alio2-cr1-flp131", "alio2-cr1-flp132", "alio2-cr1-flp131", "alio2-cr1-flp132", "alio2-cr1-flp131", "alio2-cr1-flp134", "alio2-cr1-flp133", "alio2-cr1-flp134", "alio2-cr1-flp133", "alio2-cr1-flp136", "alio2-cr1-flp135", "alio2-cr1-flp136", "alio2-cr1-flp135", "alio2-cr1-flp136", "alio2-cr1-flp135", "alio2-cr1-flp138", "alio2-cr1-flp137",
    "alio2-cr1-flp138", "alio2-cr1-flp137", "alio2-cr1-flp140", "alio2-cr1-flp139", "alio2-cr1-flp140", "alio2-cr1-flp139", "alio2-cr1-flp140", "alio2-cr1-flp139", "alio2-cr1-flp142", "alio2-cr1-flp141", "alio2-cr1-flp142", "alio2-cr1-flp141", "alio2-cr1-flp144", "alio2-cr1-flp143", "alio2-cr1-flp144", "alio2-cr1-flp143", "alio2-cr1-flp144", "alio2-cr1-flp143",
    "alio2-cr1-flp145"};

  if (mTPCLinkRej && (mTPCLinkRej & (0x1UL << RDHUtils::getLinkID(rdh)))) {
    return "";
  }

  int cru = RDHUtils::getCRUID(rdh);
  if (cru >= NFLP) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", RDHUtils::getCRUID(rdh), int(RDHUtils::getEndPointID(rdh)), RDHUtils::getFEEID(rdh));
    LOGP(error, "Unrecognized TPC flp, setting to {}", flpname);
    return flpname;
  }
  return fmt::format("{}_cru{}_{}", CRU_FLPS[cru], cru, int(RDHUtils::getEndPointID(rdh)));
}

//_____________________________________________________________________
std::string RawDump::getBaseFileNameTRD(const header::RDHAny* rdh)
{
  constexpr int NLinks = 72;
  struct TRDCRUMapping {
    int32_t flpid;       // hostname of flp
    int32_t cruHWID = 0; // cru ID taken from ecs
    int32_t HCID = 0;    // hcid of first link
  };
  constexpr TRDCRUMapping trdHWMap[NLinks / 2] = {
    {166, 250, 0}, {166, 583, 0}, {166, 585, 0}, {167, 248, 0}, {167, 249, 0}, {167, 596, 0}, {168, 246, 0}, {168, 247, 0}, {168, 594, 0}, {169, 252, 0}, {169, 253, 0}, {169, 254, 0}, {170, 245, 0}, {170, 593, 0}, {170, 595, 0}, {171, 258, 0}, {171, 259, 0}, {171, 260, 0}, {172, 579, 0}, {172, 581, 0}, {172, 586, 0}, {173, 578, 0}, {173, 580, 0}, {173, 597, 0}, {174, 256, 0}, {174, 582, 0}, {174, 587, 0}, {175, 251, 0}, {175, 255, 0}, {175, 588, 0}, {176, 264, 0}, {176, 591, 0}, {176, 592, 0}, {177, 263, 0}, {177, 589, 0}, {177, 590, 0}};

  // see DataFormatsTRD/RawData.h
  uint16_t feeID = RDHUtils::getFEEID(rdh);
  int ep = feeID & 0x1, supermodule = feeID >> 8, side = (feeID & (0x1 << 4)) & 0x1; // A=0, C=1
  int link = supermodule * 4 + side * 2 + ep, cru = link / 2;
  if (link >= NLinks) {
    auto flpname = fmt::format("flp-unknown_cru{}_ep{}_feeid0x{:05x}", cru, int(RDHUtils::getEndPointID(rdh)), RDHUtils::getFEEID(rdh));
    LOGP(error, "Got wrong link {}, setting TRF file name to unrecognized flp {}", flpname);
    return flpname;
  }
  return fmt::format("alio2-cr1-flp{}_cru{}_{}", trdHWMap[cru].flpid, trdHWMap[cru].cruHWID, ep);
}

//_____________________________________________________________________
std::string RawDump::getReadoutType(DetID id)
{
  return (id == DetID::EMC || id == DetID::HMP || id == DetID::PHS) ? "RORC" : "CRU";
}

//__________________________________________________________
DataProcessorSpec getRawDumpSpec(DetID::mask_t detMask, bool TOFUncompressed)
{
  std::vector<InputSpec> inputs;
  o2h::DataOrigin orig;
  for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
    if (detMask[id] && (orig = DetID::getDataOrigin(id)) != o2h::gDataOriginInvalid) {
      inputs.emplace_back(DetID::getName(id), ConcreteDataTypeMatcher{orig, (id != DetID::TOF || TOFUncompressed) ? RawDump::DESCRaw : RawDump::DESCCRaw}, Lifetime::Optional);
    }
  }
  return DataProcessorSpec{
    "rawdump",
    inputs,
    {},
    AlgorithmSpec{adaptFromTask<RawDump>(TOFUncompressed)},
    {ConfigParamSpec{"fatal-on-deadbeef", VariantType::Bool, false, {"produce fata if 0xdeadbeef received for some detector"}},
     ConfigParamSpec{"skip-dump", VariantType::Bool, false, {"do not produce binary data"}},
     ConfigParamSpec{"dump-verbosity", VariantType::Int, 0, {"0:minimal, 1:report Det/FeeID->filename, 2: print RDH"}},
     ConfigParamSpec{"reject-tpc-links", VariantType::String, "", {"comma-separated list TPC links to reject"}},
     ConfigParamSpec{"skip-impose-sox", VariantType::Bool, false, {"do not impose SOX for 1st TF"}},
     ConfigParamSpec{"output-directory", VariantType::String, "./", {"Output directory (create if needed)"}}}};
}

} // namespace o2::raw
