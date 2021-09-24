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

///
/// \file    DataDecoder.cxx
/// \author  Andrea Ferrero
///
/// \brief Implementation of a data processor to run the raw decoding
///

#include "MCHRawDecoder/DataDecoder.h"

#include <fstream>
#include <FairMQLogger.h>
#include "Headers/RAWDataHeader.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MCHMappingInterface/Segmentation.h"
#include "Framework/Logger.h"

#define MCH_DECODER_MAX_ERROR_COUNT 100

namespace o2
{
namespace mch
{
namespace raw
{

using namespace o2;
//using namespace o2::framework;
using namespace o2::mch::mapping;
using RDH = o2::header::RDHAny;

static constexpr uint32_t bcRollOver = (1 << 20);
static constexpr uint32_t twentyBitsAtOne = 0xFFFFF;
static constexpr uint32_t bcInOrbit = o2::constants::lhc::LHCMaxBunches;

// conversion matrix between the original channels numbering of the RUN2 readout electronics and the final version of the RUN3 DualSAMPA-based readout
static std::array<int, 64> refManu2ds_st345_v5 = {
  63, 62, 61, 60, 59, 57, 56, 53, 51, 50, 47, 45, 44, 41, 38, 35,
  36, 33, 34, 37, 32, 39, 40, 42, 43, 46, 48, 49, 52, 54, 55, 58,
  7, 8, 5, 2, 6, 1, 3, 0, 4, 9, 10, 15, 17, 18, 22, 25,
  31, 30, 29, 28, 27, 26, 24, 23, 20, 21, 16, 19, 12, 14, 11, 13};

// conversion matrix between the original channels numbering of the RUN2 readout electronics and the intermediate version of the RUN3 DualSAMPA-based readout
static std::array<int, 64> refManu2ds_st345_v2 = {
  62, 61, 63, 60, 59, 55, 58, 57, 56, 54, 50, 46, 42, 39, 37, 41,
  35, 36, 33, 34, 32, 38, 43, 40, 45, 44, 47, 48, 49, 52, 51, 53,
  7, 6, 5, 4, 2, 3, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23,
  31, 30, 29, 28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8};

#define refManu2ds_st345 refManu2ds_st345_v5

// inverse channel conversion matrix
static std::array<int, 64> refDs2manu_st345;

// function returning the RUN3 DualSAMPA channel number given the original RUN2 channel
static int manu2ds(int i)
{
  return refManu2ds_st345[i];
}

// function returning the original RUN2 channel number given the RUN3 DualSAMPA channel
static int ds2manu(int i)
{
  return refDs2manu_st345[i];
}

//_________________________________________________________________________________________________

static void patchPage(gsl::span<const std::byte> rdhBuffer, bool verbose)
{
  auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(rdhBuffer[0])));

  auto existingFeeId = o2::raw::RDHUtils::getFEEID(rdhAny);
  if (existingFeeId == 0) {
    // early versions of raw data did not set the feeId
    // which we need to select the right decoder

    auto cruId = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF;
    auto endpoint = o2::raw::RDHUtils::getEndPointID(rdhAny);
    auto flags = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF00;

    uint32_t feeId = cruId * 2 + endpoint + flags;
    o2::raw::RDHUtils::setFEEID(rdhAny, feeId);
  }
};

//_________________________________________________________________________________________________

bool operator<(const DataDecoder::RawDigit& d1, const DataDecoder::RawDigit& d2)
{
  if (d1.getTime() == d2.getTime()) {
    if (d1.getDetID() == d2.getDetID()) {
      return d1.getPadID() < d2.getPadID();
    }
    return d1.getDetID() < d2.getDetID();
  }
  return (d1.getTime() < d2.getTime());
}

std::string asString(const DataDecoder::RawDigit& d)
{
  return fmt::format("DE {:4d} PADID {:5d} ADC {:5d} TIME {} BX {}",
                     d.getDetID(), d.getPadID(), d.getADC(), d.getTime(), d.getBunchCrossing());
}

std::ostream& operator<<(std::ostream& os, const DataDecoder::RawDigit& d)
{
  os << asString(d);
  return os;
}

//_________________________________________________________________________________________________

static bool isValidDeID(int deId)
{
  for (auto id : deIdsForAllMCH) {
    if (id == deId) {
      return true;
    }
  }

  return false;
}

//=======================
// Data decoder

bool DataDecoder::SampaInfo::operator==(const DataDecoder::SampaInfo& other) const
{
  return chip == other.chip &&
         ds == other.ds &&
         solar == other.solar &&
         sampaTime == other.sampaTime &&
         bunchCrossing == other.bunchCrossing &&
         orbit == other.orbit &&
         tfTime == other.tfTime;
  ;
}

bool DataDecoder::RawDigit::operator==(const DataDecoder::RawDigit& other) const
{
  return digit == other.digit && info == other.info;
}

void DataDecoder::TimeFrameStartRecord::update(uint32_t orbit, uint32_t bunchCrossing)
{
  mOrbitPrev = mOrbit;
  mBunchCrossingPrev = mBunchCrossing;

  mOrbit = orbit;
  mBunchCrossing = bunchCrossing;
}

bool DataDecoder::TimeFrameStartRecord::check()
{
  if (mOrbitPrev < 0) {
    return true;
  }

  if (mOrbit < mOrbitPrev) {
    return false;
  }

  uint64_t dOrbit = mOrbit - mOrbitPrev;
  uint64_t bcExpected = dOrbit * bcInOrbit + mBunchCrossingPrev;

  uint64_t bcExpected20bits = bcExpected & twentyBitsAtOne;

  return (bcExpected20bits == mBunchCrossing);
}

//_________________________________________________________________________________________________

DataDecoder::DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler,
                         uint32_t sampaBcOffset,
                         std::string mapCRUfile, std::string mapFECfile,
                         bool ds2manu, bool verbose, bool useDummyElecMap)
  : mChannelHandler(channelHandler), mRdhHandler(rdhHandler), mSampaTimeOffset(sampaBcOffset), mMapCRUfile(mapCRUfile), mMapFECfile(mapFECfile), mDs2manu(ds2manu), mDebug(verbose), mUseDummyElecMap(useDummyElecMap)
{
  init();
}

//_________________________________________________________________________________________________

void DataDecoder::setFirstOrbitInTF(uint32_t orbit)
{
  mFirstOrbitInTF = orbit;
}

//_________________________________________________________________________________________________

void DataDecoder::decodeBuffer(gsl::span<const std::byte> buf)
{
  if (mDebug) {
    std::cout << "\n\n============================\nStart of new buffer\n";
  }
  size_t bufSize = buf.size();
  size_t pageStart = 0;
  while (bufSize > pageStart) {
    RDH* rdh = reinterpret_cast<RDH*>(const_cast<std::byte*>(&(buf[pageStart])));
    if (mDebug) {
      if (pageStart == 0) {
        std::cout << "+++\n[decodeBuffer]" << std::endl;
      } else {
        std::cout << "---\n[decodeBuffer]" << std::endl;
      }
      o2::raw::RDHUtils::printRDH(rdh);
    }
    auto rdhVersion = o2::raw::RDHUtils::getVersion(rdh);
    auto rdhHeaderSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    if (rdhHeaderSize != 64) {
      break;
    }
    auto pageSize = o2::raw::RDHUtils::getOffsetToNext(rdh);

    gsl::span<const std::byte> page(reinterpret_cast<const std::byte*>(rdh), pageSize);
    decodePage(page);

    pageStart += pageSize;
  }

  if (mDebug) {
    std::cout << "[decodeBuffer] mOrbits size: " << mOrbits.size() << std::endl;
    dumpOrbits(mOrbits);
    std::cout << "[decodeBuffer] mDigits size: " << mDigits.size() << std::endl;
    dumpDigits();
  }
}

//_________________________________________________________________________________________________

void DataDecoder::dumpDigits()
{
  for (size_t di = 0; di < mDigits.size(); di++) {
    auto& d = mDigits[di];
    auto detID = d.getDetID();
    auto padID = d.getPadID();
    if (padID < 0) {
      continue;
    }
    const Segmentation& segment = segmentation(detID);
    bool bend = segment.isBendingPad(padID);
    float X = segment.padPositionX(padID);
    float Y = segment.padPositionY(padID);
    uint32_t orbit = d.getOrbit();
    uint32_t bunchCrossing = d.getBunchCrossing();
    uint32_t sampaTime = d.getSampaTime();
    std::cout << fmt::format("  DE {:4d}  PAD {:5d}  ADC {:6d}  TIME ({} {} {:4d})",
                             detID, padID, d.getADC(), orbit, bunchCrossing, sampaTime);
    std::cout << fmt::format("\tC {}  PAD_XY {:+2.2f} , {:+2.2f}", (bend ? (int)0 : (int)1), X, Y);
    std::cout << std::endl;
  }
};

//_________________________________________________________________________________________________

void dumpOrbits(const std::unordered_set<OrbitInfo, OrbitInfoHash>& mOrbits)
{
  std::set<OrbitInfo> ordered_orbits(mOrbits.begin(), mOrbits.end());
  for (auto o : ordered_orbits) {
    std::cout << " FEEID " << o.getFeeID() << "  LINK " << (int)o.getLinkID() << "  ORBIT " << o.getOrbit() << std::endl;
  }
};

//_________________________________________________________________________________________________

int32_t DataDecoder::getMergerChannelId(const DsElecId& dsElecId, DualSampaChannelId channel)
{
  auto solarId = dsElecId.solarId();
  uint32_t dsId = static_cast<uint32_t>(dsElecId.elinkGroupId()) * 5 + dsElecId.elinkIndexInGroup();
  if (solarId > DataDecoder::sMaxSolarId || dsId >= 40 || channel >= 64) {
    return -1;
  }

  return (solarId * 40 * 64) + (dsId * 64) + channel;
}

//_________________________________________________________________________________________________

bool DataDecoder::mergeDigits(uint32_t mergerChannelId, o2::mch::raw::SampaCluster& sc)
{
  static constexpr uint32_t BCROLLOVER = (1 << 20);
  static constexpr uint32_t ONEADCCLOCK = 4;
  static constexpr uint32_t MAXNOFSAMPLES = 0x3FF;
  static constexpr uint32_t TWENTYBITSATONE = 0xFFFFF;

  // only digits that start at the beginning of the SAMPA window can be merged
  if (sc.sampaTime != 0) {
    return false;
  }

  auto& mergerCh = mMergerRecords[mergerChannelId];

  // if there is not previous digit for this channel then no merging is possible
  if (mergerCh.digitId < 0) {
    return false;
  }

  // time stamp of the digits to be merged
  uint32_t bcStart = sc.bunchCrossing;
  // correct for bunch crossing counter rollover if needed
  if (bcStart < mergerCh.bcEnd) {
    bcStart += BCROLLOVER;
  }

  // if the new digit starts just one ADC clock cycle after the end of the previous,
  // the it can be merged into the existing one
  if ((bcStart - mergerCh.bcEnd) != ONEADCCLOCK) {
    return false;
  }

  // add total charge and number of samples to existing digit
  auto& digit = mDigits[mergerCh.digitId].digit;
  digit.setADC(digit.getADC() + sc.sum());
  uint32_t newNofSamples = digit.getNofSamples() + sc.nofSamples();
  if (newNofSamples > MAXNOFSAMPLES) {
    newNofSamples = MAXNOFSAMPLES;
  }
  digit.setNofSamples(newNofSamples);

  // update the time stamp of the signal's end
  mergerCh.bcEnd = (bcStart & TWENTYBITSATONE) + (sc.nofSamples() - 1) * 4;

  return true;
}

//_________________________________________________________________________________________________

void DataDecoder::updateMergerRecord(uint32_t mergerChannelId, uint32_t digitId)
{
  auto& mergerCh = mMergerRecords[mergerChannelId];
  auto& digit = mDigits[digitId];
  mergerCh.digitId = digitId;
  mergerCh.bcEnd = digit.info.bunchCrossing + (digit.info.sampaTime + digit.digit.getNofSamples() - 1) * 4;
}

//_________________________________________________________________________________________________

bool DataDecoder::getPadMapping(const DsElecId& dsElecId, DualSampaChannelId channel, int& deId, int& dsIddet, int& padId)
{
  deId = -1;
  dsIddet = -1;
  padId = -1;

  if (auto opt = mElec2Det(dsElecId); opt.has_value()) {
    DsDetId dsDetId = opt.value();
    dsIddet = dsDetId.dsId();
    deId = dsDetId.deId();
  }
  if (mDebug) {
    auto s = asString(dsElecId);
    auto ch = fmt::format("{}-CH{:02d}", s, channel);
    std::cout << ch << "  "
              << "deId " << deId << "  dsIddet " << dsIddet << std::endl;
  }

  if (deId < 0 || dsIddet < 0 || !isValidDeID(deId)) {
    LOGP(error, "got invalid DsDetId from dsElecId={}", asString(dsElecId));
    return false;
  }

  const Segmentation& segment = segmentation(deId);
  padId = segment.findPadByFEE(dsIddet, int(channel));

  if (padId < 0) {
    return false;
  }
  return true;
}

//_________________________________________________________________________________________________

bool DataDecoder::addDigit(const DsElecId& dsElecId, DualSampaChannelId channel, const o2::mch::raw::SampaCluster& sc)
{
  int deId, dsIddet, padId;
  if (!getPadMapping(dsElecId, channel, deId, dsIddet, padId)) {
    return false;
  }

  uint32_t digitadc = sc.sum();

  if (mDebug) {
    auto s = asString(dsElecId);
    auto ch = fmt::format("{}-CH{:02d}", s, channel);
    LOG(info) << ch << "  "
              << fmt::format("PAD ({:04d} {:04d} {:04d})\tADC {:06d}  TIME ({} {} {:02d})  SIZE {}  END {}",
                             deId, dsIddet, padId, digitadc, mOrbit, sc.bunchCrossing, sc.sampaTime, sc.nofSamples(), (sc.sampaTime + sc.nofSamples() - 1))
              << (((sc.sampaTime + sc.nofSamples() - 1) >= 98) ? " *" : "");
  }

  // skip channels not associated to any pad
  if (padId < 0) {
    LOGP(error, "got invalid padId from dsElecId={} dualSampaId={} channel={}", asString(dsElecId), dsIddet, channel);
    return false;
  }

  RawDigit digit;
  digit.digit = o2::mch::Digit(deId, padId, digitadc, 0, sc.nofSamples());
  digit.info.chip = channel / 32;
  digit.info.ds = dsElecId.elinkId();
  digit.info.solar = dsElecId.solarId();
  digit.info.sampaTime = sc.sampaTime;
  digit.info.bunchCrossing = sc.bunchCrossing;
  digit.info.orbit = mOrbit;

  mDigits.emplace_back(digit);

  if (mDebug) {
    RawDigit& lastDigit = mDigits.back();
    LOGP(info, "DIGIT STORED: ORBIT {} ADC {} DE {} PADID {} TIME {} BXCOUNT {}",
         mOrbit, lastDigit.getADC(), lastDigit.getDetID(), lastDigit.getPadID(),
         lastDigit.getSampaTime(), lastDigit.getBunchCrossing());
  }
  return true;
}

uint64_t DataDecoder::getChipId(uint32_t solar, uint32_t ds, uint32_t chip)
{
  return solar * 40 * 2 + ds * 2 + chip;
}

//_________________________________________________________________________________________________

uint32_t DataDecoder::subtractBcOffset(uint32_t bc, uint32_t offset)
{
  int64_t bcTF = static_cast<int64_t>(bc) - offset;

  if (bcTF < 0) {
    bcTF += bcRollOver;
  }
  uint32_t bc20bits = static_cast<uint32_t>(bcTF & twentyBitsAtOne);

  return bc20bits;
}

//_________________________________________________________________________________________________

void DataDecoder::updateTimeFrameStartRecord(uint64_t chipId, uint32_t mFirstOrbitInTF, uint32_t bcTF)
{
  if (chipId < DataDecoder::sReadoutChipsNum) {
    mTimeFrameStartRecords[chipId].update(mFirstOrbitInTF, bcTF);
  }
}

//_________________________________________________________________________________________________

void DataDecoder::decodePage(gsl::span<const std::byte> page)
{
  uint8_t isStopRDH = 0;
  uint32_t orbit;
  uint32_t feeId;
  uint32_t linkId;

  auto heartBeatHandler = [&](DsElecId dsElecId, uint8_t chip, uint32_t bunchCrossing) {
    auto ds = dsElecId.elinkId();
    auto solar = dsElecId.solarId();
    uint64_t chipId = getChipId(solar, ds, chip);

    uint32_t bcTF = subtractBcOffset(bunchCrossing, mSampaTimeOffset);

    updateTimeFrameStartRecord(chipId, mFirstOrbitInTF, bcTF);

    if (mDebug) {
      auto s = asString(dsElecId);
      LOGP(info, "HeartBeat: {}-CHIP{} -> {}/{} offset {} -> bcTF {}",
           s, chip, mFirstOrbitInTF, bunchCrossing, mSampaTimeOffset, bcTF);
    }

    if (!mTimeFrameStartRecords[chipId].check()) {
      if (mErrorCount < MCH_DECODER_MAX_ERROR_COUNT) {
        auto s = asString(dsElecId);
        LOGP(warning, "Inconsistent HeartBeat packet received: {}-CHIP{} {}/{} (last {}/{})",
             s, chip, mFirstOrbitInTF, bcTF, mTimeFrameStartRecords[chipId].mOrbitPrev, mTimeFrameStartRecords[chipId].mBunchCrossingPrev);
        mErrorCount += 1;
      }
    }
  };

  auto channelHandler = [&](DsElecId dsElecId, DualSampaChannelId channel,
                            o2::mch::raw::SampaCluster sc) {
    if (mChannelHandler) {
      mChannelHandler(dsElecId, channel, sc);
    }

    if (mDs2manu) {
      LOGP(error, "using ds2manu");
      channel = ds2manu(int(channel));
    }

    int32_t mergerChannelId = getMergerChannelId(dsElecId, channel);
    if (mergerChannelId < 0) {
      LOGP(error, "dsElecId={} is out-of-bounds", asString(dsElecId));
      return;
    }

    if (mergeDigits(mergerChannelId, sc)) {
      return;
    }

    if (!addDigit(dsElecId, channel, sc)) {
      return;
    }

    updateMergerRecord(mergerChannelId, mDigits.size() - 1);
  };

  patchPage(page, mDebug);

  auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(page[0])));
  mOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhAny);
  if (mDebug) {
    LOGP(info, "[decodeBuffer] mOrbit set to {}", mOrbit);
  }

  if (mRdhHandler) {
    mRdhHandler(&rdhAny);
  }

  // add orbit to vector if not present yet
  mOrbits.emplace(page);

  if (!mDecoder) {
    DecodedDataHandlers handlers;
    handlers.sampaChannelHandler = channelHandler;
    handlers.sampaHeartBeatHandler = heartBeatHandler;
    mDecoder = mFee2Solar ? o2::mch::raw::createPageDecoder(page, handlers, mFee2Solar)
                          : o2::mch::raw::createPageDecoder(page, handlers);
  }

  mDecoder(page);
};

//_________________________________________________________________________________________________

int32_t DataDecoder::getDigitTime(uint32_t orbitStart, uint32_t bcStart, uint32_t orbitDigit, uint32_t bcDigit)
{
  // We use the difference of orbits values to estimate the minimum and maximum allowed
  // difference in bunch crossings
  int64_t dOrbit = static_cast<int64_t>(orbitDigit) - static_cast<int64_t>(orbitStart);

  // Digits might be sent out later than the orbit in which they were recorded.
  // We account for this by allowing an extra -3 / +10 orbits when converting the
  // difference from orbit numbers to bunch crossings.
  int64_t dBcMin = (dOrbit - 10) * bcInOrbit;
  int64_t dBcMax = (dOrbit + 3) * bcInOrbit;

  // Difference in bunch crossing values
  int64_t dBc = static_cast<int64_t>(bcDigit) - static_cast<int64_t>(bcStart);

  if (dBc < dBcMin) {
    // the difference is too small, so we assume that it needs to be
    // incremented by one rollover factor
    dBc += bcRollOver;
  } else if (dBc > dBcMax) {
    // the difference is too big, so we assume that it needs to be
    // decremented by one rollover factor
    dBc -= bcRollOver;
  }

  return static_cast<int32_t>(dBc);
}

//_________________________________________________________________________________________________

bool DataDecoder::getTimeFrameStartRecord(const RawDigit& digit, uint32_t& orbitTF, uint32_t& bcTF)
{
  static constexpr uint32_t bcInOrbit = o2::constants::lhc::LHCMaxBunches;
  static constexpr uint32_t twentyBitsAtOne = 0xFFFFF;

  // first orbit of the current TF
  orbitTF = mFirstOrbitInTF;

  auto& d = digit.digit;
  auto& info = digit.info;

  auto chipId = getChipId(info.solar, info.ds, info.chip);
  auto& tfStart = mTimeFrameStartRecords[chipId];

  if (tfStart.mOrbit < 0) {
    if (mErrorCount < MCH_DECODER_MAX_ERROR_COUNT) {
      LOGP(warning, "Missing TF start record for S{}-J{}-DS{}-CHIP{}", info.solar, info.ds / 5 + 1, info.ds % 5, info.chip);
      mErrorCount += 1;
    }
    return false;
  }

  // orbit and BC from the last received HB packet
  uint32_t orbitHBP = tfStart.mOrbit;
  // SAMPA BC at the beginning of the current TF
  bcTF = tfStart.mBunchCrossing;

  if (orbitHBP != orbitTF) {
    // we correct the BC from the last received HB packet, if it was recorded from an older TF
    bcTF += (orbitTF - orbitHBP) * bcInOrbit;
    // only keep 20 bits
    bcTF &= twentyBitsAtOne;

    //std::cout << fmt::format(" -> {}/{}", orbitTF, bcTF) << std::endl;

    // update the time frame start information for this chip, to speed-up the computations
    // in case another digit from the same chip is found in the same time frame.
    tfStart.update(orbitTF, bcTF);
  }

  return true;
}

//_________________________________________________________________________________________________

void DataDecoder::computeDigitsTime()
{
  static constexpr int32_t timeInvalid = DataDecoder::tfTimeInvalid;

  auto setDigitTime = [&](Digit& d, int32_t tfTime) {
    d.setTime(tfTime);
  };

  for (auto& digit : mDigits) {
    auto& d = digit.digit;
    auto& info = digit.info;

    uint32_t orbitTF;
    uint32_t bcTF;
    int32_t tfTime = timeInvalid;

    auto orbitDigit = info.orbit;
    auto bcDigit = info.getBXTime();

    if (getTimeFrameStartRecord(digit, orbitTF, bcTF)) {
      int solar = info.solar;
      int ds = info.ds;
      int chip = info.chip;
      tfTime = DataDecoder::getDigitTime(orbitTF, bcTF, orbitDigit, bcDigit);
    }

    setDigitTime(d, tfTime);
    info.tfTime = tfTime;
  }
}

//_________________________________________________________________________________________________

static std::string readFileContent(std::string& filename)
{
  std::string content;
  std::string s;
  std::ifstream in(filename);
  while (std::getline(in, s)) {
    content += s;
    content += "\n";
  }
  return content;
};

//_________________________________________________________________________________________________

void DataDecoder::initElec2DetMapper(std::string filename)
{
  if (filename.empty()) {
    if (mUseDummyElecMap) {
      LOGP(warning, "[initElec2DetMapper] Using dummy electronic mapping");
      mElec2Det = createElec2DetMapper<ElectronicMapperDummy>();
    } else {
      mElec2Det = createElec2DetMapper<ElectronicMapperGenerated>();
    }
  } else {
    LOGP(info, "[initElec2DetMapper] filename={}", filename);
    ElectronicMapperString::sFecMap = readFileContent(filename);
    mElec2Det = createElec2DetMapper<ElectronicMapperString>();
  }
};

//_________________________________________________________________________________________________

void DataDecoder::initFee2SolarMapper(std::string filename)
{
  if (filename.empty()) {
    if (mUseDummyElecMap) {
      LOGP(warning, "[initFee2SolarMapper] Using dummy electronic mapping");
      mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperDummy>();
    } else {
      mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
    }
  } else {
    LOGP(info, "[initFee2SolarMapper] filename={}", filename);
    ElectronicMapperString::sCruMap = readFileContent(filename);
    mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperString>();
  }
};

//_________________________________________________________________________________________________

void DataDecoder::init()
{
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 64; j++) {
      if (refManu2ds_st345[j] != i) {
        continue;
      }
      refDs2manu_st345[i] = j;
      break;
    }
  }

  initFee2SolarMapper(mMapCRUfile);
  initElec2DetMapper(mMapFECfile);
};

//_________________________________________________________________________________________________

void DataDecoder::reset()
{
  mDigits.clear();
  mOrbits.clear();
  for (auto& mergerCh : mMergerRecords) {
    mergerCh.digitId = -1;
    mergerCh.bcEnd = -1;
  }
}

} // namespace raw
} // namespace mch
} // end namespace o2
