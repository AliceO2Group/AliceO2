// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

DataDecoder::DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler,
                         uint32_t sampaBcOffset,
                         std::string mapCRUfile, std::string mapFECfile,
                         bool ds2manu, bool verbose, bool useDummyElecMap)
  : mChannelHandler(channelHandler), mRdhHandler(rdhHandler), mSampaTimeOffset(sampaBcOffset), mMapCRUfile(mapCRUfile), mMapFECfile(mapFECfile), mDs2manu(ds2manu), mDebug(verbose), mUseDummyElecMap(useDummyElecMap)
{
  init();
}

static bool isValidDeID(int deId)
{
  for (auto id : deIdsForAllMCH) {
    if (id == deId) {
      return true;
    }
  }

  return false;
}

void DataDecoder::setFirstOrbitInTF(uint32_t orbit)
{
  constexpr int BCINORBIT = o2::constants::lhc::LHCMaxBunches;
  constexpr int TWENTYBITSATONE = 0xFFFFF;
  if (!mFirstOrbitInRun) {
    LOG(ERROR) << "[setFirstOrbitInTF] first orbit in run not set!";
    return;
  }
  if (orbit < mFirstOrbitInRun) {
    LOG(ERROR) << "[setFirstOrbitInTF] first TF orbit smaller than first orbit in run!";
    return;
  }

  // the SAMPA BC value at the beginning of the TF is computed by counting the number of orbits
  // since the beginning of the run, and then multiplying this number by the number of BC in one orbit.
  // We then take the first 20 bits of the result to emulate the internal SAMPA BC counter.
  uint64_t nOrbitsInRun = orbit - mFirstOrbitInRun.value();
  uint64_t bc = (nOrbitsInRun * BCINORBIT + mSampaTimeOffset);
  uint32_t bc20bits = bc & TWENTYBITSATONE;
  mSampaTimeFrameStart = SampaTimeFrameStart(orbit, bc20bits);
}

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

void dumpOrbits(const std::unordered_set<OrbitInfo, OrbitInfoHash>& mOrbits)
{
  std::set<OrbitInfo> ordered_orbits(mOrbits.begin(), mOrbits.end());
  for (auto o : ordered_orbits) {
    std::cout << " FEEID " << o.getFeeID() << "  LINK " << (int)o.getLinkID() << "  ORBIT " << o.getOrbit() << std::endl;
  }
};

void DataDecoder::decodePage(gsl::span<const std::byte> page)
{
  /*
   * TODO: we should use the HBPackets to verify the synchronization between the SAMPA chips
  auto heartBeatHandler = [&](DsElecId dsElecId, uint8_t chip, uint32_t bunchCrossing) {
    SampaInfo sampaId;
    sampaId.chip = chip;
    sampaId.ds = dsElecId.elinkId();
    sampaId.solar = dsElecId.solarId();

    if (mDebug) {
      auto s = asString(dsElecId);
      LOGP(info, "HeartBeat: {} ID {} SOLAR {} ds {} chip {} -> bxcount {} orbit {} [ bc {} ]",
           s, sampaId.id, csampaId.solar, csampaId.ds, csampaId.chip,
           bunchCrossingCounter, mOrbit,
           bunchCrossingCounter % 3564);
    }
  };
   */

  auto channelHandler = [&](DsElecId dsElecId, DualSampaChannelId channel,
                            o2::mch::raw::SampaCluster sc) {
    if (mChannelHandler) {
      mChannelHandler(dsElecId, channel, sc);
    }

    if (mDs2manu) {
      LOGP(error, "using ds2manu");
      channel = ds2manu(int(channel));
    }

    uint32_t digitadc = sc.sum();

    int deId{-1};
    int dsIddet{-1};
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
      return;
    }

    int padId = -1;
    const Segmentation& segment = segmentation(deId);

    padId = segment.findPadByFEE(dsIddet, int(channel));
    if (mDebug) {
      auto s = asString(dsElecId);
      auto ch = fmt::format("{}-CH{:02d}", s, channel);
      std::cout << ch << "  "
                << fmt::format("PAD ({:04d} {:04d} {:04d})\tADC {:06d}  TIME ({} {} {:02d})  SIZE {}  END {}",
                               deId, dsIddet, padId, digitadc, mOrbit, sc.bunchCrossing, sc.sampaTime, sc.nofSamples(), (sc.sampaTime + sc.nofSamples() - 1))
                << (((sc.sampaTime + sc.nofSamples() - 1) >= 98) ? " *" : "") << std::endl;
    }

    // skip channels not associated to any pad
    if (padId < 0) {
      LOGP(error, "got invalid padId from dsElecId={} dualSampaId={} channel={}", asString(dsElecId), dsIddet, channel);
      return;
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
  };

  patchPage(page, mDebug);

  auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(page[0])));
  mOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhAny);
  if (mDebug) {
    std::cout << "[decodeBuffer] mOrbit set to " << mOrbit << std::endl;
  }

  if (mRdhHandler) {
    mRdhHandler(&rdhAny);
  }

  // add orbit to vector if not present yet
  mOrbits.emplace(page);

  if (!mDecoder) {
    DecodedDataHandlers handlers;
    handlers.sampaChannelHandler = channelHandler;
    //handlers.sampaHeartBeatHandler = heartBeatHandler;
    mDecoder = mFee2Solar ? o2::mch::raw::createPageDecoder(page, handlers, mFee2Solar)
                          : o2::mch::raw::createPageDecoder(page, handlers);
  }
  mDecoder(page);
};

int32_t DataDecoder::digitsTimeDiff(uint32_t orbit1, uint32_t bc1, uint32_t orbit2, uint32_t bc2)
{
  // bunch crossings are stored with 20 bits
  static const int32_t BCROLLOVER = (1 << 20);
  // number of bunch crossings in one orbit
  constexpr int BCINORBIT = o2::constants::lhc::LHCMaxBunches;

  // We use the difference of orbits values to estimate the minimum and maximum allowed
  // difference in bunch crossings
  int64_t dOrbit = static_cast<int64_t>(orbit2) - static_cast<int64_t>(orbit1);

  // Digits might be sent out later than the orbit in which they were recorded.
  // We account for this by allowing an extra +/- 3 orbits when converting the
  // difference from orbit numbers to bunch crossings.
  int64_t dBcMin = (dOrbit - 3) * BCINORBIT;
  int64_t dBcMax = (dOrbit + 3) * BCINORBIT;

  // Difference in bunch crossing values
  int64_t dBc = static_cast<int64_t>(bc2) - static_cast<int64_t>(bc1);

  if (dBc < dBcMin) {
    // the difference is too small, so we assume that it needs to be
    // incremented by one rollover factor
    dBc += BCROLLOVER;
  } else if (dBc > dBcMax) {
    // the difference is too big, so we assume that it needs to be
    // decremented by one rollover factor
    dBc -= BCROLLOVER;
  }

  return static_cast<int32_t>(dBc);
}

void DataDecoder::computeDigitsTime_(RawDigitVector& digits, SampaTimeFrameStart& sampaTimeFrameStart, bool debug)
{
  constexpr int32_t timeInvalid = DataDecoder::tfTimeInvalid;
  auto setDigitTime = [&](Digit& d, int32_t tfTime) {
    d.setTime(tfTime);
    if (debug) {
      std::cout << "[computeDigitsTime_] hit time set to " << d.getTime() << std::endl;
    }
  };

  for (size_t di = 0; di < digits.size(); di++) {
    Digit& d = digits[di].digit;
    SampaInfo& info = digits[di].info;

    int32_t tfTime = 0;
    uint32_t bc = sampaTimeFrameStart.mBunchCrossing;
    uint32_t orbit = sampaTimeFrameStart.mOrbit;
    tfTime = DataDecoder::digitsTimeDiff(orbit, bc, info.orbit, info.getBXTime());
    if (debug) {
      std::cout << "\n[computeDigitsTime_] hit " << info.orbit << "," << info.getBXTime()
                << "    tfTime(1) " << orbit << "," << bc << "    diff " << tfTime << std::endl;
    }
    setDigitTime(d, tfTime);
    info.tfTime = tfTime;

    if (debug) {
      std::cout << "                     solar " << info.solar << "  ds " << info.ds << "  chip " << info.chip << std::endl;
      std::cout << "                     pad " << d.getDetID() << "," << d.getPadID() << " "
                << info.orbit << " " << info.tfTime << " " << info.getBXTime() << std::endl;
    }
  }
}

static std::string readFileContent(std::string& filename)
{
  std::string content;
  std::string s;
  std::ifstream in(filename);
  while (std::getline(in, s)) {
    content += s;
    content += "\n";
  }
  std::cout << "readFileContent(" << filename << "):" << std::endl
            << content << std::endl;
  return content;
};

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
}

} // namespace raw
} // namespace mch
} // end namespace o2
