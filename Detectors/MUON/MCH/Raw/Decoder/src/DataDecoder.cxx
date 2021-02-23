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
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MCHMappingInterface/Segmentation.h"

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
  static int sNrdhs = 0;
  auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(rdhBuffer[0])));
  sNrdhs++;

  auto cruId = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF;
  auto flags = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF00;
  auto endpoint = o2::raw::RDHUtils::getEndPointID(rdhAny);
  auto existingFeeId = o2::raw::RDHUtils::getFEEID(rdhAny);
  if (existingFeeId == 0) {
    // early versions of raw data did not set the feeId
    // which we need to select the right decoder
    uint32_t feeId = cruId * 2 + endpoint + flags;
    o2::raw::RDHUtils::setFEEID(rdhAny, feeId);
  }

  if (verbose) {
    std::cout << "RDH number " << sNrdhs << "--\n";
    o2::raw::RDHUtils::printRDH(rdhAny);
  }
};

//=======================
// Data decoder

DataDecoder::DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler, std::string mapCRUfile, std::string mapFECfile, bool ds2manu, bool verbose)
  : mChannelHandler(channelHandler), mRdhHandler(rdhHandler), mMapCRUfile(mapCRUfile), mMapFECfile(mapFECfile), mDs2manu(ds2manu), mDebug(verbose)
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

void DataDecoder::decodeBuffer(gsl::span<const std::byte> buf)
{
  if (mDebug) {
    std::cout << "\n\n============================\nStart of new buffer\n";
  }
  size_t bufSize = buf.size();
  size_t pageStart = 0;
  while (bufSize > pageStart) {
    RDH* rdh = reinterpret_cast<RDH*>(const_cast<std::byte*>(&(buf[pageStart])));
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
}

void DataDecoder::decodePage(gsl::span<const std::byte> page)
{
  if (mDebug) {
    std::cout << "\n----------------------------\nStart of new page\n\n";
  }
  size_t ndigits{0};

  auto heartBeatHandler = [&](DsElecId dsElecId, uint8_t chip, uint32_t bunchCrossing) {
    SampaInfo sampaId;
    sampaId.chip = chip;
    sampaId.ds = dsElecId.elinkId();
    sampaId.solar = dsElecId.solarId();

    mTimeFrameInfos[sampaId.id].emplace_back(mOrbit, bunchCrossing);
    if (mDebug) {
      std::cout << "HeartBeat: solar " << sampaId.solar << "  ds " << sampaId.ds << "  chip " << sampaId.chip
                << "  mOrbit " << mOrbit << "  bunchCrossing " << bunchCrossing << std::endl;
    }
  };

  auto channelHandler = [&](DsElecId dsElecId, uint8_t channel, o2::mch::raw::SampaCluster sc) {
    if (mChannelHandler) {
      mChannelHandler(dsElecId, channel, sc);
    }

    if (mDs2manu) {
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
      return;
    }

    SampaInfo sampaInfo;
    sampaInfo.chip = channel / 32;
    sampaInfo.ds = dsElecId.elinkId();
    sampaInfo.solar = dsElecId.solarId();
    sampaInfo.sampaTime = sc.sampaTime;
    sampaInfo.bunchCrossing = sc.bunchCrossing;
    sampaInfo.orbit = mOrbit;

    mOutputDigits.emplace_back(o2::mch::Digit(deId, padId, digitadc, 0, sc.nofSamples()));
    mSampaInfos.emplace_back(sampaInfo);

    if (mDebug) {
      std::cout << "DIGIT STORED:\nADC " << mOutputDigits.back().getADC() << " DE# " << mOutputDigits.back().getDetID() << " PadId " << mOutputDigits.back().getPadID() << " time " << mSampaInfos.back().sampaTime << std::endl;
    }
    ++ndigits;
  };

  const auto dumpDigits = [&](bool bending) {
    if (mOutputDigits.size() != mSampaInfos.size()) {
      return;
    }

    for (size_t di = 0; di < mOutputDigits.size(); di++) {
      Digit& d = mOutputDigits[di];
      SampaInfo& t = mSampaInfos[di];
      if (d.getPadID() < 0) {
        continue;
      }
      const Segmentation& segment = segmentation(d.getDetID());
      bool bend = segment.isBendingPad(d.getPadID());
      if (bending != segment.isBendingPad(d.getPadID())) {
        continue;
      }
      float X = segment.padPositionX(d.getPadID());
      float Y = segment.padPositionY(d.getPadID());
      uint32_t orbit = t.orbit;
      uint32_t bunchCrossing = t.bunchCrossing;
      uint32_t sampaTime = t.sampaTime;
      std::cout << fmt::format("  DE {:4d}  PAD {:5d}  ADC {:6d}  TIME ({} {} {:4d})",
                               d.getDetID(), d.getPadID(), d.getADC(), orbit, bunchCrossing, sampaTime);
      std::cout << fmt::format("\tC {}  PAD_XY {:+2.2f} , {:+2.2f}", (bending ? (int)0 : (int)1), X, Y);
      std::cout << std::endl;
    }
  };

  const auto dumpOrbits = [&]() {
    std::set<OrbitInfo> ordered_orbits(mOrbits.begin(), mOrbits.end());
    for (auto o : ordered_orbits) {
      std::cout << " FEEID " << o.getFeeID() << "  LINK " << (int)o.getLinkID() << "  ORBIT " << o.getOrbit() << std::endl;
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
    handlers.sampaHeartBeatHandler = heartBeatHandler;
    mDecoder = mFee2Solar ? o2::mch::raw::createPageDecoder(page, handlers, mFee2Solar)
                          : o2::mch::raw::createPageDecoder(page, handlers);
  }
  mDecoder(page);

  if (mDebug) {
    std::cout << "[decodeBuffer] mOrbits size: " << mOrbits.size() << std::endl;
    //dumpOrbits();
    std::cout << "[decodeBuffer] mOutputDigits size: " << mOutputDigits.size() << std::endl;
    dumpDigits(true);
    dumpDigits(false);
  }
};

int32_t digitsTimeDiff(uint32_t orbit1, uint32_t bc1, uint32_t orbit2, uint32_t bc2)
{
  // bunch crossings are stored with 20 bits
  static const int32_t BCROLLOVER = (1 << 20);
  // bunch crossings half range
  //static const int32_t BCHALFRANGE = (1 << 19);
  // number of bunch crossings in one orbit
  static const int32_t BCINORBIT = 3564;

  // We use the difference of orbits values to estimate the minimum and maximum allowed
  // difference in bunch crossings
  int64_t dOrbit = static_cast<int64_t>(orbit2) - static_cast<int64_t>(orbit1);

  // Digits might be sent out later than the orbit in which they were recorded.
  // We account for this by allowing an extra +/- 2 orbits when converting the
  // difference from orbit numbers to bunch crossings.
  int64_t dBcMin = (dOrbit - 2) * BCINORBIT;
  int64_t dBcMax = (dOrbit + 2) * BCINORBIT;

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

void DataDecoder::computeDigitsTime_(std::vector<o2::mch::Digit>& digits, std::vector<SampaInfo>& sampaInfo, TimeFrameInfos& timeFrameInfos, bool debug)
{
  if (digits.size() != sampaInfo.size()) {
    return;
  }

  auto setDigitTime = [&](Digit& d, int32_t tfTime1, int32_t tfTime2) {
    d.setTime(tfTime1);
    d.setTimeValid(true);
    if (tfTime2 < 0) {
      d.setTime(tfTime1);
      if (tfTime1 >= 0) {
        d.setTFindex(1);
      } else {
        d.setTFindex(0);
      }
    } else {
      d.setTFindex(2);
    }
    if (debug) {
      std::cout << "[computeDigitsTime_] hit time set to " << d.getTime() << ", TF index is " << (int)d.getTFindex() << std::endl;
    }
  };

  for (size_t di = 0; di < digits.size(); di++) {
    Digit& d = digits[di];
    SampaInfo& info = sampaInfo[di];
    std::vector<TimeFrameInfo>& tfInfo = timeFrameInfos[info.id];

    int32_t tfTime1 = 0, tfTime2 = -1;

    if (tfInfo.empty()) {
      d.setTimeValid(false);
      continue;
    }
    uint32_t bc = tfInfo[0].mBunchCrossing;
    uint32_t orbit = tfInfo[0].mOrbit;
    tfTime1 = digitsTimeDiff(orbit, bc, info.orbit, info.getBXTime());
    if (debug) {
      std::cout << "[computeDigitsTime_] hit " << info.orbit << "," << info.getBXTime()
                << "    tfTime(1) " << orbit << "," << bc << "    diff " << tfTime1 << std::endl;
    }

    if (tfInfo.size() > 1) {
      bc = tfInfo[1].mBunchCrossing;
      orbit = tfInfo[1].mOrbit;
      tfTime2 = digitsTimeDiff(orbit, bc, info.orbit, info.getBXTime());
      if (debug) {
        std::cout << "[computeDigitsTime_] hit " << info.orbit << "," << info.getBXTime()
                  << "    tfTime(1) " << orbit << "," << bc << "    diff " << tfTime2 << std::endl;
      }
    }

    setDigitTime(d, tfTime1, tfTime2);
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
  std::cout << "[initElec2DetMapper] filename=" << filename << std::endl;
  if (filename.empty()) {
    mElec2Det = createElec2DetMapper<ElectronicMapperGenerated>();
  } else {
    ElectronicMapperString::sFecMap = readFileContent(filename);
    mElec2Det = createElec2DetMapper<ElectronicMapperString>();
  }
};

void DataDecoder::initFee2SolarMapper(std::string filename)
{
  std::cout << "[initFee2SolarMapper] filename=" << filename << std::endl;
  if (filename.empty()) {
    mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
  } else {
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
  mOutputDigits.clear();
  mSampaInfos.clear();
  mOrbits.clear();
}

} // namespace raw
} // namespace mch
} // end namespace o2
