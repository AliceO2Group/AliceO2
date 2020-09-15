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
#include "DigitsMerger.h"
#include "MCHMappingInterface/Segmentation.h"

/*
#include <random>
#include <iostream>
#include <stdexcept>
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include "DPLUtils/DPLRawParser.h"
#include "MCHBase/Digit.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawElecMap/Mapper.h"
*/

static bool mPrint = false;

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
  static int mNrdhs = 0;
  auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(rdhBuffer[0])));
  mNrdhs++;

  auto cruId = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF;
  auto flags = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF00;
  auto endpoint = o2::raw::RDHUtils::getEndPointID(rdhAny);
  uint32_t feeId = cruId * 2 + endpoint + flags;
  o2::raw::RDHUtils::setFEEID(rdhAny, feeId);

  if (verbose) {
    std::cout << mNrdhs << "--\n";
    o2::raw::RDHUtils::printRDH(rdhAny);
  }
};

//=======================
// Data decoder

DataDecoder::DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler, std::string mapCRUfile, std::string mapFECfile, bool ds2manu, bool verbose)
  : mChannelHandler(channelHandler), mRdhHandler(rdhHandler), mMapCRUfile(mapCRUfile), mMapFECfile(mapFECfile), mDs2manu(ds2manu), mPrint(verbose)
{
  init();
}

void DataDecoder::decodeBuffer(gsl::span<const std::byte> page)
{
  size_t ndigits{0};

  uint8_t isStopRDH = 0;
  uint32_t orbit;
  uint32_t feeId;
  uint32_t linkId;

  const auto storeDigit = [&](const Digit& d) {
    mOutputDigits.emplace_back(d);
    if (mPrint) {
      std::cout << "[storeDigit]: digit stored, mOutputDigits size: " << mOutputDigits.size() << std::endl;
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
    if (mPrint) {
      auto s = asString(dsElecId);
      auto ch = fmt::format("{}-CH{:02d}", s, channel);
      std::cout << ch << "  "
                << "deId " << deId << "  dsIddet " << dsIddet << std::endl;
    }

    if (deId < 0 || dsIddet < 0) {
      return;
    }

    int padId = -1;
    const Segmentation& segment = segmentation(deId);
    if ((&segment) == nullptr) {
      return;
    }

    padId = segment.findPadByFEE(dsIddet, int(channel));
    if (mPrint) {
      auto s = asString(dsElecId);
      auto ch = fmt::format("{}-CH{:02d}", s, channel);
      std::cout << ch << "  "
                << fmt::format("PAD ({:04d} {:04d} {:04d})\tADC {:06d}  TIME ({} {} {:02d})  SIZE {}  END {}",
                               deId, dsIddet, padId, digitadc, orbit, sc.bunchCrossing, sc.sampaTime, sc.nofSamples(), (sc.sampaTime + sc.nofSamples() - 1))
                << (((sc.sampaTime + sc.nofSamples() - 1) >= 98) ? " *" : "") << std::endl;
    }

    // skip channels not associated to any pad
    if (padId < 0) {
      return;
    }

    Digit::Time time;
    time.sampaTime = sc.sampaTime;
    time.bunchCrossing = sc.bunchCrossing;
    time.orbit = orbit;

    mMerger->addDigit(feeId, static_cast<int>(dsElecId.solarId()), static_cast<int>(dsElecId.elinkId()), static_cast<int>(channel),
                      deId, padId, digitadc, time, sc.nofSamples());
    ++ndigits;
  };

  const auto updateMerger = [&](gsl::span<const std::byte> rdhBuffer) {
    auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(rdhBuffer[0])));

    auto feeId = o2::raw::RDHUtils::getFEEID(rdhAny) & 0xFF;
    auto orbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhAny);
    auto linkId = o2::raw::RDHUtils::getLinkID(rdhAny);
    auto isStopRDH = o2::raw::RDHUtils::getStop(rdhAny);

    if (!mMerger) {
      if (linkId == 15) {
        mMerger = new Merger;
      } else {
        mMerger = new SimpleMerger;
      }

      mMerger->setDigitHandler(storeDigit);
    }
    mMerger->setOrbit(feeId, orbit, isStopRDH);
  };

  const auto dumpDigits = [&](bool bending) {
    for (auto d : mOutputDigits) {
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
      std::cout << fmt::format("  DE {:4d}  PAD {:5d}  ADC {:6d}  TIME ({} {} {:4d})",
                               d.getDetID(), d.getPadID(), d.getADC(), d.getTime().orbit, d.getTime().bunchCrossing, d.getTime().sampaTime);
      std::cout << fmt::format("\tC {}  PAD_XY {:+2.2f} , {:+2.2f}", (bending ? (int)0 : (int)1), X, Y);
      std::cout << std::endl;
    }
  };

  patchPage(page, mPrint);

  if (mRdhHandler) {
    auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(page
                                                                      [0])));
    mRdhHandler(&rdhAny);
  }

  // add orbit to vector if not present yet
  OrbitInfo orbitInfo(page);
  if (std::find(mOrbits.begin(), mOrbits.end(), orbitInfo) == mOrbits.end()) {
    if (mPrint) {
      printf("Orbit info: %lX (%u %u %u)\n", orbitInfo.get(), (uint32_t)orbitInfo.getOrbit(), (uint32_t)orbitInfo.getLinkID(), (uint32_t)orbitInfo.getFeeID());
    }
    mOrbits.push_back(orbitInfo);
  }

  // initialize the merger on the first call, and pass the current orbit number
  updateMerger(page);
  //mMerger->mergeDigits(feeId);

  if (!mDecoder) {
    mDecoder = mFee2Solar ? o2::mch::raw::createPageDecoder(page, channelHandler, mFee2Solar)
                          : o2::mch::raw::createPageDecoder(page, channelHandler);
  }
  mDecoder(page);

  if (mPrint) {
    std::cout << "[decodeBuffer] mOutputDigits size: " << mOutputDigits.size() << std::endl;
    dumpDigits(true);
    dumpDigits(false);
  }
};

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
  mNrdhs = 0;

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
  mOrbits.clear();
}

} // namespace raw
} // namespace mch
} // end namespace o2
