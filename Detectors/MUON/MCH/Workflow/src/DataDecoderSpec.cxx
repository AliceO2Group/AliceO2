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
///
/// \file    DatDecoderSpec.cxx
/// \author  Andrea Ferrero
///
/// \brief Implementation of a data processor to run the raw decoding
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>

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
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/OrbitInfo.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHWorkflow/DataDecoderSpec.h"
#include "DetectorsRaw/RDHUtils.h"

namespace o2
{
namespace mch
{
namespace raw
{

using namespace o2;
using namespace o2::framework;
using namespace o2::mch::mapping;
using RDH = o2::header::RDHAny;

std::array<int, 64> refManu2ds_st345 = {
  63, 62, 61, 60, 59, 57, 56, 53, 51, 50, 47, 45, 44, 41, 38, 35,
  36, 33, 34, 37, 32, 39, 40, 42, 43, 46, 48, 49, 52, 54, 55, 58,
  7, 8, 5, 2, 6, 1, 3, 0, 4, 9, 10, 15, 17, 18, 22, 25,
  31, 30, 29, 28, 27, 26, 24, 23, 20, 21, 16, 19, 12, 14, 11, 13};
std::array<int, 64> refDs2manu_st345;

int manu2ds(int i)
{
  return refManu2ds_st345[i];
}

int ds2manu(int i)
{
  return refDs2manu_st345[i];
}

// custom hash can be a standalone function object:
struct OrbitInfoHash {
  std::size_t operator()(OrbitInfo const& info) const noexcept
  {
    return std::hash<uint64_t>{}(info.get());
  }
};

//=======================
// Data decoder
class DataDecoderTask
{
 private:
  void decodeBuffer(gsl::span<const std::byte> page, std::vector<o2::mch::Digit>& digits)
  {
    size_t ndigits{0};

    uint32_t orbit;

    auto channelHandler = [&](DsElecId dsElecId, uint8_t channel, o2::mch::raw::SampaCluster sc) {
      if (mDs2manu) {
        channel = ds2manu(int(channel));
      }
      if (mPrint) {
        auto s = asString(dsElecId);
        auto ch = fmt::format("{}-CH{}", s, channel);
        std::cout << "dsElecId: " << ch << std::endl;
      }
      uint32_t digitadc(0);
      if (sc.isClusterSum()) {
        digitadc = sc.chargeSum;
      } else {
        for (auto& s : sc.samples) {
          digitadc += s;
        }
      }

      int deId{-1};
      int dsIddet{-1};
      if (auto opt = mElec2Det(dsElecId); opt.has_value()) {
        DsDetId dsDetId = opt.value();
        dsIddet = dsDetId.dsId();
        deId = dsDetId.deId();
      }
      if (mPrint) {
        std::cout << "deId " << deId << "  dsIddet " << dsIddet << "  channel " << (int)channel << std::endl;
      }

      if (deId < 0 || dsIddet < 0) {
        return;
      }

      int padId = -1;
      try {
        const Segmentation& segment = segmentation(deId);
        padId = segment.findPadByFEE(dsIddet, int(channel));
        if (mPrint) {
          std::cout << "DS " << (int)dsElecId.elinkId() << "  CHIP " << ((int)channel) / 32 << "  CH " << ((int)channel) % 32 << "  ADC " << digitadc << "  DE# " << deId << "  DSid " << dsIddet << "  PadId " << padId << std::endl;
        }
      } catch (const std::exception& e) {
        std::cout << "Failed to get padId: " << e.what() << std::endl;
        return;
      }

      Digit::Time time;
      time.sampaTime = sc.sampaTime;
      time.bunchCrossing = sc.bunchCrossing;
      time.orbit = orbit;

      digits.emplace_back(o2::mch::Digit(deId, padId, digitadc, time, sc.nofSamples()));

      if (mPrint) {
        std::cout << "DIGIT STORED:\nADC " << digits.back().getADC() << " DE# " << digits.back().getDetID() << " PadId " << digits.back().getPadID() << " time " << digits.back().getTime().sampaTime << std::endl;
      }
      ++ndigits;
    };

    const auto patchPage = [&](gsl::span<const std::byte> rdhBuffer) {
      auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(rdhBuffer[0])));
      mNrdhs++;
      auto cruId = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF;
      auto flags = o2::raw::RDHUtils::getCRUID(rdhAny) & 0xFF00;
      auto endpoint = o2::raw::RDHUtils::getEndPointID(rdhAny);
      auto feeId = cruId * 2 + endpoint + flags;
      auto linkId = o2::raw::RDHUtils::getLinkID(rdhAny);
      o2::raw::RDHUtils::setFEEID(rdhAny, feeId);
      orbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhAny);
      if (mPrint) {
        std::cout << mNrdhs << "--\n";
        o2::raw::RDHUtils::printRDH(rdhAny);
      }
    };

    patchPage(page);

    // add orbit to vector if not present yet
    mOrbits.emplace(page);

    if (!mDecoder) {
      mDecoder = mFee2Solar ? o2::mch::raw::createPageDecoder(page, channelHandler, mFee2Solar)
                            : o2::mch::raw::createPageDecoder(page, channelHandler);
    }
    mDecoder(page);
  }

 private:
  std::string readFileContent(const std::string& filename)
  {
    std::string content;
    std::string s;
    std::ifstream in(filename);
    while (std::getline(in, s)) {
      content += s;
      content += " ";
    }
    return content;
  }

  void initElec2DetMapper(const std::string& filename)
  {
    if (filename.empty()) {
      mElec2Det = createElec2DetMapper<ElectronicMapperGenerated>();
    } else {
      ElectronicMapperString::sFecMap = readFileContent(filename);
      mElec2Det = createElec2DetMapper<ElectronicMapperString>();
    }
  }

  void initFee2SolarMapper(const std::string& filename)
  {
    if (filename.empty()) {
      mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
    } else {
      ElectronicMapperString::sCruMap = readFileContent(filename);
      mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperString>();
    }
  }

 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
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

    mDs2manu = ic.options().get<bool>("ds2manu");
    mPrint = ic.options().get<bool>("print");

    auto mapCRUfile = ic.options().get<std::string>("cru-map");
    auto mapFECfile = ic.options().get<std::string>("fec-map");

    initFee2SolarMapper(mapCRUfile);
    initElec2DetMapper(mapFECfile);
  }

  //_________________________________________________________________________________________________
  void
    decodeTF(framework::ProcessingContext& pc, std::vector<o2::mch::Digit>& digits)
  {
    // get the input buffer
    auto& inputs = pc.inputs();
    DPLRawParser parser(inputs, o2::framework::select("TF:MCH/RAWDATA"));

    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* rdh = it.get_if<RDH>();
      auto const* raw = it.raw();
      size_t payloadSize = it.size();
      if (payloadSize == 0) {
        continue;
      }

      gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(raw), sizeof(RDH) + payloadSize);
      decodeBuffer(buffer, digits);
    }
  }

  //_________________________________________________________________________________________________
  void decodeReadout(const o2::framework::DataRef& input, std::vector<o2::mch::Digit>& digits)
  {
    static int nFrame = 1;
    // get the input buffer
    if (input.spec->binding != "readout") {
      return;
    }

    const auto* header = o2::header::get<header::DataHeader*>(input.header);
    if (!header) {
      return;
    }

    auto const* raw = input.payload;
    // size of payload
    size_t payloadSize = header->payloadSize;

    if (mPrint) {
      std::cout << nFrame << "  payloadSize=" << payloadSize << std::endl;
    }
    nFrame += 1;
    if (payloadSize == 0) {
      return;
    }

    gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(raw), payloadSize);
    decodeBuffer(buffer, digits);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    std::vector<o2::mch::Digit> digits;

    mOrbits.clear();

    decodeTF(pc, digits);
    for (auto&& input : pc.inputs()) {
      if (input.spec->binding == "readout") {
        decodeReadout(input, digits);
      }
    }

    if (mPrint) {
      for (auto d : digits) {
        std::cout << " DE# " << d.getDetID() << " PadId " << d.getPadID() << " ADC " << d.getADC() << " time " << d.getTime().sampaTime << std::endl;
      }
    }

    auto createBuffer = [&](auto& vec, size_t& size) {
      size = vec.empty() ? 0 : sizeof(*(vec.begin())) * vec.size();
      char* buf = nullptr;
      if (size > 0) {
        buf = (char*)malloc(size);
        if (buf) {
          char* p = buf;
          size_t sizeofElement = sizeof(*(vec.begin()));
          for (auto& element : vec) {
            memcpy(p, &element, sizeofElement);
            p += sizeofElement;
          }
        }
      }
      return buf;
    };

    // send the output buffer via DPL
    size_t digitsSize, orbitsSize;
    char* digitsBuffer = createBuffer(digits, digitsSize);
    char* orbitsBuffer = createBuffer(mOrbits, orbitsSize);

    // create the output message
    auto freefct = [](void* data, void*) { free(data); };
    pc.outputs().adoptChunk(Output{"MCH", "DIGITS", 0}, digitsBuffer, digitsSize, freefct, nullptr);
    pc.outputs().adoptChunk(Output{"MCH", "ORBITS", 0}, orbitsBuffer, orbitsSize, freefct, nullptr);
  }

 private:
  Elec2DetMapper mElec2Det{nullptr};
  FeeLink2SolarMapper mFee2Solar{nullptr};
  o2::mch::raw::PageDecoder mDecoder;
  size_t mNrdhs{0};
  std::unordered_set<OrbitInfo, OrbitInfoHash> mOrbits; ///< list of orbits in the processed buffer

  std::ifstream mInputFile{}; ///< input file
  bool mDs2manu = false;      ///< print convert channel numbering from Run3 to Run1-2 order
  bool mPrint = false;        ///< print digits
};                            // namespace raw

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDecodingSpec(std::string inputSpec)
{
  return DataProcessorSpec{
    "DataDecoder",
    o2::framework::select(inputSpec.c_str()),
    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe}, OutputSpec{"MCH", "ORBITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DataDecoderTask>()},
    Options{{"print", VariantType::Bool, false, {"print digits"}},
            {"cru-map", VariantType::String, "", {"custom CRU mapping"}},
            {"fec-map", VariantType::String, "", {"custom FEC mapping"}},
            {"ds2manu", VariantType::Bool, false, {"convert channel numbering from Run3 to Run1-2 order"}}}};
}

} // namespace raw
} // namespace mch
} // end namespace o2
