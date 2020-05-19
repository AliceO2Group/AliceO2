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
/// \file    DatDecoderSpec.cxx
/// \author  Andrea Ferrero
///
/// \brief Implementation of a data processor to run the raw decoding
///

#include <random>
#include <iostream>
#include <fstream>
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
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHWorkflow/DataDecoderSpec.h"
#include <array>
#include "DetectorsRaw/RDHUtils.h"

namespace o2::header
{
extern std::ostream& operator<<(std::ostream&, const o2::header::RAWDataHeaderV4&);
}

namespace o2
{
namespace mch
{
namespace raw
{

using namespace o2;
using namespace o2::framework;
using namespace o2::mch::mapping;
using RDHv4 = o2::header::RAWDataHeaderV4;

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
        std::cout << ch << std::endl;
      }
      double digitadc(0);
      //for (auto d = 0; d < sc.nofSamples(); d++) {
      for (auto d = 0; d < sc.samples.size(); d++) {
        digitadc += sc.samples[d];
      }

      int deId{-1};
      int dsIddet{-1};
      if (auto opt = mElec2Det(dsElecId); opt.has_value()) {
        DsDetId dsDetId = opt.value();
        dsIddet = dsDetId.dsId();
        deId = dsDetId.deId();
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

      digits.emplace_back(o2::mch::Digit(deId, padId, digitadc, time));

      if (mPrint)
        std::cout << "DIGIT STORED:\nADC " << digits.back().getADC() << " DE# " << digits.back().getDetID() << " PadId " << digits.back().getPadID() << " time " << digits.back().getTime().sampaTime << std::endl;
      ++ndigits;
    };

    const auto patchPage = [&](gsl::span<const std::byte> rdhBuffer) {
      auto rdhPtr = const_cast<void*>(reinterpret_cast<const void*>(rdhBuffer.data()));
      mNrdhs++;
      auto cruId = o2::raw::RDHUtils::getCRUID(rdhPtr);
      auto endpoint = o2::raw::RDHUtils::getEndPointID(rdhPtr);
      o2::raw::RDHUtils::setFEEID(rdhPtr, cruId * 2 + endpoint);
      orbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhPtr);
      if (mPrint) {
        std::cout << mNrdhs << "--\n";
        o2::raw::RDHUtils::printRDH(rdhPtr);
      }
    };

    if (!mDecoder) {
      mDecoder = mFee2Solar ? o2::mch::raw::createPageDecoder(page, channelHandler, mFee2Solar)
                            : o2::mch::raw::createPageDecoder(page, channelHandler);
    }

    patchPage(page);
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
      // retrieving RDH v4
      auto const* rdh = it.get_if<o2::header::RAWDataHeaderV4>();
      // retrieving the raw pointer of the page
      auto const* raw = it.raw();
      // size of payload
      size_t payloadSize = it.size();

      if (payloadSize == 0) {
        continue;
      }

      gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(raw), sizeof(o2::header::RAWDataHeaderV4) + payloadSize);
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

    decodeTF(pc, digits);
    for (auto&& input : pc.inputs()) {
      if (input.spec->binding == "readout")
        decodeReadout(input, digits);
    }

    if (mPrint) {
      for (auto d : digits) {
        std::cout << " DE# " << d.getDetID() << " PadId " << d.getPadID() << " ADC " << d.getADC() << " time " << d.getTime().sampaTime << std::endl;
      }
    }

    const size_t OUT_SIZE = sizeof(o2::mch::Digit) * digits.size();

    // send the output buffer via DPL
    char* outbuffer = nullptr;
    outbuffer = (char*)realloc(outbuffer, OUT_SIZE);
    memcpy(outbuffer, digits.data(), OUT_SIZE);

    // create the output message
    auto freefct = [](void* data, void*) { free(data); };
    pc.outputs().adoptChunk(Output{"MCH", "DIGITS", 0}, outbuffer, OUT_SIZE, freefct, nullptr);
  }

 private:
  Elec2DetMapper mElec2Det{nullptr};
  FeeLink2SolarMapper mFee2Solar{nullptr};
  o2::mch::raw::PageDecoder mDecoder;
  size_t mNrdhs{0};

  std::ifstream mInputFile{}; ///< input file
  bool mDs2manu = false;      ///< print convert channel numbering from Run3 to Run1-2 order
  bool mPrint = false;        ///< print digits
};                            // namespace raw

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDecodingSpec()
{
  return DataProcessorSpec{
    "DataDecoder",
    //o2::framework::select("TF:MCH/RAWDATA, re:ROUT/RAWDATA"),
    o2::framework::select("readout:ROUT/RAWDATA"),
    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DataDecoderTask>()},
    Options{{"print", VariantType::Bool, false, {"print digits"}},
            {"cru-map", VariantType::String, "", {"custom CRU mapping"}},
            {"fec-map", VariantType::String, "", {"custom FEC mapping"}},
            {"ds2manu", VariantType::Bool, false, {"convert channel numbering from Run3 to Run1-2 order"}}}};
}

} // namespace raw
} // namespace mch
} // end namespace o2
