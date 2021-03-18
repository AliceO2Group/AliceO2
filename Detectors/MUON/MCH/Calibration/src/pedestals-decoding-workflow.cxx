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
#include <array>
#include <functional>
#include <chrono>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

#include "Headers/RDHAny.h"
#include "MCHRawDecoder/PageDecoder.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "MCHRawCommon/DataFormats.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"

#include "MCHBase/DecoderError.h"
#include "MCHCalibration/PedestalDigit.h"

static const size_t SOLAR_ID_MAX = 100 * 8;

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
class PedestalsTask
{
 public:
  PedestalsTask(std::string spec) : mInputSpec(spec) {}

  void initElec2DetMapper(std::string filename)
  {
    LOG(INFO) << "[initElec2DetMapper] filename=" << filename;
    if (filename.empty()) {
      mElec2Det = createElec2DetMapper<ElectronicMapperGenerated>();
    } else {
      ElectronicMapperString::sFecMap = readFileContent(filename);
      mElec2Det = createElec2DetMapper<ElectronicMapperString>();
    }
  };

  void initFee2SolarMapper(std::string filename)
  {
    LOG(INFO) << "[initFee2SolarMapper] filename=" << filename;
    if (filename.empty()) {
      mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
    } else {
      ElectronicMapperString::sCruMap = readFileContent(filename);
      mFee2Solar = createFeeLink2SolarMapper<ElectronicMapperString>();
    }
  };

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    mDebug = ic.options().get<bool>("debug");
    mPatchRDH = ic.options().get<bool>("patch-rdh");

    auto mapCRUfile = ic.options().get<std::string>("cru-map");
    auto mapFECfile = ic.options().get<std::string>("fec-map");
    initFee2SolarMapper(mMapCRUfile);
    initElec2DetMapper(mMapFECfile);
    auto stop = [this]() {
      if (mTFcount > 0) {
        LOG(INFO) << "time spent for decoding (ms): min=" << mTimeDecoderMin->count() << ", max="
                  << mTimeDecoderMax->count() << ", mean=" << mTimeDecoder.count() / mTFcount;
      }
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
    ic.services().get<CallbackService>().set(CallbackService::Id::Reset, [this]() { reset(); });
  }

  //_________________________________________________________________________________________________
  void reset()
  {
    mDigits.clear();
  }

  //_________________________________________________________________________________________________
  void decodePage(gsl::span<const std::byte> page)
  {
    size_t ndigits{0};

    uint32_t orbit;

    auto tStart = std::chrono::high_resolution_clock::now();

    auto channelHandler = [&](DsElecId dsElecId, uint8_t channel, o2::mch::raw::SampaCluster sc) {
      auto solarId = dsElecId.solarId();
      auto dsId = dsElecId.elinkId();

      mDigits.emplace_back(o2::mch::calibration::PedestalDigit(solarId, dsId, channel, sc.bunchCrossing, 0, sc.samples));
      ++ndigits;
    };

    auto errorHandler = [&](DsElecId dsElecId, int8_t chip, uint32_t error) {
      auto solarId = dsElecId.solarId();
      auto dsId = dsElecId.elinkId();

      mErrors.emplace_back(o2::mch::DecoderError(solarId, dsId, chip, error));
    };

    if (mPatchRDH) {
      patchPage(page, mDebug);
    }

    if (!mDecoder) {
      DecodedDataHandlers handlers;
      handlers.sampaChannelHandler = channelHandler;
      handlers.sampaErrorHandler = errorHandler;
      mDecoder = mFee2Solar ? o2::mch::raw::createPageDecoder(page, handlers, mFee2Solar)
                            : o2::mch::raw::createPageDecoder(page, handlers);
    }
    try {
      mDecoder(page);
    } catch (std::exception& e) {
      std::cout << e.what() << '\n';
    }
  }

  //_________________________________________________________________________________________________
  void decodeBuffer(gsl::span<const std::byte> buf)
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

  //_________________________________________________________________________________________________
  // the decodeTF() function processes the messages generated by the (sub)TimeFrame builder
  void decodeTF(framework::ProcessingContext& pc)
  {
    // get the input buffer
    auto& inputs = pc.inputs();
    DPLRawParser parser(inputs, o2::framework::select(mInputSpec.c_str()));

    auto tStart = std::chrono::high_resolution_clock::now();
    size_t totPayloadSize = 0;
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* raw = it.raw();
      if (!raw) {
        continue;
      }
      size_t payloadSize = it.size();
      totPayloadSize += payloadSize;

      gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(raw), sizeof(RDH) + payloadSize);
      decodeBuffer(buffer);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();

    if (totPayloadSize > 0) {
      std::chrono::duration<double, std::milli> elapsed = tEnd - tStart;
      mTimeDecoder += elapsed;
      if (!mTimeDecoderMin || (elapsed < mTimeDecoderMin)) {
        mTimeDecoderMin = elapsed;
      }
      if (!mTimeDecoderMax || (elapsed > mTimeDecoderMax)) {
        mTimeDecoderMax = elapsed;
      }
      mTFcount += 1;
    }
  }

  //_________________________________________________________________________________________________
  // the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
  void decodeReadout(const o2::framework::DataRef& input)
  {
    const auto* header = o2::header::get<header::DataHeader*>(input.header);
    if (!header) {
      return;
    }

    auto const* raw = input.payload;
    size_t payloadSize = header->payloadSize;

    gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(raw), payloadSize);
    decodeBuffer(buffer);
  }

  //_________________________________________________________________________________________________
  void logStats()
  {
    static auto loggerStart = std::chrono::high_resolution_clock::now();
    static auto loggerEnd = loggerStart;
    static uint64_t nDigits = 0;
    static uint64_t nTF = 0;

    nDigits += mDigits.size();
    nTF += 1;

    loggerEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> loggerElapsed = loggerEnd - loggerStart;
    if (loggerElapsed.count() > 1000) {
      LOG(INFO) << "Processed " << nDigits << " digits in " << nTF << " time frames";
      nDigits = 0;
      nTF = 0;
      loggerStart = std::chrono::high_resolution_clock::now();
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
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

    reset();
    for (auto&& input : pc.inputs()) {
      if (input.spec->binding == "TF") {
        decodeTF(pc);
      }
      if (input.spec->binding == "readout") {
        decodeReadout(input);
      }
    }

    size_t digitsSize;
    char* digitsBuffer = createBuffer(mDigits, digitsSize);

    size_t errorsSize;
    char* errorsBuffer = createBuffer(mErrors, errorsSize);

    // create the output message
    auto freefct = [](void* data, void*) { free(data); };
    pc.outputs().adoptChunk(Output{"MCH", "PDIGITS", 0}, digitsBuffer, digitsSize, freefct, nullptr);
    pc.outputs().adoptChunk(Output{"MCH", "ERRORS", 0}, errorsBuffer, errorsSize, freefct, nullptr);

    logStats();
  }

 private:
  o2::mch::raw::PageDecoder mDecoder;
  SampaChannelHandler mChannelHandler;
  std::vector<o2::mch::calibration::PedestalDigit> mDigits;
  std::vector<o2::mch::DecoderError> mErrors;

  Elec2DetMapper mElec2Det{nullptr};
  FeeLink2SolarMapper mFee2Solar{nullptr};
  std::string mMapCRUfile;
  std::string mMapFECfile;

  std::string mInputSpec;   /// selection string for the input data
  bool mPatchRDH = {false}; /// flag to enable verbose output
  bool mDebug = {false};    /// flag to enable verbose output

  std::chrono::duration<double, std::milli> mTimeDecoder{};
  std::optional<std::chrono::duration<double, std::milli>> mTimeDecoderMin{};
  std::optional<std::chrono::duration<double, std::milli>> mTimeDecoderMax{};
  size_t mTFcount{0};
};

} // namespace raw
} // namespace mch
} // end namespace o2

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"dataspec", VariantType::String, "TF:MCH/RAWDATA", {"selection string for the input data"}});
}

#include "Framework/runDataProcessing.h"

using namespace o2;
using namespace o2::framework;

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPedestalsSpec(std::string inputSpec)
{
  //o2::mch::raw::PedestalsTask task();
  return DataProcessorSpec{
    "Pedestals",
    o2::framework::select(inputSpec.c_str()),
    Outputs{OutputSpec{"MCH", "PDIGITS", 0, Lifetime::Timeframe}, OutputSpec{"MCH", "ERRORS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<o2::mch::raw::PedestalsTask>(inputSpec)},
    Options{{"debug", VariantType::Bool, false, {"enable verbose output"}},
            {"patch-rdh", VariantType::Bool, false, {"fill the FEEID RDH field using the CRUID value"}},
            {"noise-threshold", VariantType::Float, (float)2.0, {"maximum acceptable noise value"}},
            {"pedestal-threshold", VariantType::Float, (float)150, {"maximum acceptable pedestal value"}},
            {"cru-map", VariantType::String, "", {"custom CRU mapping"}},
            {"fec-map", VariantType::String, "", {"custom FEC mapping"}}}};
}

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  auto inputSpec = config.options().get<std::string>("dataspec");

  WorkflowSpec specs;

  DataProcessorSpec producer = getPedestalsSpec(inputSpec);
  specs.push_back(producer);

  return specs;
}
