// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCWorkflow/ZSSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/WorkflowSpec.h" // o2::framework::mergeInputs
#include "Framework/DataRefUtils.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/GPUCATracking.h"
#include "DataFormatsTPC/Digit.h"
#include "GPUDataTypes.h"
#include "GPUHostDataTypes.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCBase/Sector.h"
#include "Algorithm/Parser.h"
#include <FairMQLogger.h>
#include <memory> // for make_shared
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <array>
#include <unistd.h>
#include "GPUParam.h"
#include "GPUReconstructionConvert.h"
#include "GPURawData.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"
#include "TPCBase/RDHUtils.h"
#include "TPCBase/ZeroSuppress.h"
#include "DPLUtils/DPLRawParser.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsTPC/WorkflowHelper.h"

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;

namespace o2
{
namespace tpc
{

DataProcessorSpec getZSEncoderSpec(std::vector<int> const& inputIds, bool zs10bit, float threshold = 2.f, bool outRaw = false)
{
  std::string processorName = "tpc-zsEncoder";
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = o2::gpu::GPUTrackingInOutZS::NENDPOINTS;
  using DigitArray = std::array<gsl::span<const o2::tpc::Digit>, o2::tpc::Sector::MAXSECTOR>;

  struct ProcessAttributes {
    std::unique_ptr<unsigned long long int[]> zsoutput;
    std::vector<unsigned int> sizes;
    std::vector<int> inputIds;
    bool verify = false;
    int verbosity = 1;
    bool finished = false;
  };

  auto initFunction = [inputIds, zs10bit, threshold, outRaw](InitContext& ic) {
    auto processAttributes = std::make_shared<ProcessAttributes>();
    auto& zsoutput = processAttributes->zsoutput;
    processAttributes->inputIds = inputIds;
    auto& verify = processAttributes->verify;
    auto& sizes = processAttributes->sizes;
    auto& verbosity = processAttributes->verbosity;

    auto processingFct = [processAttributes, zs10bit, threshold, outRaw](
                           ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      auto& zsoutput = processAttributes->zsoutput;
      auto& verify = processAttributes->verify;
      auto& sizes = processAttributes->sizes;
      auto& verbosity = processAttributes->verbosity;

      std::array<gsl::span<const o2::tpc::Digit>, NSectors> inputDigits;

      GPUParam _GPUParam;
      GPUO2InterfaceConfiguration config;
      config.configEvent.solenoidBz = 5.00668;
      config.ReadConfigurableParam();
      _GPUParam.SetDefaults(&config.configEvent, &config.configReconstruction, &config.configProcessing, nullptr);

      const auto& inputs = getWorkflowTPCInput(pc, 0, false, false, 0xFFFFFFFFF, true);
      sizes.resize(NSectors * NEndpoints);
      bool zs12bit = !zs10bit;
      o2::InteractionRecord ir = o2::raw::HBFUtils::Instance().getFirstIR();
      o2::gpu::GPUReconstructionConvert::RunZSEncoder<o2::tpc::Digit, DigitArray>(inputs->inputDigits, &zsoutput, sizes.data(), nullptr, &ir, _GPUParam, zs12bit, verify, threshold);
      ZeroSuppressedContainer8kb* page = reinterpret_cast<ZeroSuppressedContainer8kb*>(zsoutput.get());
      unsigned int offset = 0;
      for (unsigned int i = 0; i < NSectors; i++) {
        unsigned int pageSector = 0;
        for (unsigned int j = 0; j < NEndpoints; j++) {
          if (sizes[i * NEndpoints + j] != 0) {
            pageSector += sizes[i * NEndpoints + j];
          }
        }
        offset += pageSector;
      }
      o2::tpc::TPCSectorHeader sh{0};
      gsl::span<ZeroSuppressedContainer8kb> outp(&page[0], offset);
      pc.outputs().snapshot(Output{gDataOriginTPC, "TPCZS", 0, Lifetime::Timeframe, sh}, outp);
      pc.outputs().snapshot(Output{gDataOriginTPC, "ZSSIZES", 0, Lifetime::Timeframe, sh}, sizes);

      if (outRaw) {
        // ===| set up raw writer |===================================================
        std::string inputGRP = o2::base::NameConf::getGRPFileName();
        const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
        o2::raw::RawFileWriter writer{"TPC"}; // to set the RDHv6.sourceID if V6 is used
        writer.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::TPC)); // must be set explicitly
        uint32_t rdhV = o2::raw::RDHUtils::getVersion<o2::gpu::RAWDataHeaderGPU>();
        writer.useRDHVersion(rdhV);
        std::string outDir = "./";
        const unsigned int defaultLink = rdh_utils::UserLogicLinkID;
        enum LinksGrouping { All,
                             Sector,
                             Link };
        auto useGrouping = Sector;

        for (unsigned int i = 0; i < NSectors; i++) {
          for (unsigned int j = 0; j < NEndpoints; j++) {
            const unsigned int cruInSector = j / 2;
            const unsigned int cruID = i * 10 + cruInSector;
            const rdh_utils::FEEIDType feeid = rdh_utils::getFEEID(cruID, j & 1, defaultLink);
            std::string outfname;
            if (useGrouping == LinksGrouping::All) { // single file for all links
              outfname = fmt::format("{}tpc_all.raw", outDir);
            } else if (useGrouping == LinksGrouping::Sector) { // file per sector
              outfname = fmt::format("{}tpc_sector{}.raw", outDir, i);
            } else if (useGrouping == LinksGrouping::Link) { // file per link
              outfname = fmt::format("{}cru{}_{}.raw", outDir, cruID, j & 1);
            }
            writer.registerLink(feeid, cruID, defaultLink, j & 1, outfname);
          }
        }
        if (useGrouping != LinksGrouping::Link) {
          writer.useCaching();
        }
        o2::gpu::GPUReconstructionConvert::RunZSEncoder<o2::tpc::Digit>(inputDigits, nullptr, nullptr, &writer, &ir, _GPUParam, zs12bit, false, threshold);
        writer.writeConfFile("TPC", "RAWDATA", fmt::format("{}tpcraw.cfg", outDir));
      }
      zsoutput.reset(nullptr);
      sizes = {};
      inputDigits = {};
    };
    return processingFct;
  };

  auto createInputSpecs = [inputIds]() {
    Inputs inputs;
    //    inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe});
    inputs.emplace_back(InputSpec{"input", gDataOriginTPC, "DIGITS", 0, Lifetime::Timeframe});
    return std::move(mergeInputs(inputs, inputIds.size(),
                                 [inputIds](InputSpec& input, size_t index) {
                                   // using unique input names for the moment but want to find
                                   // an input-multiplicity-agnostic way of processing
                                   input.binding += std::to_string(inputIds[index]);
                                   DataSpecUtils::updateMatchingSubspec(input, inputIds[index]);
                                 }));
    return inputs;
  };

  auto createOutputSpecs = []() {
    std::vector<OutputSpec> outputSpecs = {};
    OutputLabel label{"TPCZS"};
    constexpr o2::header::DataDescription datadesc("TPCZS");
    OutputLabel label2{"sizes"};
    constexpr o2::header::DataDescription datadesc2("ZSSIZES");
    outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    outputSpecs.emplace_back(label2, gDataOriginTPC, datadesc2, 0, Lifetime::Timeframe);
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{"tpc-zsEncoder",
                           {createInputSpecs()},
                           {createOutputSpecs()},
                           AlgorithmSpec(initFunction)};
} //spec end

DataProcessorSpec getZStoDigitsSpec(std::vector<int> const& inputIds)
{
  std::string processorName = "tpc-zs-to-Digits";
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = o2::gpu::GPUTrackingInOutZS::NENDPOINTS;

  struct ProcessAttributes {
    std::array<std::vector<Digit>, NSectors> outDigits;
    std::unique_ptr<unsigned long long int[]> zsinput;
    std::vector<unsigned int> sizes;
    std::unique_ptr<o2::tpc::ZeroSuppress> decoder;
    std::vector<int> inputIds;
    bool verify = false;
    int verbosity = 1;
    bool finished = false;

    /// Digit sorting according to expected output from simulation
    void sortDigits()
    {
      // sort digits
      for (auto& digits : outDigits) {
        std::sort(digits.begin(), digits.end(), [](const auto& a, const auto& b) {
          if (a.getTimeStamp() < b.getTimeStamp()) {
            return true;
          }
          if ((a.getTimeStamp() == b.getTimeStamp()) && (a.getRow() < b.getRow())) {
            return true;
          }
          return false;
        });
      }
    }
  };

  auto initFunction = [inputIds](InitContext& ic) {
    auto processAttributes = std::make_shared<ProcessAttributes>();
    processAttributes->inputIds = inputIds;
    auto& outDigits = processAttributes->outDigits;
    auto& decoder = processAttributes->decoder;
    decoder = std::make_unique<o2::tpc::ZeroSuppress>();
    auto& verbosity = processAttributes->verbosity;

    auto processingFct = [processAttributes](
                           ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }
      std::array<unsigned int, NEndpoints * NSectors> sizes;
      gsl::span<const ZeroSuppressedContainer8kb> inputZS;
      std::array<gsl::span<const ZeroSuppressedContainer8kb>, NSectors> inputZSSector;
      auto& outDigits = processAttributes->outDigits;
      auto& decoder = processAttributes->decoder;
      auto& verbosity = processAttributes->verbosity;
      unsigned int firstOrbit = 0;

      for (unsigned int i = 0; i < NSectors; i++) {
        outDigits[i].clear();
      }
      o2::InteractionRecord ir = o2::raw::HBFUtils::Instance().getFirstIR();
      firstOrbit = ir.orbit;
      std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
      for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
        const o2::header::DataHeader* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const gsl::span<const char> raw = pc.inputs().get<gsl::span<char>>(ref);
        o2::framework::RawParser parser(raw.data(), raw.size());

        const unsigned char* ptr = nullptr;
        int count = 0;
        int cruID = 0;
        rdh_utils::FEEIDType lastFEE = -1;
        size_t totalSize = 0;
        for (auto it = parser.begin(); it != parser.end(); it++) {
          const unsigned char* current = it.raw();
          const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)current;
          if (current == nullptr || it.size() == 0 || (current - ptr) % TPCZSHDR::TPC_ZS_PAGE_SIZE || o2::raw::RDHUtils::getFEEID(*rdh) != lastFEE) {
            if (count) {
              unsigned int sector = cruID / 10;
              gsl::span<const ZeroSuppressedContainer8kb> z0in(reinterpret_cast<const ZeroSuppressedContainer8kb*>(ptr), count);
              decoder->DecodeZSPages(&z0in, &outDigits[sector], firstOrbit);
            }
            count = 0;
            lastFEE = o2::raw::RDHUtils::getFEEID(*rdh);
            cruID = int(o2::raw::RDHUtils::getCRUID(*rdh));
            if (it.size() == 0) {
              ptr = nullptr;
              continue;
            }
            ptr = current;
          }
          count++;
        }
        if (count) {
          unsigned int sector = cruID / 10;
          gsl::span<const ZeroSuppressedContainer8kb> z0in(reinterpret_cast<const ZeroSuppressedContainer8kb*>(ptr), count);
          decoder->DecodeZSPages(&z0in, &outDigits[sector], firstOrbit);
        }
      }
      for (int i = 0; i < NSectors; i++) {
        LOG(INFO) << "digits in sector " << i << " : " << outDigits[i].size();
        o2::tpc::TPCSectorHeader sh{i};
        pc.outputs().snapshot(Output{gDataOriginTPC, "DIGITS", (unsigned int)i, Lifetime::Timeframe, sh}, outDigits[i]);
      }
    };
    return processingFct;
  };

  auto createInputSpecs = []() {
    Inputs inputs;
    inputs.emplace_back(InputSpec{"zsraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Timeframe});

    return std::move(inputs);
  };

  auto createOutputSpecs = []() {
    std::vector<OutputSpec> outputSpecs = {};
    OutputLabel label{"tpcdigits"};
    constexpr o2::header::DataDescription datadesc("DIGITS");
    for (int i = 0; i < NSectors; i++) {
      outputSpecs.emplace_back(gDataOriginTPC, "DIGITS", i, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{"decode-zs-to-digits",
                           {createInputSpecs()},
                           {createOutputSpecs()},
                           AlgorithmSpec(initFunction)};
} // spec end

} // namespace tpc
} // namespace o2
