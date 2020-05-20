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
/// \file    runFileReader.cxx
/// \author  Andrea Ferrero
///
/// \brief This is an executable that reads a data file from disk and sends the data to QC via DPL.
///
/// This is an executable that reads a data file from disk and sends the data to QC via the Data Processing Layer.
/// It can be used as a data source for QC development. For example, one can do:
/// \code{.sh}
/// o2-qc-run-file-reader --infile=some_data_file | o2-qc --config json://${QUALITYCONTROL_ROOT}/etc/your_config.json
/// \endcode
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
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"

#include "DPLUtils/DPLRawParser.h"
#include "MCHBase/Digit.h"
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2;
using namespace o2::framework;

struct CRUheader {
  uint8_t header_version;
  uint8_t header_size;
  uint16_t block_length;
  uint16_t fee_id;
  uint8_t priority_bit;
  uint8_t reserved_1;
  uint16_t next_packet_offset;
  uint16_t memory_size;
  uint8_t link_id;
  uint8_t packet_counter;
  uint16_t cru_id : 12;
  uint8_t dpw_id : 4;
  uint32_t hb_orbit;
  //uint16_t cru_id;
  //uint8_t dummy1;
  //uint64_t dummy2;
};

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

class FileReaderTask
{
  void decodeBuffer(gsl::span<const std::byte> page, std::vector<o2::mch::Digit>& digits)
  {
    size_t ndigits{0};

    auto channelHandler = [&](DsElecId dsElecId, uint8_t channel, o2::mch::raw::SampaCluster sc) {
      auto s = asString(dsElecId);
      channel = ds2manu(int(channel));
      if (mPrint) {
        auto ch = fmt::format("{}-CH{} samples={}", s, channel, sc.samples.size());
        std::cout << ch << std::endl;
      }
      double digitadc(0);
      //for (auto d = 0; d < sc.nofSamples(); d++) {
      for (auto d = 0; d < sc.samples.size(); d++) {
        digitadc += sc.samples[d];
      }

      int deId = -1;
      int dsIddet = -1;
      if (auto opt = Elec2Det(dsElecId); opt.has_value()) {
        DsDetId dsDetId = opt.value();
        dsIddet = dsDetId.dsId();
        deId = dsDetId.deId();
      }
      if (dsIddet < 0 || deId < 0) {
        std::cout << "SOLAR " << (int)dsElecId.solarId()
                  << "  DS " << (int)dsElecId.elinkId() << " (" << (int)dsElecId.elinkGroupId() << "," << (int)dsElecId.elinkIndexInGroup() << ")"
                  << "  CHIP " << ((int)channel) / 32 << "  CH " << ((int)channel) % 32 << "  ADC " << digitadc << "  DE# " << deId << "  DSid " << dsIddet << std::endl;
        return;
      }

      int padId = -1;
      try {
        const Segmentation& segment = segmentation(deId);
        //Segmentation segment(deId);

        padId = segment.findPadByFEE(dsIddet, int(channel));
        if (mPrint)
          std::cout << "DS " << (int)dsElecId.elinkId() << "  CHIP " << ((int)channel) / 32 << "  CH " << ((int)channel) % 32 << "  ADC " << digitadc << "  DE# " << deId << "  DSid " << dsIddet << "  PadId " << padId << std::endl;
      } catch (const std::exception& e) {
        return;
      }

      o2::mch::Digit::Time time;

      digits.emplace_back(o2::mch::Digit(deId, padId, digitadc, time));
      //o2::mch::Digit& mchdigit = digits.back();
      //mchdigit.setDetID(deId);
      //mchdigit.setPadID(padId);
      //mchdigit.setADC(digitadc);
      //mchdigit.setTimeStamp(time);

      if (mPrint)
        std::cout << "DIGIT STORED:\nADC " << digits.back().getADC() << " DE# " << digits.back().getDetID() << " PadId " << digits.back().getPadID() << " time " << digits.back().getTime().sampaTime << std::endl;
      ++ndigits;
    };

    const auto patchPage = [&](gsl::span<const std::byte> rdhBuffer) {
      auto rdhPtr = const_cast<void*>(reinterpret_cast<const void*>(rdhBuffer.data()));
      nrdhs++;
      auto cruId = o2::raw::RDHUtils::getCRUID(rdhPtr);
      auto endpoint = o2::raw::RDHUtils::getEndPointID(rdhPtr);
      o2::raw::RDHUtils::setFEEID(rdhPtr, cruId * 2 + endpoint);
      if (mPrint) {
        std::cout << nrdhs << "--\n";
        o2::raw::RDHUtils::printRDH(rdhPtr);
      }
    };

    patchPage(page);
    if (!decoder.has_value())
      decoder = o2::mch::raw::createPageDecoder(page, channelHandler);
    decoder.value()(page);
  }

 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file and other options from the context
    LOG(INFO) << "initializing file reader";

    for (int i = 0; i < 64; i++) {
      for (int j = 0; j < 64; j++) {
        if (refManu2ds_st345[j] != i)
          continue;
        refDs2manu_st345[i] = j;
        break;
      }
    }

    Elec2Det = createElec2DetMapper<ElectronicMapperGenerated>();
    fee2Solar = o2::mch::raw::createFeeLink2SolarMapper<ElectronicMapperGenerated>();
    nrdhs = 0;

    mFrameMax = ic.options().get<int>("frames");
    mPrint = ic.options().get<bool>("print");

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, std::ios::binary);
    if (!mInputFile.is_open()) {
      throw std::invalid_argument("Cannot open input file \"" + inputFileName + "\"");
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop file reader";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    std::vector<o2::mch::Digit> digits;

    uint32_t CRUbuf[4 * 4];
    CRUheader CRUh;
    /// send one RDH block via DPL

    int RDH_BLOCK_SIZE = 8192;

    if (mFrameMax == 0)
      return;
    //printf("mFrameMax: %d\n", mFrameMax);
    if (mFrameMax > 0)
      mFrameMax -= 1;

    mInputFile.read((char*)(&CRUbuf), sizeof(CRUbuf));
    memcpy(&CRUh, CRUbuf, sizeof(CRUheader));
    if (CRUh.header_version != 4 || CRUh.header_size != 64)
      return;

    RDH_BLOCK_SIZE = CRUh.next_packet_offset;

    char* buf = (char*)malloc(RDH_BLOCK_SIZE);
    memcpy(buf, CRUbuf, CRUh.header_size);

    mInputFile.read(buf + CRUh.header_size, RDH_BLOCK_SIZE - CRUh.header_size);
    if (mInputFile.fail()) {
      if (mPrint) {
        LOG(INFO) << "end of file reached";
      }
      free(buf);
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }

    gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(buf), RDH_BLOCK_SIZE);
    decodeBuffer(buffer, digits);

    if (mPrint) {
      for (auto d : digits) {
        std::cout << " DE# " << d.getDetID() << " PadId " << d.getPadID() << " ADC " << d.getADC() << " time " << d.getTime().sampaTime << std::endl;
      }
    }

    const size_t OUT_SIZE = sizeof(o2::mch::Digit) * digits.size();

    /// send the output buffer via DPL
    char* outbuffer = nullptr;
    outbuffer = (char*)realloc(outbuffer, OUT_SIZE);
    memcpy(outbuffer, digits.data(), OUT_SIZE);

    // create the output message
    auto freefct = [](void* data, void*) { free(data); };
    pc.outputs().adoptChunk(Output{"MCH", "DIGITS", 0}, outbuffer, OUT_SIZE, freefct, nullptr);
  }

 private:
  std::function<std::optional<DsDetId>(DsElecId)> Elec2Det;
  std::function<std::optional<uint16_t>(FeeLinkId id)> fee2Solar;
  std::optional<o2::mch::raw::PageDecoder> decoder;
  size_t nrdhs{0};

  std::ifstream mInputFile{}; ///< input file
  int mFrameMax;              ///< number of frames to process
  bool mPrint = false;        ///< print digits
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getFileReaderSpec()
{
  return DataProcessorSpec{
    "FileReader",
    Inputs{},
    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<FileReaderTask>()},
    Options{{"infile", VariantType::String, "data.raw", {"input file name"}}}};
}

} // end namespace raw
} // end namespace mch
} // end namespace o2

using namespace o2;
using namespace o2::framework;

// clang-format off
WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  // The producer to generate some data in the workflow
  DataProcessorSpec producer{
    "FileReader",
    Inputs{},
    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<o2::mch::raw::FileReaderTask>()},
    Options{ { "infile", VariantType::String, "", { "input file name" } },
      {"print", VariantType::Bool, false, {"print digits"}},
      { "frames", VariantType::Int, -1, { "number of frames to process" } }}
  };
  specs.push_back(producer);

  return specs;
}
// clang-format on
