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

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "MCHBase/Digit.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/DataDecoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHWorkflow/DataDecoderSpec.h"

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

//=======================
// Data decoder
class DataDecoderTask
{
 public:
  DataDecoderTask(std::string spec) : mInputSpec(spec) {}

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    SampaChannelHandler channelHandler;
    RdhHandler rdhHandler;

    auto ds2manu = ic.options().get<bool>("ds2manu");
    mDebug = ic.options().get<bool>("debug");
    auto mapCRUfile = ic.options().get<std::string>("cru-map");
    auto mapFECfile = ic.options().get<std::string>("fec-map");

    mDecoder = new DataDecoder(channelHandler, rdhHandler, mapCRUfile, mapFECfile, ds2manu, mDebug);
  }

  //_________________________________________________________________________________________________
  // the decodeTF() function processes the messages generated by the (sub)TimeFrame builder
  void decodeTF(framework::ProcessingContext& pc)
  {
    // get the input buffer
    auto& inputs = pc.inputs();
    DPLRawParser parser(inputs, o2::framework::select(mInputSpec.c_str()));

    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* raw = it.raw();
      if (!raw) {
        continue;
      }
      size_t payloadSize = it.size();

      gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(raw), sizeof(RDH) + payloadSize);
      mDecoder->decodeBuffer(buffer);
    }
  }

  //_________________________________________________________________________________________________
  // the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
  void decodeReadout(const o2::framework::DataRef& input)
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

    if (mDebug) {
      std::cout << nFrame << "  payloadSize=" << payloadSize << std::endl;
    }
    nFrame += 1;
    if (payloadSize == 0) {
      return;
    }

    gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(raw), payloadSize);
    mDecoder->decodeBuffer(buffer);
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

    mDecoder->reset();
    for (auto&& input : pc.inputs()) {
      //std::cout << "input: " << input.spec->binding << std::endl;
      if (input.spec->binding == "readout") {
        decodeReadout(input);
      }
      if (input.spec->binding == "TF") {
        decodeTF(pc);
      }
    }
    mDecoder->computeDigitsTime();

    auto& digits = mDecoder->getOutputDigits();
    auto& orbits = mDecoder->getOrbits();

    if (mDebug) {
      dumpOrbits(orbits);
      for (auto d : digits) {
        std::cout << " DE# " << d.getDetID() << " PadId " << d.getPadID() << " ADC " << d.getADC() << " time " << d.getTime() << std::endl;
      }
    }
    // send the output buffer via DPL
    size_t digitsSize, orbitsSize;
    char* digitsBuffer = createBuffer(digits, digitsSize);
    char* orbitsBuffer = createBuffer(orbits, orbitsSize);

    // create the output message
    auto freefct = [](void* data, void*) { free(data); };
    pc.outputs().adoptChunk(Output{"MCH", "DIGITS", 0}, digitsBuffer, digitsSize, freefct, nullptr);
    pc.outputs().adoptChunk(Output{"MCH", "ORBITS", 0}, orbitsBuffer, orbitsSize, freefct, nullptr);
  }

 private:
  std::string mInputSpec;            /// selection string for the input data
  bool mDebug = {false};             /// flag to enable verbose output
  DataDecoder* mDecoder = {nullptr}; /// pointer to the data decoder instance
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDecodingSpec(std::string inputSpec)
{
  o2::mch::raw::DataDecoderTask task(inputSpec);
  return DataProcessorSpec{
    "DataDecoder",
    o2::framework::select(inputSpec.c_str()),
    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe}, OutputSpec{"MCH", "ORBITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DataDecoderTask>(std::move(task))},
    Options{{"debug", VariantType::Bool, false, {"enable verbose output"}},
            {"cru-map", VariantType::String, "", {"custom CRU mapping"}},
            {"fec-map", VariantType::String, "", {"custom FEC mapping"}},
            {"ds2manu", VariantType::Bool, false, {"convert channel numbering from Run3 to Run1-2 order"}}}};
}

} // namespace raw
} // namespace mch
} // end namespace o2
