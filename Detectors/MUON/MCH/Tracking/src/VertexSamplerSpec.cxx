// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file VertexSamplerSpec.cxx
/// \brief Implementation of a data processor to read and send vertices
///
/// \author Philippe Pillot, Subatech

#include "VertexSamplerSpec.h"

#include <iostream>
#include <fstream>
#include <string>

#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MathUtils/Cartesian3D.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class VertexSamplerSpec
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file from the context
    LOG(INFO) << "initializing vertex sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    if (!inputFileName.empty()) {
      mInputFile.open(inputFileName, ios::binary);
      if (!mInputFile.is_open()) {
        throw invalid_argument("Cannot open input file" + inputFileName);
      }
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop vertex sampler";
      if (mInputFile.is_open()) {
        this->mInputFile.close();
      }
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the vertex of the current event if an input file is provided
    /// or the default vertex (0.,0.,0.) otherwise

    // get the current event
    auto msgIn = pc.inputs().get<gsl::span<char>>("tracks");
    if (msgIn.size() < SSizeOfInt) {
      throw out_of_range("missing event header");
    }
    const int& eventTracks = *reinterpret_cast<const int*>(msgIn.data());

    // read the corresponding vertex or set it to (0,0,0)
    int eventVtx(-1);
    Point3D<double> vertex(0., 0., 0.);
    if (mInputFile.is_open()) {
      do {
        mInputFile.read(reinterpret_cast<char*>(&eventVtx), SSizeOfInt);
        if (mInputFile.fail()) {
          throw out_of_range(std::string("missing vertex for event ") + eventTracks);
        }
        VertexStruct vtx{};
        mInputFile.read(reinterpret_cast<char*>(&vtx), SSizeOfVertexStruct);
        vertex.SetCoordinates(vtx.x, vtx.y, vtx.z);
      } while (eventVtx != eventTracks);
    }

    // create the output message
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "VERTEX", 0, Lifetime::Timeframe}, SSizeOfInt + SSizeOfPoint3D);
    if (msgOut.size() != SSizeOfInt + SSizeOfPoint3D) {
      throw length_error("incorrect message payload");
    }

    // fill it
    auto bufferPtr = msgOut.data();
    memcpy(bufferPtr, &eventVtx, SSizeOfInt);
    bufferPtr += SSizeOfInt;
    memcpy(bufferPtr, &vertex, SSizeOfPoint3D);
  }

 private:
  struct VertexStruct {
    double x;
    double y;
    double z;
  };
  static constexpr int SSizeOfInt = sizeof(int);
  static constexpr int SSizeOfVertexStruct = sizeof(VertexStruct);
  static constexpr int SSizeOfPoint3D = sizeof(Point3D<double>);

  std::ifstream mInputFile{}; ///< input file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getVertexSamplerSpec()
{
  return DataProcessorSpec{
    "VertexSampler",
    Inputs{InputSpec{"tracks", "MCH", "TRACKS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "VERTEX", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<VertexSamplerSpec>()},
    Options{{"infile", VariantType::String, "", {"input filename"}}}};
}

} // end namespace mch
} // end namespace o2
