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
#include "Framework/Logger.h"

#include "MathUtils/Cartesian.h"

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

    // read the corresponding vertex or set it to (0,0,0)
    math_utils::Point3D<double> vertex(0., 0., 0.);
    if (mInputFile.is_open()) {
      int event(-1);
      mInputFile.read(reinterpret_cast<char*>(&event), sizeof(int));
      if (mInputFile.fail()) {
        throw out_of_range("missing vertex");
      }
      VertexStruct vtx{};
      mInputFile.read(reinterpret_cast<char*>(&vtx), sizeof(VertexStruct));
      vertex.SetCoordinates(vtx.x, vtx.y, vtx.z);
    }

    // create the output message
    pc.outputs().snapshot(Output{"MCH", "VERTEX", 0, Lifetime::Timeframe}, vertex);
  }

 private:
  struct VertexStruct {
    double x;
    double y;
    double z;
  };

  std::ifstream mInputFile{}; ///< input file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getVertexSamplerSpec()
{
  return DataProcessorSpec{
    "VertexSampler",
    // the input message is just used to synchronize the sending of the vertex with the track reconstruction
    Inputs{InputSpec{"tracks", "MCH", "TRACKS", 0, Lifetime::Timeframe},
           InputSpec{"clusters", "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "VERTEX", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<VertexSamplerSpec>()},
    Options{{"infile", VariantType::String, "", {"input filename"}}}};
}

} // end namespace mch
} // end namespace o2
