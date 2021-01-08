// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterSamplerSpec.cxx
/// \brief Implementation of a data processor to read and send clusters
///
/// \author Philippe Pillot, Subatech

#include "ClusterSamplerSpec.h"

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
#include "Framework/Logger.h"

#include "MCHBase/ClusterBlock.h"
#include "MCHBase/Digit.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class ClusterSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file from the context
    LOG(INFO) << "initializing cluster sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, ios::binary);
    if (!mInputFile.is_open()) {
      throw invalid_argument("Cannot open input file" + inputFileName);
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop cluster sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the clusters of the current event

    // get the number of clusters and associated digits
    int nClusters(0);
    mInputFile.read(reinterpret_cast<char*>(&nClusters), sizeof(int));
    if (mInputFile.fail()) {
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }
    int nDigits(0);
    mInputFile.read(reinterpret_cast<char*>(&nDigits), sizeof(int));

    if (nClusters < 0 || nDigits < 0) {
      throw length_error("incorrect message payload");
    }

    // create the output message
    auto clusters = pc.outputs().make<ClusterStruct>(Output{"MCH", "CLUSTERS", 0, Lifetime::Timeframe}, nClusters);

    // fill clusters in O2 format, if any
    if (nClusters > 0) {
      mInputFile.read(reinterpret_cast<char*>(clusters.data()), clusters.size_bytes());
    } else {
      LOG(INFO) << "event is empty";
    }

    // skip the digits if any
    if (nDigits > 0) {
      mInputFile.seekg(nDigits * sizeof(Digit), std::ios::cur);
    }
  }

 private:
  std::ifstream mInputFile{}; ///< input file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterSamplerSpec()
{
  return DataProcessorSpec{
    "ClusterSampler",
    Inputs{},
    Outputs{OutputSpec{"MCH", "CLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input filename"}}}};
}

} // end namespace mch
} // end namespace o2
