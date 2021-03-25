// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "HMPIDReconstruction/Clusterer.h"
#include "DataFormatsHMP/Cluster.h"
#include "DataFormatsHMP/Digit.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;
using namespace o2::hmpid::raw;

namespace o2
{
namespace hmpid
{

// use the tasking system of DPL
// just need to implement 2 special methods init + run (there is no need to inherit from anything)
class HMPIDDPLClustererTask
{
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  bool mUseMC = true;

 public:
  explicit HMPIDDPLClustererTask(bool useMC) : mUseMC(useMC) {}
  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    // get digit data
    auto digits = pc.inputs().get<std::vector<o2::hmpid::raw::Digit>*>("hmpiddigits");
    LOG(INFO) << "RECEIVED " << digits->size() << " DIGITS";
    auto labelvector = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
    if (mUseMC) {
      auto digitlabels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("hmpiddigitlabels");
      *labelvector.get() = std::move(*digitlabels);
      mClusterer.setMCTruthContainer(&mClsLabels);
      mClsLabels.clear();
    }
    // call actual clustering routine
    mClustersArray.clear();

    if (mUseMC) {
      mClusterer.process(*digits.get(), mClustersArray, labelvector.get());
    } else {
      mClusterer.process(*digits.get(), mClustersArray, nullptr);
    }

    LOG(INFO) << "HMPID CLUSTERER : TRANSFORMED " << digits->size()
              << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send clusters
    //pc.outputs().snapshot(Output{"HMP", "CLUSTERS", 0, Lifetime::Timeframe}, mClustersArray);
    // send labels
    if (mUseMC) {
      //pc.outputs().snapshot(Output{"HMP", "CLUSTERSMCTR", 0, Lifetime::Timeframe}, mClsLabels);
    }

    // declare done
    finished = true;
    pc.services().get<ControlService>().readyToQuit(false);
  }

 private:
  Clusterer mClusterer; ///< Cluster finder

  std::vector<Cluster> mClustersArray; ///< Array of clusters
  MCLabelContainer mClsLabels;
};

o2::framework::DataProcessorSpec getHMPIDClusterizerSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("hmpiddigits", "HMP", "DIGITS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("hmpiddigitlabels", "HMP", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "HMPClusterer",
    inputs,
    //    Outputs{OutputSpec{"HMP", "CLUSTERS", 0, Lifetime::Timeframe},
    //      OutputSpec{"HMP", "CLUSTERSMCTR", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<HMPIDDPLClustererTask>(useMC)},
    Options{/* for the moment no options */}};
}

} // end namespace hmpid
} // end namespace o2
