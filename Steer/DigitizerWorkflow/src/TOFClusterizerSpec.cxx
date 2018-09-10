// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFClusterizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "TOFReconstruction/Clusterer.h"
#include "TOFReconstruction/DataReader.h"
#include "DataFormatsTOF/Cluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

// use the tasking system of DPL
// just need to implement 2 special methods init + run (there is no need to inherit from anything)
class TOFDPLClustererTask
{
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

 public:
  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
  }

  void run(framework::ProcessingContext& pc)
  {
    // get digit data
    auto digits = pc.inputs().get<std::vector<o2::tof::Digit>*>("tofdigits");
    mReader.setDigitArray(digits.get());

    // call actual clustering routine
    mClustersArray.clear();
    mClusterer.process(mReader, mClustersArray, nullptr /*digit labels*/);

    LOG(INFO) << "TOF CLUSTERER : TRANSFORMED " << digits->size()
              << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send clusters
    pc.outputs().snapshot(Output{ "TOF", "CLUSTERS", 0, Lifetime::Timeframe }, mClustersArray);
    // declare done
    pc.services().get<ControlService>().readyToQuit(false);
  }

 private:
  DigitDataReader mReader; ///< Digit reader
  Clusterer mClusterer;    ///< Cluster finder

  std::vector<Cluster> mClustersArray; ///< Array of clusters
};

o2::framework::DataProcessorSpec getTOFClusterizerSpec()
{
  return DataProcessorSpec{
    "TOFClusterer",
    Inputs{ InputSpec{ "tofdigits", "TOF", "DIGITS", 0, Lifetime::Timeframe } },
    Outputs{ OutputSpec{ "TOF", "CLUSTERS", 0, Lifetime::Timeframe } },
    AlgorithmSpec{ adaptFromTask<TOFDPLClustererTask>() },
    Options{ /* for the moment no options */ }
  };
}

} // end namespace tof
} // end namespace o2
