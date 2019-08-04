// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClustererSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "ITSWorkflow/ClustererSpec.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

void ClustererDPL::init(InitContext& ic)
{
  o2::base::GeometryManager::loadGeometry(); // for generating full clusters
  o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L));

  mClusterer = std::make_unique<o2::itsmft::Clusterer>();
  mClusterer->setGeometry(geom);
  mClusterer->setNChips(o2::itsmft::ChipMappingITS::getNChips());

  auto filenameGRP = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filenameGRP.c_str());

  if (grp) {
    mClusterer->setContinuousReadOut(grp->isDetContinuousReadOut("ITS"));
  } else {
    LOG(ERROR) << "Cannot retrieve GRP from the " << filenameGRP.c_str() << " file !";
    mState = 0;
    return;
  }

  // settings for the fired pixel overflow masking
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  mClusterer->setMaxBCSeparationToMask(alpParams.roFrameLength / o2::constants::lhc::LHCBunchSpacingNS + 10);

  auto filename = ic.options().get<std::string>("its-dictionary-file");
  mFile = std::make_unique<std::ifstream>(filename.c_str(), std::ios::in | std::ios::binary);
  if (mFile->good()) {
    mClusterer->loadDictionary(filename);
    LOG(INFO) << "ITSClusterer running with a provided dictionary: " << filename.c_str();
    mState = 1;
  } else {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    mState = 0;
  }

  mClusterer->print();
}

void ClustererDPL::run(ProcessingContext& pc)
{
  if (mState > 1)
    return;

  auto digits = pc.inputs().get<const std::vector<o2::itsmft::Digit>>("digits");
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  std::vector<o2::itsmft::MC2ROFRecord> mc2rofs;
  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    mc2rofs = pc.inputs().get<const std::vector<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
  }

  LOG(INFO) << "ITSClusterer pulled " << digits.size() << " digits, in "
            << rofs.size() << " RO frames";

  o2::itsmft::DigitPixelReader reader;
  reader.setDigits(&digits);
  reader.setROFRecords(&rofs);
  if (mUseMC) {
    reader.setMC2ROFRecords(&mc2rofs);
    reader.setDigitsMCTruth(labels.get());
  }
  reader.init();

  std::vector<o2::itsmft::CompClusterExt> compClusters;
  std::vector<o2::itsmft::Cluster> clusters;
  std::vector<o2::itsmft::ROFRecord> clusterROframes; // To be filled in future

  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
  if (mUseMC) {
    clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  }
  mClusterer->process(reader, &clusters, &compClusters, clusterLabels.get(), &clusterROframes);
  // TODO: in principle, after masking "overflow" pixels the MC2ROFRecord maxROF supposed to change, nominally to minROF
  // -> consider recalculationg maxROF

  LOG(INFO) << "ITSClusterer pushed " << clusters.size() << " clusters, in "
            << clusterROframes.size() << " RO frames";

  pc.outputs().snapshot(Output{"ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe}, compClusters);
  pc.outputs().snapshot(Output{"ITS", "CLUSTERS", 0, Lifetime::Timeframe}, clusters);
  pc.outputs().snapshot(Output{"ITS", "ITSClusterROF", 0, Lifetime::Timeframe}, clusterROframes);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe}, *clusterLabels.get());
    std::vector<o2::itsmft::MC2ROFRecord>& clusterMC2ROframes = mc2rofs; // Simply, replicate it from digits ?
    pc.outputs().snapshot(Output{"ITS", "ITSClusterMC2ROF", 0, Lifetime::Timeframe}, clusterMC2ROframes);
  }

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getClustererSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "ITS", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "ITSDigitROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "CLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "ITSClusterROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "ITS", "DIGITSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "ITSDigitMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "ITSClusterMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-clusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ClustererDPL>(useMC)},
    Options{
      {"its-dictionary-file", VariantType::String, "complete_dictionary.bin", {"Name of the cluster-topology dictionary file"}},
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}}}};
}

} // namespace its
} // namespace o2
