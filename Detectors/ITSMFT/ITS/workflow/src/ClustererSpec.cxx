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
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/ClustererSpec.h"
#include "DataFormatsITSMFT/Digit.h"
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
#include "DetectorsCommonDataFormats/NameConf.h"

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

  mFullClusters = ic.options().get<bool>("full-clusters");
  mPatterns = !ic.options().get<bool>("no-patterns");

  // settings for the fired pixel overflow masking
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  mClusterer->setMaxBCSeparationToMask(alpParams.roFrameLength / o2::constants::lhc::LHCBunchSpacingNS + 10);

  std::string dictPath = ic.options().get<std::string>("its-dictionary-path");
  std::string dictFile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, dictPath, ".bin");
  if (o2::base::NameConf::pathExists(dictFile)) {
    mClusterer->loadDictionary(dictFile);
    LOG(INFO) << "ITSClusterer running with a provided dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, ITSClusterer expects cluster patterns";
  }
  mState = 1;
  mClusterer->print();
}

void ClustererDPL::run(ProcessingContext& pc)
{
  auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  gsl::span<const o2::itsmft::MC2ROFRecord> mc2rofs;
  if (mUseMC) {
    labels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    mc2rofs = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
  }

  LOG(INFO) << "ITSClusterer pulled " << digits.size() << " digits, in "
            << rofs.size() << " RO frames";

  o2::itsmft::DigitPixelReader reader;
  reader.setDigits(digits);
  reader.setROFRecords(rofs);
  if (mUseMC) {
    reader.setMC2ROFRecords(mc2rofs);
    reader.setDigitsMCTruth(labels.get());
  }
  reader.init();
  auto orig = o2::header::gDataOriginITS;
  std::vector<o2::itsmft::Cluster, boost::container::pmr::polymorphic_allocator<o2::itsmft::Cluster>>* clusVec = nullptr;
  std::vector<o2::itsmft::CompClusterExt, boost::container::pmr::polymorphic_allocator<o2::itsmft::CompClusterExt>>* clusCompVec = nullptr;
  std::vector<o2::itsmft::ROFRecord, boost::container::pmr::polymorphic_allocator<o2::itsmft::ROFRecord>>* clusROFVec = nullptr;
  std::vector<unsigned char, boost::container::pmr::polymorphic_allocator<unsigned char>>* clusPattVec = nullptr;

  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
  if (mUseMC) {
    clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  }

  clusCompVec = &pc.outputs().make<std::vector<o2::itsmft::CompClusterExt>>(Output{orig, "COMPCLUSTERS", 0, Lifetime::Timeframe});
  clusROFVec = &pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{orig, "ITSClusterROF", 0, Lifetime::Timeframe});
  //  if (mFullClusters) // in principle, we don't need to send empty array, but other devices expecting it do not know about options of this device: problem?
  clusVec = &pc.outputs().make<std::vector<o2::itsmft::Cluster>>(Output{orig, "CLUSTERS", 0, Lifetime::Timeframe});
  //  if (mPatterns) // idem
  clusPattVec = &pc.outputs().make<std::vector<unsigned char>>(Output{orig, "PATTERNS", 0, Lifetime::Timeframe});

  mClusterer->process(reader, mFullClusters ? clusVec : nullptr, clusCompVec, mPatterns ? clusPattVec : nullptr, clusROFVec, clusterLabels.get());

  if (mUseMC) {
    pc.outputs().snapshot(Output{orig, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, *clusterLabels.get()); // at the moment requires snapshot
    std::vector<o2::itsmft::MC2ROFRecord> clusterMC2ROframes(mc2rofs.size());
    for (int i = mc2rofs.size(); i--;) {
      clusterMC2ROframes[i] = mc2rofs[i]; // Simply, replicate it from digits ?
    }
    pc.outputs().snapshot(Output{orig, "ITSClusterMC2ROF", 0, Lifetime::Timeframe}, clusterMC2ROframes);
  }

  // TODO: in principle, after masking "overflow" pixels the MC2ROFRecord maxROF supposed to change, nominally to minROF
  // -> consider recalculationg maxROF
  LOG(INFO) << "ITSClusterer pushed " << clusCompVec->size() << " clusters, in " << clusROFVec->size() << " RO frames";
}

DataProcessorSpec getClustererSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "ITS", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "ITSDigitROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "PATTERNS", 0, Lifetime::Timeframe);
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
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}},
      {"full-clusters", o2::framework::VariantType::Bool, false, {"Produce full clusters"}},
      {"no-patterns", o2::framework::VariantType::Bool, false, {"Do not save rare cluster patterns"}}}};
}

} // namespace its
} // namespace o2
