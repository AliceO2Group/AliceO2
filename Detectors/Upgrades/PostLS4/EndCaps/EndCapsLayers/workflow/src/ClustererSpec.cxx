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
#include "EC0Workflow/ClustererSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "EndCapsReconstruction/ChipMappingITS.h"
#include "EndCapsReconstruction/ClustererParam.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "EndCapsReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "ECLayersBase/GeometryTGeo.h"
#include "EndCapsBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace ecl
{

void ClustererDPL::init(InitContext& ic)
{
  o2::base::GeometryManager::loadGeometry(); // for generating full clusters
  o2::ecl::GeometryTGeo* geom = o2::ecl::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L));

  mClusterer = std::make_unique<o2::endcaps::Clusterer>();
  mClusterer->setGeometry(geom);
  mClusterer->setNChips(o2::endcaps::ChipMappingITS::getNChips());

  auto filenameGRP = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filenameGRP.c_str());

  if (grp) {
    mClusterer->setContinuousReadOut(grp->isDetContinuousReadOut("EC0"));
  } else {
    LOG(ERROR) << "Cannot retrieve GRP from the " << filenameGRP.c_str() << " file !";
    mState = 0;
    return;
  }

  mFullClusters = ic.options().get<bool>("full-clusters");
  mPatterns = !ic.options().get<bool>("no-patterns");
  mNThreads = ic.options().get<int>("nthreads");

  // settings for the fired pixel overflow masking
  const auto& alpParams = o2::endcaps::DPLAlpideParam<o2::detectors::DetID::EC0>::Instance();
  const auto& clParams = o2::endcaps::ClustererParam<o2::detectors::DetID::EC0>::Instance();
  auto nbc = clParams.maxBCDiffToMaskBias;
  nbc += mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS);
  mClusterer->setMaxBCSeparationToMask(nbc);
  mClusterer->setMaxRowColDiffToMask(clParams.maxRowColDiffToMask);

  std::string dictPath = ic.options().get<std::string>("ecl-dictionary-path");
  std::string dictFile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::EC0, dictPath, ".bin");
  if (o2::base::NameConf::pathExists(dictFile)) {
    mClusterer->loadDictionary(dictFile);
    LOG(INFO) << "EC0Clusterer running with a provided dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, EC0Clusterer expects cluster patterns";
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

  LOG(INFO) << "EC0Clusterer pulled " << digits.size() << " digits, in "
            << rofs.size() << " RO frames";

  o2::endcaps::DigitPixelReader reader;
  reader.setDigits(digits);
  reader.setROFRecords(rofs);
  if (mUseMC) {
    reader.setMC2ROFRecords(mc2rofs);
    reader.setDigitsMCTruth(labels.get());
  }
  reader.init();
  auto orig = o2::header::gDataOriginEC0;
  std::vector<o2::itsmft::Cluster> clusVec;
  std::vector<o2::itsmft::CompClusterExt> clusCompVec;
  std::vector<o2::itsmft::ROFRecord> clusROFVec;
  std::vector<unsigned char> clusPattVec;

  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
  if (mUseMC) {
    clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  }
  mClusterer->process(mNThreads, reader, mFullClusters ? &clusVec : nullptr, &clusCompVec, mPatterns ? &clusPattVec : nullptr, &clusROFVec, clusterLabels.get());
  pc.outputs().snapshot(Output{orig, "COMPCLUSTERS", 0, Lifetime::Timeframe}, clusCompVec);
  pc.outputs().snapshot(Output{orig, "CLUSTERSROF", 0, Lifetime::Timeframe}, clusROFVec);
  pc.outputs().snapshot(Output{orig, "CLUSTERS", 0, Lifetime::Timeframe}, clusVec);
  pc.outputs().snapshot(Output{orig, "PATTERNS", 0, Lifetime::Timeframe}, clusPattVec);

  if (mUseMC) {
    pc.outputs().snapshot(Output{orig, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, *clusterLabels.get()); // at the moment requires snapshot
    std::vector<o2::itsmft::MC2ROFRecord> clusterMC2ROframes(mc2rofs.size());
    for (int i = mc2rofs.size(); i--;) {
      clusterMC2ROframes[i] = mc2rofs[i]; // Simply, replicate it from digits ?
    }
    pc.outputs().snapshot(Output{orig, "CLUSTERSMC2ROF", 0, Lifetime::Timeframe}, clusterMC2ROframes);
  }

  // TODO: in principle, after masking "overflow" pixels the MC2ROFRecord maxROF supposed to change, nominally to minROF
  // -> consider recalculationg maxROF
  LOG(INFO) << "EC0Clusterer pushed " << clusCompVec.size() << " clusters, in " << clusROFVec.size() << " RO frames";
}

DataProcessorSpec getClustererSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "EC0", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "EC0", "DIGITSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("EC0", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("EC0", "PATTERNS", 0, Lifetime::Timeframe);
  outputs.emplace_back("EC0", "CLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("EC0", "CLUSTERSROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "EC0", "DIGITSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "EC0", "DIGITSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("EC0", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("EC0", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "ecl-clusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ClustererDPL>(useMC)},
    Options{
      {"ecl-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}},
      {"full-clusters", o2::framework::VariantType::Bool, false, {"Produce full clusters"}},
      {"no-patterns", o2::framework::VariantType::Bool, false, {"Do not save rare cluster patterns"}},
      {"nthreads", VariantType::Int, 0, {"Number of clustering threads (<1: rely on openMP default)"}}}};
}

} // namespace ecl
} // namespace o2
