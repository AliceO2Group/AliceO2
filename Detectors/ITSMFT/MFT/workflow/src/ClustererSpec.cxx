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
#include "MFTWorkflow/ClustererSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "MFTBase/GeometryTGeo.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "ITSMFTReconstruction/ClustererParam.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{

void ClustererDPL::init(InitContext& ic)
{

  mClusterer = std::make_unique<o2::itsmft::Clusterer>();
  mClusterer->setNChips(o2::itsmft::ChipMappingMFT::getNChips());
  LOG(INFO) << "MFT ClustererDPL::init total number of sensors " << o2::itsmft::ChipMappingMFT::getNChips() << "\n";

  //mClusterer->setMaskOverflowPixels(false);

  auto filenameGRP = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filenameGRP.c_str());

  if (grp) {
    mClusterer->setContinuousReadOut(grp->isDetContinuousReadOut("MFT"));
  } else {
    LOG(ERROR) << "Cannot retrieve GRP from the " << filenameGRP.c_str() << " file !";
    mState = 0;
    return;
  }

  mPatterns = !ic.options().get<bool>("no-patterns");
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));

  // settings for the fired pixel overflow masking
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
  const auto& clParams = o2::itsmft::ClustererParam<o2::detectors::DetID::MFT>::Instance();
  auto nbc = clParams.maxBCDiffToMaskBias;
  nbc += mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS);
  mClusterer->setMaxBCSeparationToMask(nbc);
  mClusterer->setMaxRowColDiffToMask(clParams.maxRowColDiffToMask);

  std::string dictPath = ic.options().get<std::string>("mft-dictionary-path");
  std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::MFT, dictPath, ".bin");
  if (o2::utils::Str::pathExists(dictFile)) {
    mClusterer->loadDictionary(dictFile);
    LOG(INFO) << "MFTClusterer running with a provided dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, MFTClusterer expects cluster patterns";
  }
  mState = 1;
  mClusterer->print();
}

void ClustererDPL::run(ProcessingContext& pc)
{
  auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

  gsl::span<const o2::itsmft::MC2ROFRecord> mc2rofs;
  gsl::span<const char> labelbuffer;
  if (mUseMC) {
    labelbuffer = pc.inputs().get<gsl::span<char>>("labels");
    mc2rofs = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
  }
  const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);

  LOG(INFO) << "MFTClusterer pulled " << digits.size() << " digits, in "
            << rofs.size() << " RO frames";

  o2::itsmft::DigitPixelReader reader;
  reader.setDigits(digits);
  reader.setROFRecords(rofs);
  if (mUseMC) {
    reader.setMC2ROFRecords(mc2rofs);
    reader.setDigitsMCTruth(labels.getIndexedSize() > 0 ? &labels : nullptr);
  }
  reader.init();
  auto orig = o2::header::gDataOriginMFT;
  std::vector<o2::itsmft::CompClusterExt> clusCompVec;
  std::vector<o2::itsmft::ROFRecord> clusROFVec;
  std::vector<unsigned char> clusPattVec;

  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
  if (mUseMC) {
    clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  }
  mClusterer->process(mNThreads, reader, &clusCompVec, mPatterns ? &clusPattVec : nullptr, &clusROFVec, clusterLabels.get());
  pc.outputs().snapshot(Output{orig, "COMPCLUSTERS", 0, Lifetime::Timeframe}, clusCompVec);
  pc.outputs().snapshot(Output{orig, "CLUSTERSROF", 0, Lifetime::Timeframe}, clusROFVec);
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
  LOG(INFO) << "MFTClusterer pushed " << clusCompVec.size() << " compressed clusters, in " << clusROFVec.size() << " RO frames";
}

DataProcessorSpec getClustererSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "MFT", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "DIGITSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "PATTERNS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "MFT", "DIGITSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "MFT", "DIGITSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-clusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ClustererDPL>(useMC)},
    Options{
      {"mft-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}},
      {"no-patterns", o2::framework::VariantType::Bool, false, {"Do not save rare cluster patterns"}},
      {"nthreads", VariantType::Int, 1, {"Number of clustering threads"}}}};
}

} // namespace mft
} // namespace o2
