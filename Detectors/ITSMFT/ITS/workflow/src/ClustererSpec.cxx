// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClustererSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "ITSWorkflow/ClustererSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace its
{

void ClustererDPL::init(InitContext& ic)
{
  mClusterer = std::make_unique<o2::itsmft::Clusterer>();
  mClusterer->setNChips(o2::itsmft::ChipMappingITS::getNChips());
  mUseClusterDictionary = !ic.options().get<bool>("ignore-cluster-dictionary");
  auto filenameGRP = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filenameGRP.c_str());

  if (grp) {
    mClusterer->setContinuousReadOut(grp->isDetContinuousReadOut("ITS"));
  } else {
    LOG(error) << "Cannot retrieve GRP from the " << filenameGRP.c_str() << " file !";
    mState = 0;
    return;
  }

  mPatterns = !ic.options().get<bool>("no-patterns");
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));

  // settings for the fired pixel overflow masking
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  const auto& clParams = o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance();
  auto nbc = clParams.maxBCDiffToMaskBias;
  nbc += mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS);
  mClusterer->setMaxBCSeparationToMask(nbc);
  mClusterer->setMaxRowColDiffToMask(clParams.maxRowColDiffToMask);

  mState = 1;
  mClusterer->print();
}

void ClustererDPL::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

  gsl::span<const o2::itsmft::MC2ROFRecord> mc2rofs;
  gsl::span<const char> labelbuffer;
  if (mUseMC) {
    labelbuffer = pc.inputs().get<gsl::span<char>>("labels");
    mc2rofs = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
  }
  o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);

  LOG(info) << "ITSClusterer pulled " << digits.size() << " digits, in "
            << rofs.size() << " RO frames";
  LOG(info) << "ITSClusterer pulled " << labels.getNElements() << " labels ";

  o2::itsmft::DigitPixelReader reader;
  reader.setDigits(digits);
  reader.setROFRecords(rofs);
  if (mUseMC) {
    reader.setMC2ROFRecords(mc2rofs);
    reader.setDigitsMCTruth(labels.getIndexedSize() > 0 ? &labels : nullptr);
  }
  reader.init();
  auto orig = o2::header::gDataOriginITS;
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
  LOG(info) << "ITSClusterer pushed " << clusCompVec.size() << " clusters, in " << clusROFVec.size() << " RO frames";
}

///_______________________________________
void ClustererDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  pc.inputs().get<TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
}

///_______________________________________
void ClustererDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated" << (!mUseClusterDictionary ? " but its using is disabled" : "");
    if (mUseClusterDictionary) {
      mClusterer->setDictionary((const o2::itsmft::TopologyDictionary*)obj);
    }
  }
}

DataProcessorSpec getClustererSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "ITS", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "DIGITSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("cldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "PATTERNS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "ITS", "DIGITSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "DIGITSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-clusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ClustererDPL>(useMC)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}},
      {"no-patterns", o2::framework::VariantType::Bool, false, {"Do not save rare cluster patterns"}},
      {"ignore-cluster-dictionary", VariantType::Bool, false, {"do not use cluster dictionary, always store explicit patterns"}},
      {"nthreads", VariantType::Int, 1, {"Number of clustering threads"}}}};
}

} // namespace its
} // namespace o2
