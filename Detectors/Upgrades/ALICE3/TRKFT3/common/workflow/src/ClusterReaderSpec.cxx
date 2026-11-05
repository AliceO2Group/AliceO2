// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterReaderSpec.cxx

#include "TRKWorkflow/ClusterReaderSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <stdexcept>
#include <string>

using namespace o2::framework;

namespace o2::trk
{

namespace
{
template <int DetID>
std::string detName()
{
  return o2::detectors::DetID(DetID).getName();
}

template <int DetID>
std::string lowerName()
{
  auto name = detName<DetID>();
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
  return name;
}

template <int DetID>
std::string branchName(const std::string& base, int layer)
{
  return std::format("{}{}_{}", detName<DetID>(), base, layer);
}
} // namespace

template <int DetID>
ClusterReader<DetID>::ClusterReader(bool useMC)
  : mClusters(NLayers, nullptr), mPatterns(NLayers, nullptr), mROFs(NLayers, nullptr), mLabels(NLayers, nullptr), mUseMC(useMC)
{
}

template <int DetID>
void ClusterReader<DetID>::init(InitContext& ic)
{
  connectTree(ic.options().get<std::string>((lowerName<DetID>() + "-cluster-infile").c_str()));
}

template <int DetID>
void ClusterReader<DetID>::run(ProcessingContext& pc)
{
  constexpr auto origin = DetID == o2::detectors::DetID::TRK ? o2::header::gDataOriginTRK : o2::header::gDataOriginFT3;
  const auto entry = mTree->GetReadEntry() + 1;
  // A timeframe without a collision has no entry: publish empty streams, so
  // that the consumers still see the timeframe (as the digit reader does).
  const bool noEntry = entry >= mTree->GetEntries();
  if (noEntry) {
    LOG(info) << detName<DetID>() << "ClusterReader: no entry to read, publishing empty output";
  } else {
    mTree->GetEntry(entry);
  }
  static const std::vector<o2::trkft3::Cluster<DetID>> noClusters;
  static const std::vector<unsigned char> noPatterns;
  static const std::vector<o2::trkft3::ROFRecord> noROFs;
  static const o2::dataformats::MCTruthContainer<o2::MCCompLabel> noLabels;
  for (int layer = 0; layer < NLayers; ++layer) {
    const auto subspec = static_cast<DataAllocator::SubSpecificationType>(layer);
    const auto& clusters = noEntry ? noClusters : *mClusters[layer];
    const auto& rofs = noEntry ? noROFs : *mROFs[layer];
    LOG(info) << detName<DetID>() << "ClusterReader on layer " << layer << " pushes " << clusters.size() << " clusters in " << rofs.size()
              << " RO frames at entry " << entry;
    pc.outputs().snapshot(Output{origin, "COMPCLUSTERS", subspec}, clusters);
    pc.outputs().snapshot(Output{origin, "PATTERNS", subspec}, noEntry ? noPatterns : *mPatterns[layer]);
    pc.outputs().snapshot(Output{origin, "CLUSTERSROF", subspec}, rofs);
    if (mUseMC) {
      pc.outputs().snapshot(Output{origin, "CLUSTERSMCTR", subspec}, noEntry ? noLabels : *mLabels[layer]);
    }
  }
  if (noEntry || mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

template <int DetID>
void ClusterReader<DetID>::connectTree(const std::string& filename)
{
  mTree.reset();
  mFile.reset(TFile::Open(filename.c_str()));
  if (!mFile || mFile->IsZombie()) {
    throw std::runtime_error(detName<DetID>() + "ClusterReader: cannot open " + filename);
  }
  mTree.reset(mFile->Get<TTree>("o2sim"));
  if (!mTree) {
    throw std::runtime_error(detName<DetID>() + "ClusterReader: no tree o2sim in " + filename);
  }
  for (int layer = 0; layer < NLayers; ++layer) {
    setBranchAddress("ClusterComp", layer, mClusters[layer]);
    setBranchAddress("ClusterPatt", layer, mPatterns[layer]);
    setBranchAddress("ClustersROF", layer, mROFs[layer]);
    if (mUseMC) {
      setBranchAddress("ClusterMCTruth", layer, mLabels[layer]);
    }
  }
  LOG(info) << detName<DetID>() << "ClusterReader: " << filename << " with " << mTree->GetEntries() << " entries";
}

template <int DetID>
template <typename Ptr>
void ClusterReader<DetID>::setBranchAddress(const std::string& base, int layer, Ptr& address)
{
  const auto name = branchName<DetID>(base, layer);
  if (!mTree->GetBranch(name.c_str())) {
    throw std::runtime_error(detName<DetID>() + "ClusterReader: no branch " + name + (base == "ClusterMCTruth" ? " (MC labels requested)" : ""));
  }
  if (const Int_t status = mTree->SetBranchAddress(name.c_str(), &address); status != 0) {
    throw std::runtime_error(std::format("{}ClusterReader: cannot set the address of {} ({})", detName<DetID>(), name, status));
  }
}

namespace
{
template <int DetID>
DataProcessorSpec getClusterReaderSpecT(bool useMC, std::string defname)
{
  constexpr auto origin = DetID == o2::detectors::DetID::TRK ? o2::header::gDataOriginTRK : o2::header::gDataOriginFT3;
  std::vector<OutputSpec> outputs;
  for (int layer = 0; layer < ClusterReader<DetID>::NLayers; ++layer) {
    outputs.emplace_back(origin, "COMPCLUSTERS", layer, Lifetime::Timeframe);
    outputs.emplace_back(origin, "PATTERNS", layer, Lifetime::Timeframe);
    outputs.emplace_back(origin, "CLUSTERSROF", layer, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(origin, "CLUSTERSMCTR", layer, Lifetime::Timeframe);
    }
  }
  return DataProcessorSpec{
    lowerName<DetID>() + "-cluster-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<ClusterReader<DetID>>(useMC)},
    Options{{lowerName<DetID>() + "-cluster-infile", VariantType::String, std::move(defname), {"Name of the input cluster file"}}}};
}
} // namespace

DataProcessorSpec getTRKClusterReaderSpec(bool useMC, std::string defname)
{
  return getClusterReaderSpecT<o2::detectors::DetID::TRK>(useMC, std::move(defname));
}

DataProcessorSpec getFT3ClusterReaderSpec(bool useMC, std::string defname)
{
  return getClusterReaderSpecT<o2::detectors::DetID::FT3>(useMC, std::move(defname));
}

template class ClusterReader<o2::detectors::DetID::TRK>;
template class ClusterReader<o2::detectors::DetID::FT3>;

} // namespace o2::trk
