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

/// \file ClusterReaderSpec.cxx
/// \brief The TF3 cluster reader (ClusterReaderSpec.h).

#include "IOTOFWorkflow/ClusterReaderSpec.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"

#include <format>
#include <stdexcept>

using namespace o2::framework;

namespace o2::iotof
{

void ClusterReader::init(InitContext& ic)
{
  connectTree(ic.options().get<std::string>("tf3-cluster-infile"));
}

void ClusterReader::run(ProcessingContext& pc)
{
  const auto entry = mTree->GetReadEntry() + 1;
  // A timeframe without a collision has no entry: publish empty streams, so
  // that the consumers still see the timeframe (as TRK's reader does).
  const bool noEntry = entry >= mTree->GetEntries();
  if (noEntry) {
    LOG(info) << "TF3ClusterReader: no entry to read, publishing empty output";
  } else {
    mTree->GetEntry(entry);
  }
  static const std::vector<o2::iotof::Cluster> noClusters;
  static const std::vector<unsigned char> noPatterns;
  static const std::vector<o2::itsmft::ROFRecord> noROFs;
  static const o2::dataformats::MCTruthContainer<o2::MCCompLabel> noLabels;
  const auto& clusters = noEntry ? noClusters : *mClusters;
  LOG(info) << "TF3ClusterReader pushes " << clusters.size() << " clusters at entry " << entry;
  pc.outputs().snapshot(Output{o2::header::gDataOriginTF3, "CLUSTERS", 0}, clusters);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTF3, "PATTERNS", 0}, noEntry ? noPatterns : *mPatterns);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTF3, "CLUSTERSROF", 0}, noEntry ? noROFs : *mROFs);
  if (mUseMC) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTF3, "CLUSTERSMCTR", 0}, noEntry ? noLabels : *mLabels);
  }
  if (noEntry || mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void ClusterReader::connectTree(const std::string& filename)
{
  mTree.reset();
  mFile.reset(TFile::Open(filename.c_str()));
  if (!mFile || mFile->IsZombie()) {
    throw std::runtime_error("TF3ClusterReader: cannot open " + filename);
  }
  mTree.reset(mFile->Get<TTree>("o2sim"));
  if (!mTree) {
    throw std::runtime_error("TF3ClusterReader: no tree o2sim in " + filename);
  }
  const auto connect = [this](const char* name, auto*& address) {
    if (!mTree->GetBranch(name)) {
      throw std::runtime_error(std::format("TF3ClusterReader: no branch {}", name));
    }
    if (const Int_t status = mTree->SetBranchAddress(name, &address); status < 0) {
      throw std::runtime_error(std::format("TF3ClusterReader: cannot set the address of {} ({})", name, status));
    }
  };
  connect("TF3Cluster", mClusters);
  connect("TF3ClusterPatt", mPatterns);
  connect("TF3ClusterROF", mROFs);
  if (mUseMC) {
    connect("TF3ClusterMCTruth", mLabels);
  }
  LOG(info) << "TF3ClusterReader: " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getIOTOFClusterReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs{{o2::header::gDataOriginTF3, "CLUSTERS", 0, Lifetime::Timeframe},
                                  {o2::header::gDataOriginTF3, "PATTERNS", 0, Lifetime::Timeframe},
                                  {o2::header::gDataOriginTF3, "CLUSTERSROF", 0, Lifetime::Timeframe}};
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTF3, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    .name = "tf3-cluster-reader",
    .inputs = Inputs{},
    .outputs = outputs,
    .algorithm = AlgorithmSpec{adaptFromTask<ClusterReader>(useMC)},
    .options = Options{{"tf3-cluster-infile", VariantType::String, "tf3clusters.root", {"Name of the input cluster file"}}},
  };
}

} // namespace o2::iotof
