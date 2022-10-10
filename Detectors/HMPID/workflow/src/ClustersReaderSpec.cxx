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

/// \file   ClusterReaderSpec.cxx
/// \author Annalisa Mastroserio - INFN Bari
/// \version 1.0
/// \date 22 Jun 2022
/// \brief Implementation of a data processor to read Cluster tree and provide the array for further usage
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>

#include "CommonUtils/StringUtils.h" // o2::utils::Str

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "DataFormatsHMP/Trigger.h"
#include "DataFormatsHMP/Cluster.h"
#include "HMPIDBase/Geo.h"


#include "HMPIDWorkflow/ClustersReaderSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
///using namespace o2::header;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//
void ClusterReaderTask::init(framework::InitContext& ic)
{
  LOG(info) << "[HMPID Cluster reader - init() ] ";
     // open input file

  // specify location and filename for output in case of writing to file
  if (mReadFile) {
    // Build the file name
    const auto filename = o2::utils::Str::concat_string(
      o2::utils::Str::rectifyDirectory(
        ic.options().get<std::string>("input-dir")),
      ic.options().get<std::string>("hmpid-digit-infile"));
    initFileIn(filename);
  
    int mTriggersFromFile, mDigitsReceived = 0;
  }

}
    
    // return;


void ClusterReaderTask::run(framework::ProcessingContext& pc)
{

  
  // outputs

  /*
  std::vector<o2::hmpid::Cluster> clusters;
  std::vector<o2::hmpid::Trigger> clusterTriggers;
  LOG(info) << "[HMPID DClusterization - run() ] Enter ...";
  clusters.clear();
  clusterTriggers.clear();
  */


  //===============mReadFromFile=============================================
  if (mReadFile) {
    LOG(info) << "[HMPID ClusterReader - run() ] Entries  = " << mTree->GetEntries();
    // check if more entries in tree

    if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      mExTimer.stop();
      //mExTimer.logMes("End ClusterReader !  clusters = " +
      //                std::to_string(mClustersReceived));
    }

    // there are more entries in file
    else {
      // =============== read digits and digit-triggers =====================
      // iterate through TTree

      auto entry = mTree->GetReadEntry() + 1;
      assert(entry < mTree->GetEntries());
      mTree->GetEntry(0);

      pc.outputs().snapshot(Output{ "HMP", "CLUSTERS", 0, Lifetime::Timeframe }, mClustersFromFile);
      pc.outputs().snapshot(Output{ "HMP", "INTRECORDS", 0, Lifetime::Timeframe },mClusterTriggersFromFile);
      // =============== create clusters =====================
      /* for (const auto& trig : *mTriggersFromFile) {
        if (trig.getNumberOfObjects()) {
        }
      } 
      LOGP(info, "Received {} triggers with {} digits -> {} triggers with {} clusters",
           mTriggersFromFile->size(), mDigitsFromFile->size(), clusterTriggers.size(),
           clusters.size());
      mDigitsReceived += mDigitsFromFile->size();*/
    } // <end else of num entries>
  }   //===============  <end mReadFromFile>


  else { // read from stream
    const auto triggers = pc.inputs().get<gsl::span<o2::hmpid::Trigger>>("intrecord");
    const auto clusters = pc.inputs().get<gsl::span<o2::hmpid::Trigger>>("CLUSTERS");

    // Output vectors
    pc.outputs().snapshot(Output{ "HMP", "CLUSTERS", 0, Lifetime::Timeframe }, clusters);
    pc.outputs().snapshot(Output{ "HMP", "INTRECORDS", 0, Lifetime::Timeframe }, triggers);
  }


    
  mExTimer.elapseMes("# received Clusters = " + std::to_string(mCurrentEntry));
  return;
}

void ClusterReaderTask::endOfStream(framework::EndOfStreamContext& ec)
{
  mExTimer.stop();
  mExTimer.logMes("End Cluster reader = " + std::to_string(mNumberOfEntries));
  return;
}

void ClusterReaderTask::initFileIn(const std::string& filename)
{
  // Create the TFIle
  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("o2sim"));

  if (!mTree) {
    LOG(error)
      << "HMPID ClusterReaderTask::init() : Did not find o2sim tree in "
      << filename.c_str();
    throw std::runtime_error(
      "HMPID ClusterReaderTask::init() : Did not find "
      "o2sim file in digits tree");
  }

  mTree->SetBranchAddress("CLUSTERS", &mClustersFromFile);
  mTree->SetBranchAddress("INTRECORDS1", &mClusterTriggersFromFile);
  mTree->Print("toponly");
}

//_________________________________________________________________________________________________


o2::framework::DataProcessorSpec getClusterReaderSpec(/*std::string inputSpec,*/ bool readFile)
{
  std::vector<o2::framework::InputSpec> inputs;

  inputs.emplace_back("clusters", o2::header::gDataOriginHMP, "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("intrecord", o2::header::gDataOriginHMP, "INTRECORDS1", 0, Lifetime::Timeframe);

    
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("HMP", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("HMP", "INTRECORDS1", 0, o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "HMP-ClusterReader",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ClusterReaderTask>(readFile)}, 
   // Options{{"sigma-cut", VariantType::String, "", {"sigmas as comma separated list"}}}};
    Options{ { "qc-hmpid-clusters", VariantType::String, "hmpidclusters.root", { "Name of the input file with clusters" } } } };
}

} // namespace hmpid
} // end namespace o2
