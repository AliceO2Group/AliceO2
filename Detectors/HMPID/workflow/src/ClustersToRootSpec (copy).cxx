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

/// \file   ClustersToRootSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 20 nov 2021
/// \brief Implementation of a data processor to produce Root File from Clusters Stream
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>
#include <algorithm>

#include "DPLUtils/DPLRawParser.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

#include "TTree.h"
#include "TFile.h"

#include <gsl/span>

#include "Framework/DataRefUtils.h"
#include "Framework/InputSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"
#include "Framework/InputRecordWalker.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"

#include "HMPIDBase/Geo.h"
#include "HMPIDWorkflow/ClustersToRootSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void ClustersToRootTask::init(framework::InitContext& ic)
{
  LOG(info) << "[HMPID Write Root File From Clusters stream - init()]";

  // get line command options
  mOutRootFileName = ic.options().get<std::string>("out-file");

  mTriggers.clear();
  mClusters.clear();

  TString filename;
  TString tit;

  filename = TString::Format("%s", mOutRootFileName.c_str());
  LOG(info) << "Create the ROOT file " << filename.Data();
  mfileOut.reset( new TFile(TString::Format("%s", filename.Data()), "RECREATE"));
  tit = TString::Format("HMPID Clusters File Decoding");
  mTheTree.reset(new TTree("o2hmp", tit));
  mTheTree->Branch("InteractionRecords", &mTriggers);
  mTheTree->Branch("HMPIDClusters", &mClusters);

  LOG(info) << "[HMPIDWrite L87 - init()]";

  mExTimer.start();
  return;
}

void ClustersToRootTask::run(framework::ProcessingContext& pc)
{
  std::vector<o2::hmpid::Trigger> triggers;
  std::vector<o2::hmpid::Cluster> clusters;
  LOG(info) << "[HMPIDWrite L97 - run()]"; 
  for (auto const& ref : InputRecordWalker(pc.inputs())) {
    

    
    if (DataRefUtils::match(ref, {"check", ConcreteDataTypeMatcher{header::gDataOriginHMP, "INTRECORDS1"}})) {
      /*LOG(info) << "[HMPIDWrite INTRECORDS1]";
      try{triggers = pc.inputs().get<std::vector<o2::hmpid::Trigger>>(ref);
         LOG(info) << "[HMPIDWrite INTRECORDS1 try]";} 
	catch(const std::exception& e) {LOG(info) << "Catched INTRECORDS1 exception";}
    */} else {
      LOG(info) << "[HMPID Write Root File Did not find INTRECORDS1";
    }/*

    if (DataRefUtils::match(ref, {"check", ConcreteDataTypeMatcher{header::gDataOriginHMP, "CLUSTERS"}})) {
      LOG(info) << "[HMPIDWrite CLUSTERS]";
      try{
	 LOG(info) << "[HMPIDWrite CLUSTERS tryIN]";
	 clusters = pc.inputs().get<std::vector<o2::hmpid::Cluster>>(ref);
         LOG(info) << "[HMPIDWrite CLUSTERS tryOUT]";} 
	catch(const std::exception& e) {LOG(info) << "Catched CLUSTERS exception";}
      //clusters = pc.inputs().get<std::vector<o2::hmpid::Cluster>>(ref);
      //    LOG(info) << "The size of the vector =" << clusters.size();
    } else {
      LOG(info) << "[HMPID Write Root File Did not find Clusterss";
    } */
    

    LOG(info) << "[HMPIDWrite entering TriggerLOOP";
    for (int i = 0; i < triggers.size(); i++) {
      //    LOG(info) << "Trigger Event     Orbit = " << triggers[i].getOrbit() << "  BC = " << triggers[i].getBc();
      int startClusterIndex = mClusters.size();
      int numberOfClusters = 0;
      LOG(info) << "[HMPIDWrite entering SECOND TriggerLOOP";
      for (int j = triggers[i].getFirstEntry(); j <= triggers[i].getLastEntry(); j++) {
        mClusters.push_back(clusters[j]); // append the cluster
        numberOfClusters++;
      }
      mTriggers.push_back(triggers[i]);
      mTriggers.back().setDataRange(startClusterIndex, numberOfClusters);
    } 
  } 
  LOG(info) << "[HMPIDWrite L120 - run()]";
  mExTimer.stop();
  return;
}

void ClustersToRootTask::endOfStream(framework::EndOfStreamContext& ec)
{
    LOG(info) << "[HMPIDWrite L120 - EOS()]";
  mExTimer.logMes("Received an End Of Stream !");
  //for (int i = 0; i < mClusters.size(); i++) {
  //  mClusters.at(i).print();
  //}
  mTheTree->Fill();
  mTheTree->Write();
  mfileOut->Close();
  mExTimer.logMes("Register Tree ! ");
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClustersToRootSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("clusters", o2::header::gDataOriginHMP, "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("intrecord", o2::header::gDataOriginHMP, "INTRECORDS1", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;

  return DataProcessorSpec{
    "HMP-ClustersToRoot",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ClustersToRootTask>()},
    Options{{"out-file", VariantType::String, "hmpClusters.root", {"name of the output file"}}}};
}

} // namespace hmpid
} // end namespace o2
