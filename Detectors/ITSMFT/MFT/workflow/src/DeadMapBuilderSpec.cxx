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

/// @file   DeadMapBuilderSpec.cxx
#include <algorithm>
#include <numeric>
#include "MFTWorkflow/DeadMapBuilderSpec.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/CompCluster.h"

namespace o2
{
namespace mft
{

//////////////////////////////////////////////////////////////////////////////
// Default constructor
MFTDeadMapBuilder::MFTDeadMapBuilder(std::string datasource)
  : mDataSource(datasource)
{
  mSelfName = o2::utils::Str::concat_string(ChipMappingMFT::getName(), "MFTDeadMapBuilder");
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
MFTDeadMapBuilder::~MFTDeadMapBuilder()
{
  // Clear dynamic memory
  delete mDeadMapTF;
  delete mTreeObject;
}

//////////////////////////////////////////////////////////////////////////////
void MFTDeadMapBuilder::init(InitContext& ic)
{

  LOG(info) << "MFTDeadMapBuilder init... " << mSelfName;

  mDeadMapTF = new std::vector<short int>{};

  mTreeObject = new TTree("map", "map");
  mTreeObject->Branch("orbit", &mFirstOrbitTF);
  mTreeObject->Branch("deadmap", &mDeadMapTF);

  mTFSampling = ic.options().get<int>("tf-sampling");
  DebugMode = ic.options().get<bool>("debug");
  mTFLength = ic.options().get<int>("tf-length");
  mDoLocalOutput = ic.options().get<bool>("local-output");
  mObjectName = ic.options().get<std::string>("output-filename");
  mLocalOutputDir = ic.options().get<std::string>("output-dir");

  LOG(info) << "Sampling one TF every " << mTFSampling;

  return;
}

//////////////////////////////////////////////////////////////////////////////

void MFTDeadMapBuilder::finalizeOutput()
{

  if (mDoLocalOutput) {
    std::string localoutfilename = mLocalOutputDir + "/" + mObjectName;
    TFile outfile(localoutfilename.c_str(), "RECREATE");
    outfile.cd();
    mTreeObject->Write();
    outfile.Close();
  }
  return;

} // finalizeOutput

//////////////////////////////////////////////////////////////////////////////
// Main running function
void MFTDeadMapBuilder::run(ProcessingContext& pc)
{
  if (mRunStopRequested) { // give up when run stop request arrived
    return;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;

  start = std::chrono::high_resolution_clock::now();

  mTFCounter++;

  mFirstOrbitTF = pc.services().get<o2::framework::TimingInfo>().firstTForbit;

  if ((Long64_t)(mFirstOrbitTF / mTFLength) % mTFSampling != 0) {
    return;
  }

  mStepCounter++;
  LOG(info) << "Processing step #" << mStepCounter << " out of " << mTFCounter << " TF received. First orbit " << mFirstOrbitTF;

  mLanesAlive.clear();
  mDeadMapTF->clear();

  bool newlane = false;

  if (mDataSource == "digits") {
    const auto elements = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("elements");
    const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROFs");
    for (const auto& rof : ROFs) {
      auto elementsInTF = rof.getROFData(elements);
      for (const auto& el : elementsInTF) {
        short int chipID = (short int)el.getChipIndex();
        newlane = mLanesAlive.insert(chipID).second;
      }
    }
  } else if (mDataSource == "clusters") {
    const auto elements = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("elements");
    const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROFs");
    for (const auto& rof : ROFs) {
      auto elementsInTF = rof.getROFData(elements);
      for (const auto& el : elementsInTF) {
        short int chipID = (short int)el.getSensorID();
        newlane = mLanesAlive.insert(chipID).second;
      }
    }
  } else if (mDataSource == "chipsstatus") {
    const auto elements = pc.inputs().get<std::vector<char>>("elements");
    for (short int chipID = 0; chipID < elements.size(); chipID++) {
      if (elements.at(chipID)) {
        newlane = mLanesAlive.insert(chipID).second;
      }
    }
  }

  std::set<short int> universalSet;
  std::generate_n(std::inserter(universalSet, universalSet.begin()), 936, [n = 0]() mutable { return n++; });
  std::set_difference(universalSet.begin(), universalSet.end(), // Getting vector of dead chips
                      mLanesAlive.begin(), mLanesAlive.end(),
                      std::back_inserter(*mDeadMapTF));

  // filling the tree
  mTreeObject->Fill();

  end = std::chrono::high_resolution_clock::now();
  int difference = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  LOG(info) << "Dead Chips: " << mDeadMapTF->size() << ", elapsed time in TF processing: " << difference / 1000. << " ms";

  return;
}

//////////////////////////////////////////////////////////////////////////////
void MFTDeadMapBuilder::PrepareOutputCcdb(DataAllocator& output)
{

  long tstart = o2::ccdb::getCurrentTimestamp();
  long secinyear = 365L * 24 * 3600;
  long tend = o2::ccdb::getFutureTimestamp(secinyear);

  std::map<std::string, std::string> md;

  std::string path("MFT/Calib/");
  std::string name_str = "time_dead_map";

  o2::ccdb::CcdbObjectInfo info((path + name_str), "time_dead_map", mObjectName, md, tstart, tend);

  auto image = o2::ccdb::CcdbApi::createObjectImage(mTreeObject, &info);

  info.setAdjustableEOV();

  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size() << "bytes, valid for "
            << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "MFT_TimeDeadMap", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "MFT_TimeDeadMap", 0}, info);

  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void MFTDeadMapBuilder::endOfStream(EndOfStreamContext& ec)
{
  if (!isEnded && !mRunStopRequested) {
    LOG(info) << "endOfStream report: " << mSelfName;
    finalizeOutput();
    PrepareOutputCcdb(ec.outputs());
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: simply close the latest tree
void MFTDeadMapBuilder::stop()
{
  if (!isEnded) {
    LOG(info) << "stop() report: " << mSelfName;
    finalizeOutput();
    if (mDoLocalOutput) {
      LOG(info) << "stop() not sending object as output. ccdb will not be populated.";
    } else {
      LOG(error) << "stop() not sending object as output. ccdb will not be populated.";
    }
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getMFTDeadMapBuilderSpec(std::string datasource)
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginMFT;
  std::vector<InputSpec> inputs;

  if (datasource == "digits") {
    inputs.emplace_back("elements", detOrig, "DIGITS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROFs", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  } else if (datasource == "clusters") {
    inputs.emplace_back("elements", detOrig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROFs", detOrig, "CLUSTERSROF", 0, Lifetime::Timeframe);
  } else if (datasource == "chipsstatus") {
    inputs.emplace_back("elements", detOrig, "CHIPSSTATUS", 0, Lifetime::Timeframe);
  } else {
    return DataProcessorSpec{0x0}; // TODO: ADD PROTECTION
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "MFT_TimeDeadMap"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "MFT_TimeDeadMap"});

  return DataProcessorSpec{
    "mft-deadmap-builder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<MFTDeadMapBuilder>(datasource)},
    Options{{"debug", VariantType::Bool, false, {"Developer debug mode."}},
            {"tf-sampling", VariantType::Int, 1000, {"Process every Nth TF. Selection according to first TF Orbit."}},
            {"tf-length", VariantType::Int, 32, {"Orbits per TFs."}},
            {"output-filename", VariantType::String, "mft_time_deadmap.root", {"ROOT object file name."}},
            {"local-output", VariantType::Bool, false, {"Save ROOT tree file locally."}},
            {"output-dir", VariantType::String, "./", {"ROOT tree local output directory."}}}};
}
} // namespace mft
} // namespace o2
