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

#include "ITSMFTWorkflow/DeadMapBuilderSpec.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TimeDeadMap.h"

namespace o2
{
namespace itsmft
{

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSMFTDeadMapBuilder::ITSMFTDeadMapBuilder(std::string datasource, bool doMFT)
  : mDataSource(datasource), mRunMFT(doMFT)
{
  std::string detector = doMFT ? "MFT" : "ITS";
  mSelfName = o2::utils::Str::concat_string("ITSMFTDeadMapBuilder_", detector);
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
ITSMFTDeadMapBuilder::~ITSMFTDeadMapBuilder()
{
  // Clear dynamic memory
  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSMFTDeadMapBuilder::init(InitContext& ic)
{

  LOG(info) << "ITSMFTDeadMapBuilder init... " << mSelfName;

  mTFSampling = ic.options().get<int>("tf-sampling");
  mTFLength = ic.options().get<int>("tf-length");
  mDoLocalOutput = ic.options().get<bool>("local-output");
  mObjectName = ic.options().get<std::string>("outfile");
  mLocalOutputDir = ic.options().get<std::string>("output-dir");
  mSkipStaticMap = ic.options().get<bool>("skip-static-map");

  mTimeStart = o2::ccdb::getCurrentTimestamp();

  if (mRunMFT) {
    N_CHIPS = o2::itsmft::ChipMappingMFT::getNChips();
  } else {
    N_CHIPS = o2::itsmft::ChipMappingITS::getNChips();
  }

  mDeadMapTF.clear();
  mStaticChipStatus.clear();
  mMapObject.clear();
  mMapObject.setMapVersion(MAP_VERSION);

  if (!mSkipStaticMap) {
    mStaticChipStatus.resize(N_CHIPS, false);
  }

  LOG(info) << "Sampling one TF every " << mTFSampling;

  return;
}

///////////////////////////////////////////////////////////////////
// TODO: can ChipMappingITS help here?
std::vector<uint16_t> ITSMFTDeadMapBuilder::getChipIDsOnSameCable(uint16_t chip)
{
  if (mRunMFT || chip < N_CHIPS_ITSIB) {
    return std::vector<uint16_t>{chip};
  } else {
    uint16_t firstchipcable = 7 * (uint16_t)((chip - N_CHIPS_ITSIB) / 7) + N_CHIPS_ITSIB;
    std::vector<uint16_t> chipList(7);
    std::generate(chipList.begin(), chipList.end(), [&firstchipcable]() { return firstchipcable++; });
    return chipList;
  }
}

//////////////////////////////////////////////////////////////////////////////

void ITSMFTDeadMapBuilder::finalizeOutput()
{

  if (!mSkipStaticMap) {
    std::vector<uint16_t> staticmap{};
    int staticmap_chipcounter = 0;
    for (uint16_t el = 0; el < mStaticChipStatus.size(); el++) {
      if (mStaticChipStatus[el]) {
        continue;
      }
      staticmap_chipcounter++;
      bool previous_dead = (el > 0 && !mStaticChipStatus[el - 1]);
      bool next_dead = (el < mStaticChipStatus.size() - 1 && !mStaticChipStatus[el + 1]);
      if (!previous_dead && next_dead) {
        staticmap.push_back(el | (uint16_t)(0x8000));
      } else if (previous_dead && next_dead) {
        continue;
      } else {
        staticmap.push_back(el);
      }
    }

    LOG(info) << "Filling static part of the map with " << staticmap_chipcounter << " dead chips, saved into " << staticmap.size() << " words";

    mMapObject.fillMap(staticmap);
  }

  if (mDoLocalOutput) {
    std::string localoutfilename = mLocalOutputDir + "/" + mObjectName;
    TFile outfile(localoutfilename.c_str(), "RECREATE");
    outfile.WriteObjectAny(&mMapObject, "o2::itsmft::TimeDeadMap", "ccdb_object");
    outfile.Close();
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
void ITSMFTDeadMapBuilder::run(ProcessingContext& pc)
{
  if (mRunStopRequested) { // give up when run stop request arrived
    return;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;

  start = std::chrono::high_resolution_clock::now();

  mTFCounter++;

  mFirstOrbitTF = pc.services().get<o2::framework::TimingInfo>().firstTForbit;

  if ((unsigned long)(mFirstOrbitTF / mTFLength) % mTFSampling != 0) {
    return;
  }

  mStepCounter++;
  LOG(info) << "Processing step #" << mStepCounter << " out of " << mTFCounter << " TF received. First orbit " << mFirstOrbitTF;

  mDeadMapTF.clear();

  std::vector<bool> ChipStatus(N_CHIPS, false);

  if (mDataSource == "digits") {
    const auto elements = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("elements");
    const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROFs");
    for (const auto& rof : ROFs) {
      auto elementsInTF = rof.getROFData(elements);
      for (const auto& el : elementsInTF) {
        ChipStatus.at((int)el.getChipIndex()) = true;
      }
    }
  } else if (mDataSource == "clusters") {
    const auto elements = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("elements");
    const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROFs");
    for (const auto& rof : ROFs) {
      auto elementsInTF = rof.getROFData(elements);
      for (const auto& el : elementsInTF) {
        ChipStatus.at((int)el.getSensorID()) = true;
      }
    }
  } else if (mDataSource == "chipsstatus") {
    const auto elements = pc.inputs().get<gsl::span<char>>("elements");
    for (uint16_t chipID = 0; chipID < elements.size(); chipID++) {
      if (elements[chipID]) {
        ChipStatus.at(chipID) = true;
      }
    }
  }

  // do AND operation before unmasking the full ITS lane

  if (!mSkipStaticMap) {
    for (size_t el = 0; el < mStaticChipStatus.size(); el++) {
      mStaticChipStatus[el] = mStaticChipStatus[el] || ChipStatus[el];
    }
  }

  // for ITS, declaring dead only chips belonging to lane with no hits
  if (!mRunMFT) {
    for (uint16_t el = N_CHIPS_ITSIB; el < ChipStatus.size(); el++) {
      if (ChipStatus.at(el)) {
        std::vector<uint16_t> chipincable = getChipIDsOnSameCable(el);
        for (uint16_t el2 : chipincable) {
          ChipStatus.at(el2) = true;
          el = el2;
        }
      }
    }
  }

  int CountDead = 0;

  for (uint16_t el = 0; el < ChipStatus.size(); el++) {
    if (ChipStatus.at(el)) {
      continue;
    }
    CountDead++;
    bool previous_dead = (el > 0 && !ChipStatus.at(el - 1));
    bool next_dead = (el < ChipStatus.size() - 1 && !ChipStatus.at(el + 1));
    if (!previous_dead && next_dead) {
      mDeadMapTF.push_back(el | (uint16_t)(0x8000));
    } else if (previous_dead && next_dead) {
      continue;
    } else {
      mDeadMapTF.push_back(el);
    }
  }

  LOG(info) << "TF contains " << CountDead << " dead chips, saved into " << mDeadMapTF.size() << " words.";

  // filling the map
  mMapObject.fillMap(mFirstOrbitTF, mDeadMapTF);

  end = std::chrono::high_resolution_clock::now();
  int difference = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  LOG(info) << "Elapsed time in TF processing: " << difference / 1000. << " ms";

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSMFTDeadMapBuilder::PrepareOutputCcdb(DataAllocator& output)
{

  long tend = o2::ccdb::getCurrentTimestamp();

  std::map<std::string, std::string> md = {
    {"map_version", MAP_VERSION}};

  std::string path = mRunMFT ? "MFT/Calib/" : "ITS/Calib/";
  std::string name_str = "TimeDeadMap";

  o2::ccdb::CcdbObjectInfo info((path + name_str), name_str, mObjectName, md, mTimeStart - 120 * 1000, tend + 60 * 1000);

  auto image = o2::ccdb::CcdbApi::createObjectImage(&mMapObject, &info);
  info.setFileName(mObjectName);

  info.setAdjustableEOV();

  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size() << "bytes, valid for "
            << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

  if (mRunMFT) {
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TimeDeadMap", 1}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TimeDeadMap", 1}, info);
  } else {
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TimeDeadMap", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TimeDeadMap", 0}, info);
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSMFTDeadMapBuilder::endOfStream(EndOfStreamContext& ec)
{
  if (!isEnded && !mRunStopRequested) {
    LOG(info) << "endOfStream report:" << mSelfName;
    finalizeOutput();
    if (mMapObject.getEvolvingMapSize() > 0) {
      PrepareOutputCcdb(ec.outputs());
    } else {
      LOG(warning) << "Time-dependent dead map is empty and will not be forwarded as output";
    }
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: create local output if endOfStream not processed
void ITSMFTDeadMapBuilder::stop()
{
  if (!isEnded) {
    LOG(info) << "stop() report:" << mSelfName;
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
DataProcessorSpec getITSMFTDeadMapBuilderSpec(std::string datasource, bool doMFT)
{
  o2::header::DataOrigin detOrig;
  if (doMFT) {
    detOrig = o2::header::gDataOriginMFT;
  } else {
    detOrig = o2::header::gDataOriginITS;
  }

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
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TimeDeadMap"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TimeDeadMap"}, Lifetime::Sporadic);

  std::string detector = doMFT ? "mft" : "its";
  std::string objectname_default = detector + "_time_deadmap.root";

  return DataProcessorSpec{
    "itsmft-deadmap-builder_" + detector,
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSMFTDeadMapBuilder>(datasource, doMFT)},
    Options{{"tf-sampling", VariantType::Int, 1000, {"Process every Nth TF. Selection according to first TF orbit."}},
            {"tf-length", VariantType::Int, 32, {"Orbits per TF."}},
            {"skip-static-map", VariantType::Bool, false, {"Do not fill static part of the map."}},
            {"outfile", VariantType::String, objectname_default, {"ROOT object file name."}},
            {"local-output", VariantType::Bool, false, {"Save ROOT tree file locally."}},
            {"output-dir", VariantType::String, "./", {"ROOT tree local output directory."}}}};
}

} // namespace itsmft
} // namespace o2
