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
  mTFSamplingTolerance = ic.options().get<int>("tf-sampling-tolerance");
  if (mTFSamplingTolerance > mTFSampling) {
    LOG(warning) << "Invalid request tf-sampling-tolerance larger or equal than tf-sampling. Setting tolerance to " << mTFSampling - 1;
    mTFSamplingTolerance = mTFSampling - 1;
  }
  mSampledSlidingWindowSize = ic.options().get<int>("tf-sampling-history-size");
  mTFLength = ic.options().get<int>("tf-length");
  mDoLocalOutput = ic.options().get<bool>("local-output");
  mObjectName = ic.options().get<std::string>("outfile");
  mCCDBUrl = ic.options().get<std::string>("ccdb-url");
  if (mCCDBUrl == "none") {
    mCCDBUrl = "";
  }

  mLocalOutputDir = ic.options().get<std::string>("output-dir");
  mSkipStaticMap = ic.options().get<bool>("skip-static-map");

  isEnded = false;
  mTimeStart = o2::ccdb::getCurrentTimestamp();

  if (mRunMFT) {
    N_CHIPS = o2::itsmft::ChipMappingMFT::getNChips();
  } else {
    N_CHIPS = o2::itsmft::ChipMappingITS::getNChips();
  }

  mSampledTFs.clear();
  mSampledHistory.clear();
  mDeadMapTF.clear();
  mStaticChipStatus.clear();
  mMapObject.clear();
  mMapObject.setMapVersion(MAP_VERSION);

  if (!mSkipStaticMap) {
    mStaticChipStatus.resize(N_CHIPS, false);
  }

  LOG(info) << "Sampling one TF every " << mTFSampling << " with " << mTFSamplingTolerance << " TF tolerance";

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

bool ITSMFTDeadMapBuilder::acceptTF(long orbit)
{

  // Description of the algorithm:
  // Return true if the TF index (calculated as orbit/TF_length) falls within any interval [k * tf_sampling, k * tf_sampling + tolerance) for some integer k, provided no other TFs have been found in the same interval.

  if (mTFSamplingTolerance < 1) {
    return ((orbit / mTFLength) % mTFSampling == 0);
  }

  if ((orbit / mTFLength) % mTFSampling > mTFSamplingTolerance) {
    return false;
  }

  long sampling_index = orbit / mTFLength / mTFSampling;

  if (mSampledTFs.find(sampling_index) == mSampledTFs.end()) {

    mSampledTFs.insert(sampling_index);
    mSampledHistory.push_back(sampling_index);

    if (mSampledHistory.size() > mSampledSlidingWindowSize) {
      long oldIndex = mSampledHistory.front();
      mSampledHistory.pop_front();
      mSampledTFs.erase(oldIndex);
    }

    return true;
  }

  return false;
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

  // Skip everything in case of garbage (potentially at EoS)
  if (pc.services().get<o2::framework::TimingInfo>().firstTForbit == -1U) {
    LOG(info) << "Skipping the processing of inputs for timeslice " << pc.services().get<o2::framework::TimingInfo>().timeslice << " (firstTForbit is " << pc.services().get<o2::framework::TimingInfo>().firstTForbit << ")";
    return;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;

  start = std::chrono::high_resolution_clock::now();

  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();

  if (tinfo.globalRunNumberChanged || mFirstOrbitRun == 0x0) { // new run is starting
    mRunNumber = tinfo.runNumber;
    mFirstOrbitRun = mFirstOrbitTF;
    mTFCounter = 0;
    isEnded = false;
  }

  if (isEnded) {
    return;
  }
  mFirstOrbitTF = tinfo.firstTForbit;
  mTFCounter++;

  long sampled_orbit = mFirstOrbitTF - mFirstOrbitRun;

  if (!acceptTF(sampled_orbit)) {
    return;
  }

  mStepCounter++;
  LOG(info) << "Processing step #" << mStepCounter << " out of " << mTFCounter << " good TF received. First orbit " << mFirstOrbitTF;

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

  if (pc.transitionState() == TransitionHandlingState::Requested && !isEnded) {
    std::string detname = mRunMFT ? "MFT" : "ITS";
    LOG(warning) << "Transition state requested for " << detname << " process, calling stop() and stopping the process of new data.";
    stop();
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSMFTDeadMapBuilder::PrepareOutputCcdb(EndOfStreamContext* ec, std::string ccdburl = "")
{

  // if ccdburl is specified, the object is sent to ccdb from this workflow

  long tend = o2::ccdb::getCurrentTimestamp();

  std::map<std::string, std::string> md = {{"map_version", MAP_VERSION}, {"runNumber", std::to_string(mRunNumber)}};

  std::string path = mRunMFT ? "MFT/Calib/" : "ITS/Calib/";
  std::string name_str = "TimeDeadMap";

  o2::ccdb::CcdbObjectInfo info((path + name_str), name_str, mObjectName, md, mTimeStart - 120 * 1000, tend + 60 * 1000);

  auto image = o2::ccdb::CcdbApi::createObjectImage(&mMapObject, &info);
  info.setFileName(mObjectName);

  info.setAdjustableEOV();

  if (ec != nullptr) {

    LOG(important) << "Sending object " << info.getPath() << "/" << info.getFileName()
                   << " to ccdb-populator, of size " << image->size() << " bytes, valid for "
                   << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    if (mRunMFT) {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TimeDeadMap", 1}, *image.get());
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TimeDeadMap", 1}, info);
    } else {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TimeDeadMap", 0}, *image.get());
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TimeDeadMap", 0}, info);
    }
  }

  else if (!ccdburl.empty()) { // send from this workflow

    LOG(important) << mSelfName << " sending object " << ccdburl << "/browse/" << info.getPath() << "/" << info.getFileName()
                   << " of size " << image->size() << " bytes, valid for "
                   << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    o2::ccdb::CcdbApi mApi;
    mApi.init(ccdburl);
    mApi.storeAsBinaryFile(
      &image->at(0), image->size(), info.getFileName(), info.getObjectType(),
      info.getPath(), info.getMetaData(),
      info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
    o2::ccdb::adjustOverriddenEOV(mApi, info);
  }

  else {

    LOG(warning) << "PrepareOutputCcdb called with empty arguments. Doing nothing.";
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSMFTDeadMapBuilder::endOfStream(EndOfStreamContext& ec)
{
  if (!isEnded) {
    LOG(info) << "endOfStream report: " << mSelfName;
    finalizeOutput();
    if (mMapObject.getEvolvingMapSize() > 0) {
      PrepareOutputCcdb(&ec);
    } else {
      LOG(warning) << "Time-dependent dead map is empty and will not be forwarded as output";
    }
    LOG(info) << "Stop process of new data because of endOfStream";
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: create local output if endOfStream not processed
void ITSMFTDeadMapBuilder::stop()
{
  if (!isEnded) {
    LOG(info) << "stop() report: " << mSelfName;
    finalizeOutput();
    if (!mCCDBUrl.empty()) {
      std::string detname = mRunMFT ? "MFT" : "ITS";
      LOG(warning) << "endOfStream not processed. Sending output to ccdb from the " << detname << " deadmap builder workflow.";
      PrepareOutputCcdb(nullptr, mCCDBUrl);
    } else {
      LOG(alarm) << "endOfStream not processed. Nothing forwarded as output.";
    }
    LOG(info) << "Stop process of new data because of stop() call.";
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
    Options{{"tf-sampling", VariantType::Int, 350, {"Process every Nth TF. Selection according to first TF orbit."}},
            {"tf-sampling-tolerance", VariantType::Int, 20, {"Tolerance on the tf-sampling value (sliding window size)."}},
            {"tf-sampling-history-size", VariantType::Int, 1000, {"Do not check if new TF is contained in a window that is older than N steps."}},
            {"tf-length", VariantType::Int, 32, {"Orbits per TF."}},
            {"skip-static-map", VariantType::Bool, false, {"Do not fill static part of the map."}},
            {"ccdb-url", VariantType::String, "", {"CCDB url. Ignored if endOfStream is processed."}},
            {"outfile", VariantType::String, objectname_default, {"ROOT object file name."}},
            {"local-output", VariantType::Bool, false, {"Save ROOT tree file locally."}},
            {"output-dir", VariantType::String, "./", {"ROOT tree local output directory."}}}};
}

} // namespace itsmft
} // namespace o2
