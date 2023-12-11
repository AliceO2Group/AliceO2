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

#include "ITSWorkflow/DeadMapBuilderSpec.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/CompCluster.h"

namespace o2
{
namespace its
{

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSDeadMapBuilder::ITSDeadMapBuilder(const ITSDMInpConf& inpConf, std::string datasource)
  : mDataSource(datasource)
{
  mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSDeadMapBuilder");
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
ITSDeadMapBuilder::~ITSDeadMapBuilder()
{
  // Clear dynamic memory
  delete mDeadMapTF;
  delete mTreeObject;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDeadMapBuilder::init(InitContext& ic)
{

  LOG(info) << "ITSDeadMapBuilder init... " << mSelfName;

  mDeadMapTF = new std::vector<ULong64_t>{};

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
// Grouping chips in lanes
short int ITSDeadMapBuilder::getLaneIDFromChip(short int chip)
{
  // TODO: does o2::itsmft::ChipMappingITS already contain this?
  if (chip < N_CHIPS_IB) {
    return chip;
  } else {
    return N_CHIPS_IB + (short int)((chip - N_CHIPS_IB) / 7);
  }
}

///////////////////////////////////////////////
// Not used here: to be imported by any code using the deadmap to traslate tree entries in lane lists
std::vector<short int> ITSDeadMapBuilder::decodeITSMapWord(ULong64_t word)
{

  std::vector<short int> lanelist{};

  if ((word & 0x1) == 0x0) { // _t0 aka IB
    for (int l = 0; l < 7; l++) {
      short int lanepp = (short int)((word >> (9 * l + 1)) & 0x1FF);
      if (lanepp == 0) {
        break;
      }
      lanelist.push_back(lanepp - 1);
    }
  } else if ((word & 0xF) == 0x1) { // _t1 aka OB
    for (int l = 0; l < 5; l++) {
      short int lanepp = (short int)((word >> (12 * l + 4)) & 0xFFF);
      if (lanepp == 0) {
        break;
      }
      lanelist.push_back(lanepp - 1);
    }
  } else if ((word & 0xF) == 0x3) { // _ t2 aka interval
    short int lanelowpp = (uint)((word >> 4) & 0x7FFF);
    short int laneuppp = (uint)((word >> 19) & 0x7FFF);
    for (short int lane = lanelowpp; lane <= laneuppp; lane++) {
      lanelist.push_back(lane - 1);
    }
    lanelowpp = (short int)((word >> 34) & 0x7FFF);
    if (lanelowpp) {
      laneuppp = (short int)((word >> 49) & 0x7FFF);
      for (short int lane = lanelowpp; lane <= laneuppp; lane++) {
        lanelist.push_back(lane - 1);
      }
    }
  } else { // word not recognized // for the use: add protection
    lanelist.push_back(-1);
  }
  LOG(info) << "Word " << word << " encodes " << lanelist.size() << " lanes";
  return lanelist;
}

//////////////////////////////////////////////////////////////////////////////

void ITSDeadMapBuilder::finalizeOutput()
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
void ITSDeadMapBuilder::run(ProcessingContext& pc)
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
        newlane = mLanesAlive.insert(getLaneIDFromChip(chipID)).second;
      }
    }
  } else if (mDataSource == "clusters") {
    const auto elements = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("elements");
    const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROFs");
    for (const auto& rof : ROFs) {
      auto elementsInTF = rof.getROFData(elements);
      for (const auto& el : elementsInTF) {
        short int chipID = (short int)el.getSensorID();
        newlane = mLanesAlive.insert(getLaneIDFromChip(chipID)).second;
      }
    }
  } else if (mDataSource == "chipsstatus") {
    const auto elements = pc.inputs().get<std::vector<char>>("elements");
    for (short int chipID = 0; chipID < elements.size(); chipID++) {
      if (elements.at(chipID)) {
        newlane = mLanesAlive.insert(getLaneIDFromChip(chipID)).second;
      }
    }
  }

  mLanesAlive.insert(getLaneIDFromChip(N_CHIPS));

  LOG(info) << "TF contains" << mLanesAlive.size() << " active elements";
  // filling the vector
  std::pair<int, int> fillresult = FillMapElementITS(); // first = n dead lanes, second = n words

  // filling the tree
  mTreeObject->Fill();

  end = std::chrono::high_resolution_clock::now();
  int difference = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  LOG(info) << "Elapsed time in TF processing: " << difference / 1000. << " ms";

  return;
}

std::pair<int, int> ITSDeadMapBuilder::FillMapElementITS()
{

  if (MAP_VERSION == "1") { // Force developer to change metadata in case encoding is changed

    std::set<short int>::iterator laneIt = mLanesAlive.begin();

    short int laneLow = -1, laneUp = -1;

    int dcount_t0 = 0, dcount_t1 = 0, dcount_t2 = 0;

    ULong64_t mapelement_t0 = t0_identifier;
    ULong64_t mapelement_t1 = t1_identifier;
    ULong64_t mapelement_t2 = t2_identifier;

    for (int ilane = 0; ilane < mLanesAlive.size(); ilane++) {

      laneLow = laneUp;
      laneUp = *laneIt;
      laneIt++;

      if (laneUp - laneLow - 1 > 6) { // more than 6 lanes (or full stave) --> better to save in a single word
        bool isfilled = (mapelement_t2 != t2_identifier);
        mapelement_t2 = mapelement_t2 | ((ULong64_t)((laneLow + 2) & 0xFFF) << 4 + 30 * isfilled);
        mapelement_t2 = mapelement_t2 | ((ULong64_t)((laneUp - 1 + 1) & 0xFFF) << 19 + 30 * isfilled);
        dcount_t2 += (laneUp - laneLow - 1);
        if (isfilled) {
          mDeadMapTF->push_back(mapelement_t2);
          mapelement_t2 = t2_identifier;
        }
      }

      else {
        for (short int idead = laneLow + 1; idead < laneUp; idead++) {

          if (idead < N_CHIPS_IB) { // IB;

            mapelement_t0 = mapelement_t0 | ((ULong64_t)((idead + 1) & 0x1FF) << (1 + 9 * (dcount_t0 % 7)));
            dcount_t0++;

            if (dcount_t0 % 7 == 0) {
              mDeadMapTF->push_back(mapelement_t0);
              mapelement_t0 = t0_identifier;
            }
          }

          else { // OB
            mapelement_t1 = mapelement_t1 | ((ULong64_t)((idead + 1) & 0xFFF) << (4 + 12 * (dcount_t1 % 5)));
            dcount_t1++;
            if (dcount_t1 % 5 == 0) {
              mDeadMapTF->push_back(mapelement_t1);
              mapelement_t1 = t1_identifier;
            }
          }
        } // end loop over dead lanes
      }
    } // end loop over alive lanes set

    // fill with last info in the buffer (partially empty words)
    if (mapelement_t0 != t0_identifier) {
      mDeadMapTF->push_back(mapelement_t0);
    }
    if (mapelement_t1 != t1_identifier) {
      mDeadMapTF->push_back(mapelement_t1);
    }
    if (mapelement_t2 != t2_identifier) {
      mDeadMapTF->push_back(mapelement_t2);
    }

    int dcountTot = dcount_t0 + dcount_t1 + dcount_t2;

    LOG(info) << "Dead lanes: " << dcountTot << ", type0|1|2: " << dcount_t0 << "|" << dcount_t1 << "|" << dcount_t2 << ", saved in " << mDeadMapTF->size() << " words.";

    return std::pair<int, int>{dcountTot, mDeadMapTF->size()};

  } // end of map version condition

  LOG(error) << "Invalid MAP version requested. Filling dummy vector";
  return std::pair<int, int>{-1, -1};
}

//////////////////////////////////////////////////////////////////////////////
void ITSDeadMapBuilder::PrepareOutputCcdb(DataAllocator& output)
{

  long tstart = o2::ccdb::getCurrentTimestamp();
  long secinyear = 365L * 24 * 3600;
  long tend = o2::ccdb::getFutureTimestamp(secinyear);

  std::map<std::string, std::string> md = {
    {"map_version", MAP_VERSION}};

  std::string path("ITS/Calib/");
  std::string name_str = "time_dead_map";

  o2::ccdb::CcdbObjectInfo info((path + name_str), "time_dead_map", mObjectName, md, tstart, tend);

  auto image = o2::ccdb::CcdbApi::createObjectImage(mTreeObject, &info);

  info.setAdjustableEOV();

  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size() << "bytes, valid for "
            << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_TimeDeadMap", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_TimeDeadMap", 0}, info);

  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSDeadMapBuilder::endOfStream(EndOfStreamContext& ec)
{
  if (!isEnded && !mRunStopRequested) {
    LOG(info) << "endOfStream report:" << mSelfName;
    finalizeOutput();
    PrepareOutputCcdb(ec.outputs());
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: simply close the latest tree
void ITSDeadMapBuilder::stop()
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
DataProcessorSpec getITSDeadMapBuilderSpec(const ITSDMInpConf& inpConf, std::string datasource)
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
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
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_TimeDeadMap"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_TimeDeadMap"});

  return DataProcessorSpec{
    "its-deadmap-builder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSDeadMapBuilder>(inpConf, datasource)},
    Options{{"debug", VariantType::Bool, false, {"Developer debug mode."}},
            {"tf-sampling", VariantType::Int, 1000, {"Process every Nth TF. Selection according to first TF Orbit."}},
            {"tf-length", VariantType::Int, 32, {"Orbits per TFs."}},
            {"output-filename", VariantType::String, "its_time_deadmap.root", {"ROOT object file name."}},
            {"local-output", VariantType::Bool, false, {"Save ROOT tree file locally."}},
            {"output-dir", VariantType::String, "./", {"ROOT tree local output directory."}}}};
}
} // namespace its
} // namespace o2
