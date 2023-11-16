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

#include "Generators/GeneratorService.h"
#include "Generators/GeneratorFactory.h"
#include "SimConfig/SimConfig.h"
#include "DataFormatsCalibration/MeanVertexObject.h"

using namespace o2::eventgen;

void GeneratorService::initService(std::string const& genName,
                                   std::string const& triggerName,
                                   VertexOption const& vxtOption)
{
  auto localSimConfig = o2::conf::SimConfig::make();
  localSimConfig.getConfigData().mGenerator = genName;
  localSimConfig.getConfigData().mTrigger = triggerName;

  o2::eventgen::GeneratorFactory::setPrimaryGenerator(localSimConfig, &mPrimGen);

  // initialize vertexing based on type
  if (dynamic_cast<MeanVertexObjectOption const*>(&vxtOption) != nullptr) {
    auto ccdboption = dynamic_cast<MeanVertexObjectOption const*>(&vxtOption);
    LOG(info) << "Init prim gen with MeanVertexObject";
    // assign the mean vertex object
    if (!ccdboption->meanVertexObject) {
      LOG(fatal) << "No mean vertex object found - Cannot initialize Event generator";
    }
    mPrimGen.setVertexMode(o2::conf::VertexMode::kCCDB, ccdboption->meanVertexObject);
  } else if (dynamic_cast<NoVertexOption const*>(&vxtOption) != nullptr) {
    mPrimGen.setVertexMode(o2::conf::VertexMode::kNoVertex);
  } else if (dynamic_cast<DiamondParamVertexOption const*>(&vxtOption) != nullptr) {
    mPrimGen.setVertexMode(o2::conf::VertexMode::kDiamondParam);
  } else {
    LOG(error) << "Unknown VertexOption passed to Generator initialization";
  }

  mStack.setExternalMode(true);
  mPrimGen.Init();
}

void GeneratorService::generateEvent_MCTracks(std::vector<MCTrack>& tracks, o2::dataformats::MCEventHeader& header)
{
  mPrimGen.SetEvent(&header);
  mStack.Reset();
  mPrimGen.GenerateEvent(&mStack); // this is the usual FairROOT interface going via stack

  tracks.reserve(mStack.getPrimaries().size());
  for (auto& tparticle : mStack.getPrimaries()) {
    tracks.emplace_back(tparticle);
  }
}

std::pair<std::vector<o2::MCTrack>, o2::dataformats::MCEventHeader> GeneratorService::generateEvent()
{
  std::vector<o2::MCTrack> tracks;
  o2::dataformats::MCEventHeader header;
  generateEvent_MCTracks(tracks, header);
  return std::pair<std::vector<MCTrack>, o2::dataformats::MCEventHeader>(tracks, header);
}

void GeneratorService::generateEvent_TParticles(std::vector<TParticle>& tracks, o2::dataformats::MCEventHeader& header)
{
  mPrimGen.SetEvent(&header);
  mStack.Reset();
  mPrimGen.GenerateEvent(&mStack); // this is the usual FairROOT interface going via stack

  tracks.clear();
  tracks = mStack.getPrimaries();
}
