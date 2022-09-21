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

/// @file   AODProducerWorkflowSpec.h

#ifndef O2_AODMCPRODUCER_WORKFLOW_SPEC
#define O2_AODMCPRODUCER_WORKFLOW_SPEC

#include "AODProducerHelpers.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Steer/MCKinematicsReader.h"
#include "TMap.h"
#include "TStopwatch.h"

#include <string>
#include <vector>

using namespace o2::framework;

namespace o2::aodmcproducer
{

class AODMcProducerWorkflowDPL : public Task
{
 public:
  AODMcProducerWorkflowDPL(std::string resFile) : mResFile{resFile} {}
  ~AODMcProducerWorkflowDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  int64_t mTFNumber{-1};
  int mRunNumber{-1};
  int mTruncate{1};
  int mFilterMC{0};
  o2::InteractionRecord mStartIR{}; // TF 1st IR
  TString mResFile{"AO2D"};
  TString mLPMProdTag{""};
  TString mAnchorPass{""};
  TString mAnchorProd{""};
  TString mRecoPass{""};
  TStopwatch mTimer;

  o2::aodhelpers::TripletsMap_t mToStore;

  // MC production metadata holder
  TMap mMetaData;

  // truncation is enabled by default
  uint32_t mCollisionPosition = 0xFFFFFFF0; // 19 bits mantissa
  uint32_t mMcParticleW = 0xFFFFFFF0;       // 19 bits
  uint32_t mMcParticlePos = 0xFFFFFFF0;     // 19 bits
  uint32_t mMcParticleMom = 0xFFFFFFF0;     // 19 bits

  void collectBCs(const std::vector<o2::InteractionTimeRecord>& mcRecords,
                  std::map<uint64_t, int>& bcsMap);

  template <typename MCParticlesCursorType>
  void fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                            const MCParticlesCursorType& mcParticlesCursor,
                            const std::map<std::pair<int, int>, int>& mcColToEvSrc);
};

/// create a processor spec
framework::DataProcessorSpec getAODMcProducerWorkflowSpec(std::string resFile);

} // namespace o2::aodmcproducer

#endif /* O2_AODMCPRODUCER_WORKFLOW_SPEC */
