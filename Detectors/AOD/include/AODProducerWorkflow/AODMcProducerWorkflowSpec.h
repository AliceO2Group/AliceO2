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

/// @file   AODMcProducerWorkflowSpec.h

#ifndef O2_AODMCPRODUCER_WORKFLOW_SPEC
#define O2_AODMCPRODUCER_WORKFLOW_SPEC

#include "AODProducerHelpers.h"
#include "AODMcProducerHelpers.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Steer/MCKinematicsReader.h"
#include <TStopwatch.h>

#include <string>
#include <vector>

using namespace o2::framework;

namespace o2::aodmcproducer
{

class AODMcProducerWorkflowDPL : public Task
{
 public:
  AODMcProducerWorkflowDPL() = default;
  ~AODMcProducerWorkflowDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

  /** Some types we will use */
  using MCEventHeader = dataformats::MCEventHeader;
  using McCollisions = aod::McCollisions;
  using McParticles = aod::StoredMcParticles;
  using Origins = aod::Origins;
  using XSections = aod::HepMCXSections;
  using PdfInfos = aod::HepMCPdfInfos;
  using HeavyIons = aod::HepMCHeavyIons;
  using MCKinematicsReader = steer::MCKinematicsReader;
  /** */
 private:
  /** some other types we will use */
  using CollisionCursor = aodmchelpers::CollisionCursor;
  using XSectionCursor = aodmchelpers::XSectionCursor;
  using PdfInfoCursor = aodmchelpers::PdfInfoCursor;
  using HeavyIonCursor = aodmchelpers::HeavyIonCursor;
  using HepMCUpdate = aodmchelpers::HepMCUpdate;
  /**
   * Update the header (collision and HepMC aux) information.
   *
   * When updating the HepMC aux tables, we take the relevant policies
   * into account (mXSectionUpdate, mPdfInfoUpdate, mHeavyIonUpdate).
   *
   * - If a policy is "never", then the corresponding table is never
   *   updated.
   *
   * - If the policy is "always", then the table is always
   *   update.
   *
   * - If the policy is either "anyKey" or "allKeys", _and_
   *   this is the first event, then we check if any or all keys,
   *   respectively are present in the header.
   *
   *   - If that check fails, then we do not update and set the
   *     corresponding policy to be "never".
   *
   *   - If the check succeeds, then we do update the table, and set
   *     the corresponding policty to "always".
   *
   *   In this way, we will let the first event decide what to do for
   *   subsequent events and thus avoid too many string comparisions.
   *
   * @param collisionCursor Cursor over aod::McCollisions
   * @param xSectionCursor Cursor over aod::HepMCXSections
   * @param pdfInfoCursor Cursor over aod::HepMCPdfInfos
   * @param heavyIonCursor Cursor over aod::HepMCHeavyIons
   * @param header Header to read information from
   * @param collisionID Index of collision in table
   * @param bcID Current event identifier (bcID)
   * @param time Time of event
   * @param generatorID Generator identifier, if any
   * @param sourceID Source identifier
   *
   */
  void updateHeader(CollisionCursor& collisionCursor,
                    XSectionCursor& xSectionCursor,
                    PdfInfoCursor& pdfInfoCursor,
                    HeavyIonCursor& heavyIonCursor,
                    const MCEventHeader& header,
                    int collisionID, // Index
                    int bcID,
                    float time,
                    short generatorID,
                    int sourceID);
  /** Current timeframe number */
  int64_t mTFNumber{1};
  /** Whether to filter tracks so that only primary particles,
      particles from EG, or their parents are written out */
  int mFilterMC{0};
  /** Whether to enable embedding */
  bool mEnableEmbed{false};
  /** LPM production tag, for anchoring */
  TString mLPMProdTag{""};
  /** Pass identifier for anchoring */
  TString mAnchorPass{""};
  /** Production identifier for anchoring */
  TString mAnchorProd{""};
  /** Reconstruction pass for anchoring */
  TString mRecoPass{""};
  /** Timer */
  TStopwatch mTimer;
  /** Rules for when to update HepMC tables */
  HepMCUpdate mXSectionUpdate = HepMCUpdate::anyKey;
  HepMCUpdate mPdfInfoUpdate = HepMCUpdate::anyKey;
  HepMCUpdate mHeavyIonUpdate = HepMCUpdate::anyKey;

  std::string mSimPrefix;

  // MC production metadata holder
  bool mIsMDSent{false};

  // truncation is enabled by default
  uint32_t mCollisionPosition = 0xFFFFFFF0; // 19 bits mantissa
  uint32_t mMcParticleW = 0xFFFFFFF0;       // 19 bits
  uint32_t mMcParticlePos = 0xFFFFFFF0;     // 19 bits
  uint32_t mMcParticleMom = 0xFFFFFFF0;     // 19 bits
};

/// create a processor spec
framework::DataProcessorSpec getAODMcProducerWorkflowSpec();

} // namespace o2::aodmcproducer

#endif /* O2_AODMCPRODUCER_WORKFLOW_SPEC */
