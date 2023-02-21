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

/// @file   AODProducerWorkflowSpec.cxx

#include "AODProducerWorkflow/AODProducerWorkflowSpec.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFDD/RecPoint.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/Configuration.h"
#include "DataFormatsCPV/Cluster.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMID/Track.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsZDC/ZDCEnergy.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "CommonUtils/NameConf.h"
#include "MathUtils/Utils.h"
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonConstants/Triggers.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/CCDBParamSpec.h"
#include "FDDBase/Constants.h"
#include "FT0Base/Geometry.h"
#include "FV0Base/Geometry.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "MCHTracking/TrackExtrap.h"
#include "MCHTracking/TrackParam.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsVertexing/PVertexerParams.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCUtils.h"
#include "ZDCBase/Constants.h"
#include "TPCBase/ParameterElectronics.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "TOFBase/Utils.h"
#include "O2Version.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include "Math/SMatrix.h"
#include "TMatrixD.h"
#include "TString.h"
#include "TObjString.h"
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <vector>
#include <thread>
#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;
using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

namespace o2::aodproducer
{

void AODProducerWorkflowDPL::createCTPReadout(const o2::globaltracking::RecoContainer& recoData, std::vector<o2::ctp::CTPDigit>& ctpDigits, ProcessingContext& pc)
{
  // Extraxt CTP Config from CCDB
  const auto ctpcfg = pc.inputs().get<o2::ctp::CTPConfiguration*>("ctpconfig");
  ctpcfg->printStream(std::cout);
  // o2::ctp::CTPConfiguration ctpcfg = o2::ctp::CTPRunManager::getConfigFromCCDB(-1, std::to_string(runNumber)); // how to get run
  //  Extract inputs from recoData
  uint64_t classMaskEMCAL = 0, classMaskTRD = 0, classMaskPHOSCPV = 0;
  for (const auto& trgclass : ctpcfg->getCTPClasses()) {
    if (trgclass.cluster->getClusterDetNames().find("EMC") != std::string::npos) {
      classMaskEMCAL = trgclass.classMask;
    }
    if (trgclass.cluster->getClusterDetNames().find("PHS") != std::string::npos) {
      classMaskPHOSCPV = trgclass.classMask;
    }
    if (trgclass.cluster->getClusterDetNames().find("TRD") != std::string::npos) {
      classMaskTRD = trgclass.classMask;
    }
  }
  LOG(info) << "createCTPReadout: Class Mask EMCAL -> " << classMaskEMCAL;
  LOG(info) << "createCTPReadout: Class Mask PHOS/CPV -> " << classMaskPHOSCPV;
  LOG(info) << "createCTPReadout: Class Mask TRD -> " << classMaskTRD;

  // const auto& fddRecPoints = recoData.getFDDRecPoints();
  // const auto& fv0RecPoints = recoData.getFV0RecPoints();
  const auto& triggerrecordEMCAL = recoData.getEMCALTriggers();
  const auto& triggerrecordPHOSCPV = recoData.getPHOSTriggers();
  const auto& triggerrecordTRD = recoData.getTRDTriggerRecords();
  // For EMCAL filter remove calibration triggers
  std::vector<o2::emcal::TriggerRecord> triggerRecordEMCALPhys;
  for (const auto& trg : triggerrecordEMCAL) {
    if (trg.getTriggerBits() & o2::trigger::Cal) {
      continue;
    }
    triggerRecordEMCALPhys.push_back(trg);
  }
  // const auto& triggerrecordTRD =recoData.getITSTPCTRDTriggers()
  //

  // Find TVX triggers, only TRD/EMCAL/PHOS/CPV triggers in coincidence will be accepted
  std::set<uint64_t> bcsMapT0triggers;
  const auto& ft0RecPoints = recoData.getFT0RecPoints();
  for (auto& ft0RecPoint : ft0RecPoints) {
    auto t0triggers = ft0RecPoint.getTrigger();
    if (t0triggers.getVertex()) {
      uint64_t globalBC = ft0RecPoint.getInteractionRecord().toLong();
      bcsMapT0triggers.insert(globalBC);
    }
  }

  auto genericCTPDigitizer = [&bcsMapT0triggers, &ctpDigits](auto triggerrecords, uint64_t classmask) -> int {
    // Strategy:
    // find detector trigger based on trigger record from readout and add CTPDigit if trigger there
    int cntwarnings = 0;
    uint32_t orbitPrev = 0;
    uint16_t bcPrev = 0;
    for (auto& trigger : triggerrecords) {
      auto orbitPrevT = orbitPrev;
      auto bcPrevT = bcPrev;
      bcPrev = trigger.getBCData().bc;
      orbitPrev = trigger.getBCData().orbit;
      // dedicated for TRD: remove bogus triggers
      if (orbitPrev < orbitPrevT || bcPrev >= o2::constants::lhc::LHCMaxBunches || (orbitPrev == orbitPrevT && bcPrev < bcPrevT)) {
        cntwarnings++;
        // LOGP(warning, "Bogus TRD trigger at bc:{}/orbit:{} (previous was {}/{}), with {} tracklets and {} digits",bcPrev, orbitPrev, bcPrevT, orbitPrevT, trig.getNumberOfTracklets(), trig.getNumberOfDigits());
      } else {
        uint64_t globalBC = trigger.getBCData().toLong();
        auto t0entry = bcsMapT0triggers.find(globalBC);
        if (t0entry != bcsMapT0triggers.end()) {
          auto ctpdig = std::find_if(ctpDigits.begin(), ctpDigits.end(), [globalBC](const o2::ctp::CTPDigit& dig) { return dig.intRecord.toLong() == globalBC; });
          if (ctpdig != ctpDigits.end()) {
            // CTP digit existing from other trigger, merge detector class mask
            ctpdig->CTPClassMask |= std::bitset<64>(classmask);
            LOG(debug) << "createCTPReadout: Merging " << classmask << " CTP digits with existing digit, CTP mask " << ctpdig->CTPClassMask;
          } else {
            // New CTP digit needed
            LOG(debug) << "createCTPReadout: New CTP digit needed for class " << classmask << std::endl;
            auto& ctpdigNew = ctpDigits.emplace_back();
            ctpdigNew.intRecord.setFromLong(globalBC);
            ctpdigNew.CTPClassMask = classmask;
          }
        } else {
          LOG(warning) << "createCTPReadout: Found " << classmask << " and no MTVX:" << globalBC;
        }
      }
    }
    return cntwarnings;
  };

  auto warningsTRD = genericCTPDigitizer(triggerrecordTRD, classMaskTRD);
  auto warningsEMCAL = genericCTPDigitizer(triggerRecordEMCALPhys, classMaskEMCAL);
  auto warningsPHOSCPV = genericCTPDigitizer(triggerrecordPHOSCPV, classMaskPHOSCPV);

  LOG(info) << "createCTPReadout:# of TRD bogus triggers:" << warningsTRD;
  LOG(info) << "createCTPReadout:# of EMCAL bogus triggers:" << warningsEMCAL;
  LOG(info) << "createCTPReadout:# of PHOS/CPV bogus triggers:" << warningsPHOSCPV;
}

void AODProducerWorkflowDPL::collectBCs(const o2::globaltracking::RecoContainer& data,
                                        const std::vector<o2::InteractionTimeRecord>& mcRecords,
                                        std::map<uint64_t, int>& bcsMap)
{
  const auto& primVertices = data.getPrimaryVertices();
  const auto& fddRecPoints = data.getFDDRecPoints();
  const auto& ft0RecPoints = data.getFT0RecPoints();
  const auto& fv0RecPoints = data.getFV0RecPoints();
  const auto& caloEMCCellsTRGR = data.getEMCALTriggers();
  const auto& caloPHOSCellsTRGR = data.getPHOSTriggers();
  const auto& cpvTRGR = data.getCPVTriggers();
  const auto& ctpDigits = data.getCTPDigits();
  const auto& zdcBCRecData = data.getZDCBCRecData();

  bcsMap[mStartIR.toLong()] = 1; // store the start of TF

  // collecting non-empty BCs and enumerating them
  for (auto& rec : mcRecords) {
    uint64_t globalBC = rec.toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& fddRecPoint : fddRecPoints) {
    uint64_t globalBC = fddRecPoint.getInteractionRecord().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& ft0RecPoint : ft0RecPoints) {
    uint64_t globalBC = ft0RecPoint.getInteractionRecord().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& fv0RecPoint : fv0RecPoints) {
    uint64_t globalBC = fv0RecPoint.getInteractionRecord().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& zdcRecData : zdcBCRecData) {
    uint64_t globalBC = zdcRecData.ir.toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& vertex : primVertices) {
    auto& timeStamp = vertex.getTimeStamp();
    double tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = relativeTime_to_GlobalBC(tsTimeStamp);
    bcsMap[globalBC] = 1;
  }

  for (auto& emcaltrg : caloEMCCellsTRGR) {
    uint64_t globalBC = emcaltrg.getBCData().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& phostrg : caloPHOSCellsTRGR) {
    uint64_t globalBC = phostrg.getBCData().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& cpvtrg : cpvTRGR) {
    uint64_t globalBC = cpvtrg.getBCData().toLong();
    bcsMap[globalBC] = 1;
  }

  for (auto& ctpDigit : ctpDigits) {
    uint64_t globalBC = ctpDigit.intRecord.toLong();
    bcsMap[globalBC] = 1;
  }

  int bcID = 0;
  for (auto& item : bcsMap) {
    item.second = bcID;
    bcID++;
  }
}

uint64_t AODProducerWorkflowDPL::getTFNumber(const o2::InteractionRecord& tfStartIR, int runNumber)
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  o2::ccdb::CcdbApi ccdb_api;
  const std::string rct_path = "RCT/Info/RunInformation/";
  const std::string start_orbit_path = "Trigger/StartOrbit";

  mgr.setURL(o2::base::NameConf::getCCDBServer());
  ccdb_api.init(o2::base::NameConf::getCCDBServer());

  std::map<int, int>* mapStartOrbit = mgr.get<std::map<int, int>>(start_orbit_path);
  int64_t ts = 0;
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers;
  const std::string run_path = Form("%s/%i", rct_path.data(), runNumber);
  headers = ccdb_api.retrieveHeaders(run_path, metadata, -1);
  ts = atol(headers["SOR"].c_str());

  // ccdb returns timestamp in mus
  // mus to ms
  ts = ts / 1000;

  uint32_t initialOrbit = mapStartOrbit->at(runNumber);
  uint16_t firstRecBC = tfStartIR.bc;
  uint32_t firstRecOrbit = tfStartIR.orbit;
  const o2::InteractionRecord firstRec(firstRecBC, firstRecOrbit);
  ts += firstRec.bc2ns() / 1000000;

  return ts;
};

template <typename TracksCursorType, typename TracksCovCursorType>
void AODProducerWorkflowDPL::addToTracksTable(TracksCursorType& tracksCursor, TracksCovCursorType& tracksCovCursor,
                                              const o2::track::TrackParCov& track, int collisionID)
{
  tracksCursor(0,
               collisionID,
               o2::aod::track::TrackIU,
               truncateFloatFraction(track.getX(), mTrackX),
               truncateFloatFraction(track.getAlpha(), mTrackAlpha),
               track.getY(),
               track.getZ(),
               truncateFloatFraction(track.getSnp(), mTrackSnp),
               truncateFloatFraction(track.getTgl(), mTrackTgl),
               truncateFloatFraction(track.getQ2Pt(), mTrack1Pt));
  // trackscov
  float sY = TMath::Sqrt(track.getSigmaY2()), sZ = TMath::Sqrt(track.getSigmaZ2()), sSnp = TMath::Sqrt(track.getSigmaSnp2()),
        sTgl = TMath::Sqrt(track.getSigmaTgl2()), sQ2Pt = TMath::Sqrt(track.getSigma1Pt2());
  tracksCovCursor(0,
                  truncateFloatFraction(sY, mTrackCovDiag),
                  truncateFloatFraction(sZ, mTrackCovDiag),
                  truncateFloatFraction(sSnp, mTrackCovDiag),
                  truncateFloatFraction(sTgl, mTrackCovDiag),
                  truncateFloatFraction(sQ2Pt, mTrackCovDiag),
                  (Char_t)(128. * track.getSigmaZY() / (sZ * sY)),
                  (Char_t)(128. * track.getSigmaSnpY() / (sSnp * sY)),
                  (Char_t)(128. * track.getSigmaSnpZ() / (sSnp * sZ)),
                  (Char_t)(128. * track.getSigmaTglY() / (sTgl * sY)),
                  (Char_t)(128. * track.getSigmaTglZ() / (sTgl * sZ)),
                  (Char_t)(128. * track.getSigmaTglSnp() / (sTgl * sSnp)),
                  (Char_t)(128. * track.getSigma1PtY() / (sQ2Pt * sY)),
                  (Char_t)(128. * track.getSigma1PtZ() / (sQ2Pt * sZ)),
                  (Char_t)(128. * track.getSigma1PtSnp() / (sQ2Pt * sSnp)),
                  (Char_t)(128. * track.getSigma1PtTgl() / (sQ2Pt * sTgl)));
}

template <typename TracksExtraCursorType>
void AODProducerWorkflowDPL::addToTracksExtraTable(TracksExtraCursorType& tracksExtraCursor, TrackExtraInfo& extraInfoHolder)
{
  // extra
  tracksExtraCursor(0,
                    truncateFloatFraction(extraInfoHolder.tpcInnerParam, mTrack1Pt),
                    extraInfoHolder.flags,
                    extraInfoHolder.itsClusterMap,
                    extraInfoHolder.tpcNClsFindable,
                    extraInfoHolder.tpcNClsFindableMinusFound,
                    extraInfoHolder.tpcNClsFindableMinusCrossedRows,
                    extraInfoHolder.tpcNClsShared,
                    extraInfoHolder.trdPattern,
                    truncateFloatFraction(extraInfoHolder.itsChi2NCl, mTrackChi2),
                    truncateFloatFraction(extraInfoHolder.tpcChi2NCl, mTrackChi2),
                    truncateFloatFraction(extraInfoHolder.trdChi2, mTrackChi2),
                    truncateFloatFraction(extraInfoHolder.tofChi2, mTrackChi2),
                    truncateFloatFraction(extraInfoHolder.tpcSignal, mTrackSignal),
                    truncateFloatFraction(extraInfoHolder.trdSignal, mTrackSignal),
                    truncateFloatFraction(extraInfoHolder.length, mTrackSignal),
                    truncateFloatFraction(extraInfoHolder.tofExpMom, mTrack1Pt),
                    truncateFloatFraction(extraInfoHolder.trackEtaEMCAL, mTrackPosEMCAL),
                    truncateFloatFraction(extraInfoHolder.trackPhiEMCAL, mTrackPosEMCAL),
                    truncateFloatFraction(extraInfoHolder.trackTime, mTrackTime),
                    truncateFloatFraction(extraInfoHolder.trackTimeRes, mTrackTimeError));
}

template <typename mftTracksCursorType, typename AmbigMFTTracksCursorType>
void AODProducerWorkflowDPL::addToMFTTracksTable(mftTracksCursorType& mftTracksCursor, AmbigMFTTracksCursorType& ambigMFTTracksCursor,
                                                 GIndex trackID, const o2::globaltracking::RecoContainer& data, int collisionID,
                                                 std::uint64_t collisionBC, const std::map<uint64_t, int>& bcsMap)
{
  // mft tracks
  int bcSlice[2] = {-1, -1};
  const auto& track = data.getMFTTrack(trackID);
  const auto& rof = data.getMFTTracksROFRecords()[mMFTROFs[trackID.getIndex()]];
  float trackTime = rof.getBCData().differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingNS + mMFTROFrameHalfLengthNS;
  float trackTimeRes = mMFTROFrameHalfLengthNS;
  bool needBCSlice = collisionID < 0 || trackID.isAmbiguous();
  std::uint64_t bcOfTimeRef;
  if (needBCSlice) {
    double error = mTimeMarginTrackTime + trackTimeRes;
    bcOfTimeRef = fillBCSlice(bcSlice, trackTime - error, trackTime + error, bcsMap);
  } else {
    bcOfTimeRef = collisionBC - mStartIR.toLong(); // by default (unambiguous) track time is wrt collision BC
  }
  trackTime -= bcOfTimeRef * o2::constants::lhc::LHCBunchSpacingNS;

  mftTracksCursor(0,
                  collisionID,
                  track.getX(),
                  track.getY(),
                  truncateFloatFraction(track.getZ(), mTrackX), // for the forward tracks Z has the same role as X in barrel
                  truncateFloatFraction(track.getPhi(), mTrackAlpha),
                  truncateFloatFraction(track.getTanl(), mTrackTgl),
                  truncateFloatFraction(track.getInvQPt(), mTrack1Pt),
                  track.getNumberOfPoints(),
                  truncateFloatFraction(track.getTrackChi2(), mTrackChi2),
                  truncateFloatFraction(trackTime, mTrackTime),
                  truncateFloatFraction(trackTimeRes, mTrackTimeError));
  if (needBCSlice) {
    ambigMFTTracksCursor(0, mTableTrMFTID, bcSlice);
  }
}

template <typename TracksCursorType, typename TracksCovCursorType, typename TracksExtraCursorType, typename AmbigTracksCursorType,
          typename MFTTracksCursorType, typename AmbigMFTTracksCursorType,
          typename FwdTracksCursorType, typename FwdTracksCovCursorType, typename AmbigFwdTracksCursorType>
void AODProducerWorkflowDPL::fillTrackTablesPerCollision(int collisionID,
                                                         std::uint64_t collisionBC,
                                                         const o2::dataformats::VtxTrackRef& trackRef,
                                                         const gsl::span<const GIndex>& GIndices,
                                                         const o2::globaltracking::RecoContainer& data,
                                                         TracksCursorType& tracksCursor,
                                                         TracksCovCursorType& tracksCovCursor,
                                                         TracksExtraCursorType& tracksExtraCursor,
                                                         AmbigTracksCursorType& ambigTracksCursor,
                                                         MFTTracksCursorType& mftTracksCursor,
                                                         AmbigMFTTracksCursorType& ambigMFTTracksCursor,
                                                         FwdTracksCursorType& fwdTracksCursor,
                                                         FwdTracksCovCursorType& fwdTracksCovCursor,
                                                         AmbigFwdTracksCursorType& ambigFwdTracksCursor,
                                                         const std::map<uint64_t, int>& bcsMap)
{
  for (int src = GIndex::NSources; src--;) {
    if (!GIndex::isTrackSource(src)) {
      continue;
    }
    int start = trackRef.getFirstEntryOfSource(src);
    int end = start + trackRef.getEntriesOfSource(src);
    for (int ti = start; ti < end; ti++) {
      auto& trackIndex = GIndices[ti];
      if (GIndex::includesSource(src, mInputSources)) {
        if (src == GIndex::Source::MFT) {                                                                // MFT tracks are treated separately since they are stored in a different table
          if (trackIndex.isAmbiguous() && mGIDToTableMFTID.find(trackIndex) != mGIDToTableMFTID.end()) { // was it already stored ?
            continue;
          }
          addToMFTTracksTable(mftTracksCursor, ambigMFTTracksCursor, trackIndex, data, collisionID, collisionBC, bcsMap);
          mGIDToTableMFTID.emplace(trackIndex, mTableTrMFTID);
          mTableTrMFTID++;
        } else if (src == GIndex::Source::MCH || src == GIndex::Source::MFTMCH || src == GIndex::Source::MCHMID) { // FwdTracks tracks are treated separately since they are stored in a different table
          if (trackIndex.isAmbiguous() && mGIDToTableFwdID.find(trackIndex) != mGIDToTableFwdID.end()) {           // was it already stored ?
            continue;
          }
          addToFwdTracksTable(fwdTracksCursor, fwdTracksCovCursor, ambigFwdTracksCursor, trackIndex, data, collisionID, collisionBC, bcsMap);
          mGIDToTableFwdID.emplace(trackIndex, mTableTrFwdID);
          mTableTrFwdID++;
        } else {
          // barrel track: normal tracks table
          if (trackIndex.isAmbiguous() && mGIDToTableID.find(trackIndex) != mGIDToTableID.end()) { // was it already stored ?
            continue;
          }
          auto extraInfoHolder = processBarrelTrack(collisionID, collisionBC, trackIndex, data, bcsMap);
          if (extraInfoHolder.trackTimeRes < 0.f) { // failed or rejected?
            LOG(warning) << "Barrel track " << trackIndex << " has no time set, rejection is not expected : time=" << extraInfoHolder.trackTime
                         << " timeErr=" << extraInfoHolder.trackTimeRes << " BCSlice: " << extraInfoHolder.bcSlice[0] << ":" << extraInfoHolder.bcSlice[1];
            continue;
          }
          addToTracksTable(tracksCursor, tracksCovCursor, data.getTrackParam(trackIndex), collisionID);
          addToTracksExtraTable(tracksExtraCursor, extraInfoHolder);
          // collecting table indices of barrel tracks for V0s table
          if (extraInfoHolder.bcSlice[0] >= 0) {
            ambigTracksCursor(0, mTableTrID, extraInfoHolder.bcSlice);
          }
          mGIDToTableID.emplace(trackIndex, mTableTrID);
          mTableTrID++;
        }
      }
    }
  }
}

void AODProducerWorkflowDPL::fillIndexTablesPerCollision(const o2::dataformats::VtxTrackRef& trackRef, const gsl::span<const GIndex>& GIndices, const o2::globaltracking::RecoContainer& data)
{
  const auto& mchmidMatches = data.getMCHMIDMatches();

  for (int src : {GIndex::Source::MCHMID, GIndex::Source::MFTMCH, GIndex::Source::MCH, GIndex::Source::MFT}) {
    int start = trackRef.getFirstEntryOfSource(src);
    int end = start + trackRef.getEntriesOfSource(src);
    for (int ti = start; ti < end; ti++) {
      auto& trackIndex = GIndices[ti];
      if (GIndex::includesSource(src, mInputSources)) {
        if (src == GIndex::Source::MFT) {
          if (trackIndex.isAmbiguous() && mGIDToTableMFTID.find(trackIndex) != mGIDToTableMFTID.end()) {
            continue;
          }
          mGIDToTableMFTID.emplace(trackIndex, mIndexMFTID);
          mIndexTableMFT[trackIndex.getIndex()] = mIndexMFTID;
          mIndexMFTID++;
        } else if (src == GIndex::Source::MCH || src == GIndex::Source::MFTMCH || src == GIndex::Source::MCHMID) {
          if (trackIndex.isAmbiguous() && mGIDToTableFwdID.find(trackIndex) != mGIDToTableFwdID.end()) {
            continue;
          }
          mGIDToTableFwdID.emplace(trackIndex, mIndexFwdID);
          if (src == GIndex::Source::MCH) {
            mIndexTableFwd[trackIndex.getIndex()] = mIndexFwdID;
          } else if (src == GIndex::Source::MCHMID) {
            const auto& mchmidMatch = mchmidMatches[trackIndex.getIndex()];
            const auto mchTrackID = mchmidMatch.getMCHRef().getIndex();
            mIndexTableFwd[mchTrackID] = mIndexFwdID;
          }
          mIndexFwdID++;
        }
      }
    }
  }
}

template <typename FwdTracksCursorType, typename FwdTracksCovCursorType, typename AmbigFwdTracksCursorType>
void AODProducerWorkflowDPL::addToFwdTracksTable(FwdTracksCursorType& fwdTracksCursor, FwdTracksCovCursorType& fwdTracksCovCursor,
                                                 AmbigFwdTracksCursorType& ambigFwdTracksCursor, GIndex trackID,
                                                 const o2::globaltracking::RecoContainer& data, int collisionID, std::uint64_t collisionBC,
                                                 const std::map<uint64_t, int>& bcsMap)
{
  const auto& mchTracks = data.getMCHTracks();
  const auto& midTracks = data.getMIDTracks();
  const auto& mchmidMatches = data.getMCHMIDMatches();
  const auto& mchClusters = data.getMCHTrackClusters();

  FwdTrackInfo fwdInfo;
  FwdTrackCovInfo fwdCovInfo;
  int bcSlice[2] = {-1, -1};

  // helper lambda for mch bitmap -- common for global and standalone tracks
  auto getMCHBitMap = [&](int mchTrackID) {
    if (mchTrackID != -1) { // check matching just in case
      const auto& mchTrack = mchTracks[mchTrackID];
      int first = mchTrack.getFirstClusterIdx();
      int last = mchTrack.getLastClusterIdx();
      for (int i = first; i <= last; i++) { // check chamberIds of all clusters
        const auto& cluster = mchClusters[i];
        int chamberId = cluster.getChamberId();
        fwdInfo.mchBitMap |= 1 << chamberId;
      }
    }
  };

  auto getMIDBitMapBoards = [&](int midTrackID) {
    if (midTrackID != -1) { // check matching just in case
      const auto& midTrack = midTracks[midTrackID];
      fwdInfo.midBitMap = midTrack.getHitMap();
      fwdInfo.midBoards = midTrack.getEfficiencyWord();
    }
  };

  auto extrapMCHTrack = [&](int mchTrackID) {
    const auto& track = mchTracks[mchTrackID];

    // mch standalone tracks extrapolated to vertex
    // compute 3 sets of tracks parameters :
    // - at vertex
    // - at DCA
    // - at the end of the absorber
    // extrapolate to vertex
    float vx = 0, vy = 0, vz = 0;
    if (collisionID >= 0) {
      const auto& v = data.getPrimaryVertex(collisionID);
      vx = v.getX();
      vy = v.getY();
      vz = v.getZ();
    }

    o2::mch::TrackParam trackParamAtVertex(track.getZ(), track.getParameters(), track.getCovariances());
    double errVtx{0.0}; // FIXME: get errors associated with vertex if available
    double errVty{0.0};
    if (!o2::mch::TrackExtrap::extrapToVertex(trackParamAtVertex, vx, vy, vz, errVtx, errVty)) {
      return false;
    }

    // extrapolate to DCA
    o2::mch::TrackParam trackParamAtDCA(track.getZ(), track.getParameters());
    if (!o2::mch::TrackExtrap::extrapToVertexWithoutBranson(trackParamAtDCA, vz)) {
      return false;
    }

    // extrapolate to the end of the absorber
    o2::mch::TrackParam trackParamAtRAbs(track.getZ(), track.getParameters());
    if (!o2::mch::TrackExtrap::extrapToZ(trackParamAtRAbs, -505.)) { // FIXME: replace hardcoded 505
      return false;
    }

    double dcaX = trackParamAtDCA.getNonBendingCoor() - vx;
    double dcaY = trackParamAtDCA.getBendingCoor() - vy;
    double dca = std::sqrt(dcaX * dcaX + dcaY * dcaY);

    double xAbs = trackParamAtRAbs.getNonBendingCoor();
    double yAbs = trackParamAtRAbs.getBendingCoor();

    double px = trackParamAtVertex.px();
    double py = trackParamAtVertex.py();
    double pz = trackParamAtVertex.pz();

    double pt = std::sqrt(px * px + py * py);
    double dphi = std::atan2(py, px);
    double dtanl = pz / pt;
    double dinvqpt = 1.0 / (trackParamAtVertex.getCharge() * pt);
    double dpdca = track.getP() * dca;
    double dchi2 = track.getChi2OverNDF();

    fwdInfo.x = trackParamAtVertex.getNonBendingCoor();
    fwdInfo.y = trackParamAtVertex.getBendingCoor();
    fwdInfo.z = trackParamAtVertex.getZ();
    fwdInfo.rabs = std::sqrt(xAbs * xAbs + yAbs * yAbs);
    fwdInfo.phi = dphi;
    fwdInfo.tanl = dtanl;
    fwdInfo.invqpt = dinvqpt;
    fwdInfo.chi2 = dchi2;
    fwdInfo.pdca = dpdca;
    fwdInfo.nClusters = track.getNClusters();

    fwdCovInfo.sigX = TMath::Sqrt(trackParamAtVertex.getCovariances()(0, 0));
    fwdCovInfo.sigY = TMath::Sqrt(trackParamAtVertex.getCovariances()(1, 1));
    fwdCovInfo.sigPhi = TMath::Sqrt(trackParamAtVertex.getCovariances()(2, 2));
    fwdCovInfo.sigTgl = TMath::Sqrt(trackParamAtVertex.getCovariances()(3, 3));
    fwdCovInfo.sig1Pt = TMath::Sqrt(trackParamAtVertex.getCovariances()(4, 4));
    fwdCovInfo.rhoXY = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 1) / (fwdCovInfo.sigX * fwdCovInfo.sigY));
    fwdCovInfo.rhoPhiX = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 2) / (fwdCovInfo.sigPhi * fwdCovInfo.sigX));
    fwdCovInfo.rhoPhiY = (Char_t)(128. * trackParamAtVertex.getCovariances()(1, 2) / (fwdCovInfo.sigPhi * fwdCovInfo.sigY));
    fwdCovInfo.rhoTglX = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 3) / (fwdCovInfo.sigTgl * fwdCovInfo.sigX));
    fwdCovInfo.rhoTglY = (Char_t)(128. * trackParamAtVertex.getCovariances()(1, 3) / (fwdCovInfo.sigTgl * fwdCovInfo.sigY));
    fwdCovInfo.rhoTglPhi = (Char_t)(128. * trackParamAtVertex.getCovariances()(2, 3) / (fwdCovInfo.sigTgl * fwdCovInfo.sigPhi));
    fwdCovInfo.rho1PtX = (Char_t)(128. * trackParamAtVertex.getCovariances()(0, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigX));
    fwdCovInfo.rho1PtY = (Char_t)(128. * trackParamAtVertex.getCovariances()(1, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigY));
    fwdCovInfo.rho1PtPhi = (Char_t)(128. * trackParamAtVertex.getCovariances()(2, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigPhi));
    fwdCovInfo.rho1PtTgl = (Char_t)(128. * trackParamAtVertex.getCovariances()(3, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigTgl));

    return true;
  };

  if (trackID.getSource() == GIndex::MCH) { // This is an MCH track
    int mchTrackID = trackID.getIndex();
    getMCHBitMap(mchTrackID);
    if (!extrapMCHTrack(mchTrackID)) {
      LOGF(warn, "Unable to extrapolate MCH track with ID %d! Dummy parameters will be used", mchTrackID);
    }
    fwdInfo.trackTypeId = o2::aod::fwdtrack::MCHStandaloneTrack;
    const auto& rof = data.getMCHTracksROFRecords()[mMCHROFs[mchTrackID]];
    auto time = rof.getTimeMUS(mStartIR).first;
    fwdInfo.trackTime = time.getTimeStamp() * 1.e3;
    fwdInfo.trackTimeRes = time.getTimeStampError() * 1.e3;
  } else if (trackID.getSource() == GIndex::MCHMID) { // This is an MCH-MID track
    fwdInfo.trackTypeId = o2::aod::fwdtrack::MuonStandaloneTrack;
    auto mchmidMatch = mchmidMatches[trackID.getIndex()];
    auto mchTrackID = mchmidMatch.getMCHRef().getIndex();
    if (!extrapMCHTrack(mchTrackID)) {
      LOGF(warn, "Unable to extrapolate MCH track with ID %d! Dummy parameters will be used", mchTrackID);
    }
    auto midTrackID = mchmidMatch.getMIDRef().getIndex();
    fwdInfo.chi2matchmchmid = mchmidMatch.getMatchChi2OverNDF();
    getMCHBitMap(mchTrackID);
    getMIDBitMapBoards(midTrackID);
    auto time = mchmidMatch.getTimeMUS(mStartIR).first;
    fwdInfo.trackTime = time.getTimeStamp() * 1.e3;
    fwdInfo.trackTimeRes = time.getTimeStampError() * 1.e3;
  } else { // This is a GlobalMuonTrack or a GlobalForwardTrack
    const auto& track = data.getGlobalFwdTrack(trackID);
    if (!extrapMCHTrack(track.getMCHTrackID())) {
      LOGF(warn, "Unable to extrapolate MCH track with ID %d! Dummy parameters will be used", track.getMCHTrackID());
    }
    fwdInfo.x = track.getX();
    fwdInfo.y = track.getY();
    fwdInfo.z = track.getZ();
    fwdInfo.phi = track.getPhi();
    fwdInfo.tanl = track.getTanl();
    fwdInfo.invqpt = track.getInvQPt();
    fwdInfo.chi2 = track.getTrackChi2();
    // fwdInfo.nClusters = track.getNumberOfPoints();
    fwdInfo.chi2matchmchmid = track.getMIDMatchingChi2();
    fwdInfo.chi2matchmchmft = track.getMFTMCHMatchingChi2();
    fwdInfo.matchscoremchmft = track.getMFTMCHMatchingScore();
    fwdInfo.matchmfttrackid = mIndexTableMFT[track.getMFTTrackID()];
    fwdInfo.matchmchtrackid = mIndexTableFwd[track.getMCHTrackID()];
    fwdInfo.trackTime = track.getTimeMUS().getTimeStamp() * 1.e3;
    fwdInfo.trackTimeRes = track.getTimeMUS().getTimeStampError() * 1.e3;

    getMCHBitMap(track.getMCHTrackID());
    getMIDBitMapBoards(track.getMIDTrackID());

    fwdCovInfo.sigX = TMath::Sqrt(track.getCovariances()(0, 0));
    fwdCovInfo.sigY = TMath::Sqrt(track.getCovariances()(1, 1));
    fwdCovInfo.sigPhi = TMath::Sqrt(track.getCovariances()(2, 2));
    fwdCovInfo.sigTgl = TMath::Sqrt(track.getCovariances()(3, 3));
    fwdCovInfo.sig1Pt = TMath::Sqrt(track.getCovariances()(4, 4));
    fwdCovInfo.rhoXY = (Char_t)(128. * track.getCovariances()(0, 1) / (fwdCovInfo.sigX * fwdCovInfo.sigY));
    fwdCovInfo.rhoPhiX = (Char_t)(128. * track.getCovariances()(0, 2) / (fwdCovInfo.sigPhi * fwdCovInfo.sigX));
    fwdCovInfo.rhoPhiY = (Char_t)(128. * track.getCovariances()(1, 2) / (fwdCovInfo.sigPhi * fwdCovInfo.sigY));
    fwdCovInfo.rhoTglX = (Char_t)(128. * track.getCovariances()(0, 3) / (fwdCovInfo.sigTgl * fwdCovInfo.sigX));
    fwdCovInfo.rhoTglY = (Char_t)(128. * track.getCovariances()(1, 3) / (fwdCovInfo.sigTgl * fwdCovInfo.sigY));
    fwdCovInfo.rhoTglPhi = (Char_t)(128. * track.getCovariances()(2, 3) / (fwdCovInfo.sigTgl * fwdCovInfo.sigPhi));
    fwdCovInfo.rho1PtX = (Char_t)(128. * track.getCovariances()(0, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigX));
    fwdCovInfo.rho1PtY = (Char_t)(128. * track.getCovariances()(1, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigY));
    fwdCovInfo.rho1PtPhi = (Char_t)(128. * track.getCovariances()(2, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigPhi));
    fwdCovInfo.rho1PtTgl = (Char_t)(128. * track.getCovariances()(3, 4) / (fwdCovInfo.sig1Pt * fwdCovInfo.sigTgl));

    fwdInfo.trackTypeId = (fwdInfo.chi2matchmchmid >= 0) ? o2::aod::fwdtrack::GlobalMuonTrack : o2::aod::fwdtrack::GlobalForwardTrack;
  }

  std::uint64_t bcOfTimeRef;
  bool needBCSlice = trackID.isAmbiguous() || collisionID < 0;
  if (needBCSlice) { // need to store BC slice
    float err = mTimeMarginTrackTime + fwdInfo.trackTimeRes;
    bcOfTimeRef = fillBCSlice(bcSlice, fwdInfo.trackTime - err, fwdInfo.trackTime + err, bcsMap);
  } else {
    bcOfTimeRef = collisionBC - mStartIR.toLong(); // by default (unambiguous) track time is wrt collision BC
  }
  fwdInfo.trackTime -= bcOfTimeRef * o2::constants::lhc::LHCBunchSpacingNS;

  fwdTracksCursor(0,
                  collisionID,
                  fwdInfo.trackTypeId,
                  fwdInfo.x,
                  fwdInfo.y,
                  truncateFloatFraction(fwdInfo.z, mTrackX), // for the forward tracks Z has the same role as X in the barrel
                  truncateFloatFraction(fwdInfo.phi, mTrackAlpha),
                  truncateFloatFraction(fwdInfo.tanl, mTrackTgl),
                  truncateFloatFraction(fwdInfo.invqpt, mTrack1Pt),
                  fwdInfo.nClusters,
                  truncateFloatFraction(fwdInfo.pdca, mTrackX),
                  truncateFloatFraction(fwdInfo.rabs, mTrackX),
                  truncateFloatFraction(fwdInfo.chi2, mTrackChi2),
                  truncateFloatFraction(fwdInfo.chi2matchmchmid, mTrackChi2),
                  truncateFloatFraction(fwdInfo.chi2matchmchmft, mTrackChi2),
                  truncateFloatFraction(fwdInfo.matchscoremchmft, mTrackChi2),
                  fwdInfo.matchmfttrackid,
                  fwdInfo.matchmchtrackid,
                  fwdInfo.mchBitMap,
                  fwdInfo.midBitMap,
                  fwdInfo.midBoards,
                  truncateFloatFraction(fwdInfo.trackTime, mTrackTime),
                  truncateFloatFraction(fwdInfo.trackTimeRes, mTrackTimeError));

  fwdTracksCovCursor(0,
                     truncateFloatFraction(fwdCovInfo.sigX, mTrackCovDiag),
                     truncateFloatFraction(fwdCovInfo.sigY, mTrackCovDiag),
                     truncateFloatFraction(fwdCovInfo.sigPhi, mTrackCovDiag),
                     truncateFloatFraction(fwdCovInfo.sigTgl, mTrackCovDiag),
                     truncateFloatFraction(fwdCovInfo.sig1Pt, mTrackCovDiag),
                     fwdCovInfo.rhoXY,
                     fwdCovInfo.rhoPhiX,
                     fwdCovInfo.rhoPhiY,
                     fwdCovInfo.rhoTglX,
                     fwdCovInfo.rhoTglY,
                     fwdCovInfo.rhoTglPhi,
                     fwdCovInfo.rho1PtX,
                     fwdCovInfo.rho1PtY,
                     fwdCovInfo.rho1PtPhi,
                     fwdCovInfo.rho1PtTgl);

  if (needBCSlice) {
    ambigFwdTracksCursor(0, mTableTrFwdID, bcSlice);
  }
}

void dimensionMCKeepStore(std::vector<std::vector<std::unordered_map<int, int>*>>& store, int Nsources, int NEvents)
{
  store.resize(Nsources);
  for (int s = 0; s < Nsources; ++s) {
    store[s].resize(NEvents, nullptr);
  }
}

void clearMCKeepStore(std::vector<std::vector<std::unordered_map<int, int>*>>& store)
{
  for (int s = 0; s < store.size(); ++s) {
    for (int e = 0; e < store[s].size(); ++e) {
      if (store[s][e]) {
        store[s][e]->clear();
      }
    }
  }
}

// helper function to add a particle/track to the MC keep store
void keepMCParticle(std::vector<std::vector<std::unordered_map<int, int>*>>& store, int source, int event, int track, int value = 1)
{
  if (!store[source][event]) {
    store[source][event] = new std::unordered_map<int, int>;
  }
  (*store[source][event])[track] = value;
}

template <typename MCParticlesCursorType>
void AODProducerWorkflowDPL::fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                                                  const MCParticlesCursorType& mcParticlesCursor,
                                                  const gsl::span<const o2::dataformats::VtxTrackRef>& primVer2TRefs,
                                                  const gsl::span<const GIndex>& GIndices,
                                                  const o2::globaltracking::RecoContainer& data,
                                                  const std::vector<std::vector<int>>& mcColToEvSrc)
{
  int NSources = 0;
  int NEvents = 0;
  for (auto& p : mcColToEvSrc) {
    NSources = std::max(p[1], NSources);
    NEvents = std::max(p[2], NEvents);
  }
  NSources++; // 0 - indexed
  NEvents++;
  LOG(info) << " number of events " << NEvents;
  LOG(info) << " number of sources " << NSources;
  dimensionMCKeepStore(mToStore, NSources, NEvents);

  std::vector<int> particleIDsToKeep;

  auto markMCTrackForSrc = [&](std::array<GID, GID::NSources>& contributorsGID, uint8_t src) {
    auto mcLabel = data.getTrackMCLabel(contributorsGID[src]);
    if (!mcLabel.isValid()) {
      return;
    }
    keepMCParticle(mToStore, mcLabel.getSourceID(), mcLabel.getEventID(), mcLabel.getTrackID());
  };

  // mark reconstructed MC particles to store them into the table
  for (auto& trackRef : primVer2TRefs) {
    for (int src = GIndex::NSources; src--;) {
      int start = trackRef.getFirstEntryOfSource(src);
      int end = start + trackRef.getEntriesOfSource(src);
      for (int ti = start; ti < end; ti++) {
        auto& trackIndex = GIndices[ti];
        if (GIndex::includesSource(src, mInputSources)) {
          auto mcTruth = data.getTrackMCLabel(trackIndex);
          if (!mcTruth.isValid()) {
            continue;
          }
          keepMCParticle(mToStore, mcTruth.getSourceID(), mcTruth.getEventID(), mcTruth.getTrackID());
          // treating contributors of global tracks
          auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
          if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
            markMCTrackForSrc(contributorsGID, GIndex::Source::TPC);
          }
          if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
            markMCTrackForSrc(contributorsGID, GIndex::Source::ITS);
          }
          if (contributorsGID[GIndex::Source::TOF].isIndexSet()) {
            const auto& labelsTOF = data.getTOFClustersMCLabels()->getLabels(contributorsGID[GIndex::Source::TOF]);
            for (auto& mcLabel : labelsTOF) {
              if (!mcLabel.isValid()) {
                continue;
              }
              keepMCParticle(mToStore, mcLabel.getSourceID(), mcLabel.getEventID(), mcLabel.getTrackID());
            }
          }
        }
      }
    }
  }
  int tableIndex = 1;
  for (auto& colInfo : mcColToEvSrc) { // loop over "<eventID, sourceID> <-> combined MC col. ID" key pairs
    int event = colInfo[2];
    int source = colInfo[1];
    int mcColId = colInfo[0];
    std::vector<MCTrack> const& mcParticles = mcReader.getTracks(source, event);
    // mark tracks to be stored per event
    // loop over stack of MC particles from end to beginning: daughters are stored after mothers
    if (mRecoOnly) {
      for (int particle = mcParticles.size() - 1; particle >= 0; particle--) {
        // we store all primary particles == particles put by generator
        if (mcParticles[particle].isPrimary()) {
          keepMCParticle(mToStore, source, event, particle);
        } else if (o2::mcutils::MCTrackNavigator::isPhysicalPrimary(mcParticles[particle], mcParticles)) {
          keepMCParticle(mToStore, source, event, particle);
        } else if (o2::mcutils::MCTrackNavigator::isKeepPhysics(mcParticles[particle], mcParticles)) {
          keepMCParticle(mToStore, source, event, particle);
        }

        // skip treatment if this particle has not been marked during reconstruction
        // or based on criteria just above
        if (mToStore[source][event] && mToStore[source][event]->find(particle) == mToStore[source][event]->end()) {
          continue;
        }

        int mother0 = mcParticles[particle].getMotherTrackId();
        // we store mothers and daughters of particles that are reconstructed
        if (mother0 != -1) {
          keepMCParticle(mToStore, source, event, mother0);
        }
        int mother1 = mcParticles[particle].getSecondMotherTrackId();
        if (mother1 != -1) {
          keepMCParticle(mToStore, source, event, mother1);
        }
        int daughter0 = mcParticles[particle].getFirstDaughterTrackId();
        if (daughter0 != -1) {
          keepMCParticle(mToStore, source, event, daughter0);
        }
        int daughterL = mcParticles[particle].getLastDaughterTrackId();
        if (daughterL != -1) {
          keepMCParticle(mToStore, source, event, daughterL);
        }
      }
      LOG(info) << "The fraction of MC particles kept is " << mToStore[source][event]->size() / (1. * mcParticles.size()) << " for source " << source << " and event " << event;

      particleIDsToKeep.clear();
      for (auto& p : *mToStore[source][event]) {
        particleIDsToKeep.push_back(p.first);
      }
      std::sort(particleIDsToKeep.begin(), particleIDsToKeep.end());
      for (auto pid : particleIDsToKeep) {
        (*mToStore[source][event])[pid] = tableIndex - 1;
        tableIndex++;
      }
    } else {
      // if all mc particles are stored, all mc particles will be enumerated
      particleIDsToKeep.clear();
      for (int particle = 0; particle < mcParticles.size(); particle++) {
        keepMCParticle(mToStore, source, event, particle, tableIndex - 1);
        tableIndex++;
        particleIDsToKeep.push_back(particle);
      }
    }

    // second part: fill survived mc tracks into the AOD table
    for (auto particle : particleIDsToKeep) {
      int statusCode = 0;
      uint8_t flags = 0;
      if (!mcParticles[particle].isPrimary()) {
        flags |= o2::aod::mcparticle::enums::ProducedByTransport; // mark as produced by transport
        statusCode = mcParticles[particle].getProcess();
      } else {
        statusCode = mcParticles[particle].getStatusCode().fullEncoding;
      }
      if (source == 0) {
        flags |= o2::aod::mcparticle::enums::FromBackgroundEvent; // mark as particle from background event
      }
      if (o2::mcutils::MCTrackNavigator::isPhysicalPrimary(mcParticles[particle], mcParticles)) {
        flags |= o2::aod::mcparticle::enums::PhysicalPrimary; // mark as physical primary
      }
      float weight = mcParticles[particle].getWeight();
      std::vector<int> mothers;
      int mcMother0 = mcParticles[particle].getMotherTrackId();
      auto item = mToStore[source][event]->find(mcMother0);
      if (item != mToStore[source][event]->end()) {
        mothers.push_back(item->second);
      }
      int mcMother1 = mcParticles[particle].getSecondMotherTrackId();
      item = mToStore[source][event]->find(mcMother1);
      if (item != mToStore[source][event]->end()) {
        mothers.push_back(item->second);
      }
      int daughters[2] = {-1, -1}; // slice
      int mcDaughter0 = mcParticles[particle].getFirstDaughterTrackId();
      item = mToStore[source][event]->find(mcDaughter0);
      if (item != mToStore[source][event]->end()) {
        daughters[0] = item->second;
      }
      int mcDaughterL = mcParticles[particle].getLastDaughterTrackId();
      item = mToStore[source][event]->find(mcDaughterL);
      if (item != mToStore[source][event]->end()) {
        daughters[1] = item->second;
        if (daughters[0] < 0) {
          LOG(error) << "AOD problematic daughter case observed";
          daughters[0] = daughters[1]; /// Treat the case of first negative label (pruned in the kinematics)
        }
      } else {
        daughters[1] = daughters[0];
      }
      if (daughters[0] > daughters[1]) {
        std::swap(daughters[0], daughters[1]);
      }
      auto pX = (float)mcParticles[particle].Px();
      auto pY = (float)mcParticles[particle].Py();
      auto pZ = (float)mcParticles[particle].Pz();
      auto energy = (float)mcParticles[particle].GetEnergy();
      mcParticlesCursor(0,
                        mcColId,
                        mcParticles[particle].GetPdgCode(),
                        statusCode,
                        flags,
                        mothers,
                        daughters,
                        truncateFloatFraction(weight, mMcParticleW),
                        truncateFloatFraction(pX, mMcParticleMom),
                        truncateFloatFraction(pY, mMcParticleMom),
                        truncateFloatFraction(pZ, mMcParticleMom),
                        truncateFloatFraction(energy, mMcParticleMom),
                        truncateFloatFraction((float)mcParticles[particle].Vx(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].Vy(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].Vz(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].T(), mMcParticlePos));
    }
    mcReader.releaseTracksForSourceAndEvent(source, event);
  }
}

template <typename MCTrackLabelCursorType, typename MCMFTTrackLabelCursorType, typename MCFwdTrackLabelCursorType>
void AODProducerWorkflowDPL::fillMCTrackLabelsTable(const MCTrackLabelCursorType& mcTrackLabelCursor,
                                                    const MCMFTTrackLabelCursorType& mcMFTTrackLabelCursor,
                                                    const MCFwdTrackLabelCursorType& mcFwdTrackLabelCursor,
                                                    const o2::dataformats::VtxTrackRef& trackRef,
                                                    const gsl::span<const GIndex>& primVerGIs,
                                                    const o2::globaltracking::RecoContainer& data)
{
  // labelMask (temporary) usage:
  //   bit 13 -- ITS/TPC or TPC/TOF labels are not equal
  //   bit 14 -- isNoise() == true
  //   bit 15 -- isFake() == true
  // labelID = -1 -- label is not set

  for (int src = GIndex::NSources; src--;) {
    int start = trackRef.getFirstEntryOfSource(src);
    int end = start + trackRef.getEntriesOfSource(src);
    for (int ti = start; ti < end; ti++) {
      const auto trackIndex = primVerGIs[ti];

      // check if the label was already stored (or the track was rejected for some reason in the fillTrackTablesPerCollision)
      auto needToStore = [trackIndex](std::unordered_map<GIndex, int>& mp) {
        auto entry = mp.find(trackIndex);
        if (entry == mp.end() || entry->second == -1) {
          return false;
        }
        entry->second = -1;
        return true;
      };

      if (GIndex::includesSource(src, mInputSources)) {
        auto mcTruth = data.getTrackMCLabel(trackIndex);
        MCLabels labelHolder;
        if ((src == GIndex::Source::MFT) || (src == GIndex::Source::MFTMCH) || (src == GIndex::Source::MCH) || (src == GIndex::Source::MCHMID)) { // treating mft and fwd labels separately
          if (!needToStore(src == GIndex::Source::MFT ? mGIDToTableMFTID : mGIDToTableFwdID)) {
            continue;
          }
          if (mcTruth.isValid()) { // if not set, -1 will be stored
            labelHolder.labelID = (*mToStore[mcTruth.getSourceID()][mcTruth.getEventID()])[mcTruth.getTrackID()];
          }
          if (mcTruth.isFake()) {
            labelHolder.fwdLabelMask |= (0x1 << 7);
          }
          if (mcTruth.isNoise()) {
            labelHolder.fwdLabelMask |= (0x1 << 6);
          }
          if (src == GIndex::Source::MFT) {
            mcMFTTrackLabelCursor(0,
                                  labelHolder.labelID,
                                  labelHolder.fwdLabelMask);
          } else {
            mcFwdTrackLabelCursor(0,
                                  labelHolder.labelID,
                                  labelHolder.fwdLabelMask);
          }
        } else {
          if (!needToStore(mGIDToTableID)) {
            continue;
          }
          if (mcTruth.isValid()) { // if not set, -1 will be stored
            labelHolder.labelID = (*mToStore[mcTruth.getSourceID()][mcTruth.getEventID()])[mcTruth.getTrackID()];
          }
          // treating possible mismatches and fakes for global tracks
          auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
          bool isSetTPC = contributorsGID[GIndex::Source::TPC].isIndexSet();
          bool isSetITS = contributorsGID[GIndex::Source::ITS].isIndexSet();
          bool isSetTOF = contributorsGID[GIndex::Source::TOF].isIndexSet();
          bool isTOFFake = true;
          if (isSetTPC && (isSetITS || isSetTOF)) {
            auto mcTruthTPC = data.getTrackMCLabel(contributorsGID[GIndex::Source::TPC]);
            if (mcTruthTPC.isValid()) {
              labelHolder.labelTPC = (*mToStore[mcTruthTPC.getSourceID()][mcTruthTPC.getEventID()])[mcTruthTPC.getTrackID()];
              labelHolder.labelID = labelHolder.labelTPC;
            }
            if (isSetITS) {
              auto mcTruthITS = data.getTrackMCLabel(contributorsGID[GIndex::Source::ITS]);
              if (mcTruthITS.isValid()) {
                labelHolder.labelITS = (*mToStore[mcTruthITS.getSourceID()][mcTruthITS.getEventID()])[mcTruthITS.getTrackID()];
              }
              if (labelHolder.labelITS != labelHolder.labelTPC) {
                LOG(debug) << "ITS-TPC MCTruth: labelIDs do not match at " << trackIndex.getIndex() << ", src = " << src;
                labelHolder.labelMask |= (0x1 << 13);
              }
            }
            if (isSetTOF) {
              const auto& labelsTOF = data.getTOFClustersMCLabels()->getLabels(contributorsGID[GIndex::Source::TOF]);
              for (auto& mcLabel : labelsTOF) {
                if (!mcLabel.isValid()) {
                  continue;
                }
                if (mcLabel == labelHolder.labelTPC) {
                  isTOFFake = false;
                  break;
                }
              }
            }
          }
          if (mcTruth.isFake() || (isSetTOF && isTOFFake)) {
            labelHolder.labelMask |= (0x1 << 15);
          }
          if (mcTruth.isNoise()) {
            labelHolder.labelMask |= (0x1 << 14);
          }
          mcTrackLabelCursor(0,
                             labelHolder.labelID,
                             labelHolder.labelMask);
        }
      }
    }
  }
}

template <typename V0CursorType, typename CascadeCursorType, typename Decay3BodyCursorType>
void AODProducerWorkflowDPL::fillSecondaryVertices(const o2::globaltracking::RecoContainer& recoData, V0CursorType& v0Cursor, CascadeCursorType& cascadeCursor, Decay3BodyCursorType& decay3BodyCursor)
{

  auto v0s = recoData.getV0s();
  auto cascades = recoData.getCascades();
  auto decays3Body = recoData.getDecays3Body();

  // filling v0s table
  for (size_t iv0 = 0; iv0 < v0s.size(); iv0++) {
    const auto& v0 = v0s[iv0];
    auto trPosID = v0.getProngID(0);
    auto trNegID = v0.getProngID(1);
    int posTableIdx = -1, negTableIdx = -1, collID = -1;
    auto item = mGIDToTableID.find(trPosID);
    if (item != mGIDToTableID.end()) {
      posTableIdx = item->second;
    } else {
      LOG(warn) << "Could not find a positive track index for prong ID " << trPosID;
    }
    item = mGIDToTableID.find(trNegID);
    if (item != mGIDToTableID.end()) {
      negTableIdx = item->second;
    } else {
      LOG(warn) << "Could not find a negative track index for prong ID " << trNegID;
    }
    auto itemV = mVtxToTableCollID.find(v0.getVertexID());
    if (itemV == mVtxToTableCollID.end()) {
      LOG(warn) << "Could not find V0 collisionID for the vertex ID " << v0.getVertexID();
    } else {
      collID = itemV->second;
    }
    if (posTableIdx != -1 and negTableIdx != -1 and collID != -1) {
      v0Cursor(0, collID, posTableIdx, negTableIdx);
      mV0ToTableID[int(iv0)] = mTableV0ID++;
    }
  }

  // filling cascades table
  for (auto& cascade : cascades) {
    auto itemV0 = mV0ToTableID.find(cascade.getV0ID());
    if (itemV0 == mV0ToTableID.end()) {
      continue;
    }
    int v0tableID = itemV0->second, bachTableIdx = -1, collID = -1;
    auto bachelorID = cascade.getBachelorID();
    auto item = mGIDToTableID.find(bachelorID);
    if (item != mGIDToTableID.end()) {
      bachTableIdx = item->second;
    } else {
      LOG(warn) << "Could not find a bachelor track index";
      continue;
    }
    auto itemV = mVtxToTableCollID.find(cascade.getVertexID());
    if (itemV != mVtxToTableCollID.end()) {
      collID = itemV->second;
    } else {
      LOG(warn) << "Could not find cascade collisionID for the vertex ID " << cascade.getVertexID();
      continue;
    }
    cascadeCursor(0, collID, v0tableID, bachTableIdx);
  }

  // filling 3 body decays table
  for (size_t i3Body = 0; i3Body < decays3Body.size(); i3Body++) {
    const auto& decay3Body = decays3Body[i3Body];
    GIndex trIDs[3]{
      decay3Body.getProngID(0),
      decay3Body.getProngID(1),
      decay3Body.getProngID(2)};
    int tableIdx[3]{-1, -1, -1}, collID = -1;
    bool missing{false};
    for (int i{0}; i < 3; ++i) {
      auto item = mGIDToTableID.find(trIDs[i]);
      if (item != mGIDToTableID.end()) {
        tableIdx[i] = item->second;
      } else {
        LOG(warn) << fmt::format("Could not find a track index for prong ID {}", trIDs[i]);
        missing = true;
      }
    }
    auto itemV = mVtxToTableCollID.find(decay3Body.getVertexID());
    if (itemV == mVtxToTableCollID.end()) {
      LOG(warn) << "Could not find 3 body collisionID for the vertex ID " << decay3Body.getVertexID();
      missing = true;
    } else {
      collID = itemV->second;
    }
    if (missing) {
      continue;
    }
    decay3BodyCursor(0, collID, tableIdx[0], tableIdx[1], tableIdx[2]);
  }
}

void AODProducerWorkflowDPL::countTPCClusters(const o2::globaltracking::RecoContainer& data)
{
  const auto& tpcTracks = data.getTPCTracks();
  const auto& tpcClusRefs = data.getTPCTracksClusterRefs();
  const auto& tpcClusShMap = data.clusterShMapTPC;
  const auto& tpcClusAcc = data.getTPCClusters();
  constexpr int maxRows = 152;
  constexpr int neighbour = 2;
  int ntr = tpcTracks.size();
  mTPCCounters.clear();
  mTPCCounters.resize(ntr);
#ifdef WITH_OPENMP
  int ngroup = std::min(50, std::max(1, ntr / mNThreads));
#pragma omp parallel for schedule(dynamic, ngroup) num_threads(mNThreads)
#endif
  for (int itr = 0; itr < ntr; itr++) {
    std::array<bool, maxRows> clMap{}, shMap{};
    uint8_t sectorIndex, rowIndex;
    uint32_t clusterIndex;
    auto& counters = mTPCCounters[itr];
    const auto& track = tpcTracks[itr];
    for (int i = 0; i < track.getNClusterReferences(); i++) {
      o2::tpc::TrackTPC::getClusterReference(tpcClusRefs, i, sectorIndex, rowIndex, clusterIndex, track.getClusterRef());
      unsigned int absoluteIndex = tpcClusAcc.clusterOffset[sectorIndex][rowIndex] + clusterIndex;
      clMap[rowIndex] = true;
      if (tpcClusShMap[absoluteIndex] & GPUCA_NAMESPACE::gpu::GPUTPCGMMergedTrackHit::flagShared) {
        if (!shMap[rowIndex]) {
          counters.shared++;
        }
        shMap[rowIndex] = true;
      }
    }
    int last = -1;
    for (int i = 0; i < maxRows; i++) {
      if (clMap[i]) {
        counters.crossed++;
        counters.found++;
        last = i;
      } else if ((i - last) <= neighbour) {
        counters.crossed++;
      } else {
        int lim = std::min(i + 1 + neighbour, maxRows);
        for (int j = i + 1; j < lim; j++) {
          if (clMap[j]) {
            counters.crossed++;
          }
        }
      }
    }
  }
}

uint8_t AODProducerWorkflowDPL::getTRDPattern(const o2::trd::TrackTRD& track)
{
  uint8_t pattern = 0;
  for (int il = o2::trd::TrackTRD::EGPUTRDTrack::kNLayers - 1; il >= 0; il--) {
    if (track.getTrackletIndex(il) != -1) {
      pattern |= 0x1 << il;
    }
  }
  if (track.getHasNeighbor()) {
    pattern |= 0x1 << 6;
  }
  if (track.getHasPadrowCrossing()) {
    pattern |= 0x1 << 7;
  }
  return pattern;
}

template <typename TCaloHandler, typename TCaloCursor, typename TCaloTRGCursor>
void AODProducerWorkflowDPL::addToCaloTable(const TCaloHandler& caloHandler, const TCaloCursor& caloCellCursor, const TCaloTRGCursor& caloTRGCursor,
                                            int eventID, int bcID, int8_t caloType)
{
  auto inputEvent = caloHandler.buildEvent(eventID);
  auto cellsInEvent = inputEvent.mCells; // get cells belonging to current event
  for (auto& cell : cellsInEvent) {      // loop over all cells in collision
    caloCellCursor(0,
                   bcID,
                   CellHelper::getCellNumber(cell),
                   truncateFloatFraction(CellHelper::getAmplitude(cell), mCaloAmp),
                   truncateFloatFraction(CellHelper::getTimeStamp(cell), mCaloTime),
                   cell.getType(),
                   caloType); // 1 = emcal, -1 = undefined, 0 = phos

    // todo: fix dummy values in CellHelper when it is clear what is filled for trigger information
    if (CellHelper::isTRU(cell)) { // Write only trigger cells into this table
      caloTRGCursor(0,
                    bcID,
                    CellHelper::getFastOrAbsID(cell),
                    CellHelper::getLnAmplitude(cell),
                    CellHelper::getTriggerBits(cell),
                    caloType);
    }
  }
}

// fill calo related tables (cells and calotrigger table)
template <typename TCaloCursor, typename TCaloTRGCursor>
void AODProducerWorkflowDPL::fillCaloTable(const TCaloCursor& caloCellCursor, const TCaloTRGCursor& caloTRGCursor,
                                           const std::map<uint64_t, int>& bcsMap,
                                           const o2::globaltracking::RecoContainer& data)
{
  // get calo information
  auto caloEMCCells = data.getEMCALCells();
  auto caloEMCCellsTRGR = data.getEMCALTriggers();

  auto caloPHOSCells = data.getPHOSCells();
  auto caloPHOSCellsTRGR = data.getPHOSTriggers();

  if (!mInputSources[GIndex::PHS]) {
    caloPHOSCells = {};
    caloPHOSCellsTRGR = {};
  }

  if (!mInputSources[GIndex::EMC]) {
    caloEMCCells = {};
    caloEMCCellsTRGR = {};
  }

  o2::emcal::EventHandler<o2::emcal::Cell> emcEventHandler;
  o2::phos::EventHandler<o2::phos::Cell> phsEventHandler;

  // get cell belonging to an eveffillnt instead of timeframe
  emcEventHandler.reset();
  emcEventHandler.setCellData(caloEMCCells, caloEMCCellsTRGR);

  phsEventHandler.reset();
  phsEventHandler.setCellData(caloPHOSCells, caloPHOSCellsTRGR);

  int emcNEvents = emcEventHandler.getNumberOfEvents();
  int phsNEvents = phsEventHandler.getNumberOfEvents();

  std::vector<std::tuple<uint64_t, int8_t, int>> caloEvents; // <bc, caloType, eventID>

  caloEvents.reserve(emcNEvents + phsNEvents);

  for (int iev = 0; iev < emcNEvents; ++iev) {
    uint64_t bc = emcEventHandler.getInteractionRecordForEvent(iev).toLong();
    caloEvents.emplace_back(std::make_tuple(bc, 1, iev));
  }

  for (int iev = 0; iev < phsNEvents; ++iev) {
    uint64_t bc = phsEventHandler.getInteractionRecordForEvent(iev).toLong();
    caloEvents.emplace_back(std::make_tuple(bc, 0, iev));
  }

  std::sort(caloEvents.begin(), caloEvents.end(),
            [](const auto& left, const auto& right) { return std::get<0>(left) < std::get<0>(right); });

  // loop over events
  for (int i = 0; i < emcNEvents + phsNEvents; ++i) {
    uint64_t globalBC = std::get<0>(caloEvents[i]);
    int8_t caloType = std::get<1>(caloEvents[i]);
    int eventID = std::get<2>(caloEvents[i]);
    auto item = bcsMap.find(globalBC);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(warn) << "Error: could not find a corresponding BC ID for a calo point; globalBC = " << globalBC << ", caloType = " << (int)caloType;
    }
    if (caloType == 0) { // phs
      addToCaloTable(phsEventHandler, caloCellCursor, caloTRGCursor, eventID, bcID, caloType);
    }
    if (caloType == 1) { // emc
      addToCaloTable(emcEventHandler, caloCellCursor, caloTRGCursor, eventID, bcID, caloType);
    }
  }

  caloEvents.clear();
}

void AODProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mLPMProdTag = ic.options().get<string>("lpmp-prod-tag");
  mAnchorPass = ic.options().get<string>("anchor-pass");
  mAnchorProd = ic.options().get<string>("anchor-prod");
  mRecoPass = ic.options().get<string>("reco-pass");
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mRecoOnly = ic.options().get<int>("reco-mctracks-only");
  mTruncate = ic.options().get<int>("enable-truncation");
  mRunNumber = ic.options().get<int>("run-number");
  mCTPReadout = ic.options().get<int>("ctpreadout-create");
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));
#ifdef WITH_OPENMP
  LOGP(info, "Multi-threaded parts will run with {} OpenMP threads", mNThreads);
#else
  mNThreads = 1;
  LOG(info) << "OpenMP is disabled";
#endif
  if (mTFNumber == -1L) {
    LOG(info) << "TFNumber will be obtained from CCDB";
  }
  if (mRunNumber == -1L) {
    LOG(info) << "The Run number will be obtained from DPL headers";
  }

  // set no truncation if selected by user
  if (mTruncate != 1) {
    LOG(info) << "Truncation is not used!";
    mCollisionPosition = 0xFFFFFFFF;
    mCollisionPositionCov = 0xFFFFFFFF;
    mTrackX = 0xFFFFFFFF;
    mTrackAlpha = 0xFFFFFFFF;
    mTrackSnp = 0xFFFFFFFF;
    mTrackTgl = 0xFFFFFFFF;
    mTrack1Pt = 0xFFFFFFFF;
    mTrackChi2 = 0xFFFFFFFF;
    mTrackCovDiag = 0xFFFFFFFF;
    mTrackCovOffDiag = 0xFFFFFFFF;
    mTrackSignal = 0xFFFFFFFF;
    mTrackTime = 0xFFFFFFFF;
    mTrackTimeError = 0xFFFFFFFF;
    mTrackPosEMCAL = 0xFFFFFFFF;
    mTracklets = 0xFFFFFFFF;
    mMcParticleW = 0xFFFFFFFF;
    mMcParticlePos = 0xFFFFFFFF;
    mMcParticleMom = 0xFFFFFFFF;
    mCaloAmp = 0xFFFFFFFF;
    mCaloTime = 0xFFFFFFFF;
    mCPVPos = 0xFFFFFFFF;
    mCPVAmpl = 0xFFFFFFFF;
    mMuonTr1P = 0xFFFFFFFF;
    mMuonTrThetaX = 0xFFFFFFFF;
    mMuonTrThetaY = 0xFFFFFFFF;
    mMuonTrZmu = 0xFFFFFFFF;
    mMuonTrBend = 0xFFFFFFFF;
    mMuonTrNonBend = 0xFFFFFFFF;
    mMuonTrCov = 0xFFFFFFFF;
    mMuonCl = 0xFFFFFFFF;
    mMuonClErr = 0xFFFFFFFF;
    mV0Time = 0xFFFFFFFF;
    mFDDTime = 0xFFFFFFFF;
    mT0Time = 0xFFFFFFFF;
    mV0Amplitude = 0xFFFFFFFF;
    mFDDAmplitude = 0xFFFFFFFF;
    mT0Amplitude = 0xFFFFFFFF;
  }

  // initialize zdc helper maps
  for (auto& ChannelName : o2::zdc::ChannelNames) {
    mZDCEnergyMap[(string)ChannelName] = 0;
    mZDCTDCMap[(string)ChannelName] = 999;
  }

  mTimer.Reset();
}

void AODProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc); // Make sure that this is called after the RecoContainer collect data, since some condition objects are fetched there

  mStartIR = recoData.startIR;

  auto primVertices = recoData.getPrimaryVertices();
  auto primVer2TRefs = recoData.getPrimaryVertexMatchedTrackRefs();
  auto primVerGIs = recoData.getPrimaryVertexMatchedTracks();
  auto primVerLabels = recoData.getPrimaryVertexMCLabels();

  auto fddChData = recoData.getFDDChannelsData();
  auto fddRecPoints = recoData.getFDDRecPoints();
  auto ft0ChData = recoData.getFT0ChannelsData();
  auto ft0RecPoints = recoData.getFT0RecPoints();
  auto fv0ChData = recoData.getFV0ChannelsData();
  auto fv0RecPoints = recoData.getFV0RecPoints();

  auto zdcEnergies = recoData.getZDCEnergy();
  auto zdcBCRecData = recoData.getZDCBCRecData();
  auto zdcTDCData = recoData.getZDCTDCData();

  auto cpvClusters = recoData.getCPVClusters();
  auto cpvTrigRecs = recoData.getCPVTriggers();

  auto ctpDigits = recoData.getCTPDigits();
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  std::vector<o2::ctp::CTPDigit> ctpDigitsCreated;
  if (mCTPReadout == 1) {
    LOG(info) << "CTP : creating ctpreadout in AOD producer";
    createCTPReadout(recoData, ctpDigitsCreated, pc);
    LOG(info) << "CTP : ctpreadout created from AOD";
    ctpDigits = gsl::span<o2::ctp::CTPDigit>(ctpDigitsCreated);
  }
  LOG(debug) << "FOUND " << primVertices.size() << " primary vertices";
  LOG(debug) << "FOUND " << ft0RecPoints.size() << " FT0 rec. points";
  LOG(debug) << "FOUND " << fv0RecPoints.size() << " FV0 rec. points";
  LOG(debug) << "FOUND " << fddRecPoints.size() << " FDD rec. points";
  LOG(debug) << "FOUND " << cpvClusters.size() << " CPV clusters";
  LOG(debug) << "FOUND " << cpvTrigRecs.size() << " CPV trigger records";

  LOG(info) << "FOUND " << primVertices.size() << " primary vertices";
  auto& bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto& cascadesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CASCADE_001"});
  auto& collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION_001"});
  auto& decay3BodyBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "DECAY3BODY"});
  auto& fddBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FDD_001"});
  auto& ft0Builder = pc.outputs().make<TableBuilder>(Output{"AOD", "FT0"});
  auto& fv0aBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FV0A"});
  auto& fwdTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FWDTRACK"});
  auto& fwdTracksCovBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "FWDTRACKCOV"});
  auto& mcColLabelsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISIONLABEL"});
  auto& mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
  auto& mcMFTTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCMFTTRACKLABEL"});
  auto& mcFwdTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCFWDTRACKLABEL"});
  auto& mcParticlesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE_001"});
  auto& mcTrackLabelBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCTRACKLABEL"});
  auto& mftTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MFTTRACK"});
  auto& tracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACK_IU"});
  auto& tracksCovBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKCOV_IU"});
  auto& tracksExtraBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "TRACKEXTRA"});
  auto& ambigTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "AMBIGUOUSTRACK"});
  auto& ambigMFTTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "AMBIGUOUSMFTTR"});
  auto& ambigFwdTracksBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "AMBIGUOUSFWDTR"});
  auto& v0sBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "V0_001"});
  auto& zdcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ZDC"});
  auto& caloCellsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALO"});
  auto& caloCellsTRGTableBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALOTRIGGER"});
  auto& cpvClustersBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CPVCLUSTER"});
  auto& originTableBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ORIGIN"});

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto cascadesCursor = cascadesBuilder.cursor<o2::aod::Cascades>();
  auto collisionsCursor = collisionsBuilder.cursor<o2::aod::Collisions>();
  auto decay3BodyCursor = decay3BodyBuilder.cursor<o2::aod::Decays3Body>();
  auto fddCursor = fddBuilder.cursor<o2::aod::FDDs>();
  auto ft0Cursor = ft0Builder.cursor<o2::aod::FT0s>();
  auto fv0aCursor = fv0aBuilder.cursor<o2::aod::FV0As>();
  auto fwdTracksCursor = fwdTracksBuilder.cursor<o2::aod::StoredFwdTracks>();
  auto fwdTracksCovCursor = fwdTracksCovBuilder.cursor<o2::aod::StoredFwdTracksCov>();
  auto mcColLabelsCursor = mcColLabelsBuilder.cursor<o2::aod::McCollisionLabels>();
  auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
  auto mcMFTTrackLabelCursor = mcMFTTrackLabelBuilder.cursor<o2::aod::McMFTTrackLabels>();
  auto mcFwdTrackLabelCursor = mcFwdTrackLabelBuilder.cursor<o2::aod::McFwdTrackLabels>();
  auto mcParticlesCursor = mcParticlesBuilder.cursor<o2::aod::StoredMcParticles_001>();
  auto mcTrackLabelCursor = mcTrackLabelBuilder.cursor<o2::aod::McTrackLabels>();
  auto mftTracksCursor = mftTracksBuilder.cursor<o2::aod::StoredMFTTracks>();
  auto tracksCovCursor = tracksCovBuilder.cursor<o2::aod::StoredTracksCov>();
  auto tracksCursor = tracksBuilder.cursor<o2::aod::StoredTracksIU>();
  auto tracksExtraCursor = tracksExtraBuilder.cursor<o2::aod::StoredTracksExtra>();
  auto ambigTracksCursor = ambigTracksBuilder.cursor<o2::aod::AmbiguousTracks>();
  auto ambigMFTTracksCursor = ambigMFTTracksBuilder.cursor<o2::aod::AmbiguousMFTTracks>();
  auto ambigFwdTracksCursor = ambigFwdTracksBuilder.cursor<o2::aod::AmbiguousFwdTracks>();
  auto v0sCursor = v0sBuilder.cursor<o2::aod::V0s>();
  auto zdcCursor = zdcBuilder.cursor<o2::aod::Zdcs>();
  auto caloCellsCursor = caloCellsBuilder.cursor<o2::aod::Calos>();
  auto caloCellsTRGTableCursor = caloCellsTRGTableBuilder.cursor<o2::aod::CaloTriggers>();
  auto cpvClustersCursor = cpvClustersBuilder.cursor<o2::aod::CPVClusters>();
  auto originCursor = originTableBuilder.cursor<o2::aod::Origins>();

  std::unique_ptr<o2::steer::MCKinematicsReader> mcReader;
  if (mUseMC) {
    mcReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
  }
  std::map<uint64_t, int> bcsMap;
  collectBCs(recoData, mUseMC ? mcReader->getDigitizationContext()->getEventRecords() : std::vector<o2::InteractionTimeRecord>{}, bcsMap);
  if (!primVer2TRefs.empty()) { // if the vertexing was done, the last slot refers to orphan tracks
    addRefGlobalBCsForTOF(primVer2TRefs.back(), primVerGIs, recoData, bcsMap);
  }
  // initialize the bunch crossing container for further use below
  mBCLookup.init(bcsMap);

  uint64_t tfNumber;
  const int runNumber = (mRunNumber == -1) ? int(tinfo.runNumber) : mRunNumber;
  if (mTFNumber == -1L) {
    // TODO has to use absolute time of TF
    tfNumber = uint64_t(tinfo.firstTForbit) + (uint64_t(tinfo.runNumber) << 32); // getTFNumber(mStartIR, runNumber);
  } else {
    tfNumber = mTFNumber;
  }

  std::vector<float> aAmplitudes;
  std::vector<uint8_t> aChannels;
  for (auto& fv0RecPoint : fv0RecPoints) {
    aAmplitudes.clear();
    aChannels.clear();
    const auto channelData = fv0RecPoint.getBunchChannelData(fv0ChData);
    for (auto& channel : channelData) {
      if (channel.charge > 0) {
        aAmplitudes.push_back(truncateFloatFraction(channel.charge, mV0Amplitude));
        aChannels.push_back(channel.channel);
      }
    }
    uint64_t bc = fv0RecPoint.getInteractionRecord().toLong();
    auto item = bcsMap.find(bc);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(fatal) << "Error: could not find a corresponding BC ID for a FV0 rec. point; BC = " << bc;
    }
    fv0aCursor(0,
               bcID,
               aAmplitudes,
               aChannels,
               truncateFloatFraction(fv0RecPoint.getCollisionGlobalMeanTime() * 1E-3, mV0Time), // ps to ns
               fv0RecPoint.getTrigger().getTriggersignals());
  }

  for (auto zdcRecData : zdcBCRecData) {
    uint64_t bc = zdcRecData.ir.toLong();
    auto item = bcsMap.find(bc);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(fatal) << "Error: could not find a corresponding BC ID for a ZDC rec. point; BC = " << bc;
    }
    float energyZEM1 = 0;
    float energyZEM2 = 0;
    float energyCommonZNA = 0;
    float energyCommonZNC = 0;
    float energyCommonZPA = 0;
    float energyCommonZPC = 0;
    float energySectorZNA[4] = {0.};
    float energySectorZNC[4] = {0.};
    float energySectorZPA[4] = {0.};
    float energySectorZPC[4] = {0.};
    int fe, ne, ft, nt, fi, ni;
    zdcRecData.getRef(fe, ne, ft, nt, fi, ni);
    for (int ie = 0; ie < ne; ie++) {
      auto& zdcEnergyData = zdcEnergies[fe + ie];
      float energy = zdcEnergyData.energy();
      string chName = o2::zdc::channelName(zdcEnergyData.ch());
      mZDCEnergyMap.at(chName) = energy;
    }
    for (int it = 0; it < nt; it++) {
      auto& tdc = zdcTDCData[ft + it];
      float tdcValue = tdc.value();
      int channelID = o2::zdc::TDCSignal[tdc.ch()];
      auto channelName = o2::zdc::ChannelNames[channelID];
      mZDCTDCMap.at((string)channelName) = tdcValue;
    }
    energySectorZNA[0] = mZDCEnergyMap.at("ZNA1");
    energySectorZNA[1] = mZDCEnergyMap.at("ZNA2");
    energySectorZNA[2] = mZDCEnergyMap.at("ZNA3");
    energySectorZNA[3] = mZDCEnergyMap.at("ZNA4");
    energySectorZNC[0] = mZDCEnergyMap.at("ZNC1");
    energySectorZNC[1] = mZDCEnergyMap.at("ZNC2");
    energySectorZNC[2] = mZDCEnergyMap.at("ZNC3");
    energySectorZNC[3] = mZDCEnergyMap.at("ZNC4");
    energySectorZPA[0] = mZDCEnergyMap.at("ZPA1");
    energySectorZPA[1] = mZDCEnergyMap.at("ZPA2");
    energySectorZPA[2] = mZDCEnergyMap.at("ZPA3");
    energySectorZPA[3] = mZDCEnergyMap.at("ZPA4");
    energySectorZPC[0] = mZDCEnergyMap.at("ZPC1");
    energySectorZPC[1] = mZDCEnergyMap.at("ZPC2");
    energySectorZPC[2] = mZDCEnergyMap.at("ZPC3");
    energySectorZPC[3] = mZDCEnergyMap.at("ZPC4");
    zdcCursor(0,
              bcID,
              mZDCEnergyMap.at("ZEM1"),
              mZDCEnergyMap.at("ZEM2"),
              mZDCEnergyMap.at("ZNAC"),
              mZDCEnergyMap.at("ZNCC"),
              mZDCEnergyMap.at("ZPAC"),
              mZDCEnergyMap.at("ZPCC"),
              energySectorZNA,
              energySectorZNC,
              energySectorZPA,
              energySectorZPC,
              mZDCTDCMap.at("ZEM1"),
              mZDCTDCMap.at("ZEM2"),
              mZDCTDCMap.at("ZNAC"),
              mZDCTDCMap.at("ZNCC"),
              mZDCTDCMap.at("ZPAC"),
              mZDCTDCMap.at("ZPCC"));
  }

  // keep track event/source id for each mc-collision
  // using map and not unordered_map to ensure
  // correct ordering when iterating over container elements
  std::vector<std::vector<int>> mcColToEvSrc;

  if (mUseMC) {
    // TODO: figure out collision weight
    float mcColWeight = 1.;
    // filling mcCollision table
    int nMCCollisions = mcReader->getDigitizationContext()->getNCollisions();
    const auto& mcRecords = mcReader->getDigitizationContext()->getEventRecords();
    const auto& mcParts = mcReader->getDigitizationContext()->getEventParts();
    for (int iCol = 0; iCol < nMCCollisions; iCol++) {
      auto time = mcRecords[iCol].getTimeNS();
      auto globalBC = mcRecords[iCol].toLong();
      auto item = bcsMap.find(globalBC);
      int bcID = -1;
      if (item != bcsMap.end()) {
        bcID = item->second;
      } else {
        LOG(fatal) << "Error: could not find a corresponding BC ID for MC collision; BC = " << globalBC << ", mc collision = " << iCol;
      }
      auto& colParts = mcParts[iCol];
      auto nParts = colParts.size();
      for (auto colPart : colParts) {
        auto eventID = colPart.entryID;
        auto sourceID = colPart.sourceID;
        // enable embedding: if several colParts exist, then they are saved as one collision
        if (nParts == 1 || sourceID == 0) {
          // FIXME:
          // use generators' names for generatorIDs (?)
          short generatorID = sourceID;
          auto& header = mcReader->getMCEventHeader(sourceID, eventID);
          mcCollisionsCursor(0,
                             bcID,
                             generatorID,
                             truncateFloatFraction(header.GetX(), mCollisionPosition),
                             truncateFloatFraction(header.GetY(), mCollisionPosition),
                             truncateFloatFraction(header.GetZ(), mCollisionPosition),
                             truncateFloatFraction(time, mCollisionPosition),
                             truncateFloatFraction(mcColWeight, mCollisionPosition),
                             header.GetB());
        }
        mcColToEvSrc.emplace_back(std::vector<int>{iCol, sourceID, eventID}); // point background and injected signal events to one collision
      }
    }
  }

  std::sort(mcColToEvSrc.begin(), mcColToEvSrc.end(),
            [](const std::vector<int>& left, const std::vector<int>& right) { return (left[0] < right[0]); });

  // vector of FDD amplitudes
  int16_t aFDDAmplitudesA[8] = {0u};
  int16_t aFDDAmplitudesC[8] = {0u};
  // filling FDD table
  for (const auto& fddRecPoint : fddRecPoints) {
    for (int i = 0; i < 8; i++) {
      aFDDAmplitudesA[i] = 0;
      aFDDAmplitudesC[i] = 0;
    }

    const auto channelData = fddRecPoint.getBunchChannelData(fddChData);
    for (const auto& channel : channelData) {
      if (channel.mPMNumber < 8) {
        aFDDAmplitudesC[channel.mPMNumber] = channel.mChargeADC; // amplitude
      } else {
        aFDDAmplitudesA[channel.mPMNumber - 8] = channel.mChargeADC; // amplitude
      }
    }

    uint64_t globalBC = fddRecPoint.getInteractionRecord().toLong();
    uint64_t bc = globalBC;
    auto item = bcsMap.find(bc);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(fatal) << "Error: could not find a corresponding BC ID for a FDD rec. point; BC = " << bc;
    }
    fddCursor(0,
              bcID,
              aFDDAmplitudesA,
              aFDDAmplitudesC,
              truncateFloatFraction(fddRecPoint.getCollisionTimeA() * 1E-3, mFDDTime), // ps to ns
              truncateFloatFraction(fddRecPoint.getCollisionTimeC() * 1E-3, mFDDTime), // ps to ns
              fddRecPoint.getTrigger().getTriggersignals());
  }

  // filling FT0 table
  std::vector<float> aAmplitudesA, aAmplitudesC;
  std::vector<uint8_t> aChannelsA, aChannelsC;
  for (auto& ft0RecPoint : ft0RecPoints) {
    aAmplitudesA.clear();
    aAmplitudesC.clear();
    aChannelsA.clear();
    aChannelsC.clear();
    const auto channelData = ft0RecPoint.getBunchChannelData(ft0ChData);
    for (auto& channel : channelData) {
      // TODO: switch to calibrated amplitude
      if (channel.QTCAmpl > 0) {
        constexpr int nFT0ChannelsAside = o2::ft0::Geometry::NCellsA * 4;
        if (channel.ChId < nFT0ChannelsAside) {
          aChannelsA.push_back(channel.ChId);
          aAmplitudesA.push_back(truncateFloatFraction(channel.QTCAmpl, mT0Amplitude));
        } else {
          aChannelsC.push_back(channel.ChId - nFT0ChannelsAside);
          aAmplitudesC.push_back(truncateFloatFraction(channel.QTCAmpl, mT0Amplitude));
        }
      }
    }
    uint64_t globalBC = ft0RecPoint.getInteractionRecord().toLong();
    uint64_t bc = globalBC;
    auto item = bcsMap.find(bc);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(fatal) << "Error: could not find a corresponding BC ID for a FT0 rec. point; BC = " << bc;
    }
    ft0Cursor(0,
              bcID,
              aAmplitudesA,
              aChannelsA,
              aAmplitudesC,
              aChannelsC,
              truncateFloatFraction(ft0RecPoint.getCollisionTimeA() * 1E-3, mT0Time), // ps to ns
              truncateFloatFraction(ft0RecPoint.getCollisionTimeC() * 1E-3, mT0Time), // ps to ns
              ft0RecPoint.getTrigger().getTriggersignals());
  }

  if (mUseMC) {
    // filling MC collision labels
    for (auto& label : primVerLabels) {
      auto it = std::find_if(mcColToEvSrc.begin(), mcColToEvSrc.end(),
                             [&label](const auto& mcColInfo) { return mcColInfo[1] == label.getSourceID() && mcColInfo[2] == label.getEventID(); });
      int32_t mcCollisionID = -1;
      if (it != mcColToEvSrc.end()) {
        mcCollisionID = it->at(0);
      }
      uint16_t mcMask = 0; // todo: set mask using normalized weights?
      mcColLabelsCursor(0, mcCollisionID, mcMask);
    }
  }

  cacheTriggers(recoData);
  countTPCClusters(recoData);

  int collisionID = 0;
  mIndexTableMFT.resize(recoData.getMFTTracks().size());
  mIndexTableFwd.resize(recoData.getMCHTracks().size());

  auto& trackReffwd = primVer2TRefs.back();
  fillIndexTablesPerCollision(trackReffwd, primVerGIs, recoData);
  collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& trackReffwd = primVer2TRefs[collisionID];
    fillIndexTablesPerCollision(trackReffwd, primVerGIs, recoData); // this function must follow the same track order as 'fillTrackTablesPerCollision' to fill the map of track indices
    collisionID++;
  }

  mGIDToTableFwdID.clear(); // reset the tables to be used by 'fillTrackTablesPerCollision'
  mGIDToTableMFTID.clear();

  // filling unassigned tracks first
  // so that all unassigned tracks are stored in the beginning of the table together
  auto& trackRef = primVer2TRefs.back(); // references to unassigned tracks are at the end
  // fixme: interaction time is undefined for unassigned tracks (?)
  fillTrackTablesPerCollision(-1, std::uint64_t(-1), trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor,
                              ambigTracksCursor, mftTracksCursor, ambigMFTTracksCursor,
                              fwdTracksCursor, fwdTracksCovCursor, ambigFwdTracksCursor, bcsMap);

  // filling collisions and tracks into tables
  collisionID = 0;
  for (auto& vertex : primVertices) {
    auto& cov = vertex.getCov();
    auto& timeStamp = vertex.getTimeStamp();                       // this is a relative time
    const double interactionTime = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = relativeTime_to_GlobalBC(interactionTime);
    uint64_t localBC = relativeTime_to_LocalBC(interactionTime);
    LOG(debug) << "global BC " << globalBC << " local BC " << localBC << " relative interaction time " << interactionTime;
    // collision timestamp in ns wrt the beginning of collision BC
    const float relInteractionTime = static_cast<float>(localBC * o2::constants::lhc::LHCBunchSpacingNS - interactionTime);
    auto item = bcsMap.find(globalBC);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(fatal) << "Error: could not find a corresponding BC ID for a collision; BC = " << globalBC << ", collisionID = " << collisionID;
    }
    collisionsCursor(0,
                     bcID,
                     truncateFloatFraction(vertex.getX(), mCollisionPosition),
                     truncateFloatFraction(vertex.getY(), mCollisionPosition),
                     truncateFloatFraction(vertex.getZ(), mCollisionPosition),
                     truncateFloatFraction(cov[0], mCollisionPositionCov),
                     truncateFloatFraction(cov[1], mCollisionPositionCov),
                     truncateFloatFraction(cov[2], mCollisionPositionCov),
                     truncateFloatFraction(cov[3], mCollisionPositionCov),
                     truncateFloatFraction(cov[4], mCollisionPositionCov),
                     truncateFloatFraction(cov[5], mCollisionPositionCov),
                     vertex.getFlags(),
                     truncateFloatFraction(vertex.getChi2(), mCollisionPositionCov),
                     vertex.getNContributors(),
                     truncateFloatFraction(relInteractionTime, mCollisionPosition),
                     truncateFloatFraction(timeStamp.getTimeStampError() * 1E3, mCollisionPositionCov));
    mVtxToTableCollID[collisionID] = mTableCollID++;

    auto& trackRef = primVer2TRefs[collisionID];
    // passing interaction time in [ps]
    fillTrackTablesPerCollision(collisionID, globalBC, trackRef, primVerGIs, recoData, tracksCursor, tracksCovCursor, tracksExtraCursor, ambigTracksCursor,
                                mftTracksCursor, ambigMFTTracksCursor,
                                fwdTracksCursor, fwdTracksCovCursor, ambigFwdTracksCursor, bcsMap);
    collisionID++;
  }

  fillSecondaryVertices(recoData, v0sCursor, cascadesCursor, decay3BodyCursor);

  // helper map for fast search of a corresponding class mask for a bc
  std::unordered_map<uint64_t, uint64_t> bcToClassMask;
  if (mInputSources[GID::CTP]) {
    LOG(debug) << "CTP input available";
    for (auto& ctpDigit : ctpDigits) {
      uint64_t bc = ctpDigit.intRecord.toLong();
      uint64_t classMask = ctpDigit.CTPClassMask.to_ulong();
      bcToClassMask[bc] = classMask;
      // LOG(debug) << Form("classmask:0x%llx", classMask);
    }
  }

  // filling BC table
  uint64_t triggerMask = 0;
  for (auto& item : bcsMap) {
    uint64_t bc = item.first;
    if (mInputSources[GID::CTP]) {
      auto bcClassPair = bcToClassMask.find(bc);
      if (bcClassPair != bcToClassMask.end()) {
        triggerMask = bcClassPair->second;
      } else {
        triggerMask = 0;
      }
    }
    bcCursor(0,
             runNumber,
             bc,
             triggerMask);
  }

  bcToClassMask.clear();

  if (mInputSources[GIndex::PHS] || mInputSources[GIndex::EMC]) {
    fillCaloTable(caloCellsCursor, caloCellsTRGTableCursor, bcsMap, recoData);
  }

  // fill cpvcluster table
  if (mInputSources[GIndex::CPV]) {
    float posX, posZ;
    for (auto& cpvEvent : cpvTrigRecs) {
      uint64_t bc = cpvEvent.getBCData().toLong();
      auto item = bcsMap.find(bc);
      int bcID = -1;
      if (item != bcsMap.end()) {
        bcID = item->second;
      } else {
        LOG(fatal) << "Error: could not find a corresponding BC ID for a CPV Trigger Record; BC = " << bc;
      }
      for (int iClu = cpvEvent.getFirstEntry(); iClu < cpvEvent.getFirstEntry() + cpvEvent.getNumberOfObjects(); iClu++) {
        auto& clu = cpvClusters[iClu];
        clu.getLocalPosition(posX, posZ);
        cpvClustersCursor(0,
                          bcID,
                          truncateFloatFraction(posX, mCPVPos),
                          truncateFloatFraction(posZ, mCPVPos),
                          truncateFloatFraction(clu.getEnergy(), mCPVAmpl),
                          clu.getPackedClusterStatus());
      }
    }
  }

  bcsMap.clear();

  if (mUseMC) {
    TStopwatch timer;
    timer.Start();
    // filling mc particles table
    fillMCParticlesTable(*mcReader,
                         mcParticlesCursor,
                         primVer2TRefs,
                         primVerGIs,
                         recoData,
                         mcColToEvSrc);
    timer.Stop();
    LOG(info) << "FILL MC took " << timer.RealTime() << " s";
    mcColToEvSrc.clear();

    // ------------------------------------------------------
    // filling track labels

    // need to go through labels in the same order as for tracks
    fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, primVer2TRefs.back(), primVerGIs, recoData);
    for (int iref = 0; iref < primVer2TRefs.size() - 1; iref++) {
      auto& trackRef = primVer2TRefs[iref];
      fillMCTrackLabelsTable(mcTrackLabelCursor, mcMFTTrackLabelCursor, mcFwdTrackLabelCursor, trackRef, primVerGIs, recoData);
    }
  }
  clearMCKeepStore(mToStore);
  mGIDToTableID.clear();
  mTableTrID = 0;
  mGIDToTableFwdID.clear();
  mTableTrFwdID = 0;
  mGIDToTableMFTID.clear();
  mTableTrMFTID = 0;
  mVtxToTableCollID.clear();
  mTableCollID = 0;
  mV0ToTableID.clear();
  mTableV0ID = 0;

  mIndexTableFwd.clear();
  mIndexFwdID = 0;
  mIndexTableMFT.clear();
  mIndexMFTID = 0;

  mBCLookup.clear();

  originCursor(0, tfNumber);

  // sending metadata to writer
  TString dataType = mUseMC ? "MC" : "RAW";
  TString O2Version = o2::fullVersion();
  TString ROOTVersion = ROOT_RELEASE;
  mMetaDataKeys = {"DataType", "Run", "O2Version", "ROOTVersion", "RecoPassName", "AnchorProduction", "AnchorPassName", "LPMProductionTag"};
  mMetaDataVals = {dataType, "3", O2Version, ROOTVersion, mRecoPass, mAnchorProd, mAnchorPass, mLPMProdTag};
  pc.outputs().snapshot(Output{"AMD", "AODMetadataKeys", 0, Lifetime::Timeframe}, mMetaDataKeys);
  pc.outputs().snapshot(Output{"AMD", "AODMetadataVals", 0, Lifetime::Timeframe}, mMetaDataVals);

  pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, tfNumber);
  pc.outputs().snapshot(Output{"TFF", "TFFilename", 0, Lifetime::Timeframe}, "");

  mTimer.Stop();
}

void AODProducerWorkflowDPL::cacheTriggers(const o2::globaltracking::RecoContainer& recoData)
{
  // ITS tracks->ROF
  {
    mITSROFs.clear();
    const auto& rofs = recoData.getITSTracksROFRecords();
    uint16_t count = 0;
    for (const auto& rof : rofs) {
      int first = rof.getFirstEntry(), last = first + rof.getNEntries();
      for (int i = first; i < last; i++) {
        mITSROFs.push_back(count);
      }
      count++;
    }
  }
  // MFT tracks->ROF
  {
    mMFTROFs.clear();
    const auto& rofs = recoData.getMFTTracksROFRecords();
    uint16_t count = 0;
    for (const auto& rof : rofs) {
      int first = rof.getFirstEntry(), last = first + rof.getNEntries();
      for (int i = first; i < last; i++) {
        mMFTROFs.push_back(count);
      }
      count++;
    }
  }
  // ITSTPCTRD tracks -> TRD trigger
  {
    mITSTPCTRDTriggers.clear();
    const auto& itstpctrigs = recoData.getITSTPCTRDTriggers();
    int count = 0;
    for (const auto& trig : itstpctrigs) {
      int first = trig.getFirstTrack(), last = first + trig.getNumberOfTracks();
      for (int i = first; i < last; i++) {
        mITSTPCTRDTriggers.push_back(count);
      }
      count++;
    }
  }
  // TPCTRD tracks -> TRD trigger
  {
    mTPCTRDTriggers.clear();
    const auto& tpctrigs = recoData.getTPCTRDTriggers();
    int count = 0;
    for (const auto& trig : tpctrigs) {
      int first = trig.getFirstTrack(), last = first + trig.getNumberOfTracks();
      for (int i = first; i < last; i++) {
        mTPCTRDTriggers.push_back(count);
      }
      count++;
    }
  }
  // MCH tracks->ROF
  {
    mMCHROFs.clear();
    const auto& rofs = recoData.getMCHTracksROFRecords();
    uint16_t count = 0;
    for (const auto& rof : rofs) {
      int first = rof.getFirstIdx(), last = first + rof.getNEntries();
      for (int i = first; i < last; i++) {
        mMCHROFs.push_back(count);
      }
      count++;
    }
  }
}

AODProducerWorkflowDPL::TrackExtraInfo AODProducerWorkflowDPL::processBarrelTrack(int collisionID, std::uint64_t collisionBC, GIndex trackIndex,
                                                                                  const o2::globaltracking::RecoContainer& data, const std::map<uint64_t, int>& bcsMap)
{
  TrackExtraInfo extraInfoHolder;
  if (collisionID < 0) {
    extraInfoHolder.flags |= o2::aod::track::OrphanTrack;
  }
  bool needBCSlice = collisionID < 0 || trackIndex.isAmbiguous(); // track is associated to multiple vertices
  uint64_t bcOfTimeRef = collisionBC - mStartIR.toLong();         // by default (unambiguous) track time is wrt collision BC

  auto setTrackTime = [&](double t, double terr, bool gaussian) {
    // set track time and error, for ambiguous tracks define the bcSlice as it was used in vertex-track association
    // provided track time (wrt TF start) and its error should be in ns, gaussian flag tells if the error is assumed to be gaussin or half-interval
    if (!gaussian) {
      extraInfoHolder.flags |= o2::aod::track::TrackTimeResIsRange;
    }
    extraInfoHolder.trackTimeRes = terr;
    if (needBCSlice) { // need to define BC slice
      double error = this->mTimeMarginTrackTime + (gaussian ? extraInfoHolder.trackTimeRes * this->mNSigmaTimeTrack : extraInfoHolder.trackTimeRes);
      bcOfTimeRef = fillBCSlice(extraInfoHolder.bcSlice, t - error, t + error, bcsMap);
    }
    extraInfoHolder.trackTime = float(t - bcOfTimeRef * o2::constants::lhc::LHCBunchSpacingNS);
    LOGP(debug, "time : {}/{} -> {}/{} -> trunc: {}/{} CollID: {} Amb: {}", t, terr, t - bcOfTimeRef * o2::constants::lhc::LHCBunchSpacingNS, terr,
         truncateFloatFraction(extraInfoHolder.trackTime, mTrackTime), truncateFloatFraction(extraInfoHolder.trackTimeRes, mTrackTimeError),
         collisionID, trackIndex.isAmbiguous());
  };
  auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
  const auto& trackPar = data.getTrackParam(trackIndex);
  extraInfoHolder.flags |= trackPar.getPID() << 28;
  auto src = trackIndex.getSource();
  if (contributorsGID[GIndex::Source::TOF].isIndexSet()) { // ITS-TPC-TRD-TOF, ITS-TPC-TOF, TPC-TRD-TOF, TPC-TOF
    const auto& tofMatch = data.getTOFMatch(trackIndex);
    extraInfoHolder.tofChi2 = tofMatch.getChi2();
    const auto& tofInt = tofMatch.getLTIntegralOut();
    float intLen = tofInt.getL();
    extraInfoHolder.length = intLen;
    const float mass = o2::constants::physics::MassPionCharged; // default pid = pion
    if (tofInt.getTOF(o2::track::PID::Pion) > 0.f) {
      const float expBeta = (intLen / (tofInt.getTOF(o2::track::PID::Pion) * cSpeed));
      extraInfoHolder.tofExpMom = mass * expBeta / std::sqrt(1.f - expBeta * expBeta);
    }
    // correct the time of the track
    const double massZ = o2::track::PID::getMass2Z(trackPar.getPID());
    const double energy = sqrt((massZ * massZ) + (extraInfoHolder.tofExpMom * extraInfoHolder.tofExpMom));
    const double exp = extraInfoHolder.length * energy / (cSpeed * extraInfoHolder.tofExpMom);
    auto tofSignal = (tofMatch.getSignal() - exp) * 1e-3; // time in ns wrt TF start
    setTrackTime(tofSignal, 0.2, true);                   // FIXME: calculate actual resolution (if possible?)
  }
  if (contributorsGID[GIndex::Source::TRD].isIndexSet()) {                                        // ITS-TPC-TRD-TOF, TPC-TRD-TOF, TPC-TRD, ITS-TPC-TRD
    const auto& trdOrig = data.getTrack<o2::trd::TrackTRD>(contributorsGID[GIndex::Source::TRD]); // refitted TRD trac
    extraInfoHolder.trdChi2 = trdOrig.getChi2();
    extraInfoHolder.trdSignal = trdOrig.getSignal();
    extraInfoHolder.trdPattern = getTRDPattern(trdOrig);
    if (extraInfoHolder.trackTimeRes < 0.) { // time is not set yet, this is possible only for TPC-TRD and ITS-TPC-TRD tracks, since those with TOF are set upstream
      // TRD is triggered: time uncertainty is within a BC
      const auto& trdTrig = (src == GIndex::Source::TPCTRD) ? data.getTPCTRDTriggers()[mTPCTRDTriggers[trackIndex.getIndex()]] : data.getITSTPCTRDTriggers()[mITSTPCTRDTriggers[trackIndex.getIndex()]];
      double ttrig = trdTrig.getBCData().differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingNS; // 1st get time wrt TF start
      setTrackTime(ttrig, 1., true);                                                                       // FIXME: calculate actual resolution (if possible?)
    }
  }
  if (contributorsGID[GIndex::Source::ITS].isIndexSet()) {
    const auto& itsTrack = data.getITSTrack(contributorsGID[GIndex::ITS]);
    int nClusters = itsTrack.getNClusters();
    float chi2 = itsTrack.getChi2();
    extraInfoHolder.itsChi2NCl = nClusters != 0 ? chi2 / (float)nClusters : 0;
    extraInfoHolder.itsClusterMap = itsTrack.getPattern();
    if (src == GIndex::ITS) { // standalone ITS track should set its time from the ROF
      const auto& rof = data.getITSTracksROFRecords()[mITSROFs[trackIndex.getIndex()]];
      double t = rof.getBCData().differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingNS + mITSROFrameHalfLengthNS;
      setTrackTime(t, mITSROFrameHalfLengthNS, false);
    }
  } else if (contributorsGID[GIndex::Source::ITSAB].isIndexSet()) { // this is an ITS-TPC afterburner contributor
    extraInfoHolder.itsClusterMap = data.getITSABRefs()[contributorsGID[GIndex::Source::ITSAB].getIndex()].pattern;
  }
  if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
    const auto& tpcOrig = data.getTPCTrack(contributorsGID[GIndex::TPC]);
    const auto& tpcClData = mTPCCounters[contributorsGID[GIndex::TPC]];
    extraInfoHolder.tpcInnerParam = tpcOrig.getP();
    extraInfoHolder.tpcChi2NCl = tpcOrig.getNClusters() ? tpcOrig.getChi2() / tpcOrig.getNClusters() : 0;
    extraInfoHolder.tpcSignal = tpcOrig.getdEdx().dEdxTotTPC;
    extraInfoHolder.tpcNClsFindable = tpcOrig.getNClusters();
    extraInfoHolder.tpcNClsFindableMinusFound = tpcOrig.getNClusters() - tpcClData.found;
    extraInfoHolder.tpcNClsFindableMinusCrossedRows = tpcOrig.getNClusters() - tpcClData.crossed;
    extraInfoHolder.tpcNClsShared = tpcClData.shared;
    if (src == GIndex::TPC) {                                                                                // standalone TPC track should set its time from their timebins range
      double terr = 0.5 * (tpcOrig.getDeltaTFwd() + tpcOrig.getDeltaTBwd()) * mTPCBinNS;                     // half-span of the interval
      double t = (tpcOrig.getTime0() + 0.5 * (tpcOrig.getDeltaTFwd() - tpcOrig.getDeltaTBwd())) * mTPCBinNS; // central value
      LOG(debug) << "TPC tracks t0:" << tpcOrig.getTime0() << " tbwd: " << tpcOrig.getDeltaTBwd() << " tfwd: " << tpcOrig.getDeltaTFwd() << " t: " << t << " te: " << terr;
      setTrackTime(t, terr, false);
    } else if (src == GIndex::ITSTPC) { // its-tpc matched tracks have gaussian time error and the time was not set above
      const auto& trITSTPC = data.getTPCITSTrack(trackIndex);
      auto ts = trITSTPC.getTimeMUS();
      setTrackTime(ts.getTimeStamp() * 1.e3, ts.getTimeStampError() * 1.e3, true);
    }
  }

  extrapolateToCalorimeters(extraInfoHolder, data.getTrackParamOut(trackIndex));
  // set bit encoding for PVContributor property as part of the flag field
  if (trackIndex.isPVContributor()) {
    extraInfoHolder.flags |= o2::aod::track::PVContributor;
  }
  return extraInfoHolder;
}

void AODProducerWorkflowDPL::extrapolateToCalorimeters(TrackExtraInfo& extraInfoHolder, const o2::track::TrackPar& track)
{
  constexpr float XEMCAL = 440.f, XPHOS = 460.f, XEMCAL2 = XEMCAL * XEMCAL;
  constexpr float ETAEMCAL = 0.75;                                  // eta of EMCAL/DCAL with margin
  constexpr float ETADCALINNER = 0.22;                              // eta of the DCAL PHOS Hole (at XEMCAL)
  constexpr float ETAPHOS = 0.13653194;                             // nominal eta of the PHOS acceptance (at XPHOS): -log(tan((TMath::Pi()/2 - atan2(63, 460))/2))
  constexpr float ETAPHOSMARGIN = 0.17946979;                       // etat of the PHOS acceptance with 20 cm margin (at XPHOS): -log(tan((TMath::Pi()/2 + atan2(63+20., 460))/2)), not used, for the ref only
  constexpr float ETADCALPHOSSWITCH = (ETADCALINNER + ETAPHOS) / 2; // switch to DCAL to PHOS check if eta < this value
  constexpr short SNONE = 0, SEMCAL = 0x1, SPHOS = 0x2;
  constexpr short SECTORTYPE[18] = {
    SNONE, SNONE, SNONE, SNONE,                     // 0:3
    SEMCAL, SEMCAL, SEMCAL, SEMCAL, SEMCAL, SEMCAL, // 3:9 EMCAL only
    SNONE, SNONE,                                   // 10:11
    SPHOS,                                          // 12 PHOS only
    SPHOS | SEMCAL, SPHOS | SEMCAL, SPHOS | SEMCAL, // 13:15 PHOS & DCAL
    SEMCAL,                                         // 16 DCAL only
    SNONE};                                         // 17

  o2::track::TrackPar outTr{track};
  auto prop = o2::base::Propagator::Instance();
  // 1st propagate to EMCAL nominal radius
  float xtrg = 0;
  if (!outTr.getXatLabR(XEMCAL, xtrg, prop->getNominalBz(), o2::track::DirType::DirOutward) ||
      !prop->PropagateToXBxByBz(outTr, xtrg, 0.95, 10, o2::base::Propagator::MatCorrType::USEMatCorrLUT)) {
    LOGP(debug, "preliminary step: does not reach R={} {}", XEMCAL, outTr.asString());
    return;
  }
  // we do not necessarilly reach wanted radius in a single propagation
  if ((outTr.getX() * outTr.getX() + outTr.getY() * outTr.getY() < XEMCAL2) &&
      (!outTr.rotateParam(outTr.getPhi()) ||
       !outTr.getXatLabR(XEMCAL, xtrg, prop->getNominalBz(), o2::track::DirType::DirOutward) ||
       !prop->PropagateToXBxByBz(outTr, xtrg, 0.95, 10, o2::base::Propagator::MatCorrType::USEMatCorrLUT))) {
    LOGP(debug, "does not reach R={} {}", XEMCAL, outTr.asString());
    return;
  }
  // rotate to proper sector
  int sector = o2::math_utils::angle2Sector(outTr.getPhiPos());

  auto propExactSector = [&outTr, &sector, prop](float xprop) -> bool { // propagate exactly to xprop in the proper sector frame
    int ntri = 0;
    while (ntri < 2) {
      auto outTrTmp = outTr;
      float alpha = o2::math_utils::sector2Angle(sector);
      if (!outTrTmp.rotateParam(alpha) ||
          !prop->PropagateToXBxByBz(outTrTmp, xprop, 0.95, 10, o2::base::Propagator::MatCorrType::USEMatCorrLUT)) {
        LOGP(debug, "failed on rotation to {} (sector {}) or propagation to X={} {}", alpha, sector, xprop, outTrTmp.asString());
        return false;
      }
      // make sure we are still in the target sector
      int sectorTmp = o2::math_utils::angle2Sector(outTrTmp.getPhiPos());
      if (sectorTmp == sector) {
        outTr = outTrTmp;
        break;
      }
      sector = sectorTmp;
      ntri++;
    }
    if (ntri == 2) {
      LOGP(debug, "failed to rotate to sector, {}", outTr.asString());
      return false;
    }
    return true;
  };

  // we are at the EMCAL X, check if we are in the good sector
  if (!propExactSector(XEMCAL) || SECTORTYPE[sector] == SNONE) { // propagation failed or neither EMCAL not DCAL/PHOS
    return;
  }

  // check if we are in a good eta range
  float r = std::sqrt(outTr.getX() * outTr.getX() + outTr.getY() * outTr.getY()), tg = std::atan2(r, outTr.getZ());
  float eta = -std::log(std::tan(0.5f * tg)), etaAbs = std::abs(eta);
  if (etaAbs > ETAEMCAL) {
    LOGP(debug, "eta = {} is off at EMCAL radius", eta, outTr.asString());
    return;
  }
  // are we in the PHOS hole (with margin)?
  if ((SECTORTYPE[sector] & SPHOS) && etaAbs < ETADCALPHOSSWITCH) { // propagate to PHOS radius
    if (!propExactSector(XPHOS)) {
      return;
    }
    r = std::sqrt(outTr.getX() * outTr.getX() + outTr.getY() * outTr.getY());
    tg = std::atan2(r, outTr.getZ());
    eta = -std::log(std::tan(0.5f * tg));
  } else if (!(SECTORTYPE[sector] & SEMCAL)) { // are in the sector with PHOS only
    return;
  }
  extraInfoHolder.trackPhiEMCAL = outTr.getPhiPos();
  extraInfoHolder.trackEtaEMCAL = eta;
  LOGP(debug, "eta = {} phi = {} sector {} for {}", extraInfoHolder.trackEtaEMCAL, extraInfoHolder.trackPhiEMCAL, sector, outTr.asString());
  //
}

void AODProducerWorkflowDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // Note: DPLAlpideParam for ITS and MFT will be loaded by the RecoContainer

    // apply settings
    auto grpECS = o2::base::GRPGeomHelper::instance().getGRPECS();
    o2::BunchFilling bcf = o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling();
    std::bitset<3564> bs = bcf.getBCPattern();
    for (int i = 0; i < bs.size(); i++) {
      if (bs.test(i)) {
        o2::tof::Utils::addInteractionBC(i);
      }
    }

    const auto& alpParamsITS = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    mITSROFrameHalfLengthNS = 0.5 * (grpECS->isDetContinuousReadOut(o2::detectors::DetID::ITS) ? alpParamsITS.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS : alpParamsITS.roFrameLengthTrig);

    const auto& alpParamsMFT = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    mMFTROFrameHalfLengthNS = 0.5 * (grpECS->isDetContinuousReadOut(o2::detectors::DetID::MFT) ? alpParamsMFT.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS : alpParamsMFT.roFrameLengthTrig);

    // RS FIXME: this is not yet fetched from the CCDB
    auto& elParam = o2::tpc::ParameterElectronics::Instance();
    mTPCBinNS = elParam.ZbinWidth * 1.e3;

    const auto& pvParams = o2::vertexing::PVertexerParams::Instance();
    mNSigmaTimeTrack = pvParams.nSigmaTimeTrack;
    mTimeMarginTrackTime = pvParams.timeMarginTrackTime * 1.e3;
    mFieldON = std::abs(o2::base::Propagator::Instance()->getNominalBz()) > 0.01;
  }
}

//_______________________________________
void AODProducerWorkflowDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "ITS Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "ALPIDEPARAM", 0)) {
    LOG(info) << "MFT Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    par.printKeyValues();
    return;
  }
}

void AODProducerWorkflowDPL::addRefGlobalBCsForTOF(const o2::dataformats::VtxTrackRef& trackRef, const gsl::span<const GIndex>& GIndices,
                                                   const o2::globaltracking::RecoContainer& data, std::map<uint64_t, int>& bcsMap)
{
  // Orphan tracks need to refer to some globalBC and for tracks with TOF this BC should be whithin an orbit
  // from the track abs time (to guarantee time precision). Therefore, we may need to insert some dummy globalBCs
  // to guarantee proper reference.
  // complete globalBCs by dummy entries necessary to provide BC references for TOF tracks with requested precision
  // to provide a reference for the time of the orphan tracks we should make sure that there are no gaps longer
  // than needed to store the time with sufficient precision

  // estimate max distance in BCs between TOF time and eventual reference BC
  int nbitsFrac = 24 - (32 - o2::math_utils::popcount(mTrackTime)); // number of bits used to encode the fractional part of float truncated by the mask
  int nbitsLoss = std::max(0, int(std::log2(TOFTimePrecPS)));       // allowed bit loss guaranteeing needed precision in PS
  assert(nbitsFrac > 1);
  std::uint64_t maxRangePS = std::uint64_t(0x1) << (nbitsFrac + nbitsLoss);
  int maxGapBC = maxRangePS / (o2::constants::lhc::LHCBunchSpacingNS * 1e3); // max gap in BCs allowing to store time with required precision
  LOG(info) << "Max gap of " << maxGapBC << " BCs to closest globalBC reference is needed for TOF tracks to provide precision of "
            << TOFTimePrecPS << " ps";

  // check if there are tracks orphan tracks at all
  if (!trackRef.getEntries()) {
    return;
  }
  // the bscMap has at least TF start BC
  std::uint64_t maxBC = mStartIR.toLong();
  const auto& tofClus = data.getTOFClusters();
  for (int src = GIndex::NSources; src--;) {
    if (!GIndex::getSourceDetectorsMask(src)[o2::detectors::DetID::TOF]) { // check only tracks with TOF contribution
      continue;
    }
    int start = trackRef.getFirstEntryOfSource(src);
    int end = start + trackRef.getEntriesOfSource(src);
    for (int ti = start; ti < end; ti++) {
      auto& trackIndex = GIndices[ti];
      const auto& tofMatch = data.getTOFMatch(trackIndex);
      const auto& tofInt = tofMatch.getLTIntegralOut();
      float intLen = tofInt.getL();
      float tofExpMom = 0.;
      if (tofInt.getTOF(o2::track::PID::Pion) > 0.f) {
        float expBeta = (intLen / (tofInt.getTOF(o2::track::PID::Pion) * cSpeed));
        tofExpMom = o2::constants::physics::MassPionCharged * expBeta / std::sqrt(1.f - expBeta * expBeta);
      } else {
        continue;
      }
      double massZ = o2::track::PID::getMass2Z(data.getTrackParam(trackIndex).getPID());
      double energy = sqrt((massZ * massZ) + (tofExpMom * tofExpMom));
      double exp = intLen * energy / (cSpeed * tofExpMom);
      auto tofSignal = (tofMatch.getSignal() - exp) * 1e-3; // time in ns wrt TF start
      auto bc = relativeTime_to_GlobalBC(tofSignal);

      auto it = bcsMap.lower_bound(bc);
      if (it == bcsMap.end() || it->first > bc + maxGapBC) {
        bcsMap.emplace_hint(it, bc, 1);
        LOG(debug) << "adding dummy BC " << bc;
      }
      if (bc > maxBC) {
        maxBC = bc;
      }
    }
  }
  // make sure there is a globalBC exceeding the max encountered bc
  if ((--bcsMap.end())->first <= maxBC) {
    bcsMap.emplace_hint(bcsMap.end(), maxBC + 1, 1);
  }
  // renumber BCs
  int bcID = 0;
  for (auto& item : bcsMap) {
    item.second = bcID;
    bcID++;
  }
}

std::uint64_t AODProducerWorkflowDPL::fillBCSlice(int (&slice)[2], double tmin, double tmax, const std::map<uint64_t, int>& bcsMap) const
{
  // for ambiguous tracks (no or multiple vertices) we store the BC slice corresponding to track time window used for track-vertex matching,
  // see VertexTrackMatcher::extractTracks creator method, i.e. central time estimated +- uncertainty defined as:
  // 1) for tracks having a gaussian time error: PVertexerParams.nSigmaTimeTrack * trackSigma + PVertexerParams.timeMarginTrackTime
  // 2) for tracks having time uncertainty in a fixed time interval (TPC,ITS,MFT..): half of the interval + PVertexerParams.timeMarginTrackTime
  // The track time in the TrackExtraInfo is stored in ns wrt the collision BC for unambigous tracks and wrt bcSlice[0] for ambiguous ones,
  // with convention for errors: trackSigma in case (1) and half of the time interval for case (2) above.

  // find indices of widest slice of global BCs in the map compatible with provided BC range. bcsMap is guaranteed to be non-empty.
  // We also assume that tmax >= tmin.

  uint64_t bcMin = relativeTime_to_GlobalBC(tmin), bcMax = relativeTime_to_GlobalBC(tmax);

  /*
    // brute force way of searching bcs via direct binary search in the map
    auto lower = bcsMap.lower_bound(bcMin), upper = bcsMap.upper_bound(bcMax);

    if (lower == bcsMap.end()) {
      --lower;
    }
    if (upper != lower) {
      --upper;
    }
    slice[0] = std::distance(bcsMap.begin(), lower);
    slice[1] = std::distance(bcsMap.begin(), upper);
  */

  // faster way to search in bunch crossing via the accelerated bunch crossing lookup structure
  auto p = mBCLookup.lower_bound(bcMin);
  // assuming that bcMax will be >= bcMin and close to bcMin; we can find
  // the upper bound quickly by lineary iterating from p.first to the point where
  // the time becomes larger than bcMax.
  // (if this is not the case we could determine it with a similar call to mBCLookup)
  auto& bcvector = mBCLookup.getBCTimeVector();
  auto upperindex = p.first;
  while (upperindex < bcvector.size() && bcvector[upperindex] <= bcMax) {
    upperindex++;
  }
  if (upperindex != p.first) {
    upperindex--;
  }
  slice[0] = p.first;
  slice[1] = upperindex;

  auto bcOfTimeRef = p.second - this->mStartIR.toLong();
  LOG(debug) << "BC slice t:" << tmin << " " << slice[0]
             << " t: " << tmax << " " << slice[1]
             << " bcref: " << bcOfTimeRef;
  return bcOfTimeRef;
}

void AODProducerWorkflowDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, bool enableSV, bool useMC, bool CTPConfigPerRun)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->inputs.emplace_back("ctpconfig", "CTP", "CTPCONFIG", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/Config", CTPConfigPerRun));

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  if (src[GID::CTP]) {
    LOGF(info, "Requesting CTP digits");
    dataRequest->requestCTPDigits(useMC);
  }
  if (enableSV) {
    dataRequest->requestSecondaryVertertices(useMC);
    LOGF(info, "requestSecondaryVertertices Finish");
  }
  if (src[GID::TPC]) {
    dataRequest->requestClusters(GIndex::getSourcesMask("TPC"), false); // no need to ask for TOF clusters as they are requested with TOF tracks
  }
  if (src[GID::TOF]) {
    dataRequest->requestTOFClusters(useMC);
  }
  if (src[GID::PHS]) {
    dataRequest->requestPHOSCells(useMC);
  }
  if (src[GID::TRD]) {
    dataRequest->requestTRDTracklets(false);
  }
  if (src[GID::EMC]) {
    dataRequest->requestEMCALCells(useMC);
  }
  if (src[GID::CPV]) {
    dataRequest->requestCPVClusters(useMC);
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                              // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true); // query only once all objects except mag.field

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2cascade_001"}, "AOD", "CASCADE_001", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision_001"}, "AOD", "COLLISION_001", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2decay3body"}, "AOD", "DECAY3BODY", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fdd_001"}, "AOD", "FDD_001", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ft0"}, "AOD", "FT0", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fv0a"}, "AOD", "FV0A", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fwdtrack"}, "AOD", "FWDTRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2fwdtrackcov"}, "AOD", "FWDTRACKCOV", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollisionlabel"}, "AOD", "MCCOLLISIONLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcmfttracklabel"}, "AOD", "MCMFTTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcfwdtracklabel"}, "AOD", "MCFWDTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle_001"}, "AOD", "MCPARTICLE_001", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mctracklabel"}, "AOD", "MCTRACKLABEL", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mfttrack"}, "AOD", "MFTTRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2track_iu"}, "AOD", "TRACK_IU", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackcov_iu"}, "AOD", "TRACKCOV_IU", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2trackextra"}, "AOD", "TRACKEXTRA", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ambiguoustrack"}, "AOD", "AMBIGUOUSTRACK", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ambiguousMFTtrack"}, "AOD", "AMBIGUOUSMFTTR", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2ambiguousFwdtrack"}, "AOD", "AMBIGUOUSFWDTR", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2v0_001"}, "AOD", "V0_001", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2zdc"}, "AOD", "ZDC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCell"}, "AOD", "CALO", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCellTRGR"}, "AOD", "CALOTRIGGER", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2cpvCluster"}, "AOD", "CPVCLUSTER", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2origin"}, "AOD", "ORIGIN", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});
  outputs.emplace_back(OutputSpec{"AMD", "AODMetadataKeys"});
  outputs.emplace_back(OutputSpec{"AMD", "AODMetadataVals"});

  LOGF(info, "Call for DataProcessorSPec aod-produce-workflow");
  return DataProcessorSpec{
    "aod-producer-workflow",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AODProducerWorkflowDPL>(src, dataRequest, ggRequest, enableSV, useMC)},
    Options{
      ConfigParamSpec{"run-number", VariantType::Int64, -1L, {"The run-number. If left default we try to get it from DPL header."}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}},
      ConfigParamSpec{"fill-calo-cells", VariantType::Int, 1, {"Fill calo cells into cell table"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"lpmp-prod-tag", VariantType::String, "", {"LPMProductionTag"}},
      ConfigParamSpec{"anchor-pass", VariantType::String, "", {"AnchorPassName"}},
      ConfigParamSpec{"anchor-prod", VariantType::String, "", {"AnchorProduction"}},
      ConfigParamSpec{"reco-pass", VariantType::String, "", {"RecoPassName"}},
      ConfigParamSpec{"nthreads", VariantType::Int, std::max(1, int(std::thread::hardware_concurrency() / 2)), {"Number of threads"}},
      ConfigParamSpec{"reco-mctracks-only", VariantType::Int, 0, {"Store only reconstructed MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}},
      ConfigParamSpec{"ctpreadout-create", VariantType::Int, 0, {"Create CTP digits from detector readout and CTP inputs. !=1 -- off, 1 -- on"}}}};
}

} // namespace o2::aodproducer
