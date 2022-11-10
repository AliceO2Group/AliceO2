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

/// \file RecoContainer.cxx
/// \brief Wrapper container for different reconstructed object types
/// \author ruben.shahoyan@cern.ch

#include <fmt/format.h>
#include <chrono>
#include "Framework/TimingInfo.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/IRFrame.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/DecayNbody.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTBase/DPLAlpideParam.h"
// FIXME: ideally, the data formats definition should be independent of the framework
// collectData is using the input of ProcessingContext to extract the first valid
// header and the TF orbit from it
#include "Framework/ProcessingContext.h"
#include "Framework/DataRefUtils.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::globaltracking;
using namespace o2::framework;
namespace o2d = o2::dataformats;

using GTrackID = o2d::GlobalTrackID;
using DetID = o2::detectors::DetID;

RecoContainer::RecoContainer() = default;
RecoContainer::~RecoContainer() = default;

void DataRequest::addInput(const InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

void DataRequest::requestIRFramesITS()
{
  addInput({"IRFramesITS", "ITS", "IRFRAMES", 0, Lifetime::Timeframe});
  requestMap["IRFramesITS"] = false;
}

void DataRequest::requestITSTracks(bool mc)
{
  addInput({"trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackITSROF", "ITS", "ITSTrackROF", 0, Lifetime::Timeframe});
  addInput({"trackITSClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe});
  addInput({"alpparITS", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam")});
  if (mc) {
    addInput({"trackITSMCTR", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["trackITS"] = mc;
}

void DataRequest::requestMFTTracks(bool mc)
{
  addInput({"trackMFT", "MFT", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackMFTROF", "MFT", "MFTTrackROF", 0, Lifetime::Timeframe});
  addInput({"trackMFTClIdx", "MFT", "TRACKCLSID", 0, Lifetime::Timeframe});
  addInput({"alpparMFT", "MFT", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("MFT/Config/AlpideParam")});
  if (mc) {
    addInput({"trackMFTMCTR", "MFT", "TRACKSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["trackMFT"] = mc;
}

void DataRequest::requestMCHTracks(bool mc)
{
  addInput({"trackMCH", "MCH", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackMCHROF", "MCH", "TRACKROFS", 0, Lifetime::Timeframe});
  addInput({"trackMCHTRACKCLUSTERS", "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackMCHMCTR", "MCH", "TRACKLABELS", 0, Lifetime::Timeframe});
  }
  requestMap["trackMCH"] = mc;
}

void DataRequest::requestMIDTracks(bool mc)
{
  addInput({"trackMIDROF", "MID", "TRACKROFS", 0, Lifetime::Timeframe});
  addInput({"trackClMIDROF", "MID", "TRCLUSROFS", 0, Lifetime::Timeframe});
  addInput({"trackMID", "MID", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackMIDTRACKCLUSTERS", "MID", "TRACKCLUSTERS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackMIDMCTR", "MID", "TRACKLABELS", 0, Lifetime::Timeframe});
    addInput({"trackMIDMCTRCL", "MID", "TRCLUSLABELS", 0, Lifetime::Timeframe});
  }
  requestMap["trackMID"] = mc;
}

void DataRequest::requestTPCTracks(bool mc)
{
  addInput({"trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe});
  addInput({"trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe});
  if (requestMap.find("clusTPC") != requestMap.end()) {
    addInput({"clusTPCshmap", "TPC", "CLSHAREDMAP", 0, Lifetime::Timeframe});
  }
  if (mc) {
    addInput({"trackTPCMCTR", "TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe});
  }
  requestMap["trackTPC"] = mc;
}

void DataRequest::requestITSTPCTracks(bool mc)
{
  addInput({"trackITSTPC", "GLO", "TPCITS", 0, Lifetime::Timeframe});
  addInput({"trackITSTPCABREFS", "GLO", "TPCITSAB_REFS", 0, Lifetime::Timeframe});
  addInput({"trackITSTPCABCLID", "GLO", "TPCITSAB_CLID", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackITSTPCMCTR", "GLO", "TPCITS_MC", 0, Lifetime::Timeframe});
    addInput({"trackITSTPCABMCTR", "GLO", "TPCITSAB_MC", 0, Lifetime::Timeframe});
  }
  requestMap["trackITSTPC"] = mc;
}

void DataRequest::requestGlobalFwdTracks(bool mc)
{
  addInput({"fwdtracks", "GLO", "GLFWD", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"MCTruth", "GLO", "GLFWD_MC", 0, Lifetime::Timeframe});
  }
  requestMap["fwdtracks"] = mc;
}

void DataRequest::requestMFTMCHMatches(bool mc)
{
  addInput({"matchMFTMCH", "GLO", "MTC_MFTMCH", 0, Lifetime::Timeframe});
  requestMap["matchMFTMCH"] = mc;
}

void DataRequest::requestMCHMIDMatches(bool mc)
{
  addInput({"matchMCHMID", "GLO", "MTC_MCHMID", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"matchMCHMID_MCTR", "GLO", "MCMTC_MCHMID", 0, Lifetime::Timeframe});
  }
  requestMap["matchMCHMID"] = mc;
}

void DataRequest::requestTPCTOFTracks(bool mc)
{
  auto ss = getMatchingInputSubSpec();
  addInput({"matchTPCTOF", "TOF", "MTC_TPC", ss, Lifetime::Timeframe});
  addInput({"trackTPCTOF", "TOF", "TOFTRACKS_TPC", ss, Lifetime::Timeframe});
  if (mc) {
    addInput({"clsTOF_TPC_MCTR", "TOF", "MCMTC_TPC", ss, Lifetime::Timeframe});
  }
  requestMap["trackTPCTOF"] = mc;
}

void DataRequest::requestITSTPCTRDTracks(bool mc)
{
  addInput({"trackITSTPCTRD", "TRD", "MATCH_ITSTPC", 0, Lifetime::Timeframe});
  addInput({"trigITSTPCTRD", "TRD", "TRGREC_ITSTPC", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackITSTPCTRDMCTR", "TRD", "MCLB_ITSTPC", 0, Lifetime::Timeframe});
    addInput({"trackITSTPCTRDSAMCTR", "TRD", "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe});
  }
  requestMap["trackITSTPCTRD"] = mc;
}

void DataRequest::requestTPCTRDTracks(bool mc)
{
  auto ss = getMatchingInputSubSpec();
  addInput({"trackTPCTRD", "TRD", "MATCH_TPC", ss, Lifetime::Timeframe});
  addInput({"trigTPCTRD", "TRD", "TRGREC_TPC", ss, Lifetime::Timeframe});
  if (mc) {
    addInput({"trackTPCTRDMCTR", "TRD", "MCLB_TPC", ss, Lifetime::Timeframe});
    addInput({"trackTPCTRDSAMCTR", "TRD", "MCLB_TPC_TRD", ss, Lifetime::Timeframe});
  }
  requestMap["trackTPCTRD"] = mc;
}

void DataRequest::requestTOFMatches(o2::dataformats::GlobalTrackID::mask_t src, bool mc)
{
  if (src[GTrackID::ITSTPCTOF]) {
    addInput({"matchITSTPCTOF", "TOF", "MTC_ITSTPC", 0, Lifetime::Timeframe});
    if (mc) {
      addInput({"clsTOF_GLO_MCTR", "TOF", "MCMTC_ITSTPC", 0, Lifetime::Timeframe});
    }
    requestMap["matchTOF_ITSTPC"] = mc;
  }
  if (src[GTrackID::TPCTRDTOF]) {
    addInput({"matchTPCTRDTOF", "TOF", "MTC_TPCTRD", 0, Lifetime::Timeframe});
    if (mc) {
      addInput({"clsTOF_GLO2_MCTR", "TOF", "MCMTC_TPCTRD", 0, Lifetime::Timeframe});
    }
    requestMap["matchTOF_TPCTRD"] = mc;
  }
  if (src[GTrackID::ITSTPCTRDTOF]) {
    addInput({"matchITSTPCTRDTOF", "TOF", "MTC_ITSTPCTRD", 0, Lifetime::Timeframe});
    if (mc) {
      addInput({"clsTOF_GLO3_MCTR", "TOF", "MCMTC_ITSTPCTRD", 0, Lifetime::Timeframe});
    }
    requestMap["matchTOF_ITSTPCTRD"] = mc;
  }
}

void DataRequest::requestITSClusters(bool mc)
{
  addInput({"clusITS", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe});
  addInput({"clusITSPatt", "ITS", "PATTERNS", 0, Lifetime::Timeframe});
  addInput({"clusITSROF", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe});
  addInput({"cldictITS", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary")});
  addInput({"alpparITS", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam")});
  if (mc) {
    addInput({"clusITSMC", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["clusITS"] = mc;
}

void DataRequest::requestMFTClusters(bool mc)
{
  addInput({"clusMFT", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe});
  addInput({"clusMFTPatt", "MFT", "PATTERNS", 0, Lifetime::Timeframe});
  addInput({"clusMFTROF", "MFT", "CLUSTERSROF", 0, Lifetime::Timeframe});
  addInput({"cldictMFT", "MFT", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/ClusterDictionary")});
  addInput({"alpparMFT", "MFT", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("MFT/Config/AlpideParam")});
  if (mc) {
    addInput({"clusMFTMC", "MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["clusMFT"] = mc;
}

void DataRequest::requestTPCClusters(bool mc)
{
  addInput({"clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe});
  if (requestMap.find("trackTPC") != requestMap.end()) {
    addInput({"clusTPCshmap", "TPC", "CLSHAREDMAP", 0, Lifetime::Timeframe});
  }
  if (mc) {
    addInput({"clusTPCMC", ConcreteDataTypeMatcher{"TPC", "CLNATIVEMCLBL"}, Lifetime::Timeframe});
  }
  requestMap["clusTPC"] = mc;
}

void DataRequest::requestTOFClusters(bool mc)
{
  addInput({"tofcluster", "TOF", "CLUSTERS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"tofclusterlabel", "TOF", "CLUSTERSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["clusTOF"] = mc;
}

void DataRequest::requestMCHClusters(bool mc)
{
  if (mc) {
    LOG(warn) << "MCH global clusters do not support MC lables, disabling";
    mc = false;
  }
  addInput({"clusMCH", "MCH", "GLOBALCLUSTERS", 0, Lifetime::Timeframe});
  addInput({"clusMCHROF", "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"clusMCHMC", "MCH", "CLUSTERLABELS", 0, Lifetime::Timeframe});
  }
  requestMap["clusMCH"] = mc;
}

void DataRequest::requestHMPClusters(bool mc)
{
  if (mc) { // RS: remove this once labels will be available
    LOG(warn) << "HMP clusters do not support MC lables, disabling";
    mc = false;
  }
  addInput({"hmpidcluster", "HMP", "CLUSTERS", 0, Lifetime::Timeframe});
  addInput({"hmpidtriggers", "HMP", "INTRECORDS1", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"hmpidclusterlabel", "HMP", "CLUSTERSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["clusHMP"] = mc;
}

void DataRequest::requestMIDClusters(bool mc)
{
  addInput({"clusMID", "MID", "CLUSTERS", 0, Lifetime::Timeframe});
  addInput({"clusMIDROF", "MID", "CLUSTERSROF", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"clusMIDMC", "MID", "CLUSTERSLABELS", 0, Lifetime::Timeframe});
  }
  requestMap["clusMID"] = mc;
}

void DataRequest::requestTRDTracklets(bool mc)
{
  addInput({"trdtracklets", o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe});
  addInput({"trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe});
  addInput({"trdtrigrecmask", o2::header::gDataOriginTRD, "TRIGRECMASK", 0, Lifetime::Timeframe});
  addInput({"trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"trdtrackletlabels", o2::header::gDataOriginTRD, "TRKLABELS", 0, Lifetime::Timeframe});
  }
  requestMap["trackletTRD"] = mc;
}

void DataRequest::requestFT0RecPoints(bool mc)
{
  addInput({"ft0recpoints", "FT0", "RECPOINTS", 0, Lifetime::Timeframe});
  addInput({"ft0channels", "FT0", "RECCHDATA", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(error) << "FT0 RecPoint does not support MC truth";
  }
  requestMap["FT0"] = false;
}

void DataRequest::requestFV0RecPoints(bool mc)
{
  addInput({"fv0recpoints", "FV0", "RECPOINTS", 0, Lifetime::Timeframe});
  addInput({"fv0channels", "FV0", "RECCHDATA", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(error) << "FV0 RecPoint does not support MC truth";
  }
  requestMap["FV0"] = false;
}

void DataRequest::requestFDDRecPoints(bool mc)
{
  addInput({"fddrecpoints", "FDD", "RECPOINTS", 0, Lifetime::Timeframe});
  addInput({"fddchannels", "FDD", "RECCHDATA", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(error) << "FDD RecPoint does not support MC truth";
  }
  requestMap["FDD"] = false;
}

void DataRequest::requestZDCRecEvents(bool mc)
{
  addInput({"zdcbcrec", "ZDC", "BCREC", 0, Lifetime::Timeframe});
  addInput({"zdcenergy", "ZDC", "ENERGY", 0, Lifetime::Timeframe});
  addInput({"zdctdcdata", "ZDC", "TDCDATA", 0, Lifetime::Timeframe});
  addInput({"zdcinfo", "ZDC", "INFO", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(error) << "ZDC RecEvent does not support MC truth";
  }
  requestMap["ZDC"] = false;
}

void DataRequest::requestCoscmicTracks(bool mc)
{
  addInput({"cosmics", "GLO", "COSMICTRC", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"cosmicsMC", "GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe});
  }
  requestMap["Cosmics"] = mc;
}

void DataRequest::requestPrimaryVertertices(bool mc)
{
  addInput({"pvtx", "GLO", "PVTX", 0, Lifetime::Timeframe});
  addInput({"pvtx_trmtc", "GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe});    // global ids of associated tracks
  addInput({"pvtx_tref", "GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe}); // vertex - trackID refs
  if (mc) {
    addInput({"pvtx_mc", "GLO", "PVTX_MCTR", 0, Lifetime::Timeframe});
  }
  requestMap["PVertex"] = mc;
}

void DataRequest::requestPrimaryVerterticesTMP(bool mc) // primary vertices before global vertex-track matching
{
  addInput({"pvtx", "GLO", "PVTX", 0, Lifetime::Timeframe});
  addInput({"pvtx_cont", "GLO", "PVTX_CONTID", 0, Lifetime::Timeframe});        // global ids of contributors
  addInput({"pvtx_contref", "GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe}); // vertex - trackID refs of contributors
  if (mc) {
    addInput({"pvtx_mc", "GLO", "PVTX_MCTR", 0, Lifetime::Timeframe});
  }
  requestMap["PVertexTMP"] = mc;
}

void DataRequest::requestSecondaryVertertices(bool)
{
  addInput({"v0s", "GLO", "V0S", 0, Lifetime::Timeframe});
  addInput({"p2v0s", "GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe});
  addInput({"cascs", "GLO", "CASCS", 0, Lifetime::Timeframe});
  addInput({"p2cascs", "GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe});
  addInput({"decay3body", "GLO", "DECAYS3BODY", 0, Lifetime::Timeframe});
  addInput({"p2decay3body", "GLO", "PVTX_3BODYREFS", 0, Lifetime::Timeframe});
  requestMap["SVertex"] = false; // no MC provided for secondary vertices
}

void DataRequest::requestCTPDigits(bool mc)
{
  addInput({"CTPDigits", "CTP", "DIGITS", 0, Lifetime::Timeframe});
  addInput({"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe});
  if (mc) {
    LOG(warning) << "MC truth not implemented for CTP";
    // addInput({"CTPDigitsMC", "CTP", "DIGITSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["CTPDigits"] = false;
}

void DataRequest::requestCPVClusters(bool mc)
{
  addInput({"CPVClusters", "CPV", "CLUSTERS", 0, Lifetime::Timeframe});
  addInput({"CPVTriggers", "CPV", "CLUSTERTRIGRECS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"CPVClustersMC", "CPV", "CLUSTERTRUEMC", 0, Lifetime::Timeframe});
  }
  requestMap["CPVClusters"] = mc;
}

void DataRequest::requestPHOSCells(bool mc)
{
  addInput({"PHSCells", "PHS", "CELLS", 0, Lifetime::Timeframe});
  addInput({"PHSTriggers", "PHS", "CELLTRIGREC", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"PHSCellsMC", "PHS", "CELLSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["PHSCells"] = mc;
}

void DataRequest::requestEMCALCells(bool mc)
{
  addInput({"EMCCells", "EMC", "CELLS", 0, Lifetime::Timeframe});
  addInput({"EMCTriggers", "EMC", "CELLSTRGR", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"EMCCellsMC", "EMC", "CELLSMCTR", 0, Lifetime::Timeframe});
  }
  requestMap["EMCCells"] = mc;
}

/*
void DataRequest::requestHMPMatches(bool mc)
{
  addInput({"matchHMP", "HMP", "MATCHES", 0, Lifetime::Timeframe});
  addInput({"matchTriggerHMP", "HMP", "TRACKREFS", 0, Lifetime::Timeframe});
  addInput({"matchPhotsCharge", "HMP", "PATTERNS", 0, Lifetime::Timeframe});
  if (mc) {
    addInput({"clsHMP_GLO_MCTR", "HMP", "MCLABELS", 0, Lifetime::Timeframe});
  }
  requestMap["matchHMP"] = mc;
}
*/

void DataRequest::requestTracks(GTrackID::mask_t src, bool useMC)
{
  // request tracks for sources probided by the mask
  if (src[GTrackID::ITS]) {
    requestITSTracks(useMC);
  }
  if (src[GTrackID::MFT]) {
    requestMFTTracks(useMC);
  }
  if (src[GTrackID::MCH] || src[GTrackID::MCHMID]) {
    requestMCHTracks(useMC);
  }
  if (src[GTrackID::MID]) {
    requestMIDTracks(useMC);
  }
  if (src[GTrackID::TPC]) {
    requestTPCTracks(useMC);
  }
  if (src[GTrackID::ITSTPC] || src[GTrackID::ITSTPCTOF]) {
    requestITSTPCTracks(useMC);
  }
  if (src[GTrackID::MFTMCH]) {
    requestGlobalFwdTracks(useMC);
  }
  if (src[GTrackID::MCHMID]) {
    requestMCHMIDMatches(useMC);
  }
  if (src[GTrackID::TPCTOF]) {
    requestTPCTOFTracks(useMC);
  }
  if (src[GTrackID::TPCTRD] || src[GTrackID::TPCTRDTOF]) {
    requestTPCTRDTracks(useMC);
  }
  if (src[GTrackID::ITSTPCTRD] || src[GTrackID::ITSTPCTRDTOF]) {
    requestITSTPCTRDTracks(useMC);
  }
  if (src[GTrackID::ITSTPCTRDTOF] || src[GTrackID::ITSTPCTOF] || src[GTrackID::TPCTRDTOF]) {
    requestTOFMatches(src, useMC);
    requestTOFClusters(false);
  }
  if (src[GTrackID::FT0]) {
    requestFT0RecPoints(false); // RS FIXME: at the moment does not support MC
  }
  if (src[GTrackID::FV0]) {
    requestFV0RecPoints(false); // RS FIXME: at the moment does not support MC
  }
  if (src[GTrackID::FDD]) {
    requestFDDRecPoints(false); // RS FIXME: at the moment does not support MC
  }
  if (src[GTrackID::ZDC]) {
    requestZDCRecEvents(false); // RS FIXME: at the moment does not support MC
  }
  if (GTrackID::includesDet(DetID::CTP, src)) {
    requestCTPDigits(false); // RS FIXME: at the moment does not support MC
  }
  if (GTrackID::includesDet(DetID::CPV, src)) {
    requestCPVClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::PHS, src)) {
    requestPHOSCells(useMC);
  }
  if (GTrackID::includesDet(DetID::EMC, src)) {
    requestEMCALCells(useMC);
  }
  if (GTrackID::includesDet(DetID::HMP, src)) {
    requestHMPClusters(useMC);
  }
  //  if (src[GTrackID::HMP]) {
  //    requestHMPMatches(useMC);
  //  }
}

void DataRequest::requestClusters(GTrackID::mask_t src, bool useMC, DetID::mask_t skipDetClusters)
{
  // request clusters for detectors of the sources probided by the mask
  // clusters needed for refits
  if (GTrackID::includesDet(DetID::ITS, src) && !skipDetClusters[DetID::ITS]) {
    requestITSClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::MFT, src) && !skipDetClusters[DetID::MFT]) {
    requestMFTClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::TPC, src) && !skipDetClusters[DetID::TPC]) {
    requestTPCClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::TOF, src) && !skipDetClusters[DetID::TOF]) {
    requestTOFClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::TRD, src) && !skipDetClusters[DetID::TRD]) {
    requestTRDTracklets(useMC);
  }
  if (GTrackID::includesDet(DetID::CTP, src) && !skipDetClusters[DetID::CTP]) {
    requestCTPDigits(false); // RS FIXME: at the moment does not support MC
  }
  if (GTrackID::includesDet(DetID::CPV, src) && !skipDetClusters[DetID::CPV]) {
    requestCPVClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::PHS, src) && !skipDetClusters[DetID::PHS]) {
    requestPHOSCells(useMC);
  }
  if (GTrackID::includesDet(DetID::EMC, src) && !skipDetClusters[DetID::EMC]) {
    requestEMCALCells(useMC);
  }
  if (GTrackID::includesDet(DetID::MCH, src) && !skipDetClusters[DetID::MCH]) {
    requestMCHClusters(useMC);
  }
  if (GTrackID::includesDet(DetID::HMP, src) && !skipDetClusters[DetID::HMP]) {
    requestHMPClusters(useMC);
  }
}

//__________________________________________________________________
void RecoContainer::collectData(ProcessingContext& pc, const DataRequest& requests)
{
  auto& reqMap = requests.requestMap;

  startIR = {0, pc.services().get<o2::framework::TimingInfo>().firstTForbit};

  auto req = reqMap.find("trackITS");
  if (req != reqMap.end()) {
    addITSTracks(pc, req->second);
  }

  req = reqMap.find("trackMFT");
  if (req != reqMap.end()) {
    addMFTTracks(pc, req->second);
  }

  req = reqMap.find("trackMCH");
  if (req != reqMap.end()) {
    addMCHTracks(pc, req->second);
  }

  req = reqMap.find("trackMID");
  if (req != reqMap.end()) {
    addMIDTracks(pc, req->second);
  }

  req = reqMap.find("trackTPC");
  if (req != reqMap.end()) {
    addTPCTracks(pc, req->second);
  }

  req = reqMap.find("trackITSTPC");
  if (req != reqMap.end()) {
    addITSTPCTracks(pc, req->second);
  }

  req = reqMap.find("fwdtracks");
  if (req != reqMap.end()) {
    addGlobalFwdTracks(pc, req->second);
  }

  req = reqMap.find("matchMFTMCH");
  if (req != reqMap.end()) {
    addMFTMCHMatches(pc, req->second);
  }

  req = reqMap.find("matchMCHMID");
  if (req != reqMap.end()) {
    addMCHMIDMatches(pc, req->second);
  }

  req = reqMap.find("trackITSTPCTRD");
  if (req != reqMap.end()) {
    addITSTPCTRDTracks(pc, req->second);
  }

  req = reqMap.find("trackTPCTRD");
  if (req != reqMap.end()) {
    addTPCTRDTracks(pc, req->second);
  }

  req = reqMap.find("trackTPCTOF");
  if (req != reqMap.end()) {
    addTPCTOFTracks(pc, req->second);
  }

  req = reqMap.find("matchTOF_ITSTPC");
  if (req != reqMap.end()) {
    addTOFMatchesITSTPC(pc, req->second);
  }

  req = reqMap.find("matchTOF_TPCTRD");
  if (req != reqMap.end()) {
    addTOFMatchesTPCTRD(pc, req->second);
  }

  req = reqMap.find("matchTOF_ITSTPCTRD");
  if (req != reqMap.end()) {
    addTOFMatchesITSTPCTRD(pc, req->second);
  }

  req = reqMap.find("clusITS");
  if (req != reqMap.end()) {
    addITSClusters(pc, req->second);
  }

  req = reqMap.find("clusMFT");
  if (req != reqMap.end()) {
    addMFTClusters(pc, req->second);
  }

  req = reqMap.find("clusTPC");
  if (req != reqMap.end()) {
    addTPCClusters(pc, req->second, reqMap.find("trackTPC") != reqMap.end());
  }

  req = reqMap.find("clusTOF");
  if (req != reqMap.end()) {
    addTOFClusters(pc, req->second);
  }

  req = reqMap.find("clusHMP");
  if (req != reqMap.end()) {
    addHMPClusters(pc, req->second);
  }

  req = reqMap.find("CTPDigits");
  if (req != reqMap.end()) {
    addCTPDigits(pc, req->second);
  }

  req = reqMap.find("CPVClusters");
  if (req != reqMap.end()) {
    addCPVClusters(pc, req->second);
  }

  req = reqMap.find("PHSCells");
  if (req != reqMap.end()) {
    addPHOSCells(pc, req->second);
  }

  req = reqMap.find("EMCCells");
  if (req != reqMap.end()) {
    addEMCALCells(pc, req->second);
  }

  req = reqMap.find("clusMCH");
  if (req != reqMap.end()) {
    addMCHClusters(pc, req->second);
  }

  req = reqMap.find("clusMID");
  if (req != reqMap.end()) {
    addMIDClusters(pc, req->second);
  }

  req = reqMap.find("FT0");
  if (req != reqMap.end()) {
    addFT0RecPoints(pc, req->second);
  }

  req = reqMap.find("FV0");
  if (req != reqMap.end()) {
    addFV0RecPoints(pc, req->second);
  }

  req = reqMap.find("FDD");
  if (req != reqMap.end()) {
    addFDDRecPoints(pc, req->second);
  }

  req = reqMap.find("ZDC");
  if (req != reqMap.end()) {
    addZDCRecEvents(pc, req->second);
  }

  req = reqMap.find("trackletTRD");
  if (req != reqMap.end()) {
    addTRDTracklets(pc, req->second);
  }

  req = reqMap.find("Cosmics");
  if (req != reqMap.end()) {
    addCosmicTracks(pc, req->second);
  }

  req = reqMap.find("PVertex");
  if (req != reqMap.end()) {
    addPVertices(pc, req->second);
  }

  req = reqMap.find("PVertexTMP");
  if (req != reqMap.end()) {
    addPVerticesTMP(pc, req->second);
  }

  req = reqMap.find("SVertex");
  if (req != reqMap.end()) {
    addSVertices(pc, req->second);
  }

  req = reqMap.find("IRFramesITS");
  if (req != reqMap.end()) {
    addIRFramesITS(pc);
  }
  //  req = reqMap.find("matchHMP");
  //  if (req != reqMap.end()) {
  //    addHMPMatches(pc, req->second);
  //  }
}

//____________________________________________________________
void RecoContainer::addSVertices(ProcessingContext& pc, bool)
{
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::V0>>("v0s"), V0S);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::RangeReference<int, int>>>("p2v0s"), PVTX_V0REFS);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::Cascade>>("cascs"), CASCS);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::RangeReference<int, int>>>("p2cascs"), PVTX_CASCREFS);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::DecayNbody>>("decay3body"), DECAY3BODY);
  svtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::RangeReference<int, int>>>("p2decay3body"), PVTX_3BODYREFS);
  // no mc
}

//____________________________________________________________
void RecoContainer::addPVertices(ProcessingContext& pc, bool mc)
{
  if (!pvtxPool.isLoaded(PVTX)) { // in case was loaded via addPVerticesTMP
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("pvtx"), PVTX);
  }
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("pvtx_trmtc"), PVTX_TRMTC);
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("pvtx_tref"), PVTX_TRMTCREFS);

  if (mc && !pvtxPool.isLoaded(PVTX_MCTR)) { // in case was loaded via addPVerticesTMP
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::MCEventLabel>>("pvtx_mc"), PVTX_MCTR);
  }
}

//____________________________________________________________
void RecoContainer::addPVerticesTMP(ProcessingContext& pc, bool mc)
{
  if (!pvtxPool.isLoaded(PVTX)) { // in case was loaded via addPVertices
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("pvtx"), PVTX);
  }
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("pvtx_cont"), PVTX_CONTID);
  pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("pvtx_contref"), PVTX_CONTIDREFS);

  if (mc && !pvtxPool.isLoaded(PVTX_MCTR)) { // in case was loaded via addPVertices
    pvtxPool.registerContainer(pc.inputs().get<gsl::span<o2::MCEventLabel>>("pvtx_mc"), PVTX_MCTR);
  }
}

//____________________________________________________________
void RecoContainer::addCosmicTracks(ProcessingContext& pc, bool mc)
{
  cosmPool.registerContainer(pc.inputs().get<gsl::span<o2::dataformats::TrackCosmics>>("cosmics"), COSM_TRACKS);
  if (mc) {
    cosmPool.registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("cosmicsMC"), COSM_TRACKS_MC);
  }
}

//____________________________________________________________
void RecoContainer::addITSTracks(ProcessingContext& pc, bool mc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("alpparITS"); // note: configurable param does not need finaliseCCDB
  }
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS"), TRACKS);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<int>>("trackITSClIdx"), INDICES);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("trackITSROF"), TRACKREFS);
  if (mc) {
    commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTR"), MCLABELS);
  }
}

//____________________________________________________________
void RecoContainer::addIRFramesITS(ProcessingContext& pc)
{
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("IRFramesITS"), VARIA);
}

//____________________________________________________________
void RecoContainer::addMFTTracks(ProcessingContext& pc, bool mc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>*>("alpparMFT"); // note: configurable param does not need finaliseCCDB
  }
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::mft::TrackMFT>>("trackMFT"), TRACKS);
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<int>>("trackMFTClIdx"), INDICES);
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("trackMFTROF"), TRACKREFS);
  if (mc) {
    commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackMFTMCTR"), MCLABELS);
  }
}

//____________________________________________________________
void RecoContainer::addMCHTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MCH].registerContainer(pc.inputs().get<gsl::span<o2::mch::TrackMCH>>("trackMCH"), TRACKS);
  commonPool[GTrackID::MCH].registerContainer(pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("trackMCHROF"), TRACKREFS);
  commonPool[GTrackID::MCH].registerContainer(pc.inputs().get<gsl::span<o2::mch::Cluster>>("trackMCHTRACKCLUSTERS"), INDICES);
  if (mc) {
    commonPool[GTrackID::MCH].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackMCHMCTR"), MCLABELS);
  }
}

//____________________________________________________________
void RecoContainer::addMIDTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MID].registerContainer(pc.inputs().get<gsl::span<o2::mid::Track>>("trackMID"), TRACKS);
  commonPool[GTrackID::MID].registerContainer(pc.inputs().get<gsl::span<o2::mid::ROFRecord>>("trackMIDROF"), TRACKREFS);
  commonPool[GTrackID::MID].registerContainer(pc.inputs().get<gsl::span<o2::mid::Cluster>>("trackMIDTRACKCLUSTERS"), INDICES);
  commonPool[GTrackID::MID].registerContainer(pc.inputs().get<gsl::span<o2::mid::ROFRecord>>("trackClMIDROF"), MATCHES);
  if (mc) {
    commonPool[GTrackID::MID].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackMIDMCTR"), MCLABELS);
    mcMIDTrackClusters = pc.inputs().get<const dataformats::MCTruthContainer<o2::mid::MCClusterLabel>*>("trackMIDMCTRCL");
  }
}

//________________________________________________________
const o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>* RecoContainer::getMIDTracksClusterMCLabels() const
{
  return mcMIDTrackClusters.get();
}

//________________________________________________________
const o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>* RecoContainer::getMIDClustersMCLabels() const
{
  return mcMIDClusters.get();
}

//____________________________________________________________
void RecoContainer::addTPCTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TPC].registerContainer(pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC"), TRACKS);
  commonPool[GTrackID::TPC].registerContainer(pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs"), INDICES);
  if (mc) {
    commonPool[GTrackID::TPC].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCMCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addITSTPCTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITSTPC].registerContainer(pc.inputs().get<gsl::span<o2d::TrackTPCITS>>("trackITSTPC"), TRACKS);
  commonPool[GTrackID::ITSAB].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::TrkClusRef>>("trackITSTPCABREFS"), TRACKREFS);
  commonPool[GTrackID::ITSAB].registerContainer(pc.inputs().get<gsl::span<int>>("trackITSTPCABCLID"), INDICES);
  if (mc) {
    commonPool[GTrackID::ITSTPC].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCMCTR"), MCLABELS);
    commonPool[GTrackID::ITSAB].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCABMCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addGlobalFwdTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MFTMCH].registerContainer(pc.inputs().get<gsl::span<o2d::GlobalFwdTrack>>("fwdtracks"), TRACKS);
  if (mc) {
    commonPool[GTrackID::MFTMCH].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("MCTruth"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addMFTMCHMatches(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MFTMCH].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoFwd>>("matchMFTMCH"), MATCHES);
}

//__________________________________________________________
void RecoContainer::addMCHMIDMatches(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MCHMID].registerContainer(pc.inputs().get<gsl::span<o2d::TrackMCHMID>>("matchMCHMID"), MATCHES);
  if (mc) {
    commonPool[GTrackID::MCHMID].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("matchMCHMID_MCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addITSTPCTRDTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITSTPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTRD>>("trackITSTPCTRD"), TRACKS);
  commonPool[GTrackID::ITSTPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTriggerRecord>>("trigITSTPCTRD"), TRACKREFS);
  if (mc) {
    commonPool[GTrackID::ITSTPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCTRDMCTR"), MCLABELS);
    commonPool[GTrackID::ITSTPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCTRDSAMCTR"), MCLABELSEXTRA);
  }
}

//__________________________________________________________
void RecoContainer::addTPCTRDTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTRD>>("trackTPCTRD"), TRACKS);
  commonPool[GTrackID::TPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::trd::TrackTriggerRecord>>("trigTPCTRD"), TRACKREFS);
  if (mc) {
    commonPool[GTrackID::TPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCTRDMCTR"), MCLABELS);
    commonPool[GTrackID::TPCTRD].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCTRDSAMCTR"), MCLABELSEXTRA);
  }
}

//__________________________________________________________
void RecoContainer::addTPCTOFTracks(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TPCTOF].registerContainer(pc.inputs().get<gsl::span<o2d::TrackTPCTOF>>("trackTPCTOF"), TRACKS);
  commonPool[GTrackID::TPCTOF].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchTPCTOF"), MATCHES);
  if (mc) {
    commonPool[GTrackID::TPCTOF].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_TPC_MCTR"), MCLABELS);
  }
}

//__________________________________________________________
void RecoContainer::addTOFMatchesITSTPC(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITSTPCTOF].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchITSTPCTOF"), MATCHES); // only ITS/TPC : TOF match info, no real tracks
  if (mc) {
    commonPool[GTrackID::ITSTPCTOF].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_GLO_MCTR"), MCLABELS);
  }
}
//__________________________________________________________
void RecoContainer::addTOFMatchesTPCTRD(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TPCTRDTOF].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchTPCTRDTOF"), MATCHES); // only ITS/TPC : TOF match info, no real tracks
  if (mc) {
    commonPool[GTrackID::TPCTRDTOF].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_GLO2_MCTR"), MCLABELS);
  }
}
//__________________________________________________________
void RecoContainer::addTOFMatchesITSTPCTRD(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ITSTPCTRDTOF].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoTOF>>("matchITSTPCTRDTOF"), MATCHES); // only ITS/TPC : TOF match info, no real tracks
  if (mc) {
    commonPool[GTrackID::ITSTPCTRDTOF].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_GLO3_MCTR"), MCLABELS);
  }
}

/*
//__________________________________________________________
void RecoContainer::addHMPMatches(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::HMP].registerContainer(pc.inputs().get<gsl::span<o2d::MatchInfoHMP>>("matchHMP"), MATCHES);           //  HMPID match info, no real tracks
  commonPool[GTrackID::HMP].registerContainer(pc.inputs().get<gsl::span<o2::hmpid::Trigger>>("matchTriggerHMP"), TRACKREFS); //  HMPID triggers
  commonPool[GTrackID::HMP].registerContainer(pc.inputs().get<gsl::span<float>>("matchPhotsCharge"), PATTERNS);              //  HMPID photon cluster charges
  if (mc) {
    commonPool[GTrackID::HMP].registerContainer(pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsHMP_GLO_MCTR"), MCLABELS);
  }
}
*/

//__________________________________________________________
void RecoContainer::addITSClusters(ProcessingContext& pc, bool mc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldictITS");                        // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("alpparITS"); // note: configurable param does not need finaliseCCDB
  }
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clusITSROF"), CLUSREFS);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("clusITS"), CLUSTERS);
  commonPool[GTrackID::ITS].registerContainer(pc.inputs().get<gsl::span<unsigned char>>("clusITSPatt"), PATTERNS);
  if (mc) {
    mcITSClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("clusITSMC");
  }
}

//__________________________________________________________
void RecoContainer::addMFTClusters(ProcessingContext& pc, bool mc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldictMFT");                        // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>*>("alpparMFT"); // note: configurable param does not need finaliseCCDB
  }
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clusMFTROF"), CLUSREFS);
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("clusMFT"), CLUSTERS);
  commonPool[GTrackID::MFT].registerContainer(pc.inputs().get<gsl::span<unsigned char>>("clusMFTPatt"), PATTERNS);
  if (mc) {
    mcITSClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("clusMFTMC");
  }
}

//__________________________________________________________
void RecoContainer::addTPCClusters(ProcessingContext& pc, bool mc, bool shmap)
{
  inputsTPCclusters = o2::tpc::getWorkflowTPCInput(pc, 0, mc);
  if (shmap) {
    clusterShMapTPC = pc.inputs().get<gsl::span<unsigned char>>("clusTPCshmap");
  }
}

//__________________________________________________________
void RecoContainer::addTRDTracklets(ProcessingContext& pc, bool mc)
{
  inputsTRD = o2::trd::getRecoInputContainer(pc, nullptr, this, mc);
}

//__________________________________________________________
void RecoContainer::addTOFClusters(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::TOF].registerContainer(pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster"), CLUSTERS);
  if (mc) {
    mcTOFClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("tofclusterlabel");
  }
}

//__________________________________________________________
void RecoContainer::addHMPClusters(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::HMP].registerContainer(pc.inputs().get<gsl::span<o2::hmpid::Cluster>>("hmpidcluster"), CLUSTERS);
  commonPool[GTrackID::HMP].registerContainer(pc.inputs().get<gsl::span<o2::hmpid::Trigger>>("hmpidtriggers"), CLUSREFS);
  if (mc) {
    mcHMPClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("hmpidclusterlabel");
  }
}
//__________________________________________________________
void RecoContainer::addMCHClusters(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MCH].registerContainer(pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("clusMCHROF"), CLUSREFS);
  commonPool[GTrackID::MCH].registerContainer(pc.inputs().get<gsl::span<o2::mch::Cluster>>("clusMCH"), CLUSTERS);
  if (mc) {
    mcMCHClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("clusMCHMC");
  }
}

//__________________________________________________________
void RecoContainer::addMIDClusters(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::MID].registerContainer(pc.inputs().get<gsl::span<o2::mid::ROFRecord>>("clusMIDROF"), CLUSREFS);
  commonPool[GTrackID::MID].registerContainer(pc.inputs().get<gsl::span<o2::mid::Cluster>>("clusMID"), CLUSTERS);
  if (mc) {
    mcMIDClusters = pc.inputs().get<const dataformats::MCTruthContainer<o2::mid::MCClusterLabel>*>("clusMIDMC");
  }
}

//__________________________________________________________
void RecoContainer::addCTPDigits(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::CTP].registerContainer(pc.inputs().get<gsl::span<o2::ctp::CTPDigit>>("CTPDigits"), CLUSTERS);
  mCTPLumi = pc.inputs().get<o2::ctp::LumiInfo>("CTPLumi");
  if (mc) {
    //  pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("CTPDigitsMC");
  }
}

//__________________________________________________________
void RecoContainer::addCPVClusters(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::CPV].registerContainer(pc.inputs().get<gsl::span<o2::cpv::Cluster>>("CPVClusters"), CLUSTERS);
  commonPool[GTrackID::CPV].registerContainer(pc.inputs().get<gsl::span<o2::cpv::TriggerRecord>>("CPVTriggers"), CLUSREFS);
  if (mc) {
    mcCPVClusters = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("CPVClustersMC");
  }
}

//__________________________________________________________
void RecoContainer::addPHOSCells(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::PHS].registerContainer(pc.inputs().get<gsl::span<o2::phos::Cell>>("PHSCells"), CLUSTERS);
  commonPool[GTrackID::PHS].registerContainer(pc.inputs().get<gsl::span<o2::phos::TriggerRecord>>("PHSTriggers"), CLUSREFS);
  if (mc) {
    mcPHSCells = pc.inputs().get<const dataformats::MCTruthContainer<o2::phos::MCLabel>*>("PHSCellsMC");
  }
}

//__________________________________________________________
void RecoContainer::addEMCALCells(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::EMC].registerContainer(pc.inputs().get<gsl::span<o2::emcal::Cell>>("EMCCells"), CLUSTERS);
  commonPool[GTrackID::EMC].registerContainer(pc.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("EMCTriggers"), CLUSREFS);
  if (mc) {
    mcEMCCells = pc.inputs().get<const dataformats::MCTruthContainer<o2::emcal::MCLabel>*>("EMCCellsMC");
  }
}

//__________________________________________________________
void RecoContainer::addFT0RecPoints(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::FT0].registerContainer(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("ft0recpoints"), TRACKS);
  commonPool[GTrackID::FT0].registerContainer(pc.inputs().get<gsl::span<o2::ft0::ChannelDataFloat>>("ft0channels"), CLUSTERS);

  if (mc) {
    LOG(error) << "FT0 RecPoint does not support MC truth";
  }
}

//__________________________________________________________
void RecoContainer::addFV0RecPoints(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::FV0].registerContainer(pc.inputs().get<gsl::span<o2::fv0::RecPoints>>("fv0recpoints"), TRACKS);
  commonPool[GTrackID::FV0].registerContainer(pc.inputs().get<gsl::span<o2::fv0::ChannelDataFloat>>("fv0channels"), CLUSTERS);

  if (mc) {
    LOG(error) << "FV0 RecPoint does not support MC truth";
  }
}

//__________________________________________________________
void RecoContainer::addFDDRecPoints(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::FDD].registerContainer(pc.inputs().get<gsl::span<o2::fdd::RecPoint>>("fddrecpoints"), TRACKS);
  commonPool[GTrackID::FDD].registerContainer(pc.inputs().get<gsl::span<o2::fdd::ChannelDataFloat>>("fddchannels"), CLUSTERS);

  if (mc) {
    LOG(error) << "FDD RecPoint does not support MC truth";
  }
}

//__________________________________________________________
void RecoContainer::addZDCRecEvents(ProcessingContext& pc, bool mc)
{
  commonPool[GTrackID::ZDC].registerContainer(pc.inputs().get<gsl::span<o2::zdc::BCRecData>>("zdcbcrec"), MATCHES);
  commonPool[GTrackID::ZDC].registerContainer(pc.inputs().get<gsl::span<o2::zdc::ZDCEnergy>>("zdcenergy"), TRACKS);
  commonPool[GTrackID::ZDC].registerContainer(pc.inputs().get<gsl::span<o2::zdc::ZDCTDCData>>("zdctdcdata"), CLUSTERS);
  commonPool[GTrackID::ZDC].registerContainer(pc.inputs().get<gsl::span<uint16_t>>("zdcinfo"), PATTERNS);

  if (mc) {
    LOG(error) << "ZDC RecEvent does not support MC truth";
  }
}

const o2::tpc::ClusterNativeAccess& RecoContainer::getTPCClusters() const
{
  return inputsTPCclusters->clusterIndex;
}

gsl::span<const o2::trd::Tracklet64> RecoContainer::getTRDTracklets() const
{
  return inputsTRD->mTracklets;
}

gsl::span<const o2::trd::CalibratedTracklet> RecoContainer::getTRDCalibratedTracklets() const
{
  return inputsTRD->mSpacePoints;
}

gsl::span<const o2::trd::TriggerRecord> RecoContainer::getTRDTriggerRecords() const
{
  static int countWarnings = 0;
  if (inputsTRD == nullptr) {
    if (countWarnings < 1) {
      LOG(warning) << "No TRD triggers";
      countWarnings++;
    }
    return gsl::span<const o2::trd::TriggerRecord>();
  } else {
    return inputsTRD->mTriggerRecords;
  }
}

const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* RecoContainer::getTRDTrackletsMCLabels() const
{
  return inputsTRD->mTrackletLabels.get();
}

const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* RecoContainer::getTPCClustersMCLabels() const
{
  return inputsTPCclusters->clusterIndex.clustersMCTruth;
}

//__________________________________________________________
const o2::track::TrackParCov& RecoContainer::getTrackParamOut(GTrackID gidx) const
{
  // get outer param of track
  auto trSrc = gidx.getSource();
  if (trSrc == GTrackID::ITSTPC) {
    return getTrack<o2d::TrackTPCITS>(gidx).getParamOut();
  } else if (trSrc == GTrackID::ITSTPCTOF) { // the physical tracks are in ITS-TPC, need to get reference from match info
    return getTrack<o2d::TrackTPCITS>(getTOFMatch(gidx).getTrackRef()).getParamOut();
  } else if (trSrc == GTrackID::TPCTOF) {
    return getTrack<o2d::TrackTPCTOF>(gidx).getParamOut();
  } else if (trSrc == GTrackID::ITSTPCTRDTOF) { // the physical tracks are in ITS-TPC-TRD, need to get reference from match info
    return getTrack<o2::trd::TrackTRD>(getTOFMatch(gidx).getTrackRef()).getOuterParam();
  }
  if (trSrc == GTrackID::TPCTRDTOF) { // the physical tracks are in TPC-TRD, need to get reference from match info
    return getTrack<o2::trd::TrackTRD>(getTOFMatch(gidx).getTrackRef()).getOuterParam();
  } else if (trSrc == GTrackID::ITSTPCTRD) {
    return getTrack<o2::trd::TrackTRD>(gidx).getOuterParam();
  }
  if (trSrc == GTrackID::TPCTRD) {
    return getTrack<o2::trd::TrackTRD>(gidx).getOuterParam();
  } else if (trSrc == GTrackID::ITS) {
    return getTrack<o2::its::TrackITS>(gidx).getParamOut();
  } else if (trSrc == GTrackID::TPC) {
    return getTrack<o2::tpc::TrackTPC>(gidx).getParamOut();
  } else {
    throw std::runtime_error(fmt::format("not defined for tracks of source {:d}", int(trSrc)));
  }
}

//__________________________________________________________
bool RecoContainer::isTrackSourceLoaded(int src) const
{
  if (src == GTrackID::ITSTPCTOF) {
    if (!isMatchSourceLoaded(src)) { // the physical tracks are in ITS-TPC, need to get reference from match info
      return false;
    }
    src = GTrackID::ITSTPC;
  }
  if (src == GTrackID::TPCTRDTOF) {
    if (!isMatchSourceLoaded(src)) { // the physical tracks are in ITS-TPC, need to get reference from match info
      return false;
    }
    src = GTrackID::TPCTRD;
  }
  if (src == GTrackID::ITSTPCTRDTOF) {
    if (!isMatchSourceLoaded(src)) { // the physical tracks are in ITS-TPC, need to get reference from match info
      return false;
    }
    src = GTrackID::ITSTPCTRD;
  }
  return commonPool[src].isLoaded(TRACKS);
}

//__________________________________________________________
const o2::track::TrackParCov& RecoContainer::getTrackParam(GTrackID gidx) const
{
  // get base track
  auto trSrc = gidx.getSource();
  if (trSrc == GTrackID::ITSTPCTOF || trSrc == GTrackID::TPCTRDTOF || trSrc == GTrackID::ITSTPCTRDTOF) { // the physical tracks are in ITS-TPC, need to get reference from match info
    gidx = getTOFMatch(gidx).getTrackRef();
  }
  return getObject<o2::track::TrackParCov>(gidx, TRACKS);
}

//__________________________________________________________
const o2::dataformats::TrackTPCITS& RecoContainer::getITSTPCTOFTrack(GTrackID gidx) const
{
  // get ITS-TPC track pointed by global TOF match
  return getTPCITSTrack(getTOFMatch(gidx).getTrackRef());
}

//________________________________________________________
void RecoContainer::fillTrackMCLabels(const gsl::span<GTrackID> gids, std::vector<o2::MCCompLabel>& mcinfo) const
{
  // fills the MCLabels corresponding to gids to MC info
  mcinfo.clear();
  mcinfo.reserve(gids.size());
  for (auto gid : gids) {
    mcinfo.push_back(getTrackMCLabel(gid));
  }
}

//________________________________________________________
void o2::globaltracking::RecoContainer::createTracks(std::function<bool(const o2::track::TrackParCov&, o2::dataformats::GlobalTrackID)> const& creator) const
{
  createTracksVariadic([&creator](const auto& _tr, GTrackID _origID, float t0, float terr) {
    if constexpr (std::is_base_of_v<o2::track::TrackParCov, std::decay_t<decltype(_tr)>>) {
      return creator(_tr, _origID);
    } else {
      return false;
    }
  });
}

//________________________________________________________
// get contributors from single detectors
RecoContainer::GlobalIDSet RecoContainer::getSingleDetectorRefs(GTrackID gidx) const
{
  GlobalIDSet table;
  auto src = gidx.getSource();
  table[src] = gidx;
  if (src == GTrackID::ITSTPCTRD) {
    const auto& parent0 = getITSTPCTRDTrack<o2::trd::TrackTRD>(gidx);
    const auto& parent1 = getTPCITSTrack(parent0.getRefGlobalTrackId());
    table[GTrackID::ITSTPC] = parent0.getRefGlobalTrackId();
    table[parent1.getRefITS().getSource()] = parent1.getRefITS();
    table[GTrackID::TPC] = parent1.getRefTPC();
    table[GTrackID::TRD] = gidx; // there is no standalone TRD track, so use the index for the ITSTPCTRD track array
  } else if (src == GTrackID::TPCTRD) {
    const auto& parent0 = getTPCTRDTrack<o2::trd::TrackTRD>(gidx);
    table[GTrackID::TPC] = parent0.getRefGlobalTrackId();
    table[GTrackID::TRD] = gidx; // there is no standalone TRD track, so use the index for the TPCTRD track array
  } else if (src == GTrackID::ITSTPCTOF) {
    const auto& parent0 = getTOFMatch(gidx); // ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack(parent0.getTrackRef());
    table[GTrackID::ITSTPC] = parent0.getTrackRef();
    table[GTrackID::TOF] = {unsigned(parent0.getIdxTOFCl()), GTrackID::TOF};
    table[GTrackID::TPC] = parent1.getRefTPC();
    table[parent1.getRefITS().getSource()] = parent1.getRefITS(); // ITS source might be an ITS track or ITSAB tracklet
  } else if (src == GTrackID::ITSTPCTRDTOF) {
    const auto& parent0 = getTOFMatch(gidx); // ITS/TPC : TOF
    const auto& parent1 = getITSTPCTRDTrack<o2::trd::TrackTRD>(parent0.getTrackRef());
    const auto& parent2 = getTPCITSTrack(parent1.getRefGlobalTrackId());
    table[GTrackID::ITSTPCTRD] = parent0.getTrackRef();
    table[GTrackID::ITSTPC] = parent1.getRefGlobalTrackId();
    table[GTrackID::TOF] = {unsigned(parent0.getIdxTOFCl()), GTrackID::TOF};
    table[GTrackID::TPC] = parent2.getRefTPC();
    table[parent2.getRefITS().getSource()] = parent2.getRefITS(); // ITS source might be an ITS track or ITSAB tracklet
    table[GTrackID::TRD] = parent0.getTrackRef();                 // there is no standalone TRD track, so use the index for the ITSTPCTRD track array
  } else if (src == GTrackID::TPCTRDTOF) {
    const auto& parent0 = getTOFMatch(gidx); // TPCTRD : TOF
    const auto& parent1 = getITSTPCTRDTrack<o2::trd::TrackTRD>(parent0.getTrackRef());
    const auto& parent2 = getTPCITSTrack(parent1.getRefGlobalTrackId());
    table[GTrackID::TPCTRD] = parent0.getTrackRef();
    table[GTrackID::TPC] = parent1.getRefGlobalTrackId();
    table[GTrackID::TOF] = {unsigned(parent0.getIdxTOFCl()), GTrackID::TOF};
    table[GTrackID::TRD] = parent0.getTrackRef(); // there is no standalone TRD track, so use the index for the TPCTRD track array
  } else if (src == GTrackID::TPCTOF) {
    const auto& parent0 = getTPCTOFMatch(gidx); // TPC : TOF
    table[GTrackID::TOF] = {unsigned(parent0.getIdxTOFCl()), GTrackID::TOF};
    table[GTrackID::TPC] = parent0.getTrackRef();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack(gidx);
    table[GTrackID::TPC] = parent0.getRefTPC();
    table[parent0.getRefITS().getSource()] = parent0.getRefITS(); // ITS source might be an ITS track or ITSAB tracklet
  } else if (src == GTrackID::MFTMCH || src == GTrackID::MFTMCHMID) {
    const auto& parent0 = getGlobalFwdTrack(gidx);
    table[GTrackID::MFT] = parent0.getMFTTrackID();
    table[GTrackID::MCH] = parent0.getMCHTrackID();
    if (parent0.getMIDTrackID() != -1) {
      table[GTrackID::MID] = parent0.getMIDTrackID();
    }
  } else if (src == GTrackID::MCHMID) {
    const auto& parent0 = getMCHMIDMatch(gidx);
    table[GTrackID::MCH] = parent0.getMCHRef();
    table[GTrackID::MID] = parent0.getMIDRef();
  }
  return std::move(table);
}

//________________________________________________________
// get contributing TPC GTrackID to the source. If source gidx is not contributed by TPC,
// returned GTrackID.isSourceSet()==false
GTrackID RecoContainer::getTPCContributorGID(GTrackID gidx) const
{
  auto src = gidx.getSource();
  if (src == GTrackID::ITSTPCTRD) {
    const auto& parent0 = getITSTPCTRDTrack<o2::trd::TrackTRD>(gidx);
    const auto& parent1 = getTPCITSTrack(parent0.getRefGlobalTrackId());
    return parent1.getRefTPC();
  } else if (src == GTrackID::ITSTPCTOF) {
    const auto& parent0 = getTOFMatch(gidx); // ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack(parent0.getTrackRef());
    return parent1.getRefTPC();
  } else if (src == GTrackID::ITSTPCTRDTOF) {
    const auto& parent0 = getTOFMatch(gidx); // ITS/TPC/TRD : TOF
    const auto& parent1 = getITSTPCTRDTrack<o2::trd::TrackTRD>(parent0.getTrackRef());
    const auto& parent2 = getTPCITSTrack(parent1.getRefGlobalTrackId());
    return parent2.getRefTPC();
  } else if (src == GTrackID::TPCTOF) {
    const auto& parent0 = getTPCTOFMatch(gidx); // TPC : TOF
    return parent0.getTrackRef();
  } else if (src == GTrackID::TPCTRD) {
    const auto& parent0 = getTPCTRDTrack<o2::trd::TrackTRD>(gidx);
    return parent0.getRefGlobalTrackId();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack(gidx);
    return parent0.getRefTPC();
  }
  return src == GTrackID::TPC ? gidx : GTrackID{};
}

//________________________________________________________
// get contributing ITS GTrackID to the source. If source gidx is not contributed by TPC,
// returned GTrackID.isSourceSet()==false
GTrackID RecoContainer::getITSContributorGID(GTrackID gidx) const
{
  auto src = gidx.getSource();
  if (src == GTrackID::ITSTPCTRD) {
    const auto& parent0 = getITSTPCTRDTrack<o2::trd::TrackTRD>(gidx);
    const auto& parent1 = getTPCITSTrack(parent0.getRefGlobalTrackId());
    return parent1.getRefITS();
  } else if (src == GTrackID::ITSTPCTOF) {
    const auto& parent0 = getTOFMatch(gidx); // ITS/TPC : TOF
    const auto& parent1 = getTPCITSTrack(parent0.getTrackRef());
    return parent1.getRefITS();
  } else if (src == GTrackID::ITSTPCTRDTOF) {
    const auto& parent0 = getTOFMatch(gidx); // ITS/TPC : TOF
    const auto& parent1 = getITSTPCTRDTrack<o2::trd::TrackTRD>(parent0.getTrackRef());
    const auto& parent2 = getTPCITSTrack(parent1.getRefGlobalTrackId());
    return parent2.getRefITS();
  } else if (src == GTrackID::ITSTPC) {
    const auto& parent0 = getTPCITSTrack(gidx);
    return parent0.getRefITS();
  }
  return src == GTrackID::ITS ? gidx : GTrackID{};
}

//________________________________________________________
const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>* RecoContainer::getPHOSCellsMCLabels() const
{
  return mcPHSCells.get();
}

//________________________________________________________
const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>* RecoContainer::getEMCALCellsMCLabels() const
{
  return mcEMCCells.get();
}
