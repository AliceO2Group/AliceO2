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

/// \file AlignmentSpec.cxx
/// \brief Implementation of alignment process for muon spectrometer
///
/// \author Chi ZHANG, CEA-Saclay

#include "MCHAlign/AlignmentSpec.h"

#include <cmath>
#include <string>
#include <tuple>
#include <vector>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <sstream>

#include <TCanvas.h>
#include <TChain.h>
#include <TDatabasePDG.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TGeoMatrix.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMatrixD.h>
#include <TSystem.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "Framework/CallbackService.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "MathUtils/Cartesian.h"
#include "MCHAlign/Aligner.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackExtrap.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/TrackFitter.h"
#include "MCHBase/TrackerParam.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;
using namespace o2;

class AlignmentTask
{
 public:
  const int fgNDetElemCh[10] = {4, 4, 4, 4, 18, 18, 26, 26, 26, 26};
  const int fgSNDetElemCh[11] = {0, 4, 8, 12, 16, 34, 52, 78, 104, 130, 156};
  const int fgNDetElemHalfCh[20] = {2, 2, 2, 2, 2, 2, 2, 2, 9,
                                    9, 9, 9, 13, 13, 13, 13, 13, 13, 13, 13};
  const int fgSNDetElemHalfCh[21] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 34, 44, 54, 64,
                                     78, 92, 106, 120, 134, 148, 162, 176};
  const int fgDetElemHalfCh[20][13] =
    {
      {100, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

      {200, 203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {201, 202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

      {300, 303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {301, 302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

      {400, 403, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {401, 402, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

      {500, 501, 502, 503, 504, 514, 515, 516, 517, 0, 0, 0, 0},
      {505, 506, 507, 508, 509, 510, 511, 512, 513, 0, 0, 0, 0},

      {600, 601, 602, 603, 604, 614, 615, 616, 617, 0, 0, 0, 0},
      {605, 606, 607, 608, 609, 610, 611, 612, 613, 0, 0, 0, 0},

      {700, 701, 702, 703, 704, 705, 706, 720, 721, 722, 723, 724, 725},
      {707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719},

      {800, 801, 802, 803, 804, 805, 806, 820, 821, 822, 823, 824, 825},
      {807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819},

      {900, 901, 902, 903, 904, 905, 906, 920, 921, 922, 923, 924, 925},
      {907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919},

      {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1020, 1021, 1022, 1023, 1024, 1025},
      {1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019}

  };
  std::vector<double> params;
  std::vector<double> errors;
  std::vector<double> pulls;

  constexpr double pi() { return 3.14159265358979323846; }

  //_________________________________________________________________________________________________
  AlignmentTask(shared_ptr<base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {

    LOG(info) << "Initializing aligner";
    // Initialize alignment algorithm

    int numberOfGlobalParam = 624;
    double default_value = 0;
    params = std::vector<double>(numberOfGlobalParam, default_value);
    errors = std::vector<double>(numberOfGlobalParam, default_value);
    pulls = std::vector<double>(numberOfGlobalParam, default_value);

    doAlign = ic.options().get<bool>("do-align");
    if (doAlign) {
      LOG(info) << "Alignment mode";
    } else {
      LOG(info) << "No alignment mode, only residuals will be stored";
    }

    doReAlign = ic.options().get<bool>("do-realign");
    if (doReAlign) {
      LOG(info) << "Re-alignment mode";
    }

    if (mCCDBRequest) {
      LOG(info) << "Loading magnetic field and reference geometry from CCDB";
      base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    } else {

      LOG(info) << "Loading magnetic field and reference geometry from input files";

      auto grpFile = ic.options().get<string>("grp-file");
      if (std::filesystem::exists(grpFile)) {
        const auto grp = parameters::GRPObject::loadFrom(grpFile);
        base::Propagator::initFieldFromGRP(grp);
        TrackExtrap::setField();
        mAlign.SetBFieldOn(TrackExtrap::isFieldON());
        TrackExtrap::useExtrapV2();
      } else {
        LOG(fatal) << "No GRP file";
      }

      auto geoIdealFile = ic.options().get<string>("geo-file-ideal");
      if (std::filesystem::exists(geoIdealFile)) {
        base::GeometryManager::loadGeometry(geoIdealFile.c_str());
        transformation = geo::transformationFromTGeoManager(*gGeoManager);
        for (int i = 0; i < 156; i++) {
          int iDEN = GetDetElemId(i);
          transformIdeal[iDEN] = transformation(iDEN);
        }
      } else {
        LOG(fatal) << "No ideal geometry";
      }

      auto geoRefFile = ic.options().get<string>("geo-file-ref");
      if (std::filesystem::exists(geoRefFile)) {
        base::GeometryManager::loadGeometry(geoRefFile.c_str());
        transformation = geo::transformationFromTGeoManager(*gGeoManager);
        for (int i = 0; i < 156; i++) {
          int iDEN = GetDetElemId(i);
          transformRef[iDEN] = transformation(iDEN);
        }
      } else {
        LOG(fatal) << "No reference geometry";
      }
    }

    auto doEvaluation = ic.options().get<bool>("do-evaluation");
    mAlign.SetDoEvaluation(doEvaluation);
    // Variation range for parameters
    mAlign.SetAllowedVariation(0, 2.0);
    mAlign.SetAllowedVariation(1, 0.3);
    mAlign.SetAllowedVariation(2, 0.002);
    mAlign.SetAllowedVariation(3, 2.0);

    // Configuration for track fitter
    const auto& trackerParam = TrackerParam::Instance();
    trackFitter.setBendingVertexDispersion(trackerParam.bendingVertexDispersion);
    trackFitter.setChamberResolution(trackerParam.chamberResolutionX, trackerParam.chamberResolutionY);
    trackFitter.smoothTracks(true);
    trackFitter.useChamberResolution();
    mImproveCutChi2 = 2. * trackerParam.sigmaCutForImprovement * trackerParam.sigmaCutForImprovement;

    // Fix chambers
    auto input_fixchambers = ic.options().get<string>("fix-chamber");
    std::stringstream string_chambers(input_fixchambers);
    string_chambers >> std::ws;
    while (string_chambers.good()) {
      string substr;
      std::getline(string_chambers, substr, ',');
      LOG(info) << Form("%s%d", "Fixing chamber: ", std::stoi(substr));
      mAlign.FixChamber(std::stoi(substr));
    }

    doMatched = ic.options().get<bool>("matched");
    outFileName = ic.options().get<string>("output");
    readFromRec = ic.options().get<bool>("use-record");

    if (readFromRec) {
      mAlign.SetReadOnly();
      LOG(info) << "Reading records as input";
    }
    mAlign.init();

    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>([this]() {
      LOG(info) << "Alignment duration = " << mElapsedTime.count() << " s";
    });
  }

  //_________________________________________________________________________________________________
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    /// finalize the track extrapolation setting
    if (mCCDBRequest && base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {

      if (matcher == framework::ConcreteDataMatcher("GLO", "GRPMAGFIELD", 0)) {
        TrackExtrap::setField();
        mAlign.SetBFieldOn(TrackExtrap::isFieldON());
        TrackExtrap::useExtrapV2();
      }

      if (matcher == framework::ConcreteDataMatcher("GLO", "GEOMALIGN", 0)) {
        LOG(info) << "Loading reference geometry from CCDB";
        transformation = geo::transformationFromTGeoManager(*gGeoManager);
        for (int i = 0; i < 156; i++) {
          int iDEN = GetDetElemId(i);
          transformRef[iDEN] = transformation(iDEN);
        }
      }
    }
  }

  //_________________________________________________________________________________________________
  void processWithMatching(vector<ROFRecord>& mchROFs, vector<TrackMCH>& mchTracks, vector<Cluster>& mchClusters,
                           vector<dataformats::TrackMCHMID>& muonTracks)
  {
    // processing for each track
    for (const auto& mchROF : mchROFs) {

      for (int iMCHTrack = mchROF.getFirstIdx();
           iMCHTrack <= mchROF.getLastIdx(); ++iMCHTrack) {
        // MCH-MID matching
        if (!FindMuon(iMCHTrack, muonTracks)) {
          continue;
        }

        auto mchTrack = mchTracks.at(iMCHTrack);
        int id_track = iMCHTrack;
        int nb_clusters = mchTrack.getNClusters();

        // Track selection, considering only tracks having at least 10 clusters
        if (nb_clusters <= 9) {
          continue;
        }

        // Format conversion from TrackMCH to Track(MCH internal use)
        mch::Track convertedTrack = MCHFormatConvert(mchTrack, mchClusters, doReAlign);

        // Erase removable track
        if (RemoveTrack(convertedTrack)) {
          continue;
        }

        //  Track processing, saving residuals
        mAlign.ProcessTrack(convertedTrack, transformation, doAlign, weightRecord);
      }
    }
  }

  //_________________________________________________________________________________________________
  void processWithOutMatching(vector<ROFRecord>& mchROFs, vector<TrackMCH>& mchTracks, vector<Cluster>& mchClusters)
  {

    // processing for each track
    for (const auto& mchROF : mchROFs) {

      for (int iMCHTrack = mchROF.getFirstIdx();
           iMCHTrack <= mchROF.getLastIdx(); ++iMCHTrack) {

        auto mchTrack = mchTracks.at(iMCHTrack);
        int id_track = iMCHTrack;
        int nb_clusters = mchTrack.getNClusters();

        // Track selection, saving only tracks having exactly 10 clusters
        if (nb_clusters <= 9) {
          continue;
        }

        // Format conversion from TrackMCH to Track(MCH internal use)
        Track convertedTrack = MCHFormatConvert(mchTrack, mchClusters, doReAlign);

        // Erase removable track
        if (RemoveTrack(convertedTrack)) {
          continue;
        }

        //  Track processing, saving residuals
        mAlign.ProcessTrack(convertedTrack, transformation, doAlign, weightRecord);
      }
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();
    LOG(info) << "Starting alignment process";
    if (doMatched) {
      LOG(info) << "Using MCH-MID matched tracks";
    }
    if (mCCDBRequest) {

      LOG(info) << "Checking CCDB updates with processing context";
      base::GRPGeomHelper::instance().checkUpdates(pc);

      auto geoIdeal = pc.inputs().get<TGeoManager*>("geomIdeal");
      LOG(info) << "Loading ideal geometry from CCDB";
      transformation = geo::transformationFromTGeoManager(*geoIdeal);
      for (int i = 0; i < 156; i++) {
        int iDEN = GetDetElemId(i);
        transformIdeal[iDEN] = transformation(iDEN);
      }
    }

    // Load new geometry if we need to do re-align
    if (doReAlign) {
      if (NewGeoFileName != "") {
        LOG(info) << "Loading re-alignment geometry";
        base::GeometryManager::loadGeometry(NewGeoFileName.c_str());
        transformation = geo::transformationFromTGeoManager(*gGeoManager);
        for (int i = 0; i < 156; i++) {
          int iDEN = GetDetElemId(i);
          transformNew[iDEN] = transformation(iDEN);
        }
      } else {
        LOG(fatal) << "No re-alignment geometry";
      }
    }

    if (!readFromRec) {
      // Loading input data
      LOG(info) << "Loading MCH tracks";
      auto [fMCH, mchReader] = LoadData(mchFileName, "o2sim");
      TTreeReaderValue<vector<ROFRecord>> mchROFs = {*mchReader, "trackrofs"};
      TTreeReaderValue<vector<TrackMCH>> mchTracks = {*mchReader, "tracks"};
      TTreeReaderValue<vector<Cluster>> mchClusters = {*mchReader, "trackclusters"};

      if (doMatched) {
        LOG(info) << "Loading MID tracks";
        auto [fMUON, muonReader] = LoadData(muonFileName.c_str(), "o2sim");
        TTreeReaderValue<vector<dataformats::TrackMCHMID>> muonTracks = {*muonReader, "tracks"};
        int nTF = muonReader->GetEntries(false);
        if (mchReader->GetEntries(false) != nTF) {
          LOG(error) << mchFileName << " and " << muonFileName << " do not contain the same number of TF";
          exit(-1);
        }

        LOG(info) << "Starting track processing";
        while (mchReader->Next() && muonReader->Next()) {
          int id_event = mchReader->GetCurrentEntry();
          processWithMatching(*mchROFs, *mchTracks, *mchClusters, *muonTracks);
        }
      } else {
        LOG(info) << "Starting track processing";
        while (mchReader->Next()) {
          int id_event = mchReader->GetCurrentEntry();
          processWithOutMatching(*mchROFs, *mchTracks, *mchClusters);
        }
      }
    }

    // Global fit
    if (doAlign) {
      mAlign.GlobalFit(params, errors, pulls);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    mElapsedTime = tEnd - tStart;

    // Generate new geometry w.r.t alignment results
    if (doAlign) {

      LOG(info) << "Generating aligned geometry using global parameters";
      vector<detectors::AlignParam> ParamAligned;
      mAlign.ReAlign(ParamAligned, params);

      TFile* FileAlign = TFile::Open("AlignParam.root", "RECREATE");
      FileAlign->cd();
      FileAlign->WriteObjectAny(&ParamAligned, "std::vector<o2::detectors::AlignParam>", "alignment");
      FileAlign->Close();

      string Geo_file;

      if (doReAlign) {
        Geo_file = Form("%s%s", "o2sim_geometry_ReAlign", ".root");
      } else {
        Geo_file = Form("%s%s", "o2sim_geometry_Align", ".root");
      }

      // Store aligned geometry
      gGeoManager->Export(Geo_file.c_str());

      // Store param plots
      drawHisto(params, errors, pulls, outFileName);

      // Export align params in ideal frame
      TransRef(ParamAligned);
    }

    mAlign.terminate();

    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  //_________________________________________________________________________________________________
  void TransRef(vector<detectors::AlignParam>& ParamsTrack)
  {
    LOG(info) << "Transforming align params to the frame of ideal geometry";
    vector<o2::detectors::AlignParam> ParamsRef;
    o2::detectors::AlignParam param_Ref;

    for (int hc = 0; hc < 20; hc++) {

      ParamsRef.emplace_back(ParamsTrack.at(fgSNDetElemHalfCh[hc]));

      for (int de = 0; de < fgNDetElemHalfCh[hc]; de++) {

        int iDEN = fgDetElemHalfCh[hc][de];
        o2::detectors::AlignParam param_Track = ParamsTrack.at(fgSNDetElemHalfCh[hc] + 1 + de);

        LOG(debug) << Form("%s%s", "Processing DET Elem: ", (param_Track.getSymName()).c_str());

        TGeoHMatrix delta_track;
        TGeoRotation r("Rotation/Track", param_Track.getPsi() / pi() * 180.0, param_Track.getTheta() / pi() * 180.0, param_Track.getPhi() / pi() * 180.0);
        delta_track.SetRotation(r.GetRotationMatrix());
        delta_track.SetDx(param_Track.getX());
        delta_track.SetDy(param_Track.getY());
        delta_track.SetDz(param_Track.getZ());

        TGeoHMatrix transRef = transformIdeal[iDEN];
        TGeoHMatrix transTrack = doReAlign ? transformNew[iDEN] : transformRef[iDEN];
        TGeoHMatrix transRefTrack = transTrack * transRef.Inverse();
        TGeoHMatrix delta_ref = delta_track * transRefTrack;

        param_Ref.setSymName((param_Track.getSymName()).c_str());
        param_Ref.setGlobalParams(delta_ref);
        param_Ref.applyToGeometry();
        ParamsRef.emplace_back(param_Ref);
      }
    }

    TFile* fOut = TFile::Open("AlignParam@ideal.root", "RECREATE");
    fOut->WriteObjectAny(&ParamsRef, "std::vector<o2::detectors::AlignParam>", "alignment");
    fOut->Close();
  }

  //_________________________________________________________________________________________________
  Track MCHFormatConvert(TrackMCH& mchTrack, vector<Cluster>& mchClusters, bool doReAlign)
  {

    Track convertedTrack = Track();

    // Get clusters for current track
    int id_cluster_first = mchTrack.getFirstClusterIdx();
    int id_cluster_last = mchTrack.getLastClusterIdx();

    for (int id_cluster = id_cluster_first;
         id_cluster < id_cluster_last + 1; ++id_cluster) {

      Cluster* cluster = &(mchClusters.at(id_cluster));
      const int DEId_cluster = cluster->getDEId();
      const int CId_cluster = cluster->getChamberId();
      const int ind_cluster = cluster->getClusterIndex();

      // Transformations to new geometry from reference geometry
      if (doReAlign) {

        math_utils::Point3D<double> local;
        math_utils::Point3D<double> master;

        master.SetXYZ(cluster->getX(), cluster->getY(), cluster->getZ());

        transformRef[cluster->getDEId()].MasterToLocal(master, local);
        transformNew[cluster->getDEId()].LocalToMaster(local, master);

        cluster->x = master.x();
        cluster->y = master.y();
        cluster->z = master.z();
      }
      convertedTrack.createParamAtCluster(*cluster);
    }

    return Track(convertedTrack);
  }

  //_________________________________________________________________________________________________
  bool RemoveTrack(Track& track)
  {

    bool removeTrack = false;

    try {
      trackFitter.fit(track, false);
    } catch (exception const& e) {
      removeTrack = true;
      return removeTrack;
    }

    auto itStartingParam = std::prev(track.rend());

    while (true) {

      try {
        trackFitter.fit(track, true, false, (itStartingParam == track.rbegin()) ? nullptr : &itStartingParam);
      } catch (exception const&) {
        removeTrack = true;
        break;
      }

      double worstLocalChi2 = -1.0;

      track.tagRemovableClusters(0x1F, false);

      auto itWorstParam = track.end();

      for (auto itParam = track.begin(); itParam != track.end(); ++itParam) {
        if (itParam->getLocalChi2() > worstLocalChi2) {
          worstLocalChi2 = itParam->getLocalChi2();
          itWorstParam = itParam;
        }
      }

      if (worstLocalChi2 < mImproveCutChi2) {
        break;
      }

      if (!itWorstParam->isRemovable()) {
        removeTrack = true;
        track.removable();
        break;
      }

      auto itNextParam = track.removeParamAtCluster(itWorstParam);
      auto itNextToNextParam = (itNextParam == track.end()) ? itNextParam : std::next(itNextParam);
      itStartingParam = track.rbegin();

      if (track.getNClusters() < 10) {
        removeTrack = true;
        break;
      } else {
        while (itNextToNextParam != track.end()) {
          if (itNextToNextParam->getClusterPtr()->getChamberId() != itNextParam->getClusterPtr()->getChamberId()) {
            itStartingParam = std::make_reverse_iterator(++itNextParam);
            break;
          }
          ++itNextToNextParam;
        }
      }
    }

    if (!removeTrack) {
      for (auto& param : track) {
        param.setParameters(param.getSmoothParameters());
        param.setCovariances(param.getSmoothCovariances());
      }
    }

    return removeTrack;
  }

  //_________________________________________________________________________________________________
  void drawHisto(std::vector<double>& params, std::vector<double>& errors, std::vector<double>& pulls, string outFileName)
  {

    TH1F* hPullX = new TH1F("hPullX", "hPullX", 201, -10, 10);
    TH1F* hPullY = new TH1F("hPullY", "hPullY", 201, -10, 10);
    TH1F* hPullZ = new TH1F("hPullZ", "hPullZ", 201, -10, 10);
    TH1F* hPullPhi = new TH1F("hPullPhi", "hPullPhi", 201, -10, 10);

    double deNumber[156];

    double alignX[156];
    double alignY[156];
    double alignZ[156];
    double alignPhi[156];
    double pullX[156];
    double pullY[156];
    double pullZ[156];
    double pullPhi[156];

    for (int iDEN = 0; iDEN < 156; iDEN++) {
      deNumber[iDEN] = iDEN + 0.5;
      alignX[iDEN] = params[iDEN * 4];
      alignY[iDEN] = params[iDEN * 4 + 1];
      alignZ[iDEN] = params[iDEN * 4 + 3];
      alignPhi[iDEN] = params[iDEN * 4 + 2];
      pullX[iDEN] = pulls[iDEN * 4];
      pullY[iDEN] = pulls[iDEN * 4 + 1];
      pullZ[iDEN] = pulls[iDEN * 4 + 3];
      pullPhi[iDEN] = pulls[iDEN * 4 + 2];
      if (params[iDEN * 4]) {

        hPullX->Fill(pulls[iDEN * 4]);
        hPullY->Fill(pulls[iDEN * 4 + 1]);
        hPullZ->Fill(pulls[iDEN * 4 + 3]);
        hPullPhi->Fill(pulls[iDEN * 4 + 2]);
      }
    }

    TGraph* graphAlignX = new TGraph(156, deNumber, alignX);
    TGraph* graphAlignY = new TGraph(156, deNumber, alignY);
    TGraph* graphAlignZ = new TGraph(156, deNumber, alignZ);
    TGraph* graphAlignPhi = new TGraph(156, deNumber, alignPhi);
    // TGraph* graphAlignYZ = new TGraph(156, alignY, alignZ);

    TGraph* graphPullX = new TGraph(156, deNumber, pullX);
    TGraph* graphPullY = new TGraph(156, deNumber, pullY);
    TGraph* graphPullZ = new TGraph(156, deNumber, pullZ);
    TGraph* graphPullPhi = new TGraph(156, deNumber, pullPhi);

    graphAlignX->SetMarkerStyle(24);
    graphPullX->SetMarkerStyle(25);

    //  graphAlignX->Draw("AP");

    graphAlignY->SetMarkerStyle(24);
    graphPullY->SetMarkerStyle(25);

    // graphAlignY->Draw("Psame");

    graphAlignZ->SetMarkerStyle(24);
    graphPullZ->SetMarkerStyle(25);

    //  graphAlignZ->Draw("AP");
    graphAlignPhi->SetMarkerStyle(24);
    graphPullPhi->SetMarkerStyle(25);

    // graphAlignYZ->SetMarkerStyle(24);
    // graphAlignYZ->Draw("P goff");

    // Saving plots
    string PlotFiles_name = Form("%s%s", outFileName.c_str(), "_results.root");
    TFile* PlotFiles = TFile::Open(PlotFiles_name.c_str(), "RECREATE");
    PlotFiles->WriteObjectAny(hPullX, "TH1F", "hPullX");
    PlotFiles->WriteObjectAny(hPullY, "TH1F", "hPullY");
    PlotFiles->WriteObjectAny(hPullZ, "TH1F", "hPullZ");
    PlotFiles->WriteObjectAny(hPullPhi, "TH1F", "hPullPhi");
    PlotFiles->WriteObjectAny(graphAlignX, "TGraph", "graphAlignX");
    PlotFiles->WriteObjectAny(graphAlignY, "TGraph", "graphAlignY");
    PlotFiles->WriteObjectAny(graphAlignZ, "TGraph", "graphAlignZ");
    // PlotFiles->WriteObjectAny(graphAlignYZ, "TGraph", "graphAlignYZ");

    TCanvas* cvn1 = new TCanvas("cvn1", "cvn1", 1200, 1600);
    // cvn1->Draw();
    cvn1->Divide(1, 4);
    TLine limLine(4, -5, 4, 5);
    TH1F* aHisto = new TH1F("aHisto", "AlignParam", 161, 0, 160);
    aHisto->SetXTitle("Det. Elem. Number");
    for (int i = 1; i < 5; i++) {
      cvn1->cd(i);
      double Range[4] = {5.0, 1.0, 5.0, 0.01};
      switch (i) {
        case 1:
          aHisto->SetYTitle("#delta_{#X} (cm)");
          aHisto->GetYaxis()->SetRangeUser(-5.0, 5.0);
          aHisto->DrawCopy("goff");
          graphAlignX->Draw("Psame goff");
          limLine.DrawLine(4, -Range[i - 1], 4, Range[i - 1]);
          limLine.DrawLine(8, -Range[i - 1], 8, Range[i - 1]);
          limLine.DrawLine(12, -Range[i - 1], 12, Range[i - 1]);
          limLine.DrawLine(16, -Range[i - 1], 16, Range[i - 1]);
          limLine.DrawLine(16 + 18, -Range[i - 1], 16 + 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18, -Range[i - 1], 16 + 2 * 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 26, -Range[i - 1], 16 + 2 * 18 + 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 2 * 26, -Range[i - 1], 16 + 2 * 18 + 2 * 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 3 * 26, -Range[i - 1], 16 + 2 * 18 + 3 * 26, Range[i - 1]);
          break;
        case 2:
          aHisto->SetYTitle("#delta_{#Y} (cm)");
          aHisto->GetYaxis()->SetRangeUser(-1.0, 1.0);
          aHisto->DrawCopy("goff");
          graphAlignY->Draw("Psame goff");
          limLine.DrawLine(4, -Range[i - 1], 4, Range[i - 1]);
          limLine.DrawLine(8, -Range[i - 1], 8, Range[i - 1]);
          limLine.DrawLine(12, -Range[i - 1], 12, Range[i - 1]);
          limLine.DrawLine(16, -Range[i - 1], 16, Range[i - 1]);
          limLine.DrawLine(16 + 18, -Range[i - 1], 16 + 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18, -Range[i - 1], 16 + 2 * 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 26, -Range[i - 1], 16 + 2 * 18 + 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 2 * 26, -Range[i - 1], 16 + 2 * 18 + 2 * 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 3 * 26, -Range[i - 1], 16 + 2 * 18 + 3 * 26, Range[i - 1]);
          break;
        case 3:
          aHisto->SetYTitle("#delta_{#Z} (cm)");
          aHisto->GetYaxis()->SetRangeUser(-5.0, 5.0);
          aHisto->DrawCopy("goff");
          graphAlignZ->Draw("Psame goff");
          limLine.DrawLine(4, -Range[i - 1], 4, Range[i - 1]);
          limLine.DrawLine(8, -Range[i - 1], 8, Range[i - 1]);
          limLine.DrawLine(12, -Range[i - 1], 12, Range[i - 1]);
          limLine.DrawLine(16, -Range[i - 1], 16, Range[i - 1]);
          limLine.DrawLine(16 + 18, -Range[i - 1], 16 + 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18, -Range[i - 1], 16 + 2 * 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 26, -Range[i - 1], 16 + 2 * 18 + 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 2 * 26, -Range[i - 1], 16 + 2 * 18 + 2 * 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 3 * 26, -Range[i - 1], 16 + 2 * 18 + 3 * 26, Range[i - 1]);
          break;
        case 4:
          aHisto->SetYTitle("#delta_{#varphi} (cm)");
          aHisto->GetYaxis()->SetRangeUser(-0.01, 0.01);
          aHisto->DrawCopy("goff");
          graphAlignPhi->Draw("Psame goff");
          limLine.DrawLine(4, -Range[i - 1], 4, Range[i - 1]);
          limLine.DrawLine(8, -Range[i - 1], 8, Range[i - 1]);
          limLine.DrawLine(12, -Range[i - 1], 12, Range[i - 1]);
          limLine.DrawLine(16, -Range[i - 1], 16, Range[i - 1]);
          limLine.DrawLine(16 + 18, -Range[i - 1], 16 + 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18, -Range[i - 1], 16 + 2 * 18, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 26, -Range[i - 1], 16 + 2 * 18 + 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 2 * 26, -Range[i - 1], 16 + 2 * 18 + 2 * 26, Range[i - 1]);
          limLine.DrawLine(16 + 2 * 18 + 3 * 26, -Range[i - 1], 16 + 2 * 18 + 3 * 26, Range[i - 1]);
          break;
      }
    }

    PlotFiles->WriteObjectAny(cvn1, "TCanvas", "AlignParam");
    PlotFiles->Close();
  }

  //_________________________________________________________________________________________________
  tuple<TFile*, TTreeReader*> LoadData(const string fileName, const string treeName)
  {
    /// open the input file and get the intput tree

    TFile* f = TFile::Open(fileName.c_str(), "READ");
    if (!f || f->IsZombie()) {
      LOG(error) << "Opening file " << fileName << " failed";
      exit(-1);
    }

    TTreeReader* r = new TTreeReader(treeName.c_str(), f);
    if (r->IsZombie()) {
      LOG(error) << "Tree " << treeName << " not found";
      exit(-1);
    }

    return std::make_tuple(f, r);
  }

  //_________________________________________________________________________________________________
  bool FindMuon(int iMCHTrack, vector<dataformats::TrackMCHMID>& muonTracks)
  {
    /// find the MCH-MID matched track corresponding to this MCH track
    for (const auto& muon : muonTracks) {
      // cout << "Muon track index: " << muon.getMCHRef().getIndex()<<endl;
      if (muon.getMCHRef().getIndex() == iMCHTrack) {
        return true;
      }
    }
    return false;
  }

  //_________________________________________________________________________________________________
  int GetDetElemNumber(int iDetElemId)
  {
    /// get det element number from ID
    // get chamber and element number in chamber
    const int iCh = iDetElemId / 100;
    const int iDet = iDetElemId % 100;

    // make sure detector index is valid
    if (!(iCh > 0 && iCh <= 10 && iDet < fgNDetElemCh[iCh - 1])) {
      LOG(fatal) << "Invalid detector element id: " << iDetElemId;
    }

    // add number of detectors up to this chamber
    return iDet + fgSNDetElemCh[iCh - 1];
  }

  //_________________________________________________________________________________________________
  int GetDetElemId(int iDetElemNumber)
  {
    // make sure detector number is valid
    if (!(iDetElemNumber >= fgSNDetElemCh[0] &&
          iDetElemNumber < fgSNDetElemCh[10])) {
      LOG(fatal) << "Invalid detector element number: " << iDetElemNumber;
    }
    /// get det element number from ID
    // get chamber and element number in chamber
    int iCh = 0;
    int iDet = 0;
    for (int i = 1; i <= 10; i++) {
      if (iDetElemNumber < fgSNDetElemCh[i]) {
        iCh = i;
        iDet = iDetElemNumber - fgSNDetElemCh[i - 1];
        break;
      }
    }

    // make sure detector index is valid
    if (!(iCh > 0 && iCh <= 10 && iDet < fgNDetElemCh[iCh - 1])) {
      LOG(fatal) << "Invalid detector element id: " << 100 * iCh + iDet;
    }

    // add number of detectors up to this chamber
    return 100 * iCh + iDet;
  }

  const string mchFileName{"mchtracks.root"};
  const string muonFileName{"muontracks.root"};
  string outFileName{"Alignment"};
  string RefGeoFileName{""};
  string NewGeoFileName{""};
  bool doAlign{false};
  bool doReAlign{false};
  bool doMatched{false};
  bool readFromRec{false};
  const double weightRecord{1.0};
  Aligner mAlign{};
  shared_ptr<base::GRPGeomRequest> mCCDBRequest{};

  map<int, math_utils::Transform3D> transformRef{};   // reference geometry w.r.t track data
  map<int, math_utils::Transform3D> transformNew{};   // new geometry
  map<int, math_utils::Transform3D> transformIdeal{}; // Ideal geometry

  geo::TransformationCreator transformation{};
  TrackFitter trackFitter{};
  double mImproveCutChi2{};

  std::chrono::duration<double> mElapsedTime{};
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getAlignmentSpec(bool disableCCDB)
{
  vector<framework::InputSpec> inputSpecs{};
  inputSpecs.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0);
  inputSpecs.emplace_back("geomIdeal", "GLO", "GEOMIDEAL", 0, Lifetime::Condition, framework::ccdbParamSpec("GLO/Config/Geometry"));

  vector<framework::OutputSpec> outputSpecs{};
  auto ccdbRequest = disableCCDB ? nullptr : std::make_shared<base::GRPGeomRequest>(false,                         // orbitResetTime
                                                                                    false,                         // GRPECS=true
                                                                                    false,                         // GRPLHCIF
                                                                                    true,                          // GRPMagField
                                                                                    false,                         // askMatLUT
                                                                                    base::GRPGeomRequest::Aligned, // geometry
                                                                                    inputSpecs);

  return DataProcessorSpec{
    "mch-alignment",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{o2::framework::adaptFromTask<AlignmentTask>(ccdbRequest)},
    Options{{"geo-file-ref", VariantType::String, o2::base::NameConf::getAlignedGeomFileName(), {"Name of the reference geometry file"}},
            {"geo-file-ideal", VariantType::String, o2::base::NameConf::getGeomFileName(), {"Name of the ideal geometry file"}},
            {"grp-file", VariantType::String, o2::base::NameConf::getGRPFileName(), {"Name of the grp file"}},
            {"do-align", VariantType::Bool, false, {"Switch for alignment, otherwise only residuals will be stored"}},
            {"do-evaluation", VariantType::Bool, false, {"Option for saving residuals for evaluation"}},
            {"do-realign", VariantType::Bool, false, {"Switch for re-alignment using another geometry"}},
            {"matched", VariantType::Bool, false, {"Switch for using MCH-MID matched tracks"}},
            {"fix-chamber", VariantType::String, "", {"Chamber fixing, ex 1,2,3"}},
            {"use-record", VariantType::Bool, false, {"Option for directly using record in alignment if provided"}},
            {"output", VariantType::String, "Alignment", {"Option for name of output file"}}}};
}

} // namespace mch
} // namespace o2