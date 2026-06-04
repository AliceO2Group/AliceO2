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
#include "TGeoGlobalMagField.h"
#include "ITSStudies/Efficiency.h"
#include "ITSStudies/ITSStudiesConfigParam.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/Task.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/IOUtils.h"
#include "ReconstructionDataFormats/DCA.h"
#include "SimulationDataFormat/MCTrack.h"
#include "Steer/MCKinematicsReader.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

#include <TEfficiency.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH1I.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TCanvas.h>
#include <TEfficiency.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TF1.h>
#include <TObjArray.h>
#include <THStack.h>
#include <TString.h>
#include <TAttMarker.h>
#include <TArrayD.h>
#include <numeric>

#define NLAYERS 3

namespace o2::its::study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;

class EfficiencyStudy final : public Task
{
 public:
  EfficiencyStudy(std::shared_ptr<DataRequest> dr,
                  mask_t src,
                  bool useMC,
                  std::shared_ptr<o2::steer::MCKinematicsReader> kineReader,
                  std::shared_ptr<o2::base::GRPGeomRequest> gr) : mDataRequest(dr), mTracksSrc(src), mUseMC(useMC), mKineReader(kineReader), mGGCCDBRequest(gr){};

  ~EfficiencyStudy() final = default;
  void init(InitContext&) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void initialiseRun(o2::globaltracking::RecoContainer&);
  void stileEfficiencyGraph(std::unique_ptr<TEfficiency>& eff, const char* name, const char* title, bool bidimensional, const int markerStyle, const double markersize, const int markercolor, const int linercolor);
  int getDCAClusterTrackMC(int countDuplicated);
  void studyDCAcutsMC();
  void studyClusterSelectionMC();
  void countDuplicatedAfterCuts();
  void getEfficiency(bool isMC);
  void process(o2::globaltracking::RecoContainer&);
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  bool mVerboseOutput = false;
  bool mUseMC;
  std::string mOutFileName;
  double b;
  std::shared_ptr<o2::steer::MCKinematicsReader> mKineReader;
  GeometryTGeo* mGeometry;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  float mrangesPt[NLAYERS][2] = {{0., 0.5}, {0.5, 2.}, {2., 7.5}};

  // Spans
  gsl::span<const o2::itsmft::ROFRecord> mTracksROFRecords;
  gsl::span<const o2::itsmft::ROFRecord> mClustersROFRecords;
  gsl::span<const o2::its::TrackITS> mTracks;
  gsl::span<const o2::MCCompLabel> mTracksMCLabels;
  gsl::span<const o2::itsmft::CompClusterExt> mClusters;
  gsl::span<const unsigned char> mClusPatterns;
  gsl::span<const int> mInputITSidxs;
  const o2::dataformats::MCLabelContainer* mClustersMCLCont;
  std::vector<o2::BaseCluster<float>> mITSClustersArray;

  // Data
  GTrackID::mask_t mTracksSrc{};
  std::shared_ptr<DataRequest> mDataRequest;

  // Utils
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<TFile> mOutFile;
  int mDuplicated_layer[NLAYERS] = {0};

  //// Histos

  // DCA betweeen track and original cluster
  std::unique_ptr<TH1D> mDCAxyOriginal[NLAYERS];
  std::unique_ptr<TH1D> mDCAzOriginal[NLAYERS];
  // DCA betweeen track and duplicated cluster
  std::unique_ptr<TH1D> mDCAxyDuplicated;
  std::unique_ptr<TH1D> mDCAzDuplicated;

  // DCA betweeen track and duplicated cluster per layer
  std::unique_ptr<TH1D> mDCAxyDuplicated_layer[NLAYERS];
  std::unique_ptr<TH1D> mDCAzDuplicated_layer[NLAYERS];

  // phi, eta, pt of the cluster
  std::unique_ptr<TH1D> mPhiOriginal[NLAYERS];
  std::unique_ptr<TH1D> mEtaOriginal[NLAYERS];
  std::unique_ptr<TH1D> mPtOriginal[NLAYERS];
  std::unique_ptr<TH1D> mPtDuplicated[NLAYERS];
  std::unique_ptr<TH1D> mEtaDuplicated[NLAYERS];
  std::unique_ptr<TH1D> mPhiDuplicated[NLAYERS];
  std::unique_ptr<TH1D> mPhiOriginalIfDuplicated[NLAYERS];

  std::unique_ptr<TH2D> mZvsPhiDUplicated[NLAYERS];

  // position of the clusters
  std::unique_ptr<TH3D> m3DClusterPositions;
  std::unique_ptr<TH3D> m3DDuplicatedClusterPositions;
  std::unique_ptr<TH2D> m2DClusterOriginalPositions;
  std::unique_ptr<TH2D> m2DClusterDuplicatedPositions;

  // Efficiency histos
  std::unique_ptr<TH1D> mEfficiencyGoodMatch;
  std::unique_ptr<TH1D> mEfficiencyFakeMatch;
  std::unique_ptr<TH1D> mEfficiencyTotal;
  std::unique_ptr<TH1D> mEfficiencyGoodMatch_layer[NLAYERS];
  std::unique_ptr<TH1D> mEfficiencyFakeMatch_layer[NLAYERS];
  std::unique_ptr<TH1D> mEfficiencyTotal_layer[NLAYERS];
  std::unique_ptr<TH2D> mEfficiencyGoodMatchPt_layer[NLAYERS];
  std::unique_ptr<TH2D> mEfficiencyFakeMatchPt_layer[NLAYERS];
  std::unique_ptr<TH2D> mEfficiencyGoodMatchEta_layer[NLAYERS];
  std::unique_ptr<TH2D> mEfficiencyFakeMatchEta_layer[NLAYERS];
  std::unique_ptr<TH2D> mEfficiencyGoodMatchPhi_layer[NLAYERS];
  std::unique_ptr<TH2D> mEfficiencyGoodMatchPhiOriginal_layer[NLAYERS];
  std::unique_ptr<TH2D> mEfficiencyFakeMatchPhi_layer[NLAYERS];

  // std::unique_ptr<TH2D> mEfficiencyColEta[NLAYERS];
  std::unique_ptr<TH2D> mDenColEta[NLAYERS];
  std::unique_ptr<TH2D> mNumColEta[NLAYERS];
  std::unique_ptr<TH2D> mDenRowPhi[NLAYERS];
  std::unique_ptr<TH2D> mNumRowPhi[NLAYERS];
  std::unique_ptr<TH2D> mDenRowCol[NLAYERS];
  std::unique_ptr<TH2D> mNumRowCol[NLAYERS];

  // phi, eta, pt of the duplicated cluster per layer
  std::unique_ptr<TH2D> mPt_EtaDupl[NLAYERS];

  // duplicated per layer and per cut
  std::unique_ptr<TH1D> mDuplicatedEtaAllPt[NLAYERS];
  std::unique_ptr<TH1D> mDuplicatedEta[NLAYERS][3];
  std::unique_ptr<TH1D> mDuplicatedPhiAllPt[NLAYERS];
  std::unique_ptr<TH1D> mDuplicatedPhi[NLAYERS][3];
  std::unique_ptr<TH1D> mDuplicatedPt[NLAYERS];
  std::unique_ptr<TH1D> mDuplicatedRow[NLAYERS];
  std::unique_ptr<TH1D> mDuplicatedCol[NLAYERS];
  std::unique_ptr<TH1D> mDuplicatedZ[NLAYERS];
  std::unique_ptr<TH2D> mDuplicatedPtEta[NLAYERS];
  std::unique_ptr<TH2D> mDuplicatedPtPhi[NLAYERS];
  std::unique_ptr<TH2D> mDuplicatedEtaPhi[NLAYERS];

  // matches per layer and per cut
  std::unique_ptr<TH1D> mNGoodMatchesEtaAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNGoodMatchesEta[NLAYERS][3];
  std::unique_ptr<TH1D> mNGoodMatchesPhiAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNGoodMatchesPhi[NLAYERS][3];

  std::unique_ptr<TH1D> mNFakeMatchesEtaAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesEta[NLAYERS][3];
  std::unique_ptr<TH1D> mNFakeMatchesPhiAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesPhi[NLAYERS][3];

  std::unique_ptr<TH1D> mNGoodMatchesPt[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesPt[NLAYERS];

  std::unique_ptr<TH1D> mNGoodMatchesRow[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesRow[NLAYERS];

  std::unique_ptr<TH1D> mNGoodMatchesCol[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesCol[NLAYERS];

  std::unique_ptr<TH1D> mNGoodMatchesZ[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesZ[NLAYERS];

  std::unique_ptr<TH2D> mNGoodMatchesPtEta[NLAYERS];
  std::unique_ptr<TH2D> mNFakeMatchesPtEta[NLAYERS];

  std::unique_ptr<TH2D> mNGoodMatchesPtPhi[NLAYERS];
  std::unique_ptr<TH2D> mNFakeMatchesPtPhi[NLAYERS];

  std::unique_ptr<TH2D> mNGoodMatchesEtaPhi[NLAYERS];
  std::unique_ptr<TH2D> mNFakeMatchesEtaPhi[NLAYERS];

  // calculating the efficiency with TEfficiency class
  std::unique_ptr<TEfficiency> mEffPtGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPtFake[NLAYERS];
  std::unique_ptr<TEfficiency> mEffRowGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffRowFake[NLAYERS];
  std::unique_ptr<TEfficiency> mEffColGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffColFake[NLAYERS];
  std::unique_ptr<TEfficiency> mEffZGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffZFake[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPtEtaGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPtEtaFake[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPtPhiGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPtPhiFake[NLAYERS];
  std::unique_ptr<TEfficiency> mEffEtaPhiGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffEtaPhiFake[NLAYERS];

  std::unique_ptr<TEfficiency> mEffEtaGoodAllPt[NLAYERS];
  std::unique_ptr<TEfficiency> mEffEtaGood[NLAYERS][3];
  std::unique_ptr<TEfficiency> mEffEtaFakeAllPt[NLAYERS];
  std::unique_ptr<TEfficiency> mEffEtaFake[NLAYERS][3];

  std::unique_ptr<TEfficiency> mEffPhiGoodAllPt[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPhiGood[NLAYERS][3];
  std::unique_ptr<TEfficiency> mEffPhiFakeAllPt[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPhiFake[NLAYERS][3];

  std::unique_ptr<TH2D> mnGoodMatchesPt_layer[NLAYERS];
  std::unique_ptr<TH2D> mnFakeMatchesPt_layer[NLAYERS];

  std::unique_ptr<TH2D> mnGoodMatchesEta_layer[NLAYERS];
  std::unique_ptr<TH2D> mnFakeMatchesEta_layer[NLAYERS];

  std::unique_ptr<TH2D> mnGoodMatchesPhi_layer[NLAYERS];
  std::unique_ptr<TH2D> mnGoodMatchesPhiOriginal_layer[NLAYERS];
  std::unique_ptr<TH2D> mnFakeMatchesPhi_layer[NLAYERS];

  std::unique_ptr<TH1D> DCAxyData[NLAYERS];
  std::unique_ptr<TH1D> DCAzData[NLAYERS];

  std::unique_ptr<TH1D> DCAxyRejected[NLAYERS];
  std::unique_ptr<TH1D> DCAzRejected[NLAYERS];

  std::unique_ptr<TH1D> denPt[NLAYERS];
  std::unique_ptr<TH1D> numPt[NLAYERS];
  std::unique_ptr<TH1D> numPtGood[NLAYERS];
  std::unique_ptr<TH1D> numPtFake[NLAYERS];

  std::unique_ptr<TH1D> denPhi[NLAYERS];
  std::unique_ptr<TH1D> numPhi[NLAYERS];
  std::unique_ptr<TH1D> numPhiGood[NLAYERS];
  std::unique_ptr<TH1D> numPhiFake[NLAYERS];

  std::unique_ptr<TH1D> denEta[NLAYERS];
  std::unique_ptr<TH1D> numEta[NLAYERS];
  std::unique_ptr<TH1D> numEtaGood[NLAYERS];
  std::unique_ptr<TH1D> numEtaFake[NLAYERS];

  std::unique_ptr<TH1D> denRow[NLAYERS];
  std::unique_ptr<TH1D> numRow[NLAYERS];
  std::unique_ptr<TH1D> numRowGood[NLAYERS];
  std::unique_ptr<TH1D> numRowFake[NLAYERS];

  std::unique_ptr<TH1D> denCol[NLAYERS];
  std::unique_ptr<TH1D> numCol[NLAYERS];
  std::unique_ptr<TH1D> numColGood[NLAYERS];
  std::unique_ptr<TH1D> numColFake[NLAYERS];
  std::unique_ptr<TH1D> denZ[NLAYERS];
  std::unique_ptr<TH1D> numZ[NLAYERS];
  std::unique_ptr<TH1D> numZGood[NLAYERS];
  std::unique_ptr<TH1D> numZFake[NLAYERS];

  std::unique_ptr<TH1D> numLayers;
  std::unique_ptr<TH1D> denLayers;
  std::unique_ptr<TH1D> numGoodLayers;
  std::unique_ptr<TH1D> numFakeLayers;

  int nDuplicatedClusters[NLAYERS] = {0};
  int nTracksSelected[NLAYERS] = {0}; // denominator fot the efficiency calculation

  std::unique_ptr<TH1D> IPOriginalxy[NLAYERS];
  std::unique_ptr<TH1D> IPOriginalz[NLAYERS];

  std::unique_ptr<TH1D> chipRowDuplicated[NLAYERS];
  std::unique_ptr<TH1D> chipRowOriginalIfDuplicated[NLAYERS];

  std::unique_ptr<TH1D> chi2trackAccepted;

  /// checking where the duplicated not found are (histograms filled with the orifinal cluster variables)
  std::unique_ptr<TH1D> phiFound[NLAYERS];
  std::unique_ptr<TH1D> rowFound[NLAYERS];
  std::unique_ptr<TH1D> phiNotFound[NLAYERS];
  std::unique_ptr<TH1D> rowNotFound[NLAYERS];
  std::unique_ptr<TH1D> zFound[NLAYERS];
  std::unique_ptr<TH1D> zNotFound[NLAYERS];
  std::unique_ptr<TH2D> colFoundOriginalVsDuplicated[NLAYERS];
  std::unique_ptr<TH1D> colFoundOriginal[NLAYERS];
  std::unique_ptr<TH1D> colNotFound[NLAYERS];
  std::unique_ptr<TH1D> radiusFound[NLAYERS];
  std::unique_ptr<TH1D> radiusNotFound[NLAYERS];
  std::unique_ptr<TH2D> m2DClusterFoundPositions;
  std::unique_ptr<TH2D> m2DClusterNotFoundPositions;
  std::unique_ptr<TH1D> mChipNotFound;
  std::unique_ptr<TH1D> mChipFound;
  std::unique_ptr<TH2D> l0_00;
  std::unique_ptr<TH2D> l1_15;
  std::unique_ptr<TH2D> l2_19;
  std::unique_ptr<TH2D> chipOrigVsOverlap;
  std::unique_ptr<TH2D> chipmap;
};

void EfficiencyStudy::init(InitContext& ic)
{
  LOGP(info, "init");

  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  auto& pars = o2::its::study::ITSEfficiencyParamConfig::Instance();
  mOutFileName = pars.outFileName;
  b = pars.b;

  int nbPt = 75;
  double xbins[nbPt + 1], ptcutl = 0.05, ptcuth = 7.5;
  double a = std::log(ptcuth / ptcutl) / nbPt;
  for (int i = 0; i <= nbPt; i++) {
    xbins[i] = ptcutl * std::exp(i * a);
  }

  mOutFile = std::make_unique<TFile>(mOutFileName.c_str(), "recreate");

  mDCAxyDuplicated = std::make_unique<TH1D>("dcaXYDuplicated", "Distance between track and duplicated cluster  ;DCA xy (cm); ", 200, -0.01, 0.01);
  mDCAzDuplicated = std::make_unique<TH1D>("dcaZDuplicated", "Distance between track and duplicated cluster  ;DCA z (cm); ", 200, -0.01, 0.01);

  m3DClusterPositions = std::make_unique<TH3D>("3DClusterPositions", ";x (cm);y (cm);z (cm)", 200, -10, 10, 200, -10, 10, 400, -20, 20);
  m3DDuplicatedClusterPositions = std::make_unique<TH3D>("3DDuplicatedClusterPositions", ";x (cm);y (cm);z (cm)", 200, -10, 10, 200, -10, 10, 500, -30, 30);
  m2DClusterOriginalPositions = std::make_unique<TH2D>("m2DClusterOriginalPositions", ";x (cm);y (cm)", 400, -10, 10, 400, -6, 6);
  m2DClusterDuplicatedPositions = std::make_unique<TH2D>("m2DClusterDuplicatedPositions", ";x (cm);y (cm)", 400, -10, 10, 400, -6, 6);

  mEfficiencyGoodMatch = std::make_unique<TH1D>("mEfficiencyGoodMatch", ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
  mEfficiencyFakeMatch = std::make_unique<TH1D>("mEfficiencyFakeMatch", ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
  mEfficiencyTotal = std::make_unique<TH1D>("mEfficiencyTotal", ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);

  chi2trackAccepted = std::make_unique<TH1D>("chi2trackAccepted", "; $#chi^{2}", 500, 0, 100);

  m2DClusterFoundPositions = std::make_unique<TH2D>("m2DClusterFoundPositions", ";x (cm);y (cm)", 250, -5, 5, 250, -5, 5);
  m2DClusterNotFoundPositions = std::make_unique<TH2D>("m2DClusterNotFoundPositions", ";x (cm);y (cm)", 250, -5, 5, 250, -5, 5);
  mChipNotFound = std::make_unique<TH1D>("mChipNotFound", ";chipID", 432, 0, 432);
  mChipFound = std::make_unique<TH1D>("mChipFound", ";chipID", 432, 0, 432);
  l0_00 = std::make_unique<TH2D>("l0_00", ";col; row", 2304, -0.5, 9215.5, 128, -0.5, 511.5);
  l1_15 = std::make_unique<TH2D>("l1_15", ";col; row", 2304, -0.5, 9215.5, 512, -0.5, 511.5);
  l2_19 = std::make_unique<TH2D>("l2_19", ";col; row", 2304, -0.5, 9215.5, 512, -0.5, 511.5);
  chipOrigVsOverlap = std::make_unique<TH2D>("chipOrigVsOverlap", ";chipID Overlap;chipID Original", 9, 0, 9, 9, 0, 9);
  chipmap = std::make_unique<TH2D>("chipmap", ";Column;Row", 1024, 0, 1023, 512, -0.5, 511.5);

  numLayers = std::make_unique<TH1D>("numLayers", "numLayers; ; Efficiency", 3, -0.5, 2.5);
  numGoodLayers = std::make_unique<TH1D>("numGoodLayers", "numGoodLayers; ; Efficiency", 3, -0.5, 2.5);
  numFakeLayers = std::make_unique<TH1D>("numFakeLayers", "numFakeLayers; ; Efficiency", 3, -0.5, 2.5);
  denLayers = std::make_unique<TH1D>("denLayers", "denLayers; ; Efficiency", 3, -0.5, 2.5);

  for (int i = 0; i < NLAYERS; i++) {

    chipRowDuplicated[i] = std::make_unique<TH1D>(Form("chipPosDuplicated_L%d", i), Form("L%d; row", i), 512, -0.5, 511.5);
    chipRowOriginalIfDuplicated[i] = std::make_unique<TH1D>(Form("chipPosOriginalIfDuplicated%d", i), Form("L%d; row", i), 512, -0.5, 511.5);

    DCAxyData[i] = std::make_unique<TH1D>(Form("dcaXYData_L%d", i), "Distance between track and original cluster ;DCA xy (cm); ", 4000, -0.2, 0.2);
    DCAzData[i] = std::make_unique<TH1D>(Form("dcaZData_L%d", i), "Distance between track and original cluster ;DCA z (cm); ", 4000, -0.2, 0.2);
    DCAxyRejected[i] = std::make_unique<TH1D>(Form("DCAxyRejected%d", i), "Distance between track and original cluster (rejected) ;DCA xy (cm); ", 30000, -30, 30);
    DCAzRejected[i] = std::make_unique<TH1D>(Form("DCAzRejected%d", i), "Distance between track and original cluster (rejected) ;DCA z (cm); ", 30000, -30, 30);

    mDCAxyOriginal[i] = std::make_unique<TH1D>(Form("dcaXYOriginal_L%d", i), "Distance between track and original cluster ;DCA xy (cm); ", 200, -0.01, 0.01);
    mDCAzOriginal[i] = std::make_unique<TH1D>(Form("dcaZOriginal_L%d", i), "Distance between track and original cluster ;DCA z (cm); ", 200, -0.01, 0.01);

    mPhiOriginal[i] = std::make_unique<TH1D>(Form("phiOriginal_L%d", i), ";phi (rad); ", 90, -3.2, 3.2);
    mEtaOriginal[i] = std::make_unique<TH1D>(Form("etaOriginal_L%d", i), ";eta (rad); ", 100, -2, 2);
    mPtOriginal[i] = std::make_unique<TH1D>(Form("ptOriginal_L%d", i), ";pt (GeV/c); ", 100, 0, 10);

    mZvsPhiDUplicated[i] = std::make_unique<TH2D>(Form("zvsphiDuplicated_L%d", i), ";z (cm);phi (rad)", 400, -20, 20, 90, -3.2, 3.2);

    mPtDuplicated[i] = std::make_unique<TH1D>(Form("ptDuplicated_L%d", i), ";pt (GeV/c); ", nbPt, 0, 7.5); // xbins);
    mEtaDuplicated[i] = std::make_unique<TH1D>(Form("etaDuplicated_L%d", i), ";eta; ", 40, -2, 2);
    mPhiDuplicated[i] = std::make_unique<TH1D>(Form("phiDuplicated_L%d", i), ";phi (rad); ", 90, -3.2, 3.2);
    mPhiOriginalIfDuplicated[i] = std::make_unique<TH1D>(Form("phiOriginalIfDuplicated_L%d", i), ";phi (rad); ", 90, -3.2, 3.2);
    mDCAxyDuplicated_layer[i] = std::make_unique<TH1D>(Form("dcaXYDuplicated_layer_L%d", i), "Distance between track and duplicated cluster  ;DCA xy (cm); ", 100, -0.01, 0.01);
    mDCAzDuplicated_layer[i] = std::make_unique<TH1D>(Form("dcaZDuplicated_layer_L%d", i), "Distance between track and duplicated cluster  ;DCA z (cm); ", 100, -0.01, 0.01);

    mEfficiencyGoodMatch_layer[i] = std::make_unique<TH1D>(Form("mEfficiencyGoodMatch_layer_L%d", i), ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
    mEfficiencyFakeMatch_layer[i] = std::make_unique<TH1D>(Form("mEfficiencyFakeMatch_layer_L%d", i), ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
    mEfficiencyTotal_layer[i] = std::make_unique<TH1D>(Form("mEfficiencyTotal_layer_L%d", i), ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);

    mEfficiencyGoodMatchPt_layer[i] = std::make_unique<TH2D>(Form("mEfficiencyGoodMatchPt_layer_L%d", i), ";#it{p}_{T} (GeV/c);#sigma(DCA) cut;Efficiency;", nbPt, 0, 7.5, /* xbins*/ 20, 0.5, 20.5);
    mEfficiencyFakeMatchPt_layer[i] = std::make_unique<TH2D>(Form("mEfficiencyFakeMatchPt_layer_L%d", i), ";#it{p}_{T} (GeV/c);#sigma(DCA) cut;Efficiency;", nbPt, 0, 7.5, /* xbins*/ 20, 0.5, 20.5);

    mEfficiencyGoodMatchEta_layer[i] = std::make_unique<TH2D>(Form("mEfficiencyGoodMatchEta_layer_L%d", i), ";#eta;#sigma(DCA) cut;Efficiency;", 40, -2, 2, 20, 0.5, 20.5);
    mEfficiencyFakeMatchEta_layer[i] = std::make_unique<TH2D>(Form("mEfficiencyFakeMatchEta_layer_L%d", i), ";#eta;#sigma(DCA) cut;Efficiency;", 40, -2, 2, 20, 0.5, 20.5);

    mEfficiencyGoodMatchPhi_layer[i] = std::make_unique<TH2D>(Form("mEfficiencyGoodMatchPhi_layer_L%d", i), ";#phi;#sigma(DCA) cut;Efficiency;", 90, -3.2, 3.2, 20, 0.5, 20.5);
    mEfficiencyGoodMatchPhiOriginal_layer[i] = std::make_unique<TH2D>(Form("mEfficiencyGoodMatchPhiOriginal_layer_L%d", i), ";#phi Original;#sigma(DCA) cut;Efficiency;", 90, -3.2, 3.2, 20, 0.5, 20.5);
    mEfficiencyFakeMatchPhi_layer[i] = std::make_unique<TH2D>(Form("mEfficiencyFakeMatchPhi_layer_L%d", i), ";#phi;#sigma(DCA) cut;Efficiency;", 90, -3.2, 3.2, 20, 0.5, 20.5);

    mPt_EtaDupl[i] = std::make_unique<TH2D>(Form("mPt_EtaDupl_L%d", i), ";#it{p}_{T} (GeV/c);#eta; ", 100, 0, 10, 100, -2, 2);

    mDuplicatedPt[i] = std::make_unique<TH1D>(Form("mDuplicatedPt_log_L%d", i), Form("; #it{p}_{T} (GeV/c); Number of duplciated clusters L%d", i), nbPt, 0, 7.5 /* xbins*/);
    mDuplicatedPt[i]->Sumw2();
    mNGoodMatchesPt[i] = std::make_unique<TH1D>(Form("mNGoodMatchesPt_L%d", i), Form("; #it{p}_{T} (GeV/c); Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/);
    mNGoodMatchesPt[i]->Sumw2();
    mNFakeMatchesPt[i] = std::make_unique<TH1D>(Form("mNFakeMatchesPt_L%d", i), Form("; #it{p}_{T} (GeV/c); Number of fake matches L%d", i), nbPt, 0, 7.5 /* xbins*/);
    mNFakeMatchesPt[i]->Sumw2();

    mDuplicatedRow[i] = std::make_unique<TH1D>(Form("mDuplicatedRow_L%d", i), Form("; Row; Number of duplciated clusters L%d", i), 128, -0.5, 511.5);
    mDuplicatedRow[i]->Sumw2();
    mNGoodMatchesRow[i] = std::make_unique<TH1D>(Form("mNGoodMatchesRow_L%d", i), Form("; Row; Number of good matches L%d", i), 128, -0.5, 511.5);
    mNGoodMatchesRow[i]->Sumw2();
    mNFakeMatchesRow[i] = std::make_unique<TH1D>(Form("mNFakeMatchesRow_L%d", i), Form(";Row; Number of fake matches L%d", i), 128, -0.5, 511.5);
    mNFakeMatchesRow[i]->Sumw2();

    mDuplicatedCol[i] = std::make_unique<TH1D>(Form("mDuplicatedCol_L%d", i), Form("; Col; Number of duplciated clusters L%d", i), 128, -0.5, 1023.5);
    mDuplicatedCol[i]->Sumw2();
    mNGoodMatchesCol[i] = std::make_unique<TH1D>(Form("mNGoodMatchesCol_L%d", i), Form("; Col; Number of good matches L%d", i), 128, -0.5, 1023.5);
    mNGoodMatchesCol[i]->Sumw2();
    mNFakeMatchesCol[i] = std::make_unique<TH1D>(Form("mNFakeMatchesCol_L%d", i), Form(";Col; Number of fake matches L%d", i), 128, -0.5, 1023.5);
    mNFakeMatchesCol[i]->Sumw2();

    mDuplicatedZ[i] = std::make_unique<TH1D>(Form("mDuplicatedZ_L%d", i), Form("; Z (cm); Number of duplciated clusters L%d", i), 100, -15, 15);
    mDuplicatedZ[i]->Sumw2();
    mNGoodMatchesZ[i] = std::make_unique<TH1D>(Form("mNGoodMatchesZ_L%d", i), Form("; Z (cm); Number of good matches L%d", i), 100, -15, 15);
    mNGoodMatchesZ[i]->Sumw2();
    mNFakeMatchesZ[i] = std::make_unique<TH1D>(Form("mNFakeMatchesZ_L%d", i), Form(";Z (cm); Number of fake matches L%d", i), 100, -15, 15);
    mNFakeMatchesZ[i]->Sumw2();

    mDuplicatedPtEta[i] = std::make_unique<TH2D>(Form("mDuplicatedPtEta_log_L%d", i), Form("; #it{p}_{T} (GeV/c);#eta; Number of duplciated clusters L%d", i), nbPt, 0, 7.5 /* xbins*/, 40, -2, 2);
    mDuplicatedPtEta[i]->Sumw2();
    mNGoodMatchesPtEta[i] = std::make_unique<TH2D>(Form("mNGoodMatchesPtEta_L%d", i), Form("; #it{p}_{T} (GeV/c);#eta; Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 40, -2, 2);
    mNGoodMatchesPtEta[i]->Sumw2();
    mNFakeMatchesPtEta[i] = std::make_unique<TH2D>(Form("mNFakeMatchesPtEta_L%d", i), Form("; #it{p}_{T} (GeV/c);#eta; Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 40, -2, 2);
    mNFakeMatchesPtEta[i]->Sumw2();

    mDuplicatedPtPhi[i] = std::make_unique<TH2D>(Form("mDuplicatedPtPhi_log_L%d", i), Form("; #it{p}_{T} (GeV/c);#phi (rad); Number of duplciated clusters L%d", i), nbPt, 0, 7.5 /* xbins*/, 90, -3.2, 3.2);
    mDuplicatedPtPhi[i]->Sumw2();
    mNGoodMatchesPtPhi[i] = std::make_unique<TH2D>(Form("mNGoodMatchesPtPhi_L%d", i), Form("; #it{p}_{T} (GeV/c);#phi (rad); Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 90, -3.2, 3.2);
    mNGoodMatchesPtPhi[i]->Sumw2();
    mNFakeMatchesPtPhi[i] = std::make_unique<TH2D>(Form("mNFakeMatchesPtPhi_L%d", i), Form("; #it{p}_{T} (GeV/c);#phi (rad); Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 90, -3.2, 3.2);
    mNFakeMatchesPtPhi[i]->Sumw2();

    mDuplicatedEtaPhi[i] = std::make_unique<TH2D>(Form("mDuplicatedEtaPhi_L%d", i), Form("; #eta;#phi (rad); Number of duplciated clusters L%d", i), 40, -2, 2, 90, -3.2, 3.2);
    mDuplicatedEtaPhi[i]->Sumw2();
    mNGoodMatchesEtaPhi[i] = std::make_unique<TH2D>(Form("mNGoodMatchesEtaPhi_L%d", i), Form("; #eta;#phi (rad); Number of good matches L%d", i), 40, -2, 2, 90, -3.2, 3.2);
    mNGoodMatchesEtaPhi[i]->Sumw2();
    mNFakeMatchesEtaPhi[i] = std::make_unique<TH2D>(Form("mNFakeMatchesEtaPhi_L%d", i), Form("; #eta;#phi (rad); Number of good matches L%d", i), 40, -2, 2, 90, -3.2, 3.2);
    mNFakeMatchesEtaPhi[i]->Sumw2();

    mDuplicatedEtaAllPt[i] = std::make_unique<TH1D>(Form("mDuplicatedEtaAllPt_L%d", i), Form("; #eta; Number of duplicated clusters L%d", i), 40, -2, 2);
    mNGoodMatchesEtaAllPt[i] = std::make_unique<TH1D>(Form("mNGoodMatchesEtaAllPt_L%d", i), Form("; #eta; Number of good matches L%d", i), 40, -2, 2);
    mNFakeMatchesEtaAllPt[i] = std::make_unique<TH1D>(Form("mNFakeMatchesEtaAllPt_L%d", i), Form("; #eta; Number of fake matches L%d", i), 40, -2, 2);

    mDuplicatedPhiAllPt[i] = std::make_unique<TH1D>(Form("mDuplicatedPhiAllPt_L%d", i), Form("; #phi (rad); Number of duplicated clusters L%d", i), 90, -3.2, 3.2);
    mNGoodMatchesPhiAllPt[i] = std::make_unique<TH1D>(Form("mNGoodMatchesPhiAllPt_L%d", i), Form("; #phi (rad); Number of good matches L%d", i), 90, -3.2, 3.2);
    mNFakeMatchesPhiAllPt[i] = std::make_unique<TH1D>(Form("mNFakeMatchesPhiAllPt_L%d", i), Form("; #phi (rad); Number of fake matches L%d", i), 90, -3.2, 3.2);

    mnGoodMatchesPt_layer[i] = std::make_unique<TH2D>(Form("mnGoodMatchesPt_layer_L%d", i), ";pt; nGoodMatches", nbPt, 0, 7.5 /* xbins*/, 20, 0.5, 20.5);
    mnFakeMatchesPt_layer[i] = std::make_unique<TH2D>(Form("mnFakeMatchesPt_layer_L%d", i), ";pt; nFakeMatches", nbPt, 0, 7.5 /* xbins*/, 20, 0.5, 20.5);
    mnGoodMatchesEta_layer[i] = std::make_unique<TH2D>(Form("mnGoodMatchesEta_layer_L%d", i), ";#eta; nGoodMatches", 40, -2, 2, 20, 0.5, 20.5);
    mnFakeMatchesEta_layer[i] = std::make_unique<TH2D>(Form("mnFakeMatchesEta_layer_L%d", i), ";#eta; nFakeMatches", 40, -2, 2, 20, 0.5, 20.5);
    mnGoodMatchesPhi_layer[i] = std::make_unique<TH2D>(Form("mnGoodMatchesPhi_layer_L%d", i), ";#Phi; nGoodMatches", 90, -3.2, 3.2, 20, 0.5, 20.5);
    mnGoodMatchesPhiOriginal_layer[i] = std::make_unique<TH2D>(Form("mnGoodMatchesPhiOriginal_layer_L%d", i), ";#Phi of the original Cluster; nGoodMatches", 90, -3.2, 3.2, 20, 0.5, 20.5);
    mnFakeMatchesPhi_layer[i] = std::make_unique<TH2D>(Form("mnFakeMatchesPhi_layer_L%d", i), ";#Phi; nFakeMatches", 90, -3.2, 3.2, 20, 0.5, 20.5);

    denPt[i] = std::make_unique<TH1D>(Form("denPt_L%d", i), Form("denPt_L%d", i), nbPt, 0, 7.5 /* xbins*/);
    numPt[i] = std::make_unique<TH1D>(Form("numPt_L%d", i), Form("numPt_L%d", i), nbPt, 0, 7.5 /* xbins*/);
    numPtGood[i] = std::make_unique<TH1D>(Form("numPtGood_L%d", i), Form("numPtGood_L%d", i), nbPt, 0, 7.5 /* xbins*/);
    numPtFake[i] = std::make_unique<TH1D>(Form("numPtFake_L%d", i), Form("numPtFake_L%d", i), nbPt, 0, 7.5 /* xbins*/);

    denPhi[i] = std::make_unique<TH1D>(Form("denPhi_L%d", i), Form("denPhi_L%d", i), 90, -3.2, 3.2);
    numPhi[i] = std::make_unique<TH1D>(Form("numPhi_L%d", i), Form("numPhi_L%d", i), 90, -3.2, 3.2);
    numPhiGood[i] = std::make_unique<TH1D>(Form("numPhiGood_L%d", i), Form("numPhiGood_L%d", i), 90, -3.2, 3.2);
    numPhiFake[i] = std::make_unique<TH1D>(Form("numPhiFake_L%d", i), Form("numPhiFake_L%d", i), 90, -3.2, 3.2);

    denEta[i] = std::make_unique<TH1D>(Form("denEta_L%d", i), Form("denEta_L%d", i), 200, -2, 2);
    numEta[i] = std::make_unique<TH1D>(Form("numEta_L%d", i), Form("numEta_L%d", i), 200, -2, 2);
    numEtaGood[i] = std::make_unique<TH1D>(Form("numEtaGood_L%d", i), Form("numEtaGood_L%d", i), 200, -2, 2);
    numEtaFake[i] = std::make_unique<TH1D>(Form("numEtaFake_L%d", i), Form("numEtaFake_L%d", i), 200, -2, 2);

    denRow[i] = std::make_unique<TH1D>(Form("denRow_L%d", i), Form("denRow_L%d", i), 128, -0.5, 511.5);
    numRow[i] = std::make_unique<TH1D>(Form("numRow_L%d", i), Form("numRow_L%d", i), 128, -0.5, 511.5);
    numRowGood[i] = std::make_unique<TH1D>(Form("numRowGood_L%d", i), Form("numRowGood_L%d", i), 128, -0.5, 511.5);
    numRowFake[i] = std::make_unique<TH1D>(Form("numRowFake_L%d", i), Form("numRowFake_L%d", i), 128, -0.5, 511.5);

    denCol[i] = std::make_unique<TH1D>(Form("denCol_L%d", i), Form("denCol_L%d", i), 128, -0.5, 1023.5);
    numCol[i] = std::make_unique<TH1D>(Form("numCol_L%d", i), Form("numCol_L%d", i), 128, -0.5, 1023.5);
    numColGood[i] = std::make_unique<TH1D>(Form("numColGood_L%d", i), Form("numColGood_L%d", i), 128, -0.5, 1023.5);
    numColFake[i] = std::make_unique<TH1D>(Form("numColFake_L%d", i), Form("numColFake_L%d", i), 128, -0.5, 1023.5);

    denZ[i] = std::make_unique<TH1D>(Form("denZ_L%d", i), Form("denZ_L%d", i), 100, -15, 15);
    numZ[i] = std::make_unique<TH1D>(Form("numZ_L%d", i), Form("numZ_L%d", i), 100, -15, 15);
    numZGood[i] = std::make_unique<TH1D>(Form("numZGood_L%d", i), Form("numZGood_L%d", i), 100, -15, 15);
    numZFake[i] = std::make_unique<TH1D>(Form("numZFake_L%d", i), Form("numZFake_L%d", i), 100, -15, 15);

    mDenColEta[i] = std::make_unique<TH2D>(Form("mDenColEta_L%d", i), Form("mDenColEta_L%d", i), 128, -0.5, 1023.5, 50, -1, 1);
    mNumColEta[i] = std::make_unique<TH2D>(Form("mNumColEta_L%d", i), Form("mNumColEta_L%d", i), 128, -0.5, 1023.5, 50, -1, 1);

    mDenRowPhi[i] = std::make_unique<TH2D>(Form("mDenRowPhi_L%d", i), Form("mDenRowPhi_L%d", i), 128, -0.5, 511.5, 90, -3.2, 3.2);
    mNumRowPhi[i] = std::make_unique<TH2D>(Form("mNumRowPhi_L%d", i), Form("mNumRowPhi_L%d", i), 128, -0.5, 511.5, 90, -3.2, 3.2);

    mDenRowCol[i] = std::make_unique<TH2D>(Form("mDenRowCol_L%d", i), Form("mDenRowCol_L%d", i), 128, -0.5, 511.5, 128, -0.5, 1023.5);
    mNumRowCol[i] = std::make_unique<TH2D>(Form("mNumRowCol_L%d", i), Form("mNumRowCol_L%d", i), 128, -0.5, 511.5, 128, -0.5, 1023.5);

    IPOriginalxy[i] = std::make_unique<TH1D>(Form("IPOriginalxy_L%d", i), Form("IPOriginalxy_L%d", i), 500, -0.002, 0.002);
    IPOriginalz[i] = std::make_unique<TH1D>(Form("IPOriginalz_L%d", i), Form("IPOriginalz_L%d", i), 200, -10, 10);

    phiFound[i] = std::make_unique<TH1D>(Form("phiFound_L%d", i), Form("phiFound_L%d", i), 190, -3.2, 3.2);
    rowFound[i] = std::make_unique<TH1D>(Form("rowFound_L%d", i), Form("rowFound_L%d", i), 128, -0.5, 511.5);
    phiNotFound[i] = std::make_unique<TH1D>(Form("phiNotFound_L%d", i), Form("phiNotFound_L%d", i), 90, -3.2, 3.2);
    rowNotFound[i] = std::make_unique<TH1D>(Form("rowNotFound_L%d", i), Form("rowNotFound_L%d", i), 128, -0.5, 511.5);
    zFound[i] = std::make_unique<TH1D>(Form("zFound_L%d", i), Form("zFound_L%d", i), 100, -15, 15);
    zNotFound[i] = std::make_unique<TH1D>(Form("zNotFound%d", i), Form("zNotFound%d", i), 100, -15, 15);
    colFoundOriginalVsDuplicated[i] = std::make_unique<TH2D>(Form("colFoundOriginalVsDuplicated_L%d", i), Form("colFoundOriginalVsDuplicated_L%d; Col Original cluster; Col Overlap cluster", i), 9216, -0.5, 9215.5, 9216, -0.5, 9215.5);
    colFoundOriginal[i] = std::make_unique<TH1D>(Form("colFoundOriginal_L%d", i), Form("colFoundOriginal_L%d; Col Original cluster;", i), 9216, -0.5, 9215.5);
    colNotFound[i] = std::make_unique<TH1D>(Form("colNotFound_L%d", i), Form("colNotFound_L%d", i), 9216, -0.5, 9215.5);
    radiusFound[i] = std::make_unique<TH1D>(Form("radiusFound_L%d", i), Form("radiusFound_L%d", i), 80, 0, 6);
    radiusNotFound[i] = std::make_unique<TH1D>(Form("radiusNotFound_L%d", i), Form("radiusNotFound_L%d", i), 80, 0, 4);

    for (int j = 0; j < 3; j++) {
      mDuplicatedEta[i][j] = std::make_unique<TH1D>(Form("mDuplicatedEta_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #eta; Number of duplicated clusters L%d", mrangesPt[j][0], mrangesPt[j][1], i), 40, -2, 2);
      mNGoodMatchesEta[i][j] = std::make_unique<TH1D>(Form("mNGoodMatchesEta_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #eta; Number of good matches L%d", mrangesPt[j][0], mrangesPt[j][1], i), 40, -2, 2);
      mNFakeMatchesEta[i][j] = std::make_unique<TH1D>(Form("mNFakeMatchesEta_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #eta; Number of fake matches L%d", mrangesPt[j][0], mrangesPt[j][1], i), 40, -2, 2);

      mDuplicatedPhi[i][j] = std::make_unique<TH1D>(Form("mDuplicatedPhi_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #phi; Number of duplicated clusters L%d", mrangesPt[j][0], mrangesPt[j][1], i), 90, -3.2, 3.2);
      mNGoodMatchesPhi[i][j] = std::make_unique<TH1D>(Form("mNGoodMatchesPhi_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #phi; Number of good matches L%d", mrangesPt[j][0], mrangesPt[j][1], i), 90, -3.2, 3.2);
      mNFakeMatchesPhi[i][j] = std::make_unique<TH1D>(Form("mNFakeMatchesPhi_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #phi; Number of fake matches L%d", mrangesPt[j][0], mrangesPt[j][1], i), 90, -3.2, 3.2);
    }
  }
  gStyle->SetPalette(55);
}

void EfficiencyStudy::run(ProcessingContext& pc)
{
  LOGP(info, "--------------- run");
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  initialiseRun(recoData);
  process(recoData);
}

void EfficiencyStudy::initialiseRun(o2::globaltracking::RecoContainer& recoData)
{
  LOGP(info, "--------------- initialiseRun");
  if (mUseMC) {
    mTracksMCLabels = recoData.getITSTracksMCLabels();
    mClustersMCLCont = recoData.getITSClustersMCLabels();
  }
  mITSClustersArray.clear();
  mTracksROFRecords = recoData.getITSTracksROFRecords();
  mTracks = recoData.getITSTracks();
  mClusters = recoData.getITSClusters();
  mClustersROFRecords = recoData.getITSClustersROFRecords();
  mClusPatterns = recoData.getITSClustersPatterns();
  mInputITSidxs = recoData.getITSTracksClusterRefs();
  mITSClustersArray.reserve(mClusters.size());
  auto pattIt = mClusPatterns.begin();
  o2::its::ioutils::convertCompactClusters(mClusters, pattIt, mITSClustersArray, mDict); // clusters converted to 3D spacepoints
}

void EfficiencyStudy::stileEfficiencyGraph(std::unique_ptr<TEfficiency>& eff, const char* name, const char* title, bool bidimensional = false, const int markerStyle = kFullCircle, const double markersize = 1, const int markercolor = kBlack, const int linecolor = kBlack)
{
  eff->SetName(name);
  eff->SetTitle(title);
  if (!bidimensional) {
    eff->SetMarkerStyle(markerStyle);
    eff->SetMarkerSize(markersize);
    eff->SetMarkerColor(markercolor);
    eff->SetLineColor(linecolor);
  }
}

int EfficiencyStudy::getDCAClusterTrackMC(int countDuplicated = 0)
{
  // get the DCA between the clusters and the track from MC and fill histograms: distance between original and duplicated cluster, DCA, phi, clusters
  // used to study the DCA cut to be applied
  LOGP(info, "--------------- getDCAClusterTrackMC");

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
  auto propagator = o2::base::Propagator::Instance();

  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  LOG(info) << ">>>>>>>>>>>> Magnetic field: " << bz;

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;
  int nLabels = 0;
  unsigned int totClus = 0;

  int duplicated = 0;

  std::unordered_map<o2::MCCompLabel, std::vector<int>> label_vecClus[mClustersROFRecords.size()][NLAYERS]; // array of maps nRofs x Nlayers -> {label, vec(iClus)} where vec(iClus) are the clusters that share the same label

  for (unsigned int iROF = 0; iROF < mTracksROFRecords.size(); iROF++) { // loop on ROFRecords array
    rofIndexTrack = mTracksROFRecords[iROF].getFirstEntry();
    rofNEntriesTrack = mTracksROFRecords[iROF].getNEntries();

    rofIndexClus = mClustersROFRecords[iROF].getFirstEntry();
    rofNEntriesClus = mClustersROFRecords[iROF].getNEntries();

    for (unsigned int iTrack = rofIndexTrack; iTrack < rofIndexTrack + rofNEntriesTrack; iTrack++) { // loop on tracks per ROF
      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];
      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7) {
        continue;
      }

      float ip[2]; // IP from 0,0,0 and the track should be the deplacement of the primary vertex
      track.getImpactParams(0, 0, 0, 0, ip);

      // if (abs(ip[0])>0.001 ) continue; ///pv not in (0,0,0)

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake()) {
        continue;
      }

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();

      // if (pt < mPtCuts[0] || pt > mPtCuts[1]) {
      //   continue;
      // }
      // if (eta < mEtaCuts[0] || eta > mEtaCuts[1]) {
      //   continue;
      // }

      float phioriginal = 0;
      float phiduplicated = 0;

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]]; // cluster spacepoint in the tracking system
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        UShort_t rowOriginal = clusOriginal.getRow();
        UShort_t colOriginal = clusOriginal.getCol();

        auto layer = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layer >= NLAYERS) {
          continue; // checking only selected layers
        }
        auto labsTrack = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;

        phioriginal = clusOriginalPointGlob.phi(); // * 180 / M_PI;

        mPhiOriginal[layer]->Fill(phioriginal);
        mPtOriginal[layer]->Fill(pt);
        mEtaOriginal[layer]->Fill(eta);
        m3DClusterPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y(), clusOriginalPointGlob.z());
        m2DClusterOriginalPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y());

        for (auto& labT : labsTrack) { // for each valid label iterate over ALL the clusters in the ROF to see if there are duplicates
          if (labT != tracklab) {
            continue;
          }
          nLabels++;
          if (labT.isValid()) {
            for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF

              auto clusDuplicated = mClusters[iClus];
              auto clusDuplicatedPoint = mITSClustersArray[iClus];

              auto layerClus = mGeometry->getLayer(clusDuplicated.getSensorID());
              if (layerClus != layer) {
                continue;
              }

              o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
              o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
              // phiduplicated = std::atan2(clusDuplicatedPointGlob.y(), clusDuplicatedPointGlob.x()) * 180 / M_PI + 180;
              phiduplicated = clusDuplicatedPointGlob.phi(); // * 180 / M_PI;

              auto labsClus = mClustersMCLCont->getLabels(iClus); // ideally I can have more than one label per cluster
              for (auto labC : labsClus) {
                if (labC == labT) {
                  label_vecClus[iROF][layerClus][labT].push_back(iClus); // same cluster: label from the track = label from the cluster
                                                                         // if a duplicate cluster is found, propagate the track to the duplicate cluster and compute the distance from the original cluster
                                                                         // if (clusOriginalPointGlob != clusDuplicatedPointGlob) { /// check that the duplicated cluster is not the original one just counted twice
                                                                         // if (clusDuplicated.getSensorID() != clusOriginal.getSensorID()) { /// check that the duplicated cluster is not the original one just counted twice

                  // applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
                  if (clusDuplicated.getSensorID() == clusOriginal.getSensorID()) {
                    continue;
                  }
                  auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
                  if (layerDuplicated != layerClus) {
                    continue;
                  }
                  auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
                  if (abs(staveDuplicated - staveOriginal) != 1) {
                    continue;
                  }
                  auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
                  if (abs(chipDuplicated - chipOriginal) > 1) {
                    continue;
                  }

                  duplicated++;

                  if (countDuplicated == 0) {
                    UShort_t rowDuplicated = clusDuplicated.getRow();
                    UShort_t colDuplicated = clusDuplicated.getCol();

                    chipRowDuplicated[layerDuplicated]->Fill(rowDuplicated);
                    chipRowOriginalIfDuplicated[layerDuplicated]->Fill(rowOriginal);

                    mDuplicated_layer[layerDuplicated]++; // This has to be incremented at the first call
                    mPtDuplicated[layerClus]->Fill(pt);
                    mEtaDuplicated[layerClus]->Fill(eta);
                    mPhiDuplicated[layerClus]->Fill(phiduplicated);
                    mZvsPhiDUplicated[layerClus]->Fill(clusDuplicatedPointGlob.Z(), phiduplicated);
                    mPhiOriginalIfDuplicated[layerClus]->Fill(phioriginal);
                  }

                  if (countDuplicated == 1) {
                    for (int ipt = 0; ipt < 3; ipt++) {
                      if (pt >= mrangesPt[ipt][0] && pt < mrangesPt[ipt][1]) {
                        mDuplicatedEta[layerDuplicated][ipt]->Fill(eta);
                        mDuplicatedPhi[layerDuplicated][ipt]->Fill(phiduplicated);
                      }
                    }
                    UShort_t rowDuplicated = clusDuplicated.getRow();
                    mDuplicatedRow[layerDuplicated]->Fill(rowOriginal);
                    mDuplicatedCol[layerDuplicated]->Fill(clusOriginal.getCol());
                    mDuplicatedZ[layerDuplicated]->Fill(clusOriginalPointGlob.Z());
                    mDuplicatedPt[layerDuplicated]->Fill(pt);
                    mDuplicatedPtEta[layerDuplicated]->Fill(pt, eta);
                    mDuplicatedPtPhi[layerDuplicated]->Fill(pt, phiduplicated);
                    mDuplicatedEtaPhi[layerDuplicated]->Fill(eta, phiduplicated);

                    mDuplicatedEtaAllPt[layerDuplicated]->Fill(eta);
                    mDuplicatedPhiAllPt[layerDuplicated]->Fill(phiduplicated);
                    mPt_EtaDupl[layerClus]->Fill(pt, eta);
                  }

                  m3DClusterPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y(), clusDuplicatedPointGlob.z());
                  m2DClusterDuplicatedPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y());

                  /// Compute the DCA between the cluster location and the track

                  /// first propagate to the original cluster
                  trackParCov.rotate(mGeometry->getSensorRefAlpha(clusOriginal.getSensorID()));
                  trackParCov.rotate(mGeometry->getSensorRefAlpha(clusOriginal.getSensorID()));
                  if (propagator->propagateToDCA(clusOriginalPointGlob, trackParCov, b, 2.f, matCorr, &clusOriginalDCA)) {
                    mDCAxyOriginal[layerClus]->Fill(clusOriginalDCA[0]);
                    mDCAzOriginal[layerClus]->Fill(clusOriginalDCA[1]);
                  }
                  /// then propagate to the duplicated cluster
                  trackParCov.rotate(mGeometry->getSensorRefAlpha(clusDuplicated.getSensorID()));
                  if (propagator->propagateToDCA(clusDuplicatedPointGlob, trackParCov, b, 2.f, matCorr, &clusDuplicatedDCA)) {
                    mDCAxyDuplicated->Fill(clusDuplicatedDCA[0]);
                    mDCAzDuplicated->Fill(clusDuplicatedDCA[1]);
                    mDCAxyDuplicated_layer[layerDuplicated]->Fill(clusDuplicatedDCA[0]);
                    mDCAzDuplicated_layer[layerDuplicated]->Fill(clusDuplicatedDCA[1]);
                  }
                  ///////////////////////////////////////////////////////
                }
              }
            }
          }
        }
      } // end loop on clusters
      totClus += NLAYERS; // summing only the number of clusters in the considered layers. Since the imposition of 7-clusters tracks, if the track is valid should release as clusters as the number of considered layers
    } // end loop on tracks per ROF
  } // end loop on ROFRecords array
  LOGP(info, "Total number of clusters: {} ", totClus);
  LOGP(info, "total nLabels: {}", nLabels);
  LOGP(info, "Number of duplicated clusters: {}", duplicated);

  if (mVerboseOutput && mUseMC) {
    // printing the duplicates
    for (unsigned int iROF = 0; iROF < mClustersROFRecords.size(); iROF++) {
      LOGP(info, "°°°°°°°°°°°°°°°°°°°°°°°° ROF {} °°°°°°°°°°°°°°°°°°°°°°°°", iROF);
      for (unsigned int lay = 0; lay < NLAYERS; lay++) {
        LOGP(info, "°°°°°°°°°°°°°°°°°°°°°°°° LAYER {} °°°°°°°°°°°°°°°°°°°°°°°°", lay);
        for (auto& it : label_vecClus[iROF][lay]) {
          if (it.second.size() <= 1) {
            continue; // printing only duplicates
          }
          std::cout << " \n++++++++++++ Label: ";
          auto label = it.first;
          it.first.print();
          for (auto iClus : it.second) {
            auto name = mGeometry->getSymbolicName(mClusters[iClus].getSensorID());
            auto chipid = mClusters[iClus].getChipID();
            auto clus = mClusters[iClus];
            auto clusPoint = mITSClustersArray[iClus];

            o2::math_utils::Point3D<float> clusPointTrack = {clusPoint.getX(), clusPoint.getY(), clusPoint.getZ()};
            o2::math_utils::Point3D<float> clusPointGlob = mGeometry->getMatrixT2G(clus.getSensorID()) * clusPointTrack;
            std::cout << "ROF: " << iROF << ", iClus: " << iClus << " -> chip: " << chipid << " = " << name << std::endl;
            LOGP(info, "LOCtrack: {} {} {}", clusPointTrack.x(), clusPointTrack.y(), clusPointTrack.z());
            LOGP(info, "LOCglob {} {} {}", clusPointGlob.x(), clusPointGlob.y(), clusPointGlob.z());
          }
        }
      }
    }
  }
  return duplicated;
}

void EfficiencyStudy::countDuplicatedAfterCuts()
{
  // count the effective number of duplicated cluster good matches after applying the pt eta and phi cuts on the track
  // to check the applied cuts
  LOGP(info, "--------------- countDuplicatedAfterCuts");

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
  auto propagator = o2::base::Propagator::Instance();

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;
  int nLabels = 0;
  unsigned int totClus = 0;

  int duplicated[3] = {0};
  int possibleduplicated[3] = {0};

  std::cout << "Track candidates: " << std::endl;

  std::unordered_map<o2::MCCompLabel, std::vector<int>> label_vecClus[mClustersROFRecords.size()][NLAYERS]; // array of maps nRofs x Nlayers -> {label, vec(iClus)} where vec(iClus) are the clusters that share the same label

  for (unsigned int iROF = 0; iROF < mTracksROFRecords.size(); iROF++) { // loop on ROFRecords array
    std::cout << "ROF number: " << iROF << std::endl;
    rofIndexTrack = mTracksROFRecords[iROF].getFirstEntry();
    rofNEntriesTrack = mTracksROFRecords[iROF].getNEntries();

    rofIndexClus = mClustersROFRecords[iROF].getFirstEntry();
    rofNEntriesClus = mClustersROFRecords[iROF].getNEntries();

    for (unsigned int iTrack = rofIndexTrack; iTrack < rofIndexTrack + rofNEntriesTrack; iTrack++) { // loop on tracks per ROF

      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];
      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7) {
        continue;
      }

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake()) {
        continue;
      }

      auto eta = trackParCov.getEta();

      // applying the cuts on the track - only eta

      if (eta < mEtaCuts[0] || eta >= mEtaCuts[1]) {
        continue;
      }

      float phi = -999.;
      float phiOriginal = -999.;

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]]; // cluster spacepoint in the tracking system
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        auto layer = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layer >= NLAYERS) {
          continue; // checking only selected layers
        }
        auto labsTrack = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;
        phiOriginal = clusOriginalPointGlob.phi(); // * 180 / M_PI;

        if (abs(clusOriginalPointGlob.y()) < 0.5) { ///// excluding gap between bottom and top barrels
          continue;
        }

        if (abs(clusOriginalPointGlob.z()) >= 10) { /// excluding external z
          continue;
        }

        if (clusOriginal.getRow() < 2 || (clusOriginal.getRow() > 15 && clusOriginal.getRow() < 496) || clusOriginal.getRow() > 509) { ////  cutting on the row
          continue;
        }

        if (clusOriginal.getCol() < 160 || clusOriginal.getCol() > 870) { /// excluding the gap between two chips in the same stave (comment to obtain the plot efficiency col vs eta)
          continue;
        }

        for (auto& labT : labsTrack) { // for each valid label iterate over ALL the clusters in the ROF to see if there are duplicates
          if (labT != tracklab) {
            continue;
          }

          if (labT.isValid()) {
            for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF

              auto clusDuplicated = mClusters[iClus];
              auto clusDuplicatedPoint = mITSClustersArray[iClus];

              auto layerClus = mGeometry->getLayer(clusDuplicated.getSensorID());
              if (layerClus != layer) {
                continue;
              }

              o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
              o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
              phi = clusDuplicatedPointGlob.phi(); // * 180 / M_PI;

              auto labsClus = mClustersMCLCont->getLabels(iClus); // ideally I can have more than one label per cluster
              for (auto labC : labsClus) {
                if (labC == labT) {
                  label_vecClus[iROF][layerClus][labT].push_back(iClus); // same cluster: label from the track = label from the cluster
                                                                         // if a duplicate cluster is found, propagate the track to the duplicate cluster and compute the distance from the original cluster
                                                                         // if (clusOriginalPointGlob != clusDuplicatedPointGlob) { /// check that the duplicated cluster is not the original one just counted twice
                                                                         // if (clusDuplicated.getSensorID() != clusOriginal.getSensorID()) { /// check that the duplicated cluster is not the original one just counted twice

                  // applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
                  if (clusDuplicated.getSensorID() == clusOriginal.getSensorID()) {
                    continue;
                  }
                  auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
                  if (layerDuplicated != layerClus) {
                    continue;
                  }
                  auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
                  if (abs(staveDuplicated - staveOriginal) != 1) {
                    continue;
                  }
                  auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
                  if (abs(chipDuplicated - chipOriginal) > 1) {
                    continue;
                  }

                  duplicated[layer]++;
                  std::cout << "Taken L" << layer << " # " << duplicated[layer] << " : eta, phi = " << eta << " , " << phiOriginal << " Label: " << std::endl;
                  labC.print();
                }
              }
            }
          }
        }
      } // end loop on clusters
      totClus += ncl;
    } // end loop on tracks per ROF
  } // end loop on ROFRecords array

  LOGP(info, "Total number of possible cluster duplicated in L0: {} ", possibleduplicated[0]);
  LOGP(info, "Total number of possible cluster duplicated in L1: {} ", possibleduplicated[1]);
  LOGP(info, "Total number of possible cluster duplicated in L2: {} ", possibleduplicated[2]);

  LOGP(info, "Total number of cluster duplicated in L0: {} ", duplicated[0]);
  LOGP(info, "Total number of cluster duplicated in L1: {} ", duplicated[1]);
  LOGP(info, "Total number of cluster duplicated in L2: {} ", duplicated[2]);
}

void EfficiencyStudy::studyDCAcutsMC()
{
  //// Study the DCA cuts to be applied

  LOGP(info, "--------------- studyDCAcutsMC");

  int duplicated = getDCAClusterTrackMC(0);

  double meanDCAxyDuplicated[NLAYERS] = {0};
  double meanDCAzDuplicated[NLAYERS] = {0};
  double sigmaDCAxyDuplicated[NLAYERS] = {0};
  double sigmaDCAzDuplicated[NLAYERS] = {0};

  std::ofstream ofs("dcaValues.csv", std::ofstream::out);
  ofs << "layer,dcaXY,dcaZ,sigmaDcaXY,sigmaDcaZ" << std::endl;

  for (int l = 0; l < NLAYERS; l++) {
    meanDCAxyDuplicated[l] = mDCAxyDuplicated_layer[l]->GetMean();
    meanDCAzDuplicated[l] = mDCAzDuplicated_layer[l]->GetMean();
    sigmaDCAxyDuplicated[l] = mDCAxyDuplicated_layer[l]->GetRMS();
    sigmaDCAzDuplicated[l] = mDCAzDuplicated_layer[l]->GetRMS();
    ofs << l << "," << meanDCAxyDuplicated[l] << "," << meanDCAzDuplicated[l] << "," << sigmaDCAxyDuplicated[l] << "," << sigmaDCAzDuplicated[l] << std::endl;
  }

  for (int l = 0; l < NLAYERS; l++) {
    LOGP(info, "meanDCAxyDuplicated L{}: {}, meanDCAzDuplicated: {}, sigmaDCAxyDuplicated: {}, sigmaDCAzDuplicated: {}", l, meanDCAxyDuplicated[l], meanDCAzDuplicated[l], sigmaDCAxyDuplicated[l], sigmaDCAzDuplicated[l]);
  }
  // now we have the DCA distribution:
  //  ->iterate again over tracks and over duplicated clusters and find the matching ones basing on DCA cuts (1 sigma, 2 sigma,...)
  //  then control if the matching ones according to the DCA matches have the same label
  //  if yes, then we have a good match -> increase the good match counter
  //  if not, keep it as a fake match -> increase the fake match counter
  //  the efficiency of each one will be match counter / total of the duplicated clusters
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
  auto propagator = o2::base::Propagator::Instance();

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;
  int nLabels = 0;
  unsigned int totClus = 0;

  unsigned int nDCAMatches[20] = {0};
  unsigned int nGoodMatches[20] = {0};
  unsigned int nFakeMatches[20] = {0};

  unsigned int nGoodMatches_layer[NLAYERS][20] = {0};
  unsigned int nFakeMatches_layer[NLAYERS][20] = {0};

  int nbPt = 75;
  double xbins[nbPt + 1], ptcutl = 0.05, ptcuth = 7.5;
  double a = std::log(ptcuth / ptcutl) / nbPt;
  for (int i = 0; i <= nbPt; i++) {
    xbins[i] = ptcutl * std::exp(i * a);
  }

  for (unsigned int iROF = 0; iROF < mTracksROFRecords.size(); iROF++) { // loop on ROFRecords array
    rofIndexTrack = mTracksROFRecords[iROF].getFirstEntry();
    rofNEntriesTrack = mTracksROFRecords[iROF].getNEntries();

    rofIndexClus = mClustersROFRecords[iROF].getFirstEntry();
    rofNEntriesClus = mClustersROFRecords[iROF].getNEntries();

    for (unsigned int iTrack = rofIndexTrack; iTrack < rofIndexTrack + rofNEntriesTrack; iTrack++) { // loop on tracks per ROF
      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];
      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();

      float ip[2];
      track.getImpactParams(0, 0, 0, 0, ip);

      float phi = -999.;
      float phiOriginal = -999.;
      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7) {
        continue;
      }

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake()) {
        continue;
      }

      if (mVerboseOutput) {
        LOGP(info, "--------- track Label: ");
        tracklab.print();
      }

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track to extract layer, stave and chip to restrict the possible matches to be searched with the DCA cut
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]]; // cluster spacepoint in the tracking system
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layerOriginal >= NLAYERS) {
          continue;
        }
        auto labsOriginal = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track (original)
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;

        phiOriginal = clusOriginalPointGlob.phi(); // * 180 / M_PI;

        for (auto& labT : labsOriginal) { // for each valid label iterate over ALL the clusters in the ROF to see if there are duplicates
          if (labT != tracklab) {
            continue;
          }
          if (!labT.isValid()) {
            continue;
          }

          /// for each oroginal cluster iterate over all the possible "adjacent" clusters (stave +-1, chip =,+-1) and calculate the DCA with the track. Then compare the cluster label with the track label to see if it is a true or fake match
          for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF
            auto clusDuplicated = mClusters[iClus];
            //// applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
            if (clusDuplicated.getSensorID() == clusOriginal.getSensorID()) {
              continue;
            }
            auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
            if (layerDuplicated != layerOriginal) {
              continue;
            }
            auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
            if (abs(staveDuplicated - staveOriginal) != 1) {
              continue;
            }
            auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
            if (abs(chipDuplicated - chipOriginal) > 1) {
              continue;
            }

            auto labsDuplicated = mClustersMCLCont->getLabels(iClus);

            /// if the cheks are passed, then calculate the DCA
            auto clusDuplicatedPoint = mITSClustersArray[iClus];

            o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
            o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
            phi = clusDuplicatedPointGlob.phi(); // * 180 / M_PI;

            /// Compute the DCA between the duplicated cluster location and the track
            trackParCov.rotate(mGeometry->getSensorRefAlpha(clusDuplicated.getSensorID()));
            if (propagator->propagateToDCA(clusDuplicatedPointGlob, trackParCov, b, 2.f, matCorr, &clusDuplicatedDCA)) { // check if the propagation fails
              if (mVerboseOutput) {
                LOGP(info, "Propagation ok");
              }
              /// checking the DCA for 20 different sigma ranges
              for (int i = 0; i < 20; i++) {
                if (abs(dcaXY[layerDuplicated] - clusDuplicatedDCA[0]) < (i + 1) * sigmaDcaXY[layerDuplicated] && abs(dcaZ[layerDuplicated] - clusDuplicatedDCA[1]) < (i + 1) * sigmaDcaZ[layerDuplicated]) { // check if the DCA is within the cut i*sigma

                  if (mVerboseOutput) {
                    LOGP(info, "Check DCA ok: {} < {}; {} < {}", abs(meanDCAxyDuplicated[layerDuplicated] - clusDuplicatedDCA[0]), (i + 1) * sigmaDCAxyDuplicated[layerDuplicated], abs(meanDCAzDuplicated[layerDuplicated] - clusDuplicatedDCA[1]), (i + 1) * sigmaDCAzDuplicated[layerDuplicated]);
                  }
                  nDCAMatches[i]++;
                  bool isGoodMatch = false;

                  for (auto labD : labsDuplicated) { // at this point the track has been matched with a duplicated cluster based on the DCA cut. Now we check if the matching is good ore not based on the label
                    if (mVerboseOutput) {
                      LOGP(info, "tracklab, labD:");
                      tracklab.print();
                      labD.print();
                    }
                    if (labD == tracklab) { //// check if the label of the origial cluster is equal to the label of the duplicated cluster among all the labels for a cluster
                      isGoodMatch = true;
                      continue;
                    }
                  }

                  if (isGoodMatch) {
                    nGoodMatches[i]++;
                    nGoodMatches_layer[layerDuplicated][i]++;
                    mnGoodMatchesPt_layer[layerDuplicated]->Fill(pt, i);
                    mnGoodMatchesEta_layer[layerDuplicated]->Fill(eta, i);
                    mnGoodMatchesPhi_layer[layerDuplicated]->Fill(phi, i);
                    mnGoodMatchesPhiOriginal_layer[layerDuplicated]->Fill(phiOriginal, i);
                  } else {

                    nFakeMatches[i]++;
                    nFakeMatches_layer[layerDuplicated][i]++;
                    mnFakeMatchesPt_layer[layerDuplicated]->Fill(pt, i);
                    mnFakeMatchesEta_layer[layerDuplicated]->Fill(eta, i);
                    mnFakeMatchesPhi_layer[layerDuplicated]->Fill(phi, i);
                  }
                } else if (mVerboseOutput) {
                  LOGP(info, "Check DCA failed");
                }
              }
            } else if (mVerboseOutput) {
              LOGP(info, "Propagation failed");
            }
          } // end loop on all the clusters in the rof
        }
      } // end loop on clusters associated to the track
    } // end loop on tracks per ROF
  } // end loop on ROFRecords array

  for (int i = 0; i < 20; i++) {
    LOGP(info, "Cut: {} sigma -> number of duplicated clusters: {} nDCAMatches: {} nGoodMatches: {} nFakeMatches: {}", i + 1, duplicated, nDCAMatches[i], nGoodMatches[i], nFakeMatches[i]);
    mEfficiencyGoodMatch->SetBinContent(i + 1, nGoodMatches[i]);
    mEfficiencyFakeMatch->SetBinContent(i + 1, nFakeMatches[i]);
    mEfficiencyTotal->SetBinContent(i + 1, double(nGoodMatches[i] + nFakeMatches[i]));

    for (int l = 0; l < NLAYERS; l++) {
      mEfficiencyGoodMatch_layer[l]->SetBinContent(i + 1, nGoodMatches_layer[l][i]);
      mEfficiencyFakeMatch_layer[l]->SetBinContent(i + 1, nFakeMatches_layer[l][i]);
      mEfficiencyTotal_layer[l]->SetBinContent(i + 1, double(nGoodMatches_layer[l][i] + nFakeMatches_layer[l][i]));

      for (int ipt = 0; ipt < mPtDuplicated[l]->GetNbinsX(); ipt++) {
        if (mPtDuplicated[l]->GetBinContent(ipt + 1) != 0) {
          mEfficiencyGoodMatchPt_layer[l]->SetBinContent(ipt + 1, i + 1, mnGoodMatchesPt_layer[l]->GetBinContent(ipt + 1, i + 1) / mPtDuplicated[l]->GetBinContent(ipt + 1));
        }
        mEfficiencyFakeMatchPt_layer[l]->SetBinContent(ipt + 1, i + 1, mnFakeMatchesPt_layer[l]->GetBinContent(ipt + 1, i + 1) / mPtDuplicated[l]->GetBinContent(ipt + 1));
      }

      for (int ieta = 0; ieta < mEtaDuplicated[l]->GetNbinsX(); ieta++) {
        if (mEtaDuplicated[l]->GetBinContent(ieta + 1) != 0) {
          mEfficiencyGoodMatchEta_layer[l]->SetBinContent(ieta + 1, i + 1, mnGoodMatchesEta_layer[l]->GetBinContent(ieta + 1, i + 1) / mEtaDuplicated[l]->GetBinContent(ieta + 1));
        }
        mEfficiencyFakeMatchEta_layer[l]->SetBinContent(ieta + 1, i + 1, mnFakeMatchesEta_layer[l]->GetBinContent(ieta + 1, i + 1) / mEtaDuplicated[l]->GetBinContent(ieta + 1));
      }

      for (int iphi = 0; iphi < mPhiDuplicated[l]->GetNbinsX(); iphi++) {
        if (mPhiDuplicated[l]->GetBinContent(iphi + 1) != 0) {
          mEfficiencyGoodMatchPhi_layer[l]->SetBinContent(iphi + 1, i + 1, mnGoodMatchesPhi_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiDuplicated[l]->GetBinContent(iphi + 1));
        }
        mEfficiencyFakeMatchPhi_layer[l]->SetBinContent(iphi + 1, i + 1, mnFakeMatchesPhi_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiDuplicated[l]->GetBinContent(iphi + 1));
      }

      for (int iphi = 0; iphi < mPhiOriginalIfDuplicated[l]->GetNbinsX(); iphi++) {
        if (mPhiOriginalIfDuplicated[l]->GetBinContent(iphi + 1) != 0) {
          mEfficiencyGoodMatchPhiOriginal_layer[l]->SetBinContent(iphi + 1, i + 1, mnGoodMatchesPhiOriginal_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiOriginalIfDuplicated[l]->GetBinContent(iphi + 1));
        }
      }
    }
  }
  for (int i = 0; i < NLAYERS; i++) {
    std::cout << "+++++++++ Duplicated in layer L" << i << ": " << mDuplicated_layer[i] << std::endl;
  }

  for (int l = 0; l < NLAYERS; l++) {
    mEfficiencyGoodMatch_layer[l]->Scale(1. / double(mDuplicated_layer[l]), "b");
    mEfficiencyFakeMatch_layer[l]->Scale(1. / double(mDuplicated_layer[l]), "b");
    mEfficiencyTotal_layer[l]->Scale(1. / double(mDuplicated_layer[l]), "b");
  }

  mEfficiencyGoodMatch->Scale(1. / double(duplicated), "b");
  mEfficiencyFakeMatch->Scale(1. / double(duplicated), "b");
  mEfficiencyTotal->Scale(1. / double(duplicated), "b");

  mOutFile->mkdir("EffVsDCA2D/");
  mOutFile->cd("EffVsDCA2D/");
  for (int l = 0; l < NLAYERS; l++) {
    mEfficiencyGoodMatchPt_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyGoodMatchPt_layer[l]->Write();
    mEfficiencyGoodMatchEta_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyGoodMatchEta_layer[l]->Write();
    mEfficiencyGoodMatchPhi_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyGoodMatchPhi_layer[l]->Write();
    mEfficiencyGoodMatchPhiOriginal_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyGoodMatchPhiOriginal_layer[l]->Write();
    mEfficiencyFakeMatchPt_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyFakeMatchPt_layer[l]->Write();
    mEfficiencyFakeMatchEta_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyFakeMatchEta_layer[l]->Write();
    mEfficiencyFakeMatchPhi_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyFakeMatchPhi_layer[l]->Write();
  }

  mOutFile->mkdir("Efficiency/");
  mOutFile->cd("Efficiency/");
  mEfficiencyGoodMatch->Write();
  mEfficiencyFakeMatch->Write();
  mEfficiencyTotal->Write();
  for (int l = 0; l < NLAYERS; l++) {

    mEfficiencyGoodMatch_layer[l]->Write();
    mEfficiencyFakeMatch_layer[l]->Write();
    mEfficiencyTotal_layer[l]->Write();

    mEfficiencyGoodMatch_layer[l]->GetYaxis()->SetRangeUser(-0.1, 1.1);
    mEfficiencyFakeMatch_layer[l]->GetYaxis()->SetRangeUser(-0.1, 1.1);
    mEfficiencyTotal_layer[l]->GetYaxis()->SetRangeUser(-0.1, 1.1);
  }

  mEfficiencyGoodMatch->GetYaxis()->SetRangeUser(-0.1, 1.1);
  mEfficiencyFakeMatch->GetYaxis()->SetRangeUser(-0.1, 1.1);
  mEfficiencyTotal->GetYaxis()->SetRangeUser(-0.1, 1.1);

  TCanvas c;
  c.SetName("EffVsDCA_allLayers");
  auto leg = std::make_unique<TLegend>(0.75, 0.45, 0.89, 0.65);
  leg->AddEntry(mEfficiencyGoodMatch.get(), "#frac{# good matches}{# tot duplicated clusters}", "p");
  leg->AddEntry(mEfficiencyFakeMatch.get(), "#frac{# fake matches}{# tot duplicated clusters}", "p");
  leg->AddEntry(mEfficiencyTotal.get(), "#frac{# tot matches}{# tot duplicated clusters}", "p");

  mEfficiencyGoodMatch->Draw("P l E1_NOSTAT PLC PMC ");
  mEfficiencyFakeMatch->Draw("same P l E1_NOSTAT  PLC PMC");
  mEfficiencyTotal->Draw("same P l E1_NOSTAT  PLC PMC");
  leg->Draw("same");
  c.Write();

  TCanvas cc[NLAYERS];
  for (int l = 0; l < NLAYERS; l++) {
    cc[l].cd();
    cc[l].SetName(Form("EffVsDCA_layer_L%d", l));

    auto leg = std::make_unique<TLegend>(0.75, 0.45, 0.89, 0.65);
    leg->AddEntry(mEfficiencyGoodMatch_layer[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "p");
    leg->AddEntry(mEfficiencyFakeMatch_layer[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "p");
    leg->AddEntry(mEfficiencyTotal_layer[l].get(), "#frac{# tot matches}{# tot duplicated clusters}", "p");

    mEfficiencyGoodMatch_layer[l]->SetLineColor(kBlue + 3);
    mEfficiencyGoodMatch_layer[l]->SetMarkerColor(kBlue + 3);
    mEfficiencyGoodMatch_layer[l]->Draw("P l E1_NOSTAT");
    mEfficiencyFakeMatch_layer[l]->SetLineColor(kAzure + 7);
    mEfficiencyFakeMatch_layer[l]->SetMarkerColor(kAzure + 7);
    mEfficiencyFakeMatch_layer[l]->Draw("same P l E1_NOSTAT");
    mEfficiencyTotal_layer[l]->SetLineColor(kGreen + 1);
    mEfficiencyTotal_layer[l]->SetMarkerColor(kGreen + 1);
    mEfficiencyTotal_layer[l]->Draw("same P l E1_NOSTAT");
    leg->Draw("same");
    cc[l].Write();
  }
}

void EfficiencyStudy::studyClusterSelectionMC()
{
  //// to be used only with MC
  // study to find a good selection method for the duplicated cluster, to be used for non-MC data
  // iterate over tracks an associated clusters, and find the closer cluster that is not the original one applying cuts on staveID and chipID
  // fix the DCA < 10 sigma, then compute the efficiency for each bin of pt, eta and phi and also in the rows

  LOGP(info, "--------------- studyClusterSelection");

  int duplicated = getDCAClusterTrackMC(1);

  std::cout << "duplicated: " << duplicated << std::endl;

  double meanDCAxyDuplicated[NLAYERS] = {0};
  double meanDCAzDuplicated[NLAYERS] = {0};
  double sigmaDCAxyDuplicated[NLAYERS] = {0};
  double sigmaDCAzDuplicated[NLAYERS] = {0};

  for (int l = 0; l < NLAYERS; l++) {
    meanDCAxyDuplicated[l] = mDCAxyDuplicated_layer[l]->GetMean();
    meanDCAzDuplicated[l] = mDCAzDuplicated_layer[l]->GetMean();
    sigmaDCAxyDuplicated[l] = mDCAxyDuplicated_layer[l]->GetRMS();
    sigmaDCAzDuplicated[l] = mDCAzDuplicated_layer[l]->GetRMS();
  }

  for (int l = 0; l < NLAYERS; l++) {
    LOGP(info, "meanDCAxyDuplicated L{}: {}, meanDCAzDuplicated: {}, sigmaDCAxyDuplicated: {}, sigmaDCAzDuplicated: {}", l, meanDCAxyDuplicated[l], meanDCAzDuplicated[l], sigmaDCAxyDuplicated[l], sigmaDCAzDuplicated[l]);
  }

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
  auto propagator = o2::base::Propagator::Instance();

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;
  int nLabels = 0;
  unsigned int totClus = 0;

  unsigned int nDCAMatches[15] = {0};
  unsigned int nGoodMatches[15] = {0};
  unsigned int nFakeMatches[15] = {0};

  std::map<std::tuple<int, double, o2::MCCompLabel>, bool> clusterMatchesPtEta[100][100] = {};

  for (unsigned int iROF = 0; iROF < mTracksROFRecords.size(); iROF++) { // loop on ROFRecords array
    rofIndexTrack = mTracksROFRecords[iROF].getFirstEntry();
    rofNEntriesTrack = mTracksROFRecords[iROF].getNEntries();

    rofIndexClus = mClustersROFRecords[iROF].getFirstEntry();
    rofNEntriesClus = mClustersROFRecords[iROF].getNEntries();

    //////calculcating efficiency vs pt, eta, phi
    for (unsigned int iTrack = rofIndexTrack; iTrack < rofIndexTrack + rofNEntriesTrack; iTrack++) { // loop on tracks per ROF
      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];

      /// cut on primary vertex position (?)
      float ip[2];
      track.getImpactParams(0, 0, 0, 0, ip);

      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7) {
        continue;
      }

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake()) {
        continue;
      }

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();

      float phi = -999.;
      float phiOriginal = -999.;
      float phiDuplicated = -999.;
      UShort_t row = -999;

      if (mVerboseOutput) {
        LOGP(info, "--------- track Label: ");
        tracklab.print();
      }
      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track to extract layer, stave and chip to restrict the possible matches to be searched with the DCA cut
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layerOriginal >= NLAYERS) {
          continue;
        }

        IPOriginalxy[layerOriginal]->Fill(ip[0]);
        IPOriginalz[layerOriginal]->Fill(ip[1]);

        UShort_t rowOriginal = clusOriginal.getRow();

        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]];
        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;

        auto phiOriginal = clusOriginalPointGlob.phi(); // * 180 / M_PI;

        auto labsOriginal = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track (original)
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        std::tuple<int, double, gsl::span<const o2::MCCompLabel>> clusID_rDCA_label = {0, 999., gsl::span<const o2::MCCompLabel>()}; // inizializing tuple with dummy values

        bool adjacentFound = 0;
        /// for each oroginal cluster iterate over all the possible "adjacten" clusters (stave +-1, chip =,+-1) and calculate the DCA with the track. Then choose the closest one.
        for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF
          auto clusDuplicated = mClusters[iClus];

          //// applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
          if (clusDuplicated.getSensorID() == clusOriginal.getSensorID()) {
            continue;
          }
          auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
          if (layerDuplicated != layerOriginal) {
            continue;
          }
          auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
          if (abs(staveDuplicated - staveOriginal) != 1) {
            continue;
          }
          auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
          if (abs(chipDuplicated - chipOriginal) > 1) {
            continue;
          }

          auto labsDuplicated = mClustersMCLCont->getLabels(iClus);

          /// if the cheks are passed, then calculate the DCA
          auto clusDuplicatedPoint = mITSClustersArray[iClus];

          o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
          o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;

          auto phiDuplicated = clusDuplicatedPointGlob.phi(); // * 180 / M_PI;

          /// Compute the DCA between the duplicated cluster location and the track
          trackParCov.rotate(mGeometry->getSensorRefAlpha(clusDuplicated.getSensorID()));
          if (!propagator->propagateToDCA(clusDuplicatedPointGlob, trackParCov, b, 2.f, matCorr, &clusDuplicatedDCA)) { // check if the propagation fails
            continue;
          }

          // Imposing that the distance between the original cluster and the duplicated one is less than x sigma
          if (!(clusDuplicatedDCA[0] > mDCACutsXY[layerDuplicated][0] && clusDuplicatedDCA[0] < mDCACutsXY[layerDuplicated][1] && clusDuplicatedDCA[1] > mDCACutsZ[layerDuplicated][0] && clusDuplicatedDCA[1] < mDCACutsZ[layerDuplicated][1])) {
            continue;
          }

          if (mVerboseOutput) {
            LOGP(info, "Propagation ok");
          }
          double rDCA = std::hypot(clusDuplicatedDCA[0], clusDuplicatedDCA[1]);

          // taking the closest cluster within x sigma
          if (rDCA < std::get<1>(clusID_rDCA_label)) { // updating the closest cluster
            clusID_rDCA_label = {iClus, rDCA, labsDuplicated};
            phi = phiDuplicated;
            row = rowOriginal;
          }
          adjacentFound = 1;

        } // end loop on all the clusters in the rof

        // here clusID_rDCA_label is updated with the closest cluster to the track other than the original one
        // checking if it is a good or fake match looking at the labels

        if (!adjacentFound) {
          continue;
        }

        bool isGood = false;
        for (auto lab : std::get<2>(clusID_rDCA_label)) {
          if (lab == tracklab) {
            isGood = true;

            mNGoodMatchesPt[layerOriginal]->Fill(pt);
            mNGoodMatchesRow[layerOriginal]->Fill(row);
            mNGoodMatchesCol[layerOriginal]->Fill(clusOriginal.getCol());
            mNGoodMatchesZ[layerOriginal]->Fill(clusOriginalPointGlob.Z());
            mNGoodMatchesPtEta[layerOriginal]->Fill(pt, eta);
            mNGoodMatchesPtPhi[layerOriginal]->Fill(pt, phi);
            mNGoodMatchesEtaPhi[layerOriginal]->Fill(eta, phi);

            mNGoodMatchesEtaAllPt[layerOriginal]->Fill(eta);
            mNGoodMatchesPhiAllPt[layerOriginal]->Fill(phi);
            for (int ipt = 0; ipt < 3; ipt++) {
              if (pt >= mrangesPt[ipt][0] && pt < mrangesPt[ipt][1]) {
                mNGoodMatchesEta[layerOriginal][ipt]->Fill(eta);
                mNGoodMatchesPhi[layerOriginal][ipt]->Fill(phi);
              }
            }

            break;
          }
        }
        if (!isGood) {

          mNFakeMatchesPt[layerOriginal]->Fill(pt);
          mNFakeMatchesRow[layerOriginal]->Fill(row);
          mNFakeMatchesCol[layerOriginal]->Fill(clusOriginal.getCol());
          mNFakeMatchesZ[layerOriginal]->Fill(clusOriginalPointGlob.Z());
          mNFakeMatchesPtEta[layerOriginal]->Fill(pt, eta);
          mNFakeMatchesPtPhi[layerOriginal]->Fill(pt, phi);
          mNFakeMatchesEtaPhi[layerOriginal]->Fill(eta, phi);
          mNFakeMatchesEtaAllPt[layerOriginal]->Fill(eta);
          mNFakeMatchesPhiAllPt[layerOriginal]->Fill(phi);

          for (int ipt = 0; ipt < 3; ipt++) {
            if (pt >= mrangesPt[ipt][0] && pt < mrangesPt[ipt][1]) {
              mNFakeMatchesEta[layerOriginal][ipt]->Fill(eta);
              mNFakeMatchesPhi[layerOriginal][ipt]->Fill(phi);
            }
          }
        }
      } // end loop on clusters associated to the track
    } // end loop on tracks per ROF
  } // end loop on ROFRecords array

  mOutFile->mkdir("EfficiencyCuts/");
  mOutFile->cd("EfficiencyCuts/");

  std::cout << "Calculating efficiency..." << std::endl;
  std::unique_ptr<TH1D> axpt = std::make_unique<TH1D>("axpt", "", 1, 0.05, 7.5);
  std::unique_ptr<TH1D> axRow = std::make_unique<TH1D>("axRow", "", 1, -0.5, 511.5);
  std::unique_ptr<TH1D> axCol = std::make_unique<TH1D>("axRow", "", 1, -0.5, 1023.5);
  std::unique_ptr<TH1D> axZ = std::make_unique<TH1D>("axZ", "", 1, -15, 15);
  std::unique_ptr<TH2D> axptetaGood = std::make_unique<TH2D>("axptetaGood", "", 1, 0.05, 7.5, 1, -2, 2);
  std::unique_ptr<TH2D> axptetaFake = std::make_unique<TH2D>("axptetaFake", "", 1, 0.05, 7.5, 1, -2, 2);
  std::unique_ptr<TH2D> axptphiGood = std::make_unique<TH2D>("axptphiGood", "", 1, 0.05, 7.5, 1, -3.2, 3.2);
  std::unique_ptr<TH2D> axptphiFake = std::make_unique<TH2D>("axptphiFake", "", 1, 0.05, 7.5, 1, -3.2, 3.2);
  std::unique_ptr<TH2D> axetaphiGood = std::make_unique<TH2D>("axetaphiGood", "", 1, -2, 2, 1, -3.2, 3.2);
  std::unique_ptr<TH2D> axetaphiFake = std::make_unique<TH2D>("axetaphiFake", "", 1, -2, 2, 1, -3.2, 3.2);
  std::unique_ptr<TH1D> axetaAllPt = std::make_unique<TH1D>("axetaAllPt", "", 1, -2, 2);
  std::unique_ptr<TH1D> axeta[NLAYERS];
  std::unique_ptr<TH1D> axphi[NLAYERS];
  for (int ipt = 0; ipt < 3; ipt++) {
    axeta[ipt] = std::make_unique<TH1D>(Form("axeta%d", ipt), Form("axeta%d", ipt), 1, -2, 2);
    axphi[ipt] = std::make_unique<TH1D>(Form("axphi%d", ipt), Form("axphi%d", ipt), 1, -3.2, 3.2);
  }
  std::unique_ptr<TH1D> axphiAllPt = std::make_unique<TH1D>("axphi", "", 1, -3.2, 3.2);

  std::unique_ptr<TCanvas> effPt[NLAYERS];
  std::unique_ptr<TCanvas> effRow[NLAYERS];
  std::unique_ptr<TCanvas> effCol[NLAYERS];
  std::unique_ptr<TCanvas> effZ[NLAYERS];
  std::unique_ptr<TCanvas> effPtEta[NLAYERS][2];
  std::unique_ptr<TCanvas> effPtPhi[NLAYERS][2];
  std::unique_ptr<TCanvas> effEtaPhi[NLAYERS][2];
  std::unique_ptr<TCanvas> effEtaAllPt[NLAYERS];
  std::unique_ptr<TCanvas> effEta[NLAYERS][3];
  std::unique_ptr<TCanvas> effPhiAllPt[NLAYERS];
  std::unique_ptr<TCanvas> effPhi[NLAYERS][3];

  ///////////////// plotting results
  for (int l = 0; l < 3; l++) {
    if (mVerboseOutput) {
      std::cout << "Pt L" << l << "\n\n";
    }

    // Pt
    effPt[l] = std::make_unique<TCanvas>(Form("effPt_L%d", l));

    mEffPtGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesPt[l], *mDuplicatedPt[l]);
    stileEfficiencyGraph(mEffPtGood[l], Form("mEffPtGood_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesPt[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesPt[l]->GetBinContent(ibin) > mDuplicatedPt[l]->GetBinContent(ibin)) {
        std::cout << "--- Pt: Npass = " << mNFakeMatchesPt[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPt[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        mNFakeMatchesPt[l]->SetBinContent(ibin, mDuplicatedPt[l]->GetBinContent(ibin));
      }
    }
    mEffPtFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesPt[l], *mDuplicatedPt[l]);
    stileEfficiencyGraph(mEffPtFake[l], Form("mEffPtFake_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});Efficiency", l), false, kFullDiamond, 1, kRed + 1, kRed + 1);

    axpt->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});Efficiency", l));
    axpt->GetYaxis()->SetRangeUser(-0.1, 1.1);
    axpt->GetXaxis()->SetRangeUser(0.05, 7.5);
    axpt->Draw();
    mEffPtGood[l]->Draw("same p");
    mEffPtFake[l]->Draw("same p");

    auto legpt = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
    legpt->AddEntry(mEffPtGood[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
    legpt->AddEntry(mEffPtFake[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
    legpt->Draw("same");
    effPt[l]->Write();

    // PtEtaGood
    effPtEta[l][0] = std::make_unique<TCanvas>(Form("effPtEtaGood_L%d", l));

    mEffPtEtaGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesPtEta[l], *mDuplicatedPtEta[l]);
    stileEfficiencyGraph(mEffPtEtaGood[l], Form("mEffPtEtaGood_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});#eta;Efficiency", l), true);

    axptetaGood->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});#eta;Efficiency", l));
    axptetaGood->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axptetaGood->GetYaxis()->SetRangeUser(-2., 2.);
    axptetaGood->GetXaxis()->SetRangeUser(0.05, 7.5);
    axptetaGood->Draw();
    mEffPtEtaGood[l]->Draw("same colz");
    effPtEta[l][0]->Update();
    effPtEta[l][0]->Write();

    if (mVerboseOutput) {
      std::cout << "Underflow (bin 0,0): " << mNFakeMatchesPtEta[l]->GetBinContent(0, 0) << "    " << mDuplicatedPtEta[l]->GetBinContent(0, 0) << std::endl;
      std::cout << "Overflow (bin nbinsx,nbinsy): " << mNFakeMatchesPtEta[l]->GetNbinsX() << "   " << mNFakeMatchesPtEta[l]->GetNbinsY() << "  -> " << mNFakeMatchesPtEta[l]->GetBinContent(mNFakeMatchesPtEta[l]->GetNbinsX(), mNFakeMatchesPtEta[l]->GetNbinsY()) << "    " << mDuplicatedPtEta[l]->GetBinContent(mNFakeMatchesPtEta[l]->GetNbinsX(), mNFakeMatchesPtEta[l]->GetNbinsY()) << std::endl;
    }

    for (int ibin = 1; ibin <= mNFakeMatchesPtEta[l]->GetNbinsX(); ibin++) {
      for (int jbin = 1; jbin <= mNFakeMatchesPtEta[l]->GetNbinsY(); jbin++) {
        if (mNFakeMatchesPtEta[l]->GetBinContent(ibin, jbin) > mDuplicatedPtEta[l]->GetBinContent(ibin, jbin)) {
          if (mVerboseOutput) {
            std::cout << "--- PtEta fakematches : Npass = " << mNFakeMatchesPtEta[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedPtEta[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          }
          mNFakeMatchesPtEta[l]->SetBinContent(ibin, jbin, mDuplicatedPtEta[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // Row
    effRow[l] = std::make_unique<TCanvas>(Form("effRow_L%d", l));

    for (int ibin = 1; ibin <= mNGoodMatchesRow[l]->GetNbinsX(); ibin++) {
      std::cout << "--- Good Row: Npass = " << mNGoodMatchesRow[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedRow[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
    }

    mEffRowGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesRow[l], *mDuplicatedRow[l]);
    stileEfficiencyGraph(mEffRowGood[l], Form("mEffRowGood_L%d", l), Form("L%d;Row;Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesRow[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesRow[l]->GetBinContent(ibin) > mDuplicatedRow[l]->GetBinContent(ibin)) {
        std::cout << "--- Row: Npass = " << mNFakeMatchesRow[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedRow[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        mNFakeMatchesRow[l]->SetBinContent(ibin, mDuplicatedRow[l]->GetBinContent(ibin));
      }
    }
    mEffRowFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesRow[l], *mDuplicatedRow[l]);
    stileEfficiencyGraph(mEffRowFake[l], Form("mEffRowFake_L%d", l), Form("L%d;Row;Efficiency", l), false, kFullDiamond, 1, kRed + 1, kRed + 1);

    axRow->SetTitle(Form("L%d;Row;Efficiency", l));
    axRow->GetYaxis()->SetRangeUser(-0.1, 1.1);
    axRow->GetXaxis()->SetRangeUser(0, 512);
    axRow->Draw();
    mEffRowGood[l]->Draw("same p");
    mEffRowFake[l]->Draw("same p");

    auto legRow = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
    legRow->AddEntry(mEffRowGood[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
    legRow->AddEntry(mEffRowFake[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
    legRow->Draw("same");
    effRow[l]->Write();

    // Col
    effCol[l] = std::make_unique<TCanvas>(Form("effCol_L%d", l));

    for (int ibin = 1; ibin <= mNGoodMatchesCol[l]->GetNbinsX(); ibin++) {
      std::cout << "--- Good Col: Npass = " << mNGoodMatchesCol[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedCol[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
    }

    mEffColGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesCol[l], *mDuplicatedCol[l]);
    stileEfficiencyGraph(mEffColGood[l], Form("mEffColGood_L%d", l), Form("L%d;Col;Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesCol[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesCol[l]->GetBinContent(ibin) > mDuplicatedCol[l]->GetBinContent(ibin)) {
        std::cout << "--- Col: Npass = " << mNFakeMatchesCol[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedCol[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        mNFakeMatchesCol[l]->SetBinContent(ibin, mDuplicatedCol[l]->GetBinContent(ibin));
      }
    }
    mEffColFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesCol[l], *mDuplicatedCol[l]);
    stileEfficiencyGraph(mEffColFake[l], Form("mEffColFake_L%d", l), Form("L%d;Col;Efficiency", l), false, kFullDiamond, 1, kRed + 1, kRed + 1);

    axCol->SetTitle(Form("L%d;Col;Efficiency", l));
    axCol->GetYaxis()->SetRangeUser(-0.1, 1.1);
    axCol->GetXaxis()->SetRangeUser(0, 1024);
    axCol->Draw();
    mEffColGood[l]->Draw("same p");
    mEffColFake[l]->Draw("same p");

    auto legCol = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
    legCol->AddEntry(mEffColGood[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
    legCol->AddEntry(mEffColFake[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
    legCol->Draw("same");
    effCol[l]->Write();

    // Z
    effZ[l] = std::make_unique<TCanvas>(Form("effZ_L%d", l));

    for (int ibin = 1; ibin <= mNGoodMatchesZ[l]->GetNbinsX(); ibin++) {
      std::cout << "--- Good Z: Npass = " << mNGoodMatchesZ[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedZ[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
    }

    mEffZGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesZ[l], *mDuplicatedZ[l]);
    stileEfficiencyGraph(mEffZGood[l], Form("mEffZGood_L%d", l), Form("L%d;Z;Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesZ[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesZ[l]->GetBinContent(ibin) > mDuplicatedZ[l]->GetBinContent(ibin)) {
        std::cout << "--- Z: Npass = " << mNFakeMatchesZ[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedZ[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        mNFakeMatchesZ[l]->SetBinContent(ibin, mDuplicatedZ[l]->GetBinContent(ibin));
      }
    }
    mEffZFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesZ[l], *mDuplicatedZ[l]);
    stileEfficiencyGraph(mEffZFake[l], Form("mEffZFake_L%d", l), Form("L%d;Z;Efficiency", l), false, kFullDiamond, 1, kRed + 1, kRed + 1);

    axZ->SetTitle(Form("L%d;Z;Efficiency", l));
    axZ->GetYaxis()->SetRangeUser(-0.1, 1.1);
    axZ->GetXaxis()->SetRangeUser(0, 512);
    axZ->Draw();
    mEffZGood[l]->Draw("same p");
    mEffZFake[l]->Draw("same p");

    auto legZ = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
    legZ->AddEntry(mEffZGood[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
    legZ->AddEntry(mEffZFake[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
    legZ->Draw("same");
    effZ[l]->Write();

    // PtEtaGood
    effPtEta[l][0] = std::make_unique<TCanvas>(Form("effPtEtaGood_L%d", l));

    mEffPtEtaGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesPtEta[l], *mDuplicatedPtEta[l]);
    stileEfficiencyGraph(mEffPtEtaGood[l], Form("mEffPtEtaGood_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});#eta;Efficiency", l), true);

    axptetaGood->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});#eta;Efficiency", l));
    axptetaGood->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axptetaGood->GetYaxis()->SetRangeUser(-2., 2.);
    axptetaGood->GetXaxis()->SetRangeUser(0.05, 7.5);
    axptetaGood->Draw();
    mEffPtEtaGood[l]->Draw("same colz");
    effPtEta[l][0]->Update();
    effPtEta[l][0]->Write();

    if (mVerboseOutput) {
      std::cout << "Underflow (bin 0,0): " << mNFakeMatchesPtEta[l]->GetBinContent(0, 0) << "    " << mDuplicatedPtEta[l]->GetBinContent(0, 0) << std::endl;
      std::cout << "Overflow (bin nbinsx,nbinsy): " << mNFakeMatchesPtEta[l]->GetNbinsX() << "   " << mNFakeMatchesPtEta[l]->GetNbinsY() << "  -> " << mNFakeMatchesPtEta[l]->GetBinContent(mNFakeMatchesPtEta[l]->GetNbinsX(), mNFakeMatchesPtEta[l]->GetNbinsY()) << "    " << mDuplicatedPtEta[l]->GetBinContent(mNFakeMatchesPtEta[l]->GetNbinsX(), mNFakeMatchesPtEta[l]->GetNbinsY()) << std::endl;
    }

    for (int ibin = 1; ibin <= mNFakeMatchesPtEta[l]->GetNbinsX(); ibin++) {
      for (int jbin = 1; jbin <= mNFakeMatchesPtEta[l]->GetNbinsY(); jbin++) {
        if (mNFakeMatchesPtEta[l]->GetBinContent(ibin, jbin) > mDuplicatedPtEta[l]->GetBinContent(ibin, jbin)) {
          if (mVerboseOutput) {
            std::cout << "--- PtEta fakematches : Npass = " << mNFakeMatchesPtEta[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedPtEta[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          }
          mNFakeMatchesPtEta[l]->SetBinContent(ibin, jbin, mDuplicatedPtEta[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // PtEtaFake
    effPtEta[l][1] = std::make_unique<TCanvas>(Form("effPtEtaFake_L%d", l));

    mEffPtEtaFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesPtEta[l], *mDuplicatedPtEta[l]);
    stileEfficiencyGraph(mEffPtEtaFake[l], Form("mEffPtEtaFake_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});#eta;Efficiency", l), true);
    axptetaFake->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});#eta;Efficiency", l));
    axptetaFake->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axptetaFake->GetYaxis()->SetRangeUser(-2., 2.);
    axptetaFake->GetXaxis()->SetRangeUser(0.05, 7.5);
    axptetaFake->Draw();
    mEffPtEtaFake[l]->Draw("same colz");
    effPtEta[l][1]->Update();
    effPtEta[l][1]->Write();

    // PtPhiGood
    effPtPhi[l][0] = std::make_unique<TCanvas>(Form("effPtPhiGood_L%d", l));

    mEffPtPhiGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesPtPhi[l], *mDuplicatedPtPhi[l]);
    stileEfficiencyGraph(mEffPtPhiGood[l], Form("mEffPtPhiGood_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (rad);Efficiency", l), true);

    axptphiGood->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (rad);Efficiency", l));
    axptphiGood->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axptphiGood->GetYaxis()->SetRangeUser(-3.2, 3.2);
    axptphiGood->GetXaxis()->SetRangeUser(0.05, 7.5);
    axptphiGood->Draw();
    mEffPtPhiGood[l]->Draw("same colz");
    effPtPhi[l][0]->Update();
    effPtPhi[l][0]->Write();

    for (int ibin = 1; ibin <= mNFakeMatchesPtPhi[l]->GetNbinsX(); ibin++) {
      for (int jbin = 1; jbin <= mNFakeMatchesPtPhi[l]->GetNbinsY(); jbin++) {
        if (mNFakeMatchesPtPhi[l]->GetBinContent(ibin, jbin) > mDuplicatedPtPhi[l]->GetBinContent(ibin, jbin)) {
          if (mVerboseOutput) {
            std::cout << "--- Pt: Npass = " << mNFakeMatchesPtPhi[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedPtPhi[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          }
          mNFakeMatchesPtPhi[l]->SetBinContent(ibin, jbin, mDuplicatedPtPhi[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // PtPhiFake
    effPtPhi[l][1] = std::make_unique<TCanvas>(Form("effPtPhiFake_L%d", l));

    mEffPtPhiFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesPtPhi[l], *mDuplicatedPtPhi[l]);
    stileEfficiencyGraph(mEffPtPhiFake[l], Form("mEffPtPhiFake_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (rad);Efficiency", l), true);
    axptphiFake->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (rad);Efficiency", l));
    axptphiFake->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axptphiFake->GetYaxis()->SetRangeUser(-3.2, 3.2);
    axptphiFake->GetXaxis()->SetRangeUser(0.05, 7.5);
    axptphiFake->Draw();
    mEffPtPhiFake[l]->Draw("same colz");
    effPtPhi[l][1]->Update();
    effPtPhi[l][1]->Write();

    // EtaPhiGood
    effEtaPhi[l][0] = std::make_unique<TCanvas>(Form("effEtaPhiGood_L%d", l));

    mEffEtaPhiGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesEtaPhi[l], *mDuplicatedEtaPhi[l]);
    stileEfficiencyGraph(mEffEtaPhiGood[l], Form("mEffEtaPhiGood_L%d", l), Form("L%d;#eta;#phi (rad);Efficiency", l), true);

    axetaphiGood->SetTitle(Form("L%d;#eta;#phi (rad);Efficiency", l));
    axetaphiGood->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axetaphiGood->GetYaxis()->SetRangeUser(-3.2, 3.2);
    axetaphiGood->GetXaxis()->SetRangeUser(-2, 2);
    axetaphiGood->Draw();
    mEffEtaPhiGood[l]->Draw("same colz");
    effEtaPhi[l][0]->Update();
    effEtaPhi[l][0]->Write();

    for (int ibin = 1; ibin <= mNFakeMatchesEtaPhi[l]->GetNbinsX(); ibin++) {
      for (int jbin = 1; jbin <= mNFakeMatchesEtaPhi[l]->GetNbinsY(); jbin++) {
        if (mNFakeMatchesEtaPhi[l]->GetBinContent(ibin, jbin) > mDuplicatedEtaPhi[l]->GetBinContent(ibin, jbin)) {
          if (mVerboseOutput) {
            std::cout << "--- Eta: Npass = " << mNFakeMatchesEtaPhi[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedEtaPhi[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          }
          mNFakeMatchesEtaPhi[l]->SetBinContent(ibin, jbin, mDuplicatedEtaPhi[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // EtaPhiFake
    effEtaPhi[l][1] = std::make_unique<TCanvas>(Form("effEtaPhiFake_L%d", l));

    mEffEtaPhiFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesEtaPhi[l], *mDuplicatedEtaPhi[l]);
    stileEfficiencyGraph(mEffEtaPhiFake[l], Form("mEffEtaPhiFake_L%d", l), Form("L%d;#eta;#phi (rad);Efficiency", l), true);
    axetaphiFake->SetTitle(Form("L%d;#eta;#phi (rad);Efficiency", l));
    axetaphiFake->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axetaphiFake->GetYaxis()->SetRangeUser(-3.2, 3.2);
    axetaphiFake->GetXaxis()->SetRangeUser(-2, 2);
    axetaphiFake->Draw();
    mEffEtaPhiFake[l]->Draw("same colz");
    effEtaPhi[l][1]->Update();
    effEtaPhi[l][1]->Write();

    // EtaAllPt
    if (mVerboseOutput) {
      std::cout << "Eta L" << l << "\n\n";
    }

    effEtaAllPt[l] = std::make_unique<TCanvas>(Form("effEtaAllPt_L%d", l));

    mEffEtaGoodAllPt[l] = std::make_unique<TEfficiency>(*mNGoodMatchesEtaAllPt[l], *mDuplicatedEtaAllPt[l]);
    stileEfficiencyGraph(mEffEtaGoodAllPt[l], Form("mEffEtaGoodAllPt_L%d", l), Form("L%d;#eta;Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesEtaAllPt[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesEtaAllPt[l]->GetBinContent(ibin) > mDuplicatedEtaAllPt[l]->GetBinContent(ibin)) {
        if (mVerboseOutput) {
          std::cout << "--- EtaAllPt: Npass = " << mNFakeMatchesEtaAllPt[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedEtaAllPt[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        }
        mNFakeMatchesEtaAllPt[l]->SetBinContent(ibin, mDuplicatedEtaAllPt[l]->GetBinContent(ibin));
      }
    }
    mEffEtaFakeAllPt[l] = std::make_unique<TEfficiency>(*mNFakeMatchesEtaAllPt[l], *mDuplicatedEtaAllPt[l]);
    stileEfficiencyGraph(mEffEtaFakeAllPt[l], Form("mEffEtaFakeAllPt_L%d", l), Form("L%d;#eta;Efficiency", l), false, kFullDiamond, 1, kRed + 1, kRed + 1);

    axetaAllPt->SetTitle(Form("L%d;#eta;Efficiency", l));
    axetaAllPt->GetYaxis()->SetRangeUser(-0.1, 1.1);

    axetaAllPt->Draw();
    mEffEtaGoodAllPt[l]->Draw("same p");
    mEffEtaFakeAllPt[l]->Draw("same p");

    auto legEta = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
    legEta->AddEntry(mEffEtaGoodAllPt[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
    legEta->AddEntry(mEffEtaFakeAllPt[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
    legEta->Draw("same");
    effEtaAllPt[l]->Write();

    /// eta and phi in different pt ranges
    for (int ipt = 0; ipt < 3; ipt++) {
      // eta
      effEta[l][ipt] = std::make_unique<TCanvas>(Form("effEta_L%d_pt%d", l, ipt));

      mEffEtaGood[l][ipt] = std::make_unique<TEfficiency>(*mNGoodMatchesEta[l][ipt], *mDuplicatedEta[l][ipt]);
      stileEfficiencyGraph(mEffEtaGood[l][ipt], Form("mEffEtaGood_L%d_pt%d", l, ipt), Form("L%d     %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#eta;Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

      for (int ibin = 1; ibin <= mNFakeMatchesEta[l][ipt]->GetNbinsX(); ibin++) {
        if (mNFakeMatchesEta[l][ipt]->GetBinContent(ibin) > mDuplicatedEta[l][ipt]->GetBinContent(ibin)) {
          if (mVerboseOutput) {
            std::cout << "--- Eta : Npass = " << mNFakeMatchesEta[l][ipt]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedEta[l][ipt]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
          }
          mNFakeMatchesEta[l][ipt]->SetBinContent(ibin, mDuplicatedEta[l][ipt]->GetBinContent(ibin));
        }
      }

      mEffEtaFake[l][ipt] = std::make_unique<TEfficiency>(*mNFakeMatchesEta[l][ipt], *mDuplicatedEta[l][ipt]);
      stileEfficiencyGraph(mEffEtaFake[l][ipt], Form("mEffEtaFake_L%d_pt%d", l, ipt), Form("L%d    %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#eta;Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]), false, kFullDiamond, 1, kRed + 1, kRed + 1);

      axeta[ipt]->SetTitle(Form("L%d     %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#eta;Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]));
      axeta[ipt]->GetYaxis()->SetRangeUser(-0.1, 1.1);

      axeta[ipt]->Draw();
      mEffEtaGood[l][ipt]->Draw("same p");
      mEffEtaFake[l][ipt]->Draw("same p");

      auto legEta = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
      legEta->AddEntry(mEffEtaGood[l][ipt].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
      legEta->AddEntry(mEffEtaFake[l][ipt].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
      legEta->Draw("same");
      effEta[l][ipt]->Write();

      // phi
      effPhi[l][ipt] = std::make_unique<TCanvas>(Form("effPhi_L%d_pt%d", l, ipt));

      for (int ibin = 1; ibin <= mNGoodMatchesPhi[l][ipt]->GetNbinsX(); ibin++) {
        if (mNGoodMatchesPhi[l][ipt]->GetBinContent(ibin) > mDuplicatedPhi[l][ipt]->GetBinContent(ibin)) {
          if (mVerboseOutput) {
            std::cout << "--- Phi L: Npass = " << mNGoodMatchesPhi[l][ipt]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhi[l][ipt]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
          }
          mNGoodMatchesPhi[l][ipt]->SetBinContent(ibin, 0);
        }
      }

      mEffPhiGood[l][ipt] = std::make_unique<TEfficiency>(*mNGoodMatchesPhi[l][ipt], *mDuplicatedPhi[l][ipt]);
      stileEfficiencyGraph(mEffPhiGood[l][ipt], Form("mEffPhiGood_L%d_pt%d", l, ipt), Form("L%d     %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#phi (rad);Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

      for (int ibin = 1; ibin <= mNFakeMatchesPhi[l][ipt]->GetNbinsX(); ibin++) {
        if (mNFakeMatchesPhi[l][ipt]->GetBinContent(ibin) > mDuplicatedPhi[l][ipt]->GetBinContent(ibin)) {
          if (mVerboseOutput) {
            std::cout << "--- Phi L: Npass = " << mNFakeMatchesPhi[l][ipt]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhi[l][ipt]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
          }
          mNFakeMatchesPhi[l][ipt]->SetBinContent(ibin, mDuplicatedPhi[l][ipt]->GetBinContent(ibin));
        }
      }

      mEffPhiFake[l][ipt] = std::make_unique<TEfficiency>(*mNFakeMatchesPhi[l][ipt], *mDuplicatedPhi[l][ipt]);
      stileEfficiencyGraph(mEffPhiFake[l][ipt], Form("mEffPhiFake_L%d_pt%d", l, ipt), Form("L%d    %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#phi (rad);Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]), false, kFullDiamond, 1, kRed + 1, kRed + 1);

      axphi[ipt]->SetTitle(Form("L%d     %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#phi (rad);Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]));
      axphi[ipt]->GetYaxis()->SetRangeUser(-0.1, 1.1);

      axphi[ipt]->Draw();
      mEffPhiGood[l][ipt]->Draw("same p");
      mEffPhiFake[l][ipt]->Draw("same p");

      auto legPhi = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
      legPhi->AddEntry(mEffPhiGood[l][ipt].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
      legPhi->AddEntry(mEffPhiFake[l][ipt].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
      legPhi->Draw("same");
      effPhi[l][ipt]->Write();
    }

    // PhiAllPt
    if (mVerboseOutput) {
      std::cout << "Phi L" << l << "\n\n";
    }

    effPhiAllPt[l] = std::make_unique<TCanvas>(Form("effPhiAllPt_L%d", l));

    for (int ibin = 1; ibin <= mNGoodMatchesPhiAllPt[l]->GetNbinsX(); ibin++) {
      if (mNGoodMatchesPhiAllPt[l]->GetBinContent(ibin) > mDuplicatedPhiAllPt[l]->GetBinContent(ibin)) {
        if (mVerboseOutput) {
          std::cout << "--- phi all good Npass = " << mNGoodMatchesPhiAllPt[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhiAllPt[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        }
        mNGoodMatchesPhiAllPt[l]->SetBinContent(ibin, 0);
      }
    }

    mEffPhiGoodAllPt[l] = std::make_unique<TEfficiency>(*mNGoodMatchesPhiAllPt[l], *mDuplicatedPhiAllPt[l]);
    stileEfficiencyGraph(mEffPhiGoodAllPt[l], Form("mEffPhiGoodAllPt_L%d", l), Form("L%d;#phi;Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesPhiAllPt[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesPhiAllPt[l]->GetBinContent(ibin) > mDuplicatedPhiAllPt[l]->GetBinContent(ibin)) {
        if (mVerboseOutput) {
          std::cout << "--- phi all fake Npass = " << mNFakeMatchesPhiAllPt[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhiAllPt[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        }
        mNFakeMatchesPhiAllPt[l]->SetBinContent(ibin, mDuplicatedPhiAllPt[l]->GetBinContent(ibin));
      }
    }
    mEffPhiFakeAllPt[l] = std::make_unique<TEfficiency>(*mNFakeMatchesPhiAllPt[l], *mDuplicatedPhiAllPt[l]);
    stileEfficiencyGraph(mEffPhiFakeAllPt[l], Form("mEffPhiFakeAllPt_L%d", l), Form("L%d;#phi;Efficiency", l), false, kFullDiamond, 1, kRed + 1, kRed + 1);

    axphiAllPt->SetTitle(Form("L%d;#phi;Efficiency", l));
    axphiAllPt->GetYaxis()->SetRangeUser(-0.1, 1.1);
    axphiAllPt->Draw();
    mEffPhiGoodAllPt[l]->Draw("same p");
    mEffPhiFakeAllPt[l]->Draw("same p");

    auto legPhi = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
    legPhi->AddEntry(mEffPhiGoodAllPt[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
    legPhi->AddEntry(mEffPhiFakeAllPt[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
    legPhi->Draw("same");
    effPhiAllPt[l]->Write();
  }

  /// all Row
  std::unique_ptr<TCanvas> effRowAll = std::make_unique<TCanvas>("effRowAll");
  auto numRowGoodAll = std::unique_ptr<TH1D>((TH1D*)mNGoodMatchesRow[0]->Clone("numRowGoodAll"));
  numRowGoodAll->Add(mNGoodMatchesRow[1].get());
  numRowGoodAll->Add(mNGoodMatchesRow[2].get());
  numRowGoodAll->Write();
  auto numRowFakeAll = std::unique_ptr<TH1D>((TH1D*)mNFakeMatchesRow[0]->Clone("numRowFakeAll"));
  numRowFakeAll->Add(mNFakeMatchesRow[1].get());
  numRowFakeAll->Add(mNFakeMatchesRow[2].get());
  numRowFakeAll->Write();
  auto denRowAll = std::unique_ptr<TH1D>((TH1D*)mDuplicatedRow[0]->Clone("denRowAll"));
  denRowAll->Add(mDuplicatedRow[1].get());
  denRowAll->Add(mDuplicatedRow[2].get());
  denRowAll->Write();

  std::unique_ptr<TEfficiency> mEffRowGoodAll = std::make_unique<TEfficiency>(*numRowGoodAll, *denRowAll);
  stileEfficiencyGraph(mEffRowGoodAll, "mEffRowGoodAll", "L0 + L1 + L2;Row;Efficiency", false, kFullDiamond, 1, kGreen + 3, kGreen + 3);
  std::unique_ptr<TEfficiency> mEffRowFakeAll = std::make_unique<TEfficiency>(*numRowFakeAll, *denRowAll);
  stileEfficiencyGraph(mEffRowFakeAll, "mEffRowFakeAll", "L0 + L1 + L2;Row;Efficiency", false, kFullDiamond, 1, kRed + 1, kRed + 1);
  axRow->SetTitle("L0 + L1 + L2;Row;Efficiency");
  axRow->GetYaxis()->SetRangeUser(-0.1, 1.1);
  axRow->Draw();
  mEffRowGoodAll->Draw("same p");
  mEffRowFakeAll->Draw("same p");

  auto legRow = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
  legRow->AddEntry(mEffRowGoodAll.get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
  legRow->AddEntry(mEffRowFakeAll.get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
  legRow->Draw("same");
  effRowAll->Write();

  /// all Col
  std::unique_ptr<TCanvas> effColAll = std::make_unique<TCanvas>("effColAll");
  auto numColGoodAll = std::unique_ptr<TH1D>((TH1D*)mNGoodMatchesCol[0]->Clone("numColGoodAll"));
  numColGoodAll->Add(mNGoodMatchesCol[1].get());
  numColGoodAll->Add(mNGoodMatchesCol[2].get());
  numColGoodAll->Write();
  auto numColFakeAll = std::unique_ptr<TH1D>((TH1D*)mNFakeMatchesCol[0]->Clone("numColFakeAll"));
  numColFakeAll->Add(mNFakeMatchesCol[1].get());
  numColFakeAll->Add(mNFakeMatchesCol[2].get());
  numColFakeAll->Write();
  auto denColAll = std::unique_ptr<TH1D>((TH1D*)mDuplicatedCol[0]->Clone("denColAll"));
  denColAll->Add(mDuplicatedCol[1].get());
  denColAll->Add(mDuplicatedCol[2].get());
  denColAll->Write();

  std::unique_ptr<TEfficiency> mEffColGoodAll = std::make_unique<TEfficiency>(*numColGoodAll, *denColAll);
  stileEfficiencyGraph(mEffColGoodAll, "mEffColGoodAll", "L0 + L1 + L2;Column;Efficiency", false, kFullDiamond, 1, kGreen + 3, kGreen + 3);
  std::unique_ptr<TEfficiency> mEffColFakeAll = std::make_unique<TEfficiency>(*numColFakeAll, *denColAll);
  stileEfficiencyGraph(mEffColFakeAll, "mEffColFakeAll", "L0 + L1 + L2;Column;Efficiency", false, kFullDiamond, 1, kRed + 1, kRed + 1);
  axCol->SetTitle("L0 + L1 + L2;Col;Efficiency");
  axCol->GetYaxis()->SetRangeUser(-0.1, 1.1);
  axCol->Draw();
  mEffColGoodAll->Draw("same p");
  mEffColFakeAll->Draw("same p");

  auto legCol = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
  legCol->AddEntry(mEffColGoodAll.get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
  legCol->AddEntry(mEffColFakeAll.get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
  legCol->Draw("same");
  effColAll->Write();

  /// all Z
  std::unique_ptr<TCanvas> effZAll = std::make_unique<TCanvas>("effZAll");
  auto numZGoodAll = std::unique_ptr<TH1D>((TH1D*)mNGoodMatchesZ[0]->Clone("numZGoodAll"));
  numZGoodAll->Add(mNGoodMatchesZ[1].get());
  numZGoodAll->Add(mNGoodMatchesZ[2].get());
  numZGoodAll->Write();
  auto numZFakeAll = std::unique_ptr<TH1D>((TH1D*)mNFakeMatchesZ[0]->Clone("numZFakeAll"));
  numZFakeAll->Add(mNFakeMatchesZ[1].get());
  numZFakeAll->Add(mNFakeMatchesZ[2].get());
  numZFakeAll->Write();
  auto denZAll = std::unique_ptr<TH1D>((TH1D*)mDuplicatedZ[0]->Clone("denZAll"));
  denZAll->Add(mDuplicatedZ[1].get());
  denZAll->Add(mDuplicatedZ[2].get());
  denZAll->Write();

  std::unique_ptr<TEfficiency> mEffZGoodAll = std::make_unique<TEfficiency>(*numZGoodAll, *denZAll);
  stileEfficiencyGraph(mEffZGoodAll, "mEffZGoodAll", "L0 + L1 + L2;Z;Efficiency", false, kFullDiamond, 1, kGreen + 3, kGreen + 3);
  std::unique_ptr<TEfficiency> mEffZFakeAll = std::make_unique<TEfficiency>(*numZFakeAll, *denZAll);
  stileEfficiencyGraph(mEffZFakeAll, "mEffZFakeAll", "L0 + L1 + L2;Z;Efficiency", false, kFullDiamond, 1, kRed + 1, kRed + 1);
  axZ->SetTitle("L0 + L1 + L2;Z;Efficiency");
  axZ->GetYaxis()->SetRangeUser(-0.1, 1.1);
  axZ->Draw();
  mEffZGoodAll->Draw("same p");
  mEffZFakeAll->Draw("same p");

  auto legZ = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
  legZ->AddEntry(mEffZGoodAll.get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
  legZ->AddEntry(mEffZFakeAll.get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
  legZ->Draw("same");
  effZAll->Write();

  /// all Eta
  std::unique_ptr<TCanvas> effEtaAll = std::make_unique<TCanvas>("effEtaAll");
  auto numEtaGoodAll = std::unique_ptr<TH1D>((TH1D*)mNGoodMatchesEtaAllPt[0]->Clone("numEtaGoodAll"));
  numEtaGoodAll->Add(mNGoodMatchesEtaAllPt[1].get());
  numEtaGoodAll->Add(mNGoodMatchesEtaAllPt[2].get());
  numEtaGoodAll->Write();
  auto numEtaFakeAll = std::unique_ptr<TH1D>((TH1D*)mNFakeMatchesEtaAllPt[0]->Clone("numEtaFakeAll"));
  numEtaFakeAll->Add(mNFakeMatchesEtaAllPt[1].get());
  numEtaFakeAll->Add(mNFakeMatchesEtaAllPt[2].get());
  numEtaFakeAll->Write();
  auto denEtaAll = std::unique_ptr<TH1D>((TH1D*)mDuplicatedEtaAllPt[0]->Clone("denEtaAll"));
  denEtaAll->Add(mDuplicatedEtaAllPt[1].get());
  denEtaAll->Add(mDuplicatedEtaAllPt[2].get());
  denEtaAll->Write();

  std::unique_ptr<TEfficiency> mEffEtaGoodAll = std::make_unique<TEfficiency>(*numEtaGoodAll, *denEtaAll);
  stileEfficiencyGraph(mEffEtaGoodAll, "mEffEtaGoodAll", "L0 + L1 + L2;#Eta;Efficiency", false, kFullDiamond, 1, kGreen + 3, kGreen + 3);
  std::unique_ptr<TEfficiency> mEffEtaFakeAll = std::make_unique<TEfficiency>(*numEtaFakeAll, *denEtaAll);
  stileEfficiencyGraph(mEffEtaFakeAll, "mEffEtaFakeAll", "L0 + L1 + L2;#Eta;Efficiency", false, kFullDiamond, 1, kRed + 1, kRed + 1);
  axetaAllPt->SetTitle("L0 + L1 + L2;Eta;Efficiency");
  axetaAllPt->GetYaxis()->SetRangeUser(-0.1, 1.1);
  axetaAllPt->Draw();
  mEffEtaGoodAll->Draw("same p");
  mEffEtaFakeAll->Draw("same p");

  auto legEta = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
  legEta->AddEntry(mEffEtaGoodAll.get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
  legEta->AddEntry(mEffEtaFakeAll.get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
  legEta->Draw("same");
  effEtaAll->Write();

  /// all Phi
  std::unique_ptr<TCanvas> effPhiAll = std::make_unique<TCanvas>("effPhiAll");
  auto numPhiGoodAll = std::unique_ptr<TH1D>((TH1D*)mNGoodMatchesPhiAllPt[0]->Clone("numPhiGoodAll"));
  numPhiGoodAll->Add(mNGoodMatchesPhiAllPt[1].get());
  numPhiGoodAll->Add(mNGoodMatchesPhiAllPt[2].get());
  numPhiGoodAll->Write();
  auto numPhiFakeAll = std::unique_ptr<TH1D>((TH1D*)mNFakeMatchesPhiAllPt[0]->Clone("numPhiFakeAll"));
  numPhiFakeAll->Add(mNFakeMatchesPhiAllPt[1].get());
  numPhiFakeAll->Add(mNFakeMatchesPhiAllPt[2].get());
  numPhiFakeAll->Write();
  auto denPhiAll = std::unique_ptr<TH1D>((TH1D*)mDuplicatedPhiAllPt[0]->Clone("denPhiAll"));
  denPhiAll->Add(mDuplicatedPhiAllPt[1].get());
  denPhiAll->Add(mDuplicatedPhiAllPt[2].get());
  denPhiAll->Write();

  std::unique_ptr<TEfficiency> mEffPhiGoodAll = std::make_unique<TEfficiency>(*numPhiGoodAll, *denPhiAll);
  stileEfficiencyGraph(mEffPhiGoodAll, "mEffPhiGoodAll", "L0 + L1 + L2;#Phi (rad);Efficiency", false, kFullDiamond, 1, kGreen + 3, kGreen + 3);
  std::unique_ptr<TEfficiency> mEffPhiFakeAll = std::make_unique<TEfficiency>(*numPhiFakeAll, *denPhiAll);
  stileEfficiencyGraph(mEffPhiFakeAll, "mEffPhiFakeAll", "L0 + L1 + L2;#Phi (rad);Efficiency", false, kFullDiamond, 1, kRed + 1, kRed + 1);
  axphiAllPt->SetTitle("L0 + L1 + L2;Phi;Efficiency");
  axphiAllPt->GetYaxis()->SetRangeUser(-0.1, 1.1);
  axphiAllPt->Draw();
  mEffPhiGoodAll->Draw("same p");
  mEffPhiFakeAll->Draw("same p");

  auto legPhi = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
  legPhi->AddEntry(mEffPhiGoodAll.get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
  legPhi->AddEntry(mEffPhiFakeAll.get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
  legPhi->Draw("same");
  effPhiAll->Write();
}

void EfficiencyStudy::getEfficiency(bool isMC)
{
  // Extract the efficiency for the IB, exploiting the staves overlaps and the duplicated clusters for the tracks passing through the overlaps
  // The denominator for the efficiency calculation will be the number of tracks per layer fulfilling some cuts (eta, z, row, col)
  // The numerator will be the number of duplicated clusters for the tracks passing through the overlaps

  LOGP(info, "getEfficiency()");

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
  auto propagator = o2::base::Propagator::Instance();

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;

  int nbPt = 75;
  double xbins[nbPt + 1], ptcutl = 0.05, ptcuth = 7.5;
  double a = std::log(ptcuth / ptcutl) / nbPt;
  for (int i = 0; i <= nbPt; i++) {
    xbins[i] = ptcutl * std::exp(i * a);
  }

  int totNClusters;
  int nDuplClusters;

  for (unsigned int iROF = 0; iROF < mTracksROFRecords.size(); iROF++) { // loop on ROFRecords array

    rofIndexTrack = mTracksROFRecords[iROF].getFirstEntry();
    rofNEntriesTrack = mTracksROFRecords[iROF].getNEntries();

    rofIndexClus = mClustersROFRecords[iROF].getFirstEntry();
    rofNEntriesClus = mClustersROFRecords[iROF].getNEntries();

    ////// imposing cuts on the tracks = collecting tracks for the denominator
    for (unsigned int iTrack = rofIndexTrack; iTrack < rofIndexTrack + rofNEntriesTrack; iTrack++) { // loop on tracks per ROF
      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];

      auto pt = trackParCov.getPt(); // Always 0.6 GeV/c for B = 0 T
      auto eta = trackParCov.getEta();
      float phi = -999.;
      float phiOriginal = -999.;

      float chi2 = track.getChi2();

      float ip[2];
      track.getImpactParams(0, 0, 0, 0, ip);

      // float phiTrack = trackParCov.getPhi(); // * 180 / M_PI;

      // applying the cuts on the track - only eta
      if (eta < mEtaCuts[0] || eta >= mEtaCuts[1]) {
        continue;
      }

      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      //// keeping only 7 clusters track to reduce fakes
      if (ncl < 7) {
        continue;
      }

      o2::MCCompLabel tracklab;
      if (isMC) {
        tracklab = mTracksMCLabels[iTrack];
        if (tracklab.isFake()) {
          continue;
        }
      }

      if (mVerboseOutput && isMC) {
        LOGP(info, "track Label: ");
        tracklab.print();
      }

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track to extract layer, stave and chip to restrict the possible matches to be searched with the DCA cut
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]];
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());

        UShort_t rowOriginal = clusOriginal.getRow();
        UShort_t colOriginal = clusOriginal.getCol();

        /// filling some chip maps
        if (clusOriginal.getChipID() >= 0 && clusOriginal.getChipID() <= 8) {
          l0_00->Fill(clusOriginal.getCol() + (1024 * (clusOriginal.getChipID() % 9)), clusOriginal.getRow());
        }
        if (clusOriginal.getChipID() >= 252 && clusOriginal.getChipID() <= 260) {
          l1_15->Fill(clusOriginal.getCol() + (1024 * (clusOriginal.getChipID() % 9)), clusOriginal.getRow());
        }
        if (clusOriginal.getChipID() >= 423 && clusOriginal.getChipID() <= 431) {
          l2_19->Fill(clusOriginal.getCol() + (1024 * (clusOriginal.getChipID() % 9)), clusOriginal.getRow());
        }

        //// only IB
        if (layerOriginal >= NLAYERS) {
          continue;
        }

        chipmap->Fill(clusOriginal.getCol(), clusOriginal.getRow());

        IPOriginalxy[layerOriginal]->Fill(ip[0]);
        IPOriginalz[layerOriginal]->Fill(ip[1]);

        ///// cluster point and conversion from track local coordinates to global coordinates
        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;
        phiOriginal = clusOriginalPointGlob.phi(); // * 180 / M_PI;

        if (abs(clusOriginalPointGlob.y()) < 0.5) { ///// excluding gap between bottom and top barrels
          continue;
        }

        if (abs(clusOriginalPointGlob.z()) >= 10) { /// excluding external z
          continue;
        }

        if (rowOriginal < 2 || (rowOriginal > 15 && rowOriginal < 496) || rowOriginal > 509) { ////  cutting on the row
          continue;
        }

        if (mUseMC) { //// excluding known bad chips in MC which are not bad in data --- to be checked based on the anchored run
          if (std::find(mExcludedChipMC.begin(), mExcludedChipMC.end(), clusOriginal.getChipID()) != mExcludedChipMC.end()) {
            continue;
          }
        }

        if (clusOriginal.getCol() < 160 || clusOriginal.getCol() > 870) { /// excluding the gap between two chips in the same stave (comment to obtain the plot efficiency col vs eta)
          continue;
        }

        /// if the track passes the cuts, fill the den and go ahead
        m2DClusterOriginalPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y());
        m3DClusterPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y(), clusOriginalPointGlob.z());
        chi2trackAccepted->Fill(chi2);
        denPt[layerOriginal]->Fill(pt);
        denPhi[layerOriginal]->Fill(phiOriginal);
        denEta[layerOriginal]->Fill(eta);
        denRow[layerOriginal]->Fill(rowOriginal);
        denCol[layerOriginal]->Fill(clusOriginal.getCol());
        denZ[layerOriginal]->Fill(clusOriginalPointGlob.z());
        nTracksSelected[layerOriginal]++;
        mDenColEta[layerOriginal]->Fill(clusOriginal.getCol(), eta);
        mDenRowPhi[layerOriginal]->Fill(clusOriginal.getRow(), clusOriginalPointGlob.z());
        mDenRowCol[layerOriginal]->Fill(clusOriginal.getRow(), clusOriginal.getCol());
        denLayers->Fill(layerOriginal);

        /// if the cuts up to here are passed, then search for the duplicated cluster, otherwise go to the next cluster
        gsl::span<const o2::MCCompLabel> labsOriginal = {};
        if (isMC) {
          labsOriginal = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track (original)
        }

        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        std::tuple<int, double, gsl::span<const o2::MCCompLabel>> clusID_rDCA_label = {0, 999., gsl::span<const o2::MCCompLabel>()}; // inizializing tuple with dummy values (if data, ignore the third value)

        bool adjacentFound = 0;
        float phiDuplicated = -999.;
        float ptDuplicated = -999.;
        float etaDuplicated = -999.;
        float clusZ = -999.;

        o2::itsmft::CompClusterExt clusDuplicatedSelected = o2::itsmft::CompClusterExt();

        /// for each original cluster iterate over all the possible duplicated clusters to select the "adjacent" clusters (stave +-1, chip =,+-1) and calculate the DCA with the track. Then choose the closest one.
        for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF
          auto clusDuplicated = mClusters[iClus];
          auto clusDuplicatedPoint = mITSClustersArray[iClus];

          o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
          o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
          phi = clusDuplicatedPointGlob.phi(); // * 180 / M_PI;

          //// applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
          if (clusDuplicated.getSensorID() == clusOriginal.getSensorID()) {
            continue;
          }
          auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
          if (layerDuplicated != layerOriginal) {
            continue;
          }
          auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
          if (abs(staveDuplicated - staveOriginal) != 1) {
            continue;
          }
          auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
          if (abs(chipDuplicated - chipOriginal) > 1) {
            continue;
          }

          gsl::span<const o2::MCCompLabel> labsDuplicated = {};
          if (isMC) {
            labsDuplicated = mClustersMCLCont->getLabels(iClus);
          }

          /// if the cheks are passed, then calculate the DCA
          /// Compute the DCA between the duplicated cluster location and the track
          trackParCov.rotate(mGeometry->getSensorRefAlpha(clusDuplicated.getSensorID()));
          if (!propagator->propagateToDCA(clusDuplicatedPointGlob, trackParCov, b, 2.f, matCorr, &clusDuplicatedDCA)) { // check if the propagation fails
            continue;
          }

          DCAxyData[layerDuplicated]->Fill(clusDuplicatedDCA[0]);
          DCAzData[layerDuplicated]->Fill(clusDuplicatedDCA[1]);

          // Imposing that the distance between the duplicated cluster and the track is less than x sigma
          if (!(clusDuplicatedDCA[0] > mDCACutsXY[layerDuplicated][0] && clusDuplicatedDCA[0] < mDCACutsXY[layerDuplicated][1] && clusDuplicatedDCA[1] > mDCACutsZ[layerDuplicated][0] && clusDuplicatedDCA[1] < mDCACutsZ[layerDuplicated][1])) {
            DCAxyRejected[layerDuplicated]->Fill(clusDuplicatedDCA[0]);
            DCAzRejected[layerDuplicated]->Fill(clusDuplicatedDCA[1]);
            continue;
          }

          m2DClusterDuplicatedPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y());
          m3DDuplicatedClusterPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y(), clusDuplicatedPointGlob.z());

          if (mVerboseOutput) {
            LOGP(info, "Propagation ok");
          }
          double rDCA = std::hypot(clusDuplicatedDCA[0], clusDuplicatedDCA[1]);

          // taking the closest cluster within x sigma
          if (rDCA < std::get<1>(clusID_rDCA_label)) { // updating the closest cluster
            if (isMC) {
              clusID_rDCA_label = {iClus, rDCA, labsDuplicated};
            } else {
              clusID_rDCA_label = {iClus, rDCA, gsl::span<const o2::MCCompLabel>()};
            }
            phiDuplicated = phiOriginal;
            ptDuplicated = pt;
            etaDuplicated = eta;
            clusZ = clusOriginalPointGlob.z();
            clusDuplicatedSelected = clusDuplicated;
          }
          adjacentFound = 1;
        } // end loop on all the clusters in the rof -> at this point we have the information on the closest cluster (if there is one)

        // here clusID_rDCA_label is updated with the closest cluster to the track other than the original one

        if (!adjacentFound) {
          radiusNotFound[layerOriginal]->Fill(sqrt(clusOriginalPointGlob.x() * clusOriginalPointGlob.x() + clusOriginalPointGlob.y() * clusOriginalPointGlob.y()));
          colNotFound[layerOriginal]->Fill(clusOriginal.getCol() + (1024 * (clusOriginal.getChipID() % 9)));
          rowNotFound[layerOriginal]->Fill(rowOriginal);
          zNotFound[layerOriginal]->Fill(clusOriginalPointGlob.z());
          phiNotFound[layerOriginal]->Fill(phiOriginal);
          continue;
        }

        chipOrigVsOverlap->Fill(clusOriginal.getChipID() % 9, clusDuplicatedSelected.getChipID() % 9);
        mChipFound->Fill(clusOriginal.getChipID());
        zFound[layerOriginal]->Fill(clusOriginalPointGlob.z());
        radiusFound[layerOriginal]->Fill(sqrt(clusOriginalPointGlob.x() * clusOriginalPointGlob.x() + clusOriginalPointGlob.y() * clusOriginalPointGlob.y()));
        colFoundOriginalVsDuplicated[layerOriginal]->Fill(clusOriginal.getCol() + (1024 * (clusOriginal.getChipID() % 9)), clusDuplicatedSelected.getCol() + (1024 * (clusDuplicatedSelected.getChipID() % 9)));
        colFoundOriginal[layerOriginal]->Fill(clusOriginal.getCol() + (1024 * (clusOriginal.getChipID() % 9)));
        m2DClusterFoundPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y());
        phiFound[layerOriginal]->Fill(phiOriginal);
        rowFound[layerOriginal]->Fill(rowOriginal);
        nDuplClusters++;
        nDuplicatedClusters[layerOriginal]++;
        numPt[layerOriginal]->Fill(pt);
        numPhi[layerOriginal]->Fill(phiDuplicated);
        numEta[layerOriginal]->Fill(etaDuplicated);
        numRow[layerOriginal]->Fill(rowOriginal);
        numCol[layerOriginal]->Fill(clusOriginal.getCol());
        numZ[layerOriginal]->Fill(clusOriginalPointGlob.z());
        mZvsPhiDUplicated[layerOriginal]->Fill(clusZ, phiDuplicated);
        mNumColEta[layerOriginal]->Fill(clusOriginal.getCol(), eta);
        mNumRowPhi[layerOriginal]->Fill(clusOriginal.getRow(), clusOriginalPointGlob.z());
        mNumRowCol[layerOriginal]->Fill(clusOriginal.getRow(), clusOriginal.getCol());
        numLayers->Fill(layerOriginal);

        // checking if it is a good or fake match looking at the labels (only if isMC)
        if (isMC) {
          bool isGood = false;
          for (auto lab : std::get<2>(clusID_rDCA_label)) {
            if (lab == tracklab) {
              isGood = true;
              numPtGood[layerOriginal]->Fill(ptDuplicated);
              numPhiGood[layerOriginal]->Fill(phiDuplicated);
              numEtaGood[layerOriginal]->Fill(etaDuplicated);
              numRowGood[layerOriginal]->Fill(rowOriginal);
              numColGood[layerOriginal]->Fill(clusOriginal.getCol());
              numZGood[layerOriginal]->Fill(clusOriginalPointGlob.z());
              numGoodLayers->Fill(layerOriginal);
              continue;
            }
          }
          if (!isGood) {
            numPtFake[layerOriginal]->Fill(ptDuplicated);
            numPhiFake[layerOriginal]->Fill(phiDuplicated);
            numEtaFake[layerOriginal]->Fill(etaDuplicated);
            numRowFake[layerOriginal]->Fill(rowOriginal);
            numColFake[layerOriginal]->Fill(clusOriginal.getCol());
            numZFake[layerOriginal]->Fill(clusOriginalPointGlob.z());
            numFakeLayers->Fill(layerOriginal);
          }
        }
      } // end loop on clusters associated to the track
      totNClusters += NLAYERS;
    } // end loop on tracks per ROF
  } // end loop on ROFRecords array

  std::cout << " Num of duplicated clusters L0: " << nDuplicatedClusters[0] << " N tracks selected: " << nTracksSelected[0] << std::endl;
  std::cout << " Num of duplicated clusters L1: " << nDuplicatedClusters[1] << " N tracks selected: " << nTracksSelected[1] << std::endl;
  std::cout << " Num of duplicated clusters L2: " << nDuplicatedClusters[2] << " N tracks selected: " << nTracksSelected[2] << std::endl;

  std::cout << " --------- N total clusters: " << totNClusters << std::endl;
  std::cout << " --------- N duplicated clusters: " << nDuplClusters << std::endl;
}

void EfficiencyStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  LOGP(info, "--------------- process");

  o2::base::GRPGeomHelper::instance().getGRPMagField()->print();

  if (mUseMC) {
    // getDCAClusterTrackMC();
    studyDCAcutsMC();
    // studyClusterSelectionMC();
    // countDuplicatedAfterCuts();
    getEfficiency(mUseMC);
  } else {
    getEfficiency(mUseMC);
  }

  LOGP(info, "** Found in {} rofs:\n\t- {} clusters\n\t",
       mClustersROFRecords.size(), mClusters.size());

  if (mUseMC) {
    LOGP(info, "mClusters size: {}, mClustersROFRecords size: {}, mClustersMCLCont size: {}, mClustersconverted size: {} ", mClusters.size(), mClustersROFRecords.size(), mClustersMCLCont->getNElements(), mITSClustersArray.size());
    LOGP(info, "mTracks size: {}, mTracksROFRecords size: {}, mTracksMCLabels size: {}", mTracks.size(), mTracksROFRecords.size(), mTracksMCLabels.size());
  } else {
    LOGP(info, "mClusters size: {}, mClustersROFRecords size: {}, mClustersconverted size: {} ", mClusters.size(), mClustersROFRecords.size(), mITSClustersArray.size());
    LOGP(info, "mTracks size: {}, mTracksROFRecords size: {}", mTracks.size(), mTracksROFRecords.size());
  }
}

void EfficiencyStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    mGeometry = GeometryTGeo::Instance();
    mGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G, o2::math_utils::TransformType::L2G));
  }
}

void EfficiencyStudy::endOfStream(EndOfStreamContext& ec)
{
  LOGP(info, "--------------- endOfStream");

  mOutFile->mkdir("EfficiencyFinal/");
  mOutFile->mkdir("DCAFinal/");
  mOutFile->mkdir("NotFoundChecks/");

  mOutFile->mkdir("DCA/");
  mOutFile->mkdir("Pt_Eta_Phi/");

  if (mUseMC) {

    mOutFile->cd("DCA");
    mDCAxyDuplicated->Write();
    mDCAzDuplicated->Write();
    for (int i = 0; i < NLAYERS; i++) {
      mDCAxyDuplicated_layer[i]->Write();
      mDCAzDuplicated_layer[i]->Write();

      mDCAxyOriginal[i]->Write();
      mDCAzOriginal[i]->Write();
    }

    mOutFile->cd("Pt_Eta_Phi/");
    for (int i = 0; i < NLAYERS; i++) {
      mDuplicatedPhiAllPt[i]->Write();
      mPtDuplicated[i]->Write();
      mEtaDuplicated[i]->Write();
      mPhiDuplicated[i]->Write();
      mPhiOriginalIfDuplicated[i]->Write();
      mDuplicatedPt[i]->Write();
      mDuplicatedPtEta[i]->Write();
      mDuplicatedPtPhi[i]->Write();
      mDuplicatedEtaPhi[i]->Write();
      mDuplicatedEtaAllPt[i]->Write();
      mDuplicatedRow[i]->Write();
      mDuplicatedCol[i]->Write();
      mDuplicatedZ[i]->Write();

      for (int p = 0; p < 3; p++) {
        mDuplicatedEta[i][p]->Write();
        mDuplicatedPhi[i][p]->Write();
      }
      mPt_EtaDupl[i]->Write();
    }
  }

  mOutFile->cd("Pt_Eta_Phi/");
  for (int i = 0; i < NLAYERS; i++) {
    IPOriginalxy[i]->Write();
    IPOriginalz[i]->Write();
    mPhiOriginal[i]->Write();
    mPtOriginal[i]->Write();
    mEtaOriginal[i]->Write();
    mZvsPhiDUplicated[i]->Write();
    chipRowDuplicated[i]->Write();
    chipRowOriginalIfDuplicated[i]->Write();
  }

  mOutFile->mkdir("chi2");
  mOutFile->cd("chi2/");

  chi2trackAccepted->Write();

  mOutFile->cd("EfficiencyFinal/");
  TList listNum;
  TList listDen;
  auto numPhiAll = std::unique_ptr<TH1D>((TH1D*)numPhi[0]->Clone("numPhiAll"));
  auto denPhiAll = std::unique_ptr<TH1D>((TH1D*)denPhi[0]->Clone("denPhiAll"));

  TList listNumColEta;
  TList listDenColEta;
  auto numColEtaAll = std::unique_ptr<TH1D>((TH1D*)mNumColEta[0]->Clone("numColEtaAll"));
  auto denColEtaAll = std::unique_ptr<TH1D>((TH1D*)mDenColEta[0]->Clone("denColEtaAll"));

  TList listNumRowPhi;
  TList listDenRowPhi;
  auto numRowPhiAll = std::unique_ptr<TH1D>((TH1D*)mNumRowPhi[0]->Clone("numRowPhiAll"));
  auto denRowPhiAll = std::unique_ptr<TH1D>((TH1D*)mDenRowPhi[0]->Clone("denRowPhiAll"));

  TList listNumRowCol;
  TList listDenRowCol;
  auto numRowColAll = std::unique_ptr<TH1D>((TH1D*)mNumRowCol[0]->Clone("numRowColAll"));
  auto denRowColAll = std::unique_ptr<TH1D>((TH1D*)mDenRowCol[0]->Clone("denRowColAll"));

  std::unique_ptr<TEfficiency> effLayers = std::make_unique<TEfficiency>(*numLayers, *denLayers);
  effLayers->SetName("effLayers");
  effLayers->SetTitle("; ;Efficiency");
  std::unique_ptr<TEfficiency> effLayersGood = std::make_unique<TEfficiency>(*numGoodLayers, *denLayers);
  effLayersGood->SetName("effLayersGood");
  effLayersGood->SetTitle("; ;Efficiency Good Matches");
  std::unique_ptr<TEfficiency> effLayersFake = std::make_unique<TEfficiency>(*numFakeLayers, *denLayers);
  effLayersFake->SetName("effLayersFake");
  effLayersFake->SetTitle("; ;Efficiency Fake Matches");
  effLayers->Write();
  effLayersGood->Write();
  effLayersFake->Write();
  denLayers->Write();
  numLayers->Write();
  numGoodLayers->Write();
  numFakeLayers->Write();

  for (int l = 0; l < NLAYERS; l++) {

    std::unique_ptr<TEfficiency> effPt = std::make_unique<TEfficiency>(*numPt[l], *denPt[l]);
    effPt->SetName(Form("effPt_layer%d", l));
    effPt->SetTitle(Form("L%d;p_{T} (GeV/c);Efficiency", l));
    std::unique_ptr<TEfficiency> effPtGood = std::make_unique<TEfficiency>(*numPtGood[l], *denPt[l]);
    effPtGood->SetName(Form("effPtGood_layer%d", l));
    effPtGood->SetTitle(Form("L%d;p_{T} (GeV/c);Efficiency Good Matches", l));
    std::unique_ptr<TEfficiency> effPtFake = std::make_unique<TEfficiency>(*numPtFake[l], *denPt[l]);
    effPtFake->SetName(Form("effPtFake_layer%d", l));
    effPtFake->SetTitle(Form("L%d;p_{T} (GeV/c);Efficiency Fake Matches", l));
    effPt->Write();
    effPtGood->Write();
    effPtFake->Write();

    std::unique_ptr<TEfficiency> effPhi = std::make_unique<TEfficiency>(*numPhi[l], *denPhi[l]);
    effPhi->SetName(Form("effPhi_layer%d", l));
    effPhi->SetTitle(Form("L%d;#phi;Efficiency", l));
    std::unique_ptr<TEfficiency> effPhiGood = std::make_unique<TEfficiency>(*numPhiGood[l], *denPhi[l]);
    effPhiGood->SetName(Form("effPhiGood_layer%d", l));
    effPhiGood->SetTitle(Form("L%d;#phi;Efficiency Good Matches", l));
    std::unique_ptr<TEfficiency> effPhiFake = std::make_unique<TEfficiency>(*numPhiFake[l], *denPhi[l]);
    effPhiFake->SetName(Form("effPhiFake_layer%d", l));
    effPhiFake->SetTitle(Form("L%d;#phi;Efficiency Fake Matches", l));
    effPhi->Write();
    effPhiGood->Write();
    effPhiFake->Write();
    listNum.Add(numPhi[l].get());
    listDen.Add(denPhi[l].get());

    std::unique_ptr<TEfficiency> effEta = std::make_unique<TEfficiency>(*numEta[l], *denEta[l]);
    effEta->SetName(Form("effEta_layer%d", l));
    effEta->SetTitle(Form("L%d;#eta;Efficiency", l));
    std::unique_ptr<TEfficiency> effEtaGood = std::make_unique<TEfficiency>(*numEtaGood[l], *denEta[l]);
    effEtaGood->SetName(Form("effEtaGood_layer%d", l));
    effEtaGood->SetTitle(Form("L%d;#eta;Efficiency Good Matches", l));
    std::unique_ptr<TEfficiency> effEtaFake = std::make_unique<TEfficiency>(*numEtaFake[l], *denEta[l]);
    effEtaFake->SetName(Form("effEtaFake_layer%d", l));
    effEtaFake->SetTitle(Form("L%d;#eta;Efficiency Fake Matches", l));
    effEta->Write();
    effEtaGood->Write();
    effEtaFake->Write();

    std::unique_ptr<TEfficiency> effRow = std::make_unique<TEfficiency>(*numRow[l], *denRow[l]);
    effRow->SetName(Form("effRow_layer%d", l));
    effRow->SetTitle(Form("L%d;#Row;Efficiency", l));
    std::unique_ptr<TEfficiency> effRowGood = std::make_unique<TEfficiency>(*numRowGood[l], *denRow[l]);
    effRowGood->SetName(Form("effRowGood_layer%d", l));
    effRowGood->SetTitle(Form("L%d;#Row;Efficiency Good Matches", l));
    std::unique_ptr<TEfficiency> effRowFake = std::make_unique<TEfficiency>(*numRowFake[l], *denRow[l]);
    effRowFake->SetName(Form("effRowFake_layer%d", l));
    effRowFake->SetTitle(Form("L%d;#Row;Efficiency Fake Matches", l));
    effRow->Write();
    effRowGood->Write();
    effRowFake->Write();

    std::unique_ptr<TEfficiency> effCol = std::make_unique<TEfficiency>(*numCol[l], *denCol[l]);
    effCol->SetName(Form("effCol_layer%d", l));
    effCol->SetTitle(Form("L%d;#Col;Efficiency", l));
    std::unique_ptr<TEfficiency> effColGood = std::make_unique<TEfficiency>(*numColGood[l], *denCol[l]);
    effColGood->SetName(Form("effColGood_layer%d", l));
    effColGood->SetTitle(Form("L%d;#Col;Efficiency Good Matches", l));
    std::unique_ptr<TEfficiency> effColFake = std::make_unique<TEfficiency>(*numColFake[l], *denCol[l]);
    effColFake->SetName(Form("effColFake_layer%d", l));
    effColFake->SetTitle(Form("L%d;#Col;Efficiency Fake Matches", l));
    effCol->Write();
    effColGood->Write();
    effColFake->Write();

    std::unique_ptr<TEfficiency> effZ = std::make_unique<TEfficiency>(*numZ[l], *denZ[l]);
    effZ->SetName(Form("effZ_layer%d", l));
    effZ->SetTitle(Form("L%d;#Z (cm);Efficiency", l));
    std::unique_ptr<TEfficiency> effZGood = std::make_unique<TEfficiency>(*numZGood[l], *denZ[l]);
    effZGood->SetName(Form("effZGood_layer%d", l));
    effZGood->SetTitle(Form("L%d;#Z (cm);Efficiency Good Matches", l));
    std::unique_ptr<TEfficiency> effZFake = std::make_unique<TEfficiency>(*numZFake[l], *denZ[l]);
    effZFake->SetName(Form("effZFake_layer%d", l));
    effZFake->SetTitle(Form("L%d;#Z (cm);Efficiency Fake Matches", l));
    effZ->Write();
    effZGood->Write();
    effZFake->Write();

    std::unique_ptr<TEfficiency> effColEta = std::make_unique<TEfficiency>(*mNumColEta[l], *mDenColEta[l]);
    effColEta->SetName(Form("effColEta_layer%d", l));
    effColEta->SetTitle(Form("L%d;Column;#eta", l));
    effColEta->Write();

    listNumColEta.Add(mNumColEta[l].get());
    listDenColEta.Add(mDenColEta[l].get());

    std::unique_ptr<TEfficiency> effRowPhi = std::make_unique<TEfficiency>(*mNumRowPhi[l], *mDenRowPhi[l]);
    effRowPhi->SetName(Form("effRowPhi_layer%d", l));
    effRowPhi->SetTitle(Form("L%d;Column;#eta", l));
    effRowPhi->Write();

    listNumRowPhi.Add(mNumRowPhi[l].get());
    listDenRowPhi.Add(mDenRowPhi[l].get());

    std::unique_ptr<TEfficiency> effRowCol = std::make_unique<TEfficiency>(*mNumRowCol[l], *mDenRowCol[l]);
    effRowCol->SetName(Form("effRowCol_layer%d", l));
    effRowCol->SetTitle(Form("L%d;Column;#eta", l));
    effRowCol->Write();

    listNumRowCol.Add(mNumRowCol[l].get());
    listDenRowCol.Add(mDenRowCol[l].get());

    mNumRowCol[l]->Write();
    mDenRowCol[l]->Write();
    mNumRowPhi[l]->Write();
    mDenRowPhi[l]->Write();
    mNumColEta[l]->Write();
    mDenColEta[l]->Write();
    numPhi[l]->Write();
    denPhi[l]->Write();
    numPt[l]->Write();
    denPt[l]->Write();
    numEta[l]->Write();
    denEta[l]->Write();
    numRow[l]->Write();
    denRow[l]->Write();
    numCol[l]->Write();
    denCol[l]->Write();
    numZ[l]->Write();
    denZ[l]->Write();
  }
  numPhiAll->Merge(&listNum);
  denPhiAll->Merge(&listDen);

  numColEtaAll->Merge(&listNumColEta);
  denColEtaAll->Merge(&listDenColEta);

  numRowPhiAll->Merge(&listNumRowPhi);
  denRowPhiAll->Merge(&listDenRowPhi);

  numRowColAll->Merge(&listNumRowCol);
  denRowColAll->Merge(&listDenRowCol);

  std::unique_ptr<TEfficiency> effPhiAll = std::make_unique<TEfficiency>(*numPhiAll, *denPhiAll);
  effPhiAll->SetName("effPhi_AllLayers");
  effPhiAll->SetTitle("L0 + L1 + L2;#phi;Efficiency");
  effPhiAll->Write();
  numPhiAll->Write();
  denPhiAll->Write();

  std::unique_ptr<TEfficiency> effColEtaAll = std::make_unique<TEfficiency>(*numColEtaAll, *denColEtaAll);
  effColEtaAll->SetName("effColEta_AllLayers");
  effColEtaAll->SetTitle("L0 + L1 + L2;Column;#eta");
  effColEtaAll->Write();
  numColEtaAll->Write();
  denColEtaAll->Write();

  std::unique_ptr<TEfficiency> effRowPhiAll = std::make_unique<TEfficiency>(*numRowPhiAll, *denRowPhiAll);
  effRowPhiAll->SetName("effRowPhi_AllLayers");
  effRowPhiAll->SetTitle("L0 + L1 + L2;Column;#eta");
  effRowPhiAll->Write();
  numRowPhiAll->Write();
  denRowPhiAll->Write();

  std::unique_ptr<TEfficiency> effRowColAll = std::make_unique<TEfficiency>(*numRowColAll, *denRowColAll);
  effRowColAll->SetName("effRowCol_AllLayers");
  effRowColAll->SetTitle("L0 + L1 + L2;Column;#eta");
  effRowColAll->Write();
  numRowColAll->Write();
  denRowColAll->Write();

  mOutFile->cd("DCAFinal/");

  for (int l = 0; l < NLAYERS; l++) {
    DCAxyData[l]->Write();
    DCAzData[l]->Write();
    DCAxyRejected[l]->Write();
    DCAzRejected[l]->Write();
  }

  mOutFile->cd("NotFoundChecks/");

  for (int l = 0; l < NLAYERS; l++) {
    phiFound[l]->Write();
    phiNotFound[l]->Write();
    rowFound[l]->Write();
    rowNotFound[l]->Write();
    zFound[l]->Write();
    zNotFound[l]->Write();
    radiusFound[l]->Write();
    radiusNotFound[l]->Write();
    colFoundOriginalVsDuplicated[l]->Write();
    colFoundOriginal[l]->Write();
    colNotFound[l]->Write();
  }
  mChipFound->Write();
  mChipNotFound->Write();
  m2DClusterFoundPositions->Write();
  l0_00->Write();
  l1_15->Write();
  l2_19->Write();
  chipOrigVsOverlap->Write();
  chipmap->SetContour(100);
  chipmap->Write();

  mOutFile->Close();
}

void EfficiencyStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  std::cout << "-------- finaliseCCDB" << std::endl;
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

DataProcessorSpec getEfficiencyStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestClusters(srcClustersMask, useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "its-efficiency-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EfficiencyStudy>(dataRequest, srcTracksMask, useMC, kineReader, ggRequest)},
    Options{}};
}

} // namespace o2::its::study
