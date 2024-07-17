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
#include <numeric>

#define NLAYERS 3

namespace o2::its::study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;

class EfficiencyStudy : public Task
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
  void getEfficiencyAndTrackInfo(bool isMC);
  void saveDataInfo();
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
  float mrangesPt[NLAYERS][2] = {{0, 0.5}, {0.5, 2}, {2, 7.5}};

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
  unsigned short mMask = 0x7f;

  // Utils
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<TFile> mOutFile;
  int mDuplicated_layer[NLAYERS] = {0};
  const o2::parameters::GRPMagField* mGRPMagField = nullptr;

  //// Histos
  // Distance betweeen original and duplicated clusters
  std::unique_ptr<TH1D> mDistanceClustersX[NLAYERS];
  std::unique_ptr<TH1D> mDistanceClustersY[NLAYERS];
  std::unique_ptr<TH1D> mDistanceClustersZ[NLAYERS];
  std::unique_ptr<TH1D> mDistanceClusters[NLAYERS];
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
  std::unique_ptr<TH1D> mPhiTrackOriginal[NLAYERS];
  std::unique_ptr<TH1D> mEtaOriginal[NLAYERS];
  std::unique_ptr<TH1D> mPtOriginal[NLAYERS];
  TH1D* mPtDuplicated[NLAYERS];
  TH1D* mEtaDuplicated[NLAYERS];
  TH1D* mPhiDuplicated[NLAYERS];
  TH1D* mPhiTrackDuplicated[NLAYERS];
  TH2D* mPhiTrackDuplicatedvsphiDuplicated[NLAYERS];
  TH2D* mPhiTrackoriginalvsphioriginal[NLAYERS];
  TH1D* mPhiOriginalIfDuplicated[NLAYERS];

  std::unique_ptr<TH2D> mZvsPhiDUplicated[NLAYERS];

  // position of the clusters
  std::unique_ptr<TH3D> m3DClusterPositions;
  std::unique_ptr<TH3D> m3DDuplicatedClusterPositions;
  std::unique_ptr<TH2D> m2DClusterOriginalPositions;
  std::unique_ptr<TH2D> m2DClusterDuplicatedPositions;

  std::unique_ptr<TH1D> mXoriginal;
  std::unique_ptr<TH1D> mYoriginal;
  std::unique_ptr<TH1D> mZoriginal;
  std::unique_ptr<TH1D> mXduplicated;
  std::unique_ptr<TH1D> mYduplicated;
  std::unique_ptr<TH1D> mZduplicated;

  // Efficiency histos
  std::unique_ptr<TH1D> mEfficiencyGoodMatch;
  std::unique_ptr<TH1D> mEfficiencyFakeMatch;
  std::unique_ptr<TH1D> mEfficiencyTotal;
  std::unique_ptr<TH1D> mEfficiencyGoodMatch_layer[NLAYERS];
  std::unique_ptr<TH1D> mEfficiencyFakeMatch_layer[NLAYERS];
  std::unique_ptr<TH1D> mEfficiencyTotal_layer[NLAYERS];
  TH2D* mEfficiencyGoodMatchPt_layer[NLAYERS];
  TH2D* mEfficiencyFakeMatchPt_layer[NLAYERS];
  TH2D* mEfficiencyGoodMatchEta_layer[NLAYERS];
  TH2D* mEfficiencyFakeMatchEta_layer[NLAYERS];
  TH2D* mEfficiencyGoodMatchPhi_layer[NLAYERS];
  TH2D* mEfficiencyGoodMatchPhiTrack_layer[NLAYERS];
  TH2D* mEfficiencyGoodMatchPhiOriginal_layer[NLAYERS];
  TH2D* mEfficiencyFakeMatchPhi_layer[NLAYERS];
  TH2D* mEfficiencyFakeMatchPhiTrack_layer[NLAYERS];

  // phi, eta, pt of the duplicated cluster per layer
  TH2D* mPt_EtaDupl[NLAYERS];

  // duplicated per layer and per cut
  std::unique_ptr<TH1D> mDuplicatedEtaAllPt[NLAYERS];
  std::unique_ptr<TH1D> mDuplicatedEta[NLAYERS][3];
  std::unique_ptr<TH1D> mDuplicatedPhiAllPt[NLAYERS];
  std::unique_ptr<TH1D> mDuplicatedPhi[NLAYERS][3];
  TH1D* mDuplicatedPt[NLAYERS];
  TH1D* mDuplicatedRow[NLAYERS];
  TH2D* mDuplicatedPtEta[NLAYERS];
  TH2D* mDuplicatedPtPhi[NLAYERS];
  TH2D* mDuplicatedEtaPhi[NLAYERS];

  // matches per layer and per cut
  std::unique_ptr<TH1D> mNGoodMatchesEtaAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNGoodMatchesEta[NLAYERS][3];
  std::unique_ptr<TH1D> mNGoodMatchesPhiAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNGoodMatchesPhi[NLAYERS][3];

  std::unique_ptr<TH1D> mNFakeMatchesEtaAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesEta[NLAYERS][3];
  std::unique_ptr<TH1D> mNFakeMatchesPhiAllPt[NLAYERS];
  std::unique_ptr<TH1D> mNFakeMatchesPhi[NLAYERS][3];

  TH1D* mNGoodMatchesPt[NLAYERS];
  TH1D* mNFakeMatchesPt[NLAYERS];

  TH1D* mNGoodMatchesRow[NLAYERS];
  TH1D* mNFakeMatchesRow[NLAYERS];

  TH2D* mNGoodMatchesPtEta[NLAYERS];
  TH2D* mNFakeMatchesPtEta[NLAYERS];

  TH2D* mNGoodMatchesPtPhi[NLAYERS];
  TH2D* mNFakeMatchesPtPhi[NLAYERS];

  TH2D* mNGoodMatchesEtaPhi[NLAYERS];
  TH2D* mNFakeMatchesEtaPhi[NLAYERS];

  // calculating the efficiency with TEfficiency class
  std::unique_ptr<TEfficiency> mEffPtGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffPtFake[NLAYERS];
  std::unique_ptr<TEfficiency> mEffRowGood[NLAYERS];
  std::unique_ptr<TEfficiency> mEffRowFake[NLAYERS];
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

  TH2D* mnGoodMatchesPt_layer[NLAYERS];
  TH2D* mnFakeMatchesPt_layer[NLAYERS];

  TH2D* mnGoodMatchesEta_layer[NLAYERS];
  TH2D* mnFakeMatchesEta_layer[NLAYERS];

  TH2D* mnGoodMatchesPhi_layer[NLAYERS];
  TH2D* mnGoodMatchesPhiTrack_layer[NLAYERS];
  TH2D* mnGoodMatchesPhiOriginal_layer[NLAYERS];
  TH2D* mnFakeMatchesPhi_layer[NLAYERS];
  TH2D* mnFakeMatchesPhiTrack_layer[NLAYERS];

  std::unique_ptr<TH1D> DCAxyData[NLAYERS];
  std::unique_ptr<TH1D> DCAzData[NLAYERS];

  std::unique_ptr<TH1D> DCAxyRejected[NLAYERS];
  std::unique_ptr<TH1D> DCAzRejected[NLAYERS];

  std::unique_ptr<TH1D> DistanceClustersX[NLAYERS];
  std::unique_ptr<TH1D> DistanceClustersY[NLAYERS];
  std::unique_ptr<TH1D> DistanceClustersZ[NLAYERS];
  std::unique_ptr<TH1D> DistanceClustersXAftercuts[NLAYERS];
  std::unique_ptr<TH1D> DistanceClustersYAftercuts[NLAYERS];
  std::unique_ptr<TH1D> DistanceClustersZAftercuts[NLAYERS];

  TH1D* denPt[NLAYERS];
  TH1D* numPt[NLAYERS];
  TH1D* numPtGood[NLAYERS];
  TH1D* numPtFake[NLAYERS];

  TH1D* denPhi[NLAYERS];
  TH1D* numPhi[NLAYERS];
  TH1D* numPhiGood[NLAYERS];
  TH1D* numPhiFake[NLAYERS];

  TH1D* denEta[NLAYERS];
  TH1D* numEta[NLAYERS];
  TH1D* numEtaGood[NLAYERS];
  TH1D* numEtaFake[NLAYERS];

  int nDuplicatedClusters[NLAYERS] = {0};
  int nTracksSelected[NLAYERS] = {0}; // denominator fot the efficiency calculation

  TH2D* diffPhivsPt[NLAYERS];
  TH1D* diffTheta[NLAYERS];

  TH1D* thetaOriginal[NLAYERS];
  TH1D* thetaOriginalCalc[NLAYERS];
  TH1D* thetaDuplicated[NLAYERS];
  TH1D* thetaOriginalCalcWhenDuplicated[NLAYERS];
  TH1D* thetaOriginalWhenDuplicated[NLAYERS];

  std::unique_ptr<TH1D> IPOriginalxy[NLAYERS];
  std::unique_ptr<TH1D> IPOriginalz[NLAYERS];
  std::unique_ptr<TH1D> IPOriginalifDuplicatedxy[NLAYERS];
  std::unique_ptr<TH1D> IPOriginalifDuplicatedz[NLAYERS];

  std::unique_ptr<TH1D> chipRowDuplicated[NLAYERS];
  std::unique_ptr<TH1D> chipRowOriginalIfDuplicated[NLAYERS];

  std::unique_ptr<TH1D> chi2track;
  std::unique_ptr<TH1D> chi2trackAccepted;
};

void EfficiencyStudy::init(InitContext& ic)
{
  LOGP(info, "--------------- init");

  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  auto& pars = o2::its::study::ITSEfficiencyParamConfig::Instance();
  mOutFileName = pars.outFileName;
  b = pars.b;

  int nbPt = 75;
  double xbins[nbPt + 1], ptcutl = 0.05, ptcuth = 7.5;
  double a = std::log(ptcuth / ptcutl) / nbPt;
  for (int i = 0; i <= nbPt; i++)
    xbins[i] = ptcutl * std::exp(i * a);

  mOutFile = std::make_unique<TFile>(mOutFileName.c_str(), "recreate");

  mXoriginal = std::make_unique<TH1D>("xoriginal", "x original  ;x (cm); ", 200, 0, 0);
  mYoriginal = std::make_unique<TH1D>("yoriginal", "y original  ;y (cm); ", 200, 0, 0);
  mZoriginal = std::make_unique<TH1D>("zoriginal", "z original  ;z (cm); ", 300, 0, 0);
  mXduplicated = std::make_unique<TH1D>("xduplicated", "x duplicated  ;x (cm); ", 200, -10, 10);
  mYduplicated = std::make_unique<TH1D>("yduplicated", "y duplicated  ;y (cm); ", 200, -10, 10);
  mZduplicated = std::make_unique<TH1D>("zduplicated", "z duplicated  ;z (cm); ", 300, -30, 30);

  mDCAxyDuplicated = std::make_unique<TH1D>("dcaXYDuplicated", "Distance between track and duplicated cluster  ;DCA xy (cm); ", 400, -0.2, 0.2);
  mDCAzDuplicated = std::make_unique<TH1D>("dcaZDuplicated", "Distance between track and duplicated cluster  ;DCA z (cm); ", 400, -0.2, 0.2);

  m3DClusterPositions = std::make_unique<TH3D>("3DClusterPositions", ";x (cm);y (cm);z (cm)", 200, -10, 10, 200, -10, 10, 400, -20, 20);
  m3DDuplicatedClusterPositions = std::make_unique<TH3D>("3DDuplicatedClusterPositions", ";x (cm);y (cm);z (cm)", 200, -10, 10, 200, -10, 10, 500, -30, 30);
  m2DClusterOriginalPositions = std::make_unique<TH2D>("m2DClusterOriginalPositions", ";x (cm);y (cm)", 400, -10, 10, 400, -6, 6);
  m2DClusterDuplicatedPositions = std::make_unique<TH2D>("m2DClusterDuplicatedPositions", ";x (cm);y (cm)", 400, -10, 10, 400, -6, 6);

  mEfficiencyGoodMatch = std::make_unique<TH1D>("mEfficiencyGoodMatch", ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
  mEfficiencyFakeMatch = std::make_unique<TH1D>("mEfficiencyFakeMatch", ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
  mEfficiencyTotal = std::make_unique<TH1D>("mEfficiencyTotal", ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);

  chi2track = std::make_unique<TH1D>("chi2track", "; $chi^{2}", 500, 0, 100);
  chi2trackAccepted = std::make_unique<TH1D>("chi2trackAccepted", "; $chi^{2}", 500, 0, 100);

  for (int i = 0; i < NLAYERS; i++) {

    chipRowDuplicated[i] = std::make_unique<TH1D>(Form("chipPosDuplicated_L%d", i), Form("L%d; row", i), 512, -0.5, 511.5);
    chipRowOriginalIfDuplicated[i] = std::make_unique<TH1D>(Form("chipPosOriginalIfDuplicated%d", i), Form("L%d; row", i), 512, -0.5, 511.5);

    DCAxyData[i] = std::make_unique<TH1D>(Form("dcaXYData_L%d", i), "Distance between track and original cluster ;DCA xy (cm); ", 4000, -2, 2);
    DCAzData[i] = std::make_unique<TH1D>(Form("dcaZData_L%d", i), "Distance between track and original cluster ;DCA z (cm); ", 4000, -2, 2);
    DCAxyRejected[i] = std::make_unique<TH1D>(Form("DCAxyRejected%d", i), "Distance between track and original cluster (rejected) ;DCA xy (cm); ", 30000, -30, 30);
    DCAzRejected[i] = std::make_unique<TH1D>(Form("DCAzRejected%d", i), "Distance between track and original cluster (rejected) ;DCA z (cm); ", 30000, -30, 30);

    DistanceClustersX[i] = std::make_unique<TH1D>(Form("distanceClustersX_L%d", i), ";Distance x (cm); ", 100, 0, 1);
    DistanceClustersY[i] = std::make_unique<TH1D>(Form("distanceClustersY_L%d", i), ";Distance y (cm); ", 100, 0, 1);
    DistanceClustersZ[i] = std::make_unique<TH1D>(Form("distanceClustersZ_L%d", i), ";Distance z (cm); ", 100, 0, 1);
    DistanceClustersXAftercuts[i] = std::make_unique<TH1D>(Form("distanceClustersXAftercuts_L%d", i), ";Distance x (cm); ", 100, 0, 1);
    DistanceClustersYAftercuts[i] = std::make_unique<TH1D>(Form("distanceClustersYAftercuts_L%d", i), ";Distance y (cm); ", 100, 0, 1);
    DistanceClustersZAftercuts[i] = std::make_unique<TH1D>(Form("distanceClustersZAftercuts_L%d", i), ";Distance z (cm); ", 100, 0, 1);

    mDistanceClustersX[i] = std::make_unique<TH1D>(Form("distanceClustersX_L%d", i), ";Distance x (cm); ", 100, 0, 1);
    mDistanceClustersY[i] = std::make_unique<TH1D>(Form("distanceClustersY_L%d", i), ";Distance y (cm); ", 100, 0, 1);
    mDistanceClustersZ[i] = std::make_unique<TH1D>(Form("distanceClustersZ_L%d", i), ";Distance z (cm); ", 100, 0, 1);
    mDistanceClusters[i] = std::make_unique<TH1D>(Form("distanceClusters_L%d", i), ";Distance (cm); ", 100, 0, 1);

    mDCAxyOriginal[i] = std::make_unique<TH1D>(Form("dcaXYOriginal_L%d", i), "Distance between track and original cluster ;DCA xy (cm); ", 400, -0.2, 0.2);
    mDCAzOriginal[i] = std::make_unique<TH1D>(Form("dcaZOriginal_L%d", i), "Distance between track and original cluster ;DCA z (cm); ", 400, -0.2, 0.2);

    mPhiOriginal[i] = std::make_unique<TH1D>(Form("phiOriginal_L%d", i), ";phi (deg); ", 1440, -180, 180);
    mPhiTrackOriginal[i] = std::make_unique<TH1D>(Form("phiTrackOriginal_L%d", i), ";phi Track (deg); ", 1440, 0, 360);
    mEtaOriginal[i] = std::make_unique<TH1D>(Form("etaOriginal_L%d", i), ";eta (deg); ", 100, -2, 2);
    mPtOriginal[i] = std::make_unique<TH1D>(Form("ptOriginal_L%d", i), ";pt (GeV/c); ", 100, 0, 10);

    mZvsPhiDUplicated[i] = std::make_unique<TH2D>(Form("zvsphiDuplicated_L%d", i), ";z (cm);phi (deg)", 400, -20, 20, 1440, -180, 180);

    mPtDuplicated[i] = new TH1D(Form("ptDuplicated_L%d", i), ";pt (GeV/c); ", nbPt, 0, 7.5); // xbins);
    mEtaDuplicated[i] = new TH1D(Form("etaDuplicated_L%d", i), ";eta; ", 40, -2, 2);
    mPhiDuplicated[i] = new TH1D(Form("phiDuplicated_L%d", i), ";phi (deg); ", 1440, -180, 180);
    mPhiTrackDuplicated[i] = new TH1D(Form("phiTrackDuplicated_L%d", i), ";phi Track (deg); ", 1440, 0, 360);
    mPhiOriginalIfDuplicated[i] = new TH1D(Form("phiOriginalIfDuplicated_L%d", i), ";phi (deg); ", 1440, -180, 180);
    mPhiTrackDuplicatedvsphiDuplicated[i] = new TH2D(Form("phiTrackDuplicatedvsphiDuplicated_L%d", i), ";phi track (deg);phi oridinal if duplicated (deg); ", 1440, 0, 360, 1440, -180, 180);
    mPhiTrackoriginalvsphioriginal[i] = new TH2D(Form("phiTrackoriginalvsphioriginal_L%d", i), ";phi track (deg);phi original (deg); ", 1440, 0, 360, 1440, -180, 180);
    mDCAxyDuplicated_layer[i] = std::make_unique<TH1D>(Form("dcaXYDuplicated_layer_L%d", i), "Distance between track and duplicated cluster  ;DCA xy (cm); ", 400, -0.2, 0.2);
    mDCAzDuplicated_layer[i] = std::make_unique<TH1D>(Form("dcaZDuplicated_layer_L%d", i), "Distance between track and duplicated cluster  ;DCA z (cm); ", 400, -0.2, 0.2);

    mEfficiencyGoodMatch_layer[i] = std::make_unique<TH1D>(Form("mEfficiencyGoodMatch_layer_L%d", i), ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
    mEfficiencyFakeMatch_layer[i] = std::make_unique<TH1D>(Form("mEfficiencyFakeMatch_layer_L%d", i), ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);
    mEfficiencyTotal_layer[i] = std::make_unique<TH1D>(Form("mEfficiencyTotal_layer_L%d", i), ";#sigma(DCA) cut;Efficiency;", 20, 0.5, 20.5);

    mEfficiencyGoodMatchPt_layer[i] = new TH2D(Form("mEfficiencyGoodMatchPt_layer_L%d", i), ";#it{p}_{T} (GeV/c);#sigma(DCA) cut;Efficiency;", nbPt, 0, 7.5, /* xbins*/ 20, 0.5, 20.5);
    mEfficiencyFakeMatchPt_layer[i] = new TH2D(Form("mEfficiencyFakeMatchPt_layer_L%d", i), ";#it{p}_{T} (GeV/c);#sigma(DCA) cut;Efficiency;", nbPt, 0, 7.5, /* xbins*/ 20, 0.5, 20.5);

    mEfficiencyGoodMatchEta_layer[i] = new TH2D(Form("mEfficiencyGoodMatchEta_layer_L%d", i), ";#eta;#sigma(DCA) cut;Efficiency;", 40, -2, 2, 20, 0.5, 20.5);
    mEfficiencyFakeMatchEta_layer[i] = new TH2D(Form("mEfficiencyFakeMatchEta_layer_L%d", i), ";#eta;#sigma(DCA) cut;Efficiency;", 40, -2, 2, 20, 0.5, 20.5);

    mEfficiencyGoodMatchPhi_layer[i] = new TH2D(Form("mEfficiencyGoodMatchPhi_layer_L%d", i), ";#phi;#sigma(DCA) cut;Efficiency;", 1440, -180, 180, 20, 0.5, 20.5);
    mEfficiencyGoodMatchPhiTrack_layer[i] = new TH2D(Form("mEfficiencyGoodMatchPhiTrack_layer_L%d", i), ";#phi track;#sigma(DCA) cut;Efficiency;", 1440, 0, 360, 20, 0.5, 20.5);
    mEfficiencyGoodMatchPhiOriginal_layer[i] = new TH2D(Form("mEfficiencyGoodMatchPhiOriginal_layer_L%d", i), ";#phi Original;#sigma(DCA) cut;Efficiency;", 1440, -180, 180, 20, 0.5, 20.5);
    mEfficiencyFakeMatchPhi_layer[i] = new TH2D(Form("mEfficiencyFakeMatchPhi_layer_L%d", i), ";#phi;#sigma(DCA) cut;Efficiency;", 1440, -180, 180, 20, 0.5, 20.5);
    mEfficiencyFakeMatchPhiTrack_layer[i] = new TH2D(Form("mEfficiencyFakeMatchPhiTrack_layer_L%d", i), ";#phi Track;#sigma(DCA) cut;Efficiency;", 1440, 0, 360, 20, 0.5, 20.5);

    mPt_EtaDupl[i] = new TH2D(Form("mPt_EtaDupl_L%d", i), ";#it{p}_{T} (GeV/c);#eta; ", 100, 0, 10, 100, -2, 2);

    mDuplicatedPt[i] = new TH1D(Form("mDuplicatedPt_log_L%d", i), Form("; #it{p}_{T} (GeV/c); Number of duplciated clusters L%d", i), nbPt, 0, 7.5 /* xbins*/);
    mDuplicatedPt[i]->Sumw2();
    mNGoodMatchesPt[i] = new TH1D(Form("mNGoodMatchesPt_L%d", i), Form("; #it{p}_{T} (GeV/c); Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/);
    mNGoodMatchesPt[i]->Sumw2();
    mNFakeMatchesPt[i] = new TH1D(Form("mNFakeMatchesPt_L%d", i), Form("; #it{p}_{T} (GeV/c); Number of fake matches L%d", i), nbPt, 0, 7.5 /* xbins*/);
    mNFakeMatchesPt[i]->Sumw2();

    mDuplicatedRow[i] = new TH1D(Form("mDuplicatedRow_L%d", i), Form("; Row; Number of duplciated clusters L%d", i), 512, -0.5, 511.5);
    mDuplicatedRow[i]->Sumw2();
    mNGoodMatchesRow[i] = new TH1D(Form("mNGoodMatchesRow_L%d", i), Form("; Row; Number of good matches L%d", i), 512, -0.5, 511.5);
    mNGoodMatchesRow[i]->Sumw2();
    mNFakeMatchesRow[i] = new TH1D(Form("mNFakeMatchesRow_L%d", i), Form(";Row; Number of fake matches L%d", i), 512, -0.5, 511.5);
    mNFakeMatchesRow[i]->Sumw2();

    mDuplicatedPtEta[i] = new TH2D(Form("mDuplicatedPtEta_log_L%d", i), Form("; #it{p}_{T} (GeV/c);#eta; Number of duplciated clusters L%d", i), nbPt, 0, 7.5 /* xbins*/, 40, -2, 2);
    mDuplicatedPtEta[i]->Sumw2();
    mNGoodMatchesPtEta[i] = new TH2D(Form("mNGoodMatchesPtEta_L%d", i), Form("; #it{p}_{T} (GeV/c);#eta; Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 40, -2, 2);
    mNGoodMatchesPtEta[i]->Sumw2();
    mNFakeMatchesPtEta[i] = new TH2D(Form("mNFakeMatchesPtEta_L%d", i), Form("; #it{p}_{T} (GeV/c);#eta; Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 40, -2, 2);
    mNFakeMatchesPtEta[i]->Sumw2();

    mDuplicatedPtPhi[i] = new TH2D(Form("mDuplicatedPtPhi_log_L%d", i), Form("; #it{p}_{T} (GeV/c);#phi (deg); Number of duplciated clusters L%d", i), nbPt, 0, 7.5 /* xbins*/, 1440, -180, 180);
    mDuplicatedPtPhi[i]->Sumw2();
    mNGoodMatchesPtPhi[i] = new TH2D(Form("mNGoodMatchesPtPhi_L%d", i), Form("; #it{p}_{T} (GeV/c);#phi (deg); Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 1440, -180, 180);
    mNGoodMatchesPtPhi[i]->Sumw2();
    mNFakeMatchesPtPhi[i] = new TH2D(Form("mNFakeMatchesPtPhi_L%d", i), Form("; #it{p}_{T} (GeV/c);#phi (deg); Number of good matches L%d", i), nbPt, 0, 7.5 /* xbins*/, 1440, -180, 180);
    mNFakeMatchesPtPhi[i]->Sumw2();

    mDuplicatedEtaPhi[i] = new TH2D(Form("mDuplicatedEtaPhi_L%d", i), Form("; #eta;#phi (deg); Number of duplciated clusters L%d", i), 40, -2, 2, 1440, -180, 180);
    mDuplicatedEtaPhi[i]->Sumw2();
    mNGoodMatchesEtaPhi[i] = new TH2D(Form("mNGoodMatchesEtaPhi_L%d", i), Form("; #eta;#phi (deg); Number of good matches L%d", i), 40, -2, 2, 1440, -180, 180);
    mNGoodMatchesEtaPhi[i]->Sumw2();
    mNFakeMatchesEtaPhi[i] = new TH2D(Form("mNFakeMatchesEtaPhi_L%d", i), Form("; #eta;#phi (deg); Number of good matches L%d", i), 40, -2, 2, 1440, -180, 180);
    mNFakeMatchesEtaPhi[i]->Sumw2();

    mDuplicatedEtaAllPt[i] = std::make_unique<TH1D>(Form("mDuplicatedEtaAllPt_L%d", i), Form("; #eta; Number of duplicated clusters L%d", i), 40, -2, 2);
    mNGoodMatchesEtaAllPt[i] = std::make_unique<TH1D>(Form("mNGoodMatchesEtaAllPt_L%d", i), Form("; #eta; Number of good matches L%d", i), 40, -2, 2);
    mNFakeMatchesEtaAllPt[i] = std::make_unique<TH1D>(Form("mNFakeMatchesEtaAllPt_L%d", i), Form("; #eta; Number of fake matches L%d", i), 40, -2, 2);

    mDuplicatedPhiAllPt[i] = std::make_unique<TH1D>(Form("mDuplicatedPhiAllPt_L%d", i), Form("; #phi (deg); Number of duplicated clusters L%d", i), 1440, -180, 180);
    mNGoodMatchesPhiAllPt[i] = std::make_unique<TH1D>(Form("mNGoodMatchesPhiAllPt_L%d", i), Form("; #phi (deg); Number of good matches L%d", i), 1440, -180, 180);
    mNFakeMatchesPhiAllPt[i] = std::make_unique<TH1D>(Form("mNFakeMatchesPhiAllPt_L%d", i), Form("; #phi (deg); Number of fake matches L%d", i), 1440, -180, 180);

    mnGoodMatchesPt_layer[i] = new TH2D(Form("mnGoodMatchesPt_layer_L%d", i), ";pt; nGoodMatches", nbPt, 0, 7.5 /* xbins*/, 20, 0.5, 20.5);
    mnFakeMatchesPt_layer[i] = new TH2D(Form("mnFakeMatchesPt_layer_L%d", i), ";pt; nFakeMatches", nbPt, 0, 7.5 /* xbins*/, 20, 0.5, 20.5);
    mnGoodMatchesEta_layer[i] = new TH2D(Form("mnGoodMatchesEta_layer_L%d", i), ";#eta; nGoodMatches", 40, -2, 2, 20, 0.5, 20.5);
    mnFakeMatchesEta_layer[i] = new TH2D(Form("mnFakeMatchesEta_layer_L%d", i), ";#eta; nFakeMatches", 40, -2, 2, 20, 0.5, 20.5);
    mnGoodMatchesPhi_layer[i] = new TH2D(Form("mnGoodMatchesPhi_layer_L%d", i), ";#Phi; nGoodMatches", 1440, -180, 180, 20, 0.5, 20.5);
    mnGoodMatchesPhiTrack_layer[i] = new TH2D(Form("mnGoodMatchesPhiTrack_layer_L%d", i), ";#Phi track; nGoodMatches", 1440, 0, 360, 20, 0.5, 20.5);
    mnGoodMatchesPhiOriginal_layer[i] = new TH2D(Form("mnGoodMatchesPhiOriginal_layer_L%d", i), ";#Phi of the original Cluster; nGoodMatches", 1440, -180, 180, 20, 0.5, 20.5);
    mnFakeMatchesPhi_layer[i] = new TH2D(Form("mnFakeMatchesPhi_layer_L%d", i), ";#Phi; nFakeMatches", 1440, -180, 180, 20, 0.5, 20.5);
    mnFakeMatchesPhiTrack_layer[i] = new TH2D(Form("mnFakeMatchesPhiTrack_layer_L%d", i), ";#Phi track; nFakeMatches", 1440, 0, 360, 20, 0.5, 20.5);

    denPt[i] = new TH1D(Form("denPt_L%d", i), Form("denPt_L%d", i), nbPt, 0, 7.5 /* xbins*/);
    numPt[i] = new TH1D(Form("numPt_L%d", i), Form("numPt_L%d", i), nbPt, 0, 7.5 /* xbins*/);
    numPtGood[i] = new TH1D(Form("numPtGood_L%d", i), Form("numPtGood_L%d", i), nbPt, 0, 7.5 /* xbins*/);
    numPtFake[i] = new TH1D(Form("numPtFake_L%d", i), Form("numPtFake_L%d", i), nbPt, 0, 7.5 /* xbins*/);

    denPhi[i] = new TH1D(Form("denPhi_L%d", i), Form("denPhi_L%d", i), 1440, -180, 180);
    numPhi[i] = new TH1D(Form("numPhi_L%d", i), Form("numPhi_L%d", i), 1440, -180, 180);
    numPhiGood[i] = new TH1D(Form("numPhiGood_L%d", i), Form("numPhiGood_L%d", i), 1440, -180, 180);
    numPhiFake[i] = new TH1D(Form("numPhiFake_L%d", i), Form("numPhiFake_L%d", i), 1440, -180, 180);

    denEta[i] = new TH1D(Form("denEta_L%d", i), Form("denEta_L%d", i), 200, -2, 2);
    numEta[i] = new TH1D(Form("numEta_L%d", i), Form("numEta_L%d", i), 200, -2, 2);
    numEtaGood[i] = new TH1D(Form("numEtaGood_L%d", i), Form("numEtaGood_L%d", i), 200, -2, 2);
    numEtaFake[i] = new TH1D(Form("numEtaFake_L%d", i), Form("numEtaFake_L%d", i), 200, -2, 2);

    diffPhivsPt[i] = new TH2D(Form("diffPhivsPt_L%d", i), Form("diffPhivsPt_L%d", i), nbPt, 0, 7.5 /* xbins*/, 50, 0, 5);

    IPOriginalxy[i] = std::make_unique<TH1D>(Form("IPOriginalxy_L%d", i), Form("IPOriginalxy_L%d", i), 500, -0.002, 0.002);
    IPOriginalz[i] = std::make_unique<TH1D>(Form("IPOriginalz_L%d", i), Form("IPOriginalz_L%d", i), 200, -10, 10);
    IPOriginalifDuplicatedxy[i] = std::make_unique<TH1D>(Form("IPOriginalifDuplicatedxy_L%d", i), Form("IPOriginalifDuplicatedxy_L%d", i), 1000, -0.005, 0.005);
    IPOriginalifDuplicatedz[i] = std::make_unique<TH1D>(Form("IPOriginalifDuplicatedz_L%d", i), Form("IPOriginalifDuplicatedz_L%d", i), 200, -10, 10);

    for (int j = 0; j < 3; j++) {
      mDuplicatedEta[i][j] = std::make_unique<TH1D>(Form("mDuplicatedEta_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #eta; Number of duplicated clusters L%d", mrangesPt[j][0], mrangesPt[j][1], i), 40, -2, 2);
      mNGoodMatchesEta[i][j] = std::make_unique<TH1D>(Form("mNGoodMatchesEta_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #eta; Number of good matches L%d", i, mrangesPt[j][0], mrangesPt[j][1], i), 40, -2, 2);
      mNFakeMatchesEta[i][j] = std::make_unique<TH1D>(Form("mNFakeMatchesEta_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #eta; Number of fake matches L%d", i, mrangesPt[j][0], mrangesPt[j][1], i), 40, -2, 2);

      mDuplicatedPhi[i][j] = std::make_unique<TH1D>(Form("mDuplicatedPhi_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #phi; Number of duplicated clusters L%d", mrangesPt[j][0], mrangesPt[j][1], i), 1440, -180, 180);
      mNGoodMatchesPhi[i][j] = std::make_unique<TH1D>(Form("mNGoodMatchesPhi_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #phi; Number of good matches L%d", i, mrangesPt[j][0], mrangesPt[j][1], i), 1440, -180, 180);
      mNFakeMatchesPhi[i][j] = std::make_unique<TH1D>(Form("mNFakeMatchesPhi_L%d_pt%d", i, j), Form("%f < #it{p}_{T} < %f GeV/c; #phi; Number of fake matches L%d", i, mrangesPt[j][0], mrangesPt[j][1], i), 1440, -180, 180);
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
  o2::gpu::gpustd::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
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

      if (ncl < 7)
        continue;

      float ip[2];
      track.getImpactParams(0, 0, 0, 0, ip);

      // if (abs(ip[0])>0.001 ) continue; ///pv not in (0,0,0)

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake())
        continue;

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();

      float phiTrack = trackParCov.getPhi() * 180 / M_PI;

      if (pt < mPtCuts[0] || pt > mPtCuts[1])
        continue;
      if (eta < mEtaCuts[0] || eta > mEtaCuts[1])
        continue;

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
        if (layer >= NLAYERS)
          continue;                                                            // checking only selected layers
        auto labsTrack = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;

        phioriginal = clusOriginalPointGlob.phi() * 180 / M_PI;
        mPhiTrackoriginalvsphioriginal[layer]->Fill(phiTrack, phioriginal);

        mPhiOriginal[layer]->Fill(phioriginal);
        mPtOriginal[layer]->Fill(pt);
        mEtaOriginal[layer]->Fill(eta);
        m3DClusterPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y(), clusOriginalPointGlob.z());
        m2DClusterOriginalPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y());

        for (auto& labT : labsTrack) { // for each valid label iterate over ALL the clusters in the ROF to see if there are duplicates
          if (labT != tracklab)
            continue;
          nLabels++;
          if (labT.isValid()) {
            for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF

              auto clusDuplicated = mClusters[iClus];
              auto clusDuplicatedPoint = mITSClustersArray[iClus];

              auto layerClus = mGeometry->getLayer(clusDuplicated.getSensorID());
              if (layerClus != layer)
                continue;

              o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
              o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
              // phiduplicated = std::atan2(clusDuplicatedPointGlob.y(), clusDuplicatedPointGlob.x()) * 180 / M_PI + 180;
              phiduplicated = clusDuplicatedPointGlob.phi() * 180 / M_PI;

              auto labsClus = mClustersMCLCont->getLabels(iClus); // ideally I can have more than one label per cluster
              for (auto labC : labsClus) {
                if (labC == labT) {
                  label_vecClus[iROF][layerClus][labT].push_back(iClus); // same cluster: label from the track = label from the cluster
                                                                         // if a duplicate cluster is found, propagate the track to the duplicate cluster and compute the distance from the original cluster
                                                                         // if (clusOriginalPointGlob != clusDuplicatedPointGlob) { /// check that the duplicated cluster is not the original one just counted twice
                                                                         // if (clusDuplicated.getSensorID() != clusOriginal.getSensorID()) { /// check that the duplicated cluster is not the original one just counted twice

                  // applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
                  if (clusDuplicated.getSensorID() == clusOriginal.getSensorID())
                    continue;
                  auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
                  if (layerDuplicated != layerClus)
                    continue;
                  auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
                  if (abs(staveDuplicated - staveOriginal) != 1)
                    continue;
                  auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
                  if (abs(chipDuplicated - chipOriginal) > 1)
                    continue;

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
                    mPhiTrackDuplicated[layerClus]->Fill(phiTrack);
                    mPhiTrackDuplicatedvsphiDuplicated[layerClus]->Fill(phiTrack, phioriginal);
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

                  /// compute the distance between original and dubplicated cluster
                  mDistanceClustersX[layerClus]->Fill(abs(clusOriginalPointGlob.x() - clusDuplicatedPointGlob.x()));
                  mDistanceClustersY[layerClus]->Fill(abs(clusOriginalPointGlob.y() - clusDuplicatedPointGlob.y()));
                  mDistanceClustersZ[layerClus]->Fill(abs(clusOriginalPointGlob.z() - clusDuplicatedPointGlob.z()));
                  mDistanceClusters[layerClus]->Fill(std::hypot(clusOriginalPointGlob.x() - clusDuplicatedPointGlob.x(), clusOriginalPointGlob.y() - clusDuplicatedPointGlob.y(), clusOriginalPointGlob.z() - clusDuplicatedPointGlob.z()));

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
      }                   // end loop on clusters
      totClus += NLAYERS; // summing only the number of clusters in the considered layers. Since the imposition of 7-clusters tracks, if the track is valid should release as clusters as the number of considered layers
    }                     // end loop on tracks per ROF
  }                       // end loop on ROFRecords array
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
          if (it.second.size() <= 1)
            continue; // printing only duplicates
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
  o2::gpu::gpustd::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
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
      // std::cout<<"Track number: "<<iTrack<<std::endl;

      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];
      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7)
        continue;

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake())
        continue;

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();

      // applying the cuts on the track - only pt and eta cuts since for phi the layer is needed
      if (pt < mPtCuts[0] || pt > mPtCuts[1])
        continue;
      if (eta < mEtaCuts[0] || eta > mEtaCuts[1])
        continue;

      float phi = -999.;
      float phiOriginal = -999.;

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]]; // cluster spacepoint in the tracking system
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        auto layer = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layer >= NLAYERS)
          continue;                                                            // checking only selected layers
        auto labsTrack = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;
        phiOriginal = clusOriginalPointGlob.phi() * 180 / M_PI;

        /// applying the cuts on the phi of the original cluster
        bool keepTrack = false; /// wether or not a cluster is found in an eligible track in the corresponding layer

        if (layerOriginal == 0) {
          for (int i = 0; i < 10; i++) {
            if ((phiOriginal >= mPhiCutsL0[i][0] && phiOriginal <= mPhiCutsL0[i][1])) {
              possibleduplicated[0]++;
              keepTrack = true;
            }
          }
        }
        if (layerOriginal == 1) {
          for (int i = 0; i < 12; i++) {
            if ((phiOriginal >= mPhiCutsL1[i][0] && phiOriginal <= mPhiCutsL1[i][1])) {
              possibleduplicated[1]++;
              keepTrack = true;
            }
          }
        }
        if (layerOriginal == 2) {
          for (int i = 0; i < 17; i++) {
            if ((phiOriginal >= mPhiCutsL2[i][0] && phiOriginal <= mPhiCutsL2[i][1])) {
              possibleduplicated[2]++;
              keepTrack = true;
            }
          }
        }

        if (!keepTrack)
          continue; /// if the track (cluster) is not eligible for any layer, go to the next one

        for (auto& labT : labsTrack) { // for each valid label iterate over ALL the clusters in the ROF to see if there are duplicates
          if (labT != tracklab)
            continue;

          if (labT.isValid()) {
            for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF

              auto clusDuplicated = mClusters[iClus];
              auto clusDuplicatedPoint = mITSClustersArray[iClus];

              auto layerClus = mGeometry->getLayer(clusDuplicated.getSensorID());
              if (layerClus != layer)
                continue;

              o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
              o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
              phi = clusDuplicatedPointGlob.phi() * 180 / M_PI;

              auto labsClus = mClustersMCLCont->getLabels(iClus); // ideally I can have more than one label per cluster
              for (auto labC : labsClus) {
                if (labC == labT) {
                  label_vecClus[iROF][layerClus][labT].push_back(iClus); // same cluster: label from the track = label from the cluster
                                                                         // if a duplicate cluster is found, propagate the track to the duplicate cluster and compute the distance from the original cluster
                                                                         // if (clusOriginalPointGlob != clusDuplicatedPointGlob) { /// check that the duplicated cluster is not the original one just counted twice
                                                                         // if (clusDuplicated.getSensorID() != clusOriginal.getSensorID()) { /// check that the duplicated cluster is not the original one just counted twice

                  // applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
                  if (clusDuplicated.getSensorID() == clusOriginal.getSensorID())
                    continue;
                  auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
                  if (layerDuplicated != layerClus)
                    continue;
                  auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
                  if (abs(staveDuplicated - staveOriginal) != 1)
                    continue;
                  auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
                  if (abs(chipDuplicated - chipOriginal) > 1)
                    continue;

                  duplicated[layer]++;
                  std::cout << "Taken L" << layer << " # " << duplicated[layer] << " : pt, eta, phi = " << pt << " , " << eta << " , " << phiOriginal << " Label: " << std::endl;
                  labC.print();
                }
              }
            }
          }
        }
      } // end loop on clusters
      totClus += ncl;
    } // end loop on tracks per ROF
  }   // end loop on ROFRecords array

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
  o2::gpu::gpustd::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
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
  for (int i = 0; i <= nbPt; i++)
    xbins[i] = ptcutl * std::exp(i * a);

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

      if (pt < mPtCuts[0] || pt > mPtCuts[1])
        continue;
      if (eta < mEtaCuts[0] || eta > mEtaCuts[1])
        continue;

      float phiTrack = trackParCov.getPhi() * 180 / M_PI;

      float phi = -999.;
      float phiOriginal = -999.;
      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7)
        continue;

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake())
        continue;

      if (mVerboseOutput) {
        LOGP(info, "--------- track Label: ");
        tracklab.print();
      }

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track to extract layer, stave and chip to restrict the possible matches to be searched with the DCA cut
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]]; // cluster spacepoint in the tracking system
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layerOriginal >= NLAYERS)
          continue;
        auto labsOriginal = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track (original)
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;

        phiOriginal = clusOriginalPointGlob.phi() * 180 / M_PI;

        for (auto& labT : labsOriginal) { // for each valid label iterate over ALL the clusters in the ROF to see if there are duplicates
          if (labT != tracklab)
            continue;
          if (!labT.isValid())
            continue;

          /// for each oroginal cluster iterate over all the possible "adjacent" clusters (stave +-1, chip =,+-1) and calculate the DCA with the track. Then compare the cluster label with the track label to see if it is a true or fake match
          for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF
            auto clusDuplicated = mClusters[iClus];
            //// applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
            if (clusDuplicated.getSensorID() == clusOriginal.getSensorID())
              continue;
            auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
            if (layerDuplicated != layerOriginal)
              continue;
            auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
            if (abs(staveDuplicated - staveOriginal) != 1)
              continue;
            auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
            if (abs(chipDuplicated - chipOriginal) > 1)
              continue;

            auto labsDuplicated = mClustersMCLCont->getLabels(iClus);

            /// if the cheks are passed, then calculate the DCA
            auto clusDuplicatedPoint = mITSClustersArray[iClus];

            o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
            o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
            phi = clusDuplicatedPointGlob.phi() * 180 / M_PI;

            /// Compute the DCA between the duplicated cluster location and the track
            trackParCov.rotate(mGeometry->getSensorRefAlpha(clusDuplicated.getSensorID()));
            if (propagator->propagateToDCA(clusDuplicatedPointGlob, trackParCov, b, 2.f, matCorr, &clusDuplicatedDCA)) { // check if the propagation fails
              if (mVerboseOutput)
                LOGP(info, "Propagation ok");
              /// checking the DCA for 20 different sigma ranges
              for (int i = 0; i < 20; i++) {
                if (abs(dcaXY[layerDuplicated] - clusDuplicatedDCA[0]) < (i + 1) * sigmaDcaXY[layerDuplicated] && abs(dcaZ[layerDuplicated] - clusDuplicatedDCA[1]) < (i + 1) * sigmaDcaZ[layerDuplicated]) { // check if the DCA is within the cut i*sigma

                  if (mVerboseOutput)
                    LOGP(info, "Check DCA ok: {} < {}; {} < {}", abs(meanDCAxyDuplicated[layerDuplicated] - clusDuplicatedDCA[0]), (i + 1) * sigmaDCAxyDuplicated[layerDuplicated], abs(meanDCAzDuplicated[layerDuplicated] - clusDuplicatedDCA[1]), (i + 1) * sigmaDCAzDuplicated[layerDuplicated]);
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
                    mnGoodMatchesPhiTrack_layer[layerDuplicated]->Fill(phiTrack, i);
                    mnGoodMatchesPhiOriginal_layer[layerDuplicated]->Fill(phiOriginal, i);
                  } else {

                    nFakeMatches[i]++;
                    nFakeMatches_layer[layerDuplicated][i]++;
                    mnFakeMatchesPt_layer[layerDuplicated]->Fill(pt, i);
                    mnFakeMatchesEta_layer[layerDuplicated]->Fill(eta, i);
                    mnFakeMatchesPhi_layer[layerDuplicated]->Fill(phi, i);
                    mnFakeMatchesPhiTrack_layer[layerDuplicated]->Fill(phiTrack, i);
                  }
                } else if (mVerboseOutput)
                  LOGP(info, "Check DCA failed");
              }
            } else if (mVerboseOutput)
              LOGP(info, "Propagation failed");
          } // end loop on all the clusters in the rof
        }
      } // end loop on clusters associated to the track
    }   // end loop on tracks per ROF
  }     // end loop on ROFRecords array

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
        if (mPtDuplicated[l]->GetBinContent(ipt + 1) != 0)
          mEfficiencyGoodMatchPt_layer[l]->SetBinContent(ipt + 1, i + 1, mnGoodMatchesPt_layer[l]->GetBinContent(ipt + 1, i + 1) / mPtDuplicated[l]->GetBinContent(ipt + 1));
        mEfficiencyFakeMatchPt_layer[l]->SetBinContent(ipt + 1, i + 1, mnFakeMatchesPt_layer[l]->GetBinContent(ipt + 1, i + 1) / mPtDuplicated[l]->GetBinContent(ipt + 1));
      }

      for (int ieta = 0; ieta < mEtaDuplicated[l]->GetNbinsX(); ieta++) {
        if (mEtaDuplicated[l]->GetBinContent(ieta + 1) != 0)
          mEfficiencyGoodMatchEta_layer[l]->SetBinContent(ieta + 1, i + 1, mnGoodMatchesEta_layer[l]->GetBinContent(ieta + 1, i + 1) / mEtaDuplicated[l]->GetBinContent(ieta + 1));
        mEfficiencyFakeMatchEta_layer[l]->SetBinContent(ieta + 1, i + 1, mnFakeMatchesEta_layer[l]->GetBinContent(ieta + 1, i + 1) / mEtaDuplicated[l]->GetBinContent(ieta + 1));
      }

      for (int iphi = 0; iphi < mPhiDuplicated[l]->GetNbinsX(); iphi++) {
        if (mPhiDuplicated[l]->GetBinContent(iphi + 1) != 0)
          mEfficiencyGoodMatchPhi_layer[l]->SetBinContent(iphi + 1, i + 1, mnGoodMatchesPhi_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiDuplicated[l]->GetBinContent(iphi + 1));
        mEfficiencyFakeMatchPhi_layer[l]->SetBinContent(iphi + 1, i + 1, mnFakeMatchesPhi_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiDuplicated[l]->GetBinContent(iphi + 1));
      }

      for (int iphi = 0; iphi < mPhiOriginalIfDuplicated[l]->GetNbinsX(); iphi++) {
        if (mPhiOriginalIfDuplicated[l]->GetBinContent(iphi + 1) != 0)
          mEfficiencyGoodMatchPhiOriginal_layer[l]->SetBinContent(iphi + 1, i + 1, mnGoodMatchesPhiOriginal_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiOriginalIfDuplicated[l]->GetBinContent(iphi + 1));
      }

      for (int iphi = 0; iphi < mPhiTrackDuplicated[l]->GetNbinsX(); iphi++) {
        if (mPhiTrackDuplicated[l]->GetBinContent(iphi + 1) != 0)
          mEfficiencyGoodMatchPhiTrack_layer[l]->SetBinContent(iphi + 1, i + 1, mnGoodMatchesPhiTrack_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiTrackDuplicated[l]->GetBinContent(iphi + 1));
        mEfficiencyFakeMatchPhiTrack_layer[l]->SetBinContent(iphi + 1, i + 1, mnFakeMatchesPhiTrack_layer[l]->GetBinContent(iphi + 1, i + 1) / mPhiTrackDuplicated[l]->GetBinContent(iphi + 1));
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
    mEfficiencyGoodMatchPhiTrack_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyGoodMatchPhiTrack_layer[l]->Write();
    mEfficiencyGoodMatchPhiOriginal_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyGoodMatchPhiOriginal_layer[l]->Write();
    mEfficiencyFakeMatchPt_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyFakeMatchPt_layer[l]->Write();
    mEfficiencyFakeMatchEta_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyFakeMatchEta_layer[l]->Write();
    mEfficiencyFakeMatchPhi_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyFakeMatchPhi_layer[l]->Write();
    mEfficiencyFakeMatchPhiTrack_layer[l]->GetZaxis()->SetRangeUser(0, 1);
    mEfficiencyFakeMatchPhiTrack_layer[l]->Write();
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
  c.SaveAs("prova.png");

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
    cc[l].SaveAs(Form("provaLayer%d.png", l));
  }
}

void EfficiencyStudy::studyClusterSelectionMC()
{
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
  o2::gpu::gpustd::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
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

      if (ncl < 7)
        continue;

      auto& tracklab = mTracksMCLabels[iTrack];
      if (tracklab.isFake())
        continue;

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();

      if (pt < mPtCuts[0] || pt > mPtCuts[1])
        continue;
      if (eta < mEtaCuts[0] || eta > mEtaCuts[1])
        continue;

      // auto phi = trackParCov.getPhi()*180/M_PI;
      float phi = -999.;
      float phiOriginal = -999.;
      float phiDuplicated = -999.;
      UShort_t row = -999;

      if (mVerboseOutput) {
        LOGP(info, "--------- track Label: ");
        tracklab.print();
      }
      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track to extract layer, stave and chip to restrict the possible matches to be searched with the DCA cut
        // LOGP(info, "New cluster");
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layerOriginal >= NLAYERS)
          continue;

        IPOriginalxy[layerOriginal]->Fill(ip[0]);
        IPOriginalz[layerOriginal]->Fill(ip[1]);

        UShort_t rowOriginal = clusOriginal.getRow();

        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]];
        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;

        auto phiOriginal = clusOriginalPointGlob.phi() * 180 / M_PI;

        auto labsOriginal = mClustersMCLCont->getLabels(mInputITSidxs[iclTrack]); // get labels of the cluster associated to the track (original)
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        std::tuple<int, double, gsl::span<const o2::MCCompLabel>> clusID_rDCA_label = {0, 999., gsl::span<const o2::MCCompLabel>()}; // inizializing tuple with dummy values

        bool adjacentFound = 0;
        /// for each oroginal cluster iterate over all the possible "adjacten" clusters (stave +-1, chip =,+-1) and calculate the DCA with the track. Then choose the closest one.
        for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF
          auto clusDuplicated = mClusters[iClus];

          //// applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
          if (clusDuplicated.getSensorID() == clusOriginal.getSensorID())
            continue;
          auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
          if (layerDuplicated != layerOriginal)
            continue;
          auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
          if (abs(staveDuplicated - staveOriginal) != 1)
            continue;
          auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
          if (abs(chipDuplicated - chipOriginal) > 1)
            continue;

          auto labsDuplicated = mClustersMCLCont->getLabels(iClus);

          /// if the cheks are passed, then calculate the DCA
          auto clusDuplicatedPoint = mITSClustersArray[iClus];

          o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
          o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;

          auto phiDuplicated = clusDuplicatedPointGlob.phi() * 180 / M_PI;

          /// Compute the DCA between the duplicated cluster location and the track
          trackParCov.rotate(mGeometry->getSensorRefAlpha(clusDuplicated.getSensorID()));
          if (!propagator->propagateToDCA(clusDuplicatedPointGlob, trackParCov, b, 2.f, matCorr, &clusDuplicatedDCA)) { // check if the propagation fails
            continue;
          }

          // Imposing that the distance between the original cluster and the duplicated one is less than x sigma
          if (!(abs(meanDCAxyDuplicated[layerDuplicated] - clusDuplicatedDCA[0]) < 8 * sigmaDCAxyDuplicated[layerDuplicated] && abs(meanDCAzDuplicated[layerDuplicated] - clusDuplicatedDCA[1]) < 8 * sigmaDCAzDuplicated[layerDuplicated])) {
            continue;
          }

          if (mVerboseOutput)
            LOGP(info, "Propagation ok");
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

        if (!adjacentFound)
          continue;

        bool isGood = false;
        for (auto lab : std::get<2>(clusID_rDCA_label)) {
          if (lab == tracklab) {
            isGood = true;
            diffPhivsPt[layerOriginal]->Fill(pt, abs(phi - phiOriginal));
            IPOriginalifDuplicatedxy[layerOriginal]->Fill(ip[0]);
            IPOriginalifDuplicatedz[layerOriginal]->Fill(ip[1]);

            mNGoodMatchesPt[layerOriginal]->Fill(pt);
            mNGoodMatchesRow[layerOriginal]->Fill(row);
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
    }   // end loop on tracks per ROF
  }     // end loop on ROFRecords array

  mOutFile->mkdir("EfficiencyCuts/");
  mOutFile->cd("EfficiencyCuts/");

  std::cout << "------Calculatin efficiency..." << std::endl;
  TH1D* axpt = new TH1D("axpt", "", 1, 0.05, 7.5);
  TH1D* axRow = new TH1D("axRow", "", 1, -0.5, 511.5);
  TH2D* axptetaGood = new TH2D("axptetaGood", "", 1, 0.05, 7.5, 1, -2, 2);
  TH2D* axptetaFake = new TH2D("axptetaFake", "", 1, 0.05, 7.5, 1, -2, 2);
  TH2D* axptphiGood = new TH2D("axptphiGood", "", 1, 0.05, 7.5, 1, -180, 180);
  TH2D* axptphiFake = new TH2D("axptphiFake", "", 1, 0.05, 7.5, 1, -180, 180);
  TH2D* axetaphiGood = new TH2D("axetaphiGood", "", 1, -2, 2, 1, -180, 180);
  TH2D* axetaphiFake = new TH2D("axetaphiFake", "", 1, -2, 2, 1, -180, 180);
  TH1D* axetaAllPt = new TH1D("axetaAllPt", "", 1, -2, 2);
  TH1D* axeta[NLAYERS];
  TH1D* axphi[NLAYERS];
  for (int ipt = 0; ipt < 3; ipt++) {
    axeta[ipt] = new TH1D(Form("axeta%d", ipt), Form("axeta%d", ipt), 1, -2, 2);
    axphi[ipt] = new TH1D(Form("axphi%d", ipt), Form("axphi%d", ipt), 1, -180, 180);
  }
  TH1D* axphiAllPt = new TH1D("axphi", "", 1, -180, 180);

  TCanvas* effPt[NLAYERS];
  TCanvas* effRow[NLAYERS];
  TCanvas* effPtEta[NLAYERS][2];
  TCanvas* effPtPhi[NLAYERS][2];
  TCanvas* effEtaPhi[NLAYERS][2];
  TCanvas* effEtaAllPt[NLAYERS];
  TCanvas* effEta[NLAYERS][3];
  TCanvas* effPhiAllPt[NLAYERS];
  TCanvas* effPhi[NLAYERS][3];

  ///////////////// plotting results
  for (int l = 0; l < 3; l++) {
    if (mVerboseOutput)
      std::cout << "Pt L" << l << "\n\n";

    diffPhivsPt[l]->Write();
    IPOriginalifDuplicatedxy[l]->Write();
    IPOriginalifDuplicatedz[l]->Write();

    // Pt
    effPt[l] = new TCanvas(Form("effPt_L%d", l));

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
    effPtEta[l][0] = new TCanvas(Form("effPtEtaGood_L%d", l));

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
          if (mVerboseOutput)
            std::cout << "--- PtEta fakematches : Npass = " << mNFakeMatchesPtEta[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedPtEta[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          mNFakeMatchesPtEta[l]->SetBinContent(ibin, jbin, mDuplicatedPtEta[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // Row
    effRow[l] = new TCanvas(Form("effRow_L%d", l));

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
    axRow->GetXaxis()->SetRangeUser(0.05, 7.5);
    axRow->Draw();
    mEffRowGood[l]->Draw("same p");
    mEffRowFake[l]->Draw("same p");

    auto legRow = std::make_unique<TLegend>(0.70, 0.15, 0.89, 0.35);
    legRow->AddEntry(mEffRowGood[l].get(), "#frac{# good matches}{# tot duplicated clusters}", "pl");
    legRow->AddEntry(mEffRowFake[l].get(), "#frac{# fake matches}{# tot duplicated clusters}", "pl");
    legRow->Draw("same");
    effRow[l]->Write();

    // PtEtaGood
    effPtEta[l][0] = new TCanvas(Form("effPtEtaGood_L%d", l));

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
          if (mVerboseOutput)
            std::cout << "--- PtEta fakematches : Npass = " << mNFakeMatchesPtEta[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedPtEta[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          mNFakeMatchesPtEta[l]->SetBinContent(ibin, jbin, mDuplicatedPtEta[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // PtEtaFake
    effPtEta[l][1] = new TCanvas(Form("effPtEtaFake_L%d", l));

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
    effPtPhi[l][0] = new TCanvas(Form("effPtPhiGood_L%d", l));

    mEffPtPhiGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesPtPhi[l], *mDuplicatedPtPhi[l]);
    stileEfficiencyGraph(mEffPtPhiGood[l], Form("mEffPtPhiGood_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (deg);Efficiency", l), true);

    axptphiGood->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (deg);Efficiency", l));
    axptphiGood->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axptphiGood->GetYaxis()->SetRangeUser(-180, 180);
    axptphiGood->GetXaxis()->SetRangeUser(0.05, 7.5);
    axptphiGood->Draw();
    mEffPtPhiGood[l]->Draw("same colz");
    effPtPhi[l][0]->Update();
    effPtPhi[l][0]->Write();

    for (int ibin = 1; ibin <= mNFakeMatchesPtPhi[l]->GetNbinsX(); ibin++) {
      for (int jbin = 1; jbin <= mNFakeMatchesPtPhi[l]->GetNbinsY(); jbin++) {
        if (mNFakeMatchesPtPhi[l]->GetBinContent(ibin, jbin) > mDuplicatedPtPhi[l]->GetBinContent(ibin, jbin)) {
          if (mVerboseOutput)
            std::cout << "--- Pt: Npass = " << mNFakeMatchesPtPhi[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedPtPhi[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          mNFakeMatchesPtPhi[l]->SetBinContent(ibin, jbin, mDuplicatedPtPhi[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // PtPhiFake
    effPtPhi[l][1] = new TCanvas(Form("effPtPhiFake_L%d", l));

    mEffPtPhiFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesPtPhi[l], *mDuplicatedPtPhi[l]);
    stileEfficiencyGraph(mEffPtPhiFake[l], Form("mEffPtPhiFake_L%d", l), Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (deg);Efficiency", l), true);
    axptphiFake->SetTitle(Form("L%d;#it{p}_{T} (GeV/#it{c});#phi (deg);Efficiency", l));
    axptphiFake->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axptphiFake->GetYaxis()->SetRangeUser(-180, 180);
    axptphiFake->GetXaxis()->SetRangeUser(0.05, 7.5);
    axptphiFake->Draw();
    mEffPtPhiFake[l]->Draw("same colz");
    effPtPhi[l][1]->Update();
    effPtPhi[l][1]->Write();

    // EtaPhiGood
    effEtaPhi[l][0] = new TCanvas(Form("effEtaPhiGood_L%d", l));

    mEffEtaPhiGood[l] = std::make_unique<TEfficiency>(*mNGoodMatchesEtaPhi[l], *mDuplicatedEtaPhi[l]);
    stileEfficiencyGraph(mEffEtaPhiGood[l], Form("mEffEtaPhiGood_L%d", l), Form("L%d;#eta;#phi (deg);Efficiency", l), true);

    axetaphiGood->SetTitle(Form("L%d;#eta;#phi (deg);Efficiency", l));
    axetaphiGood->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axetaphiGood->GetYaxis()->SetRangeUser(-180, 180);
    axetaphiGood->GetXaxis()->SetRangeUser(-2, 2);
    axetaphiGood->Draw();
    mEffEtaPhiGood[l]->Draw("same colz");
    effEtaPhi[l][0]->Update();
    effEtaPhi[l][0]->Write();

    for (int ibin = 1; ibin <= mNFakeMatchesEtaPhi[l]->GetNbinsX(); ibin++) {
      for (int jbin = 1; jbin <= mNFakeMatchesEtaPhi[l]->GetNbinsY(); jbin++) {
        if (mNFakeMatchesEtaPhi[l]->GetBinContent(ibin, jbin) > mDuplicatedEtaPhi[l]->GetBinContent(ibin, jbin)) {
          if (mVerboseOutput)
            std::cout << "--- Eta: Npass = " << mNFakeMatchesEtaPhi[l]->GetBinContent(ibin, jbin) << ",  Nall = " << mDuplicatedEtaPhi[l]->GetBinContent(ibin, jbin) << " for ibin = " << ibin << ", jbin = " << jbin << std::endl;
          mNFakeMatchesEtaPhi[l]->SetBinContent(ibin, jbin, mDuplicatedEtaPhi[l]->GetBinContent(ibin, jbin));
        }
      }
    }

    // EtaPhiFake
    effEtaPhi[l][1] = new TCanvas(Form("effEtaPhiFake_L%d", l));

    mEffEtaPhiFake[l] = std::make_unique<TEfficiency>(*mNFakeMatchesEtaPhi[l], *mDuplicatedEtaPhi[l]);
    stileEfficiencyGraph(mEffEtaPhiFake[l], Form("mEffEtaPhiFake_L%d", l), Form("L%d;#eta;#phi (deg);Efficiency", l), true);
    axetaphiFake->SetTitle(Form("L%d;#eta;#phi (deg);Efficiency", l));
    axetaphiFake->GetZaxis()->SetRangeUser(-0.1, 1.1);
    axetaphiFake->GetYaxis()->SetRangeUser(-180, 180);
    axetaphiFake->GetXaxis()->SetRangeUser(-2, 2);
    axetaphiFake->Draw();
    mEffEtaPhiFake[l]->Draw("same colz");
    effEtaPhi[l][1]->Update();
    effEtaPhi[l][1]->Write();

    // EtaAllPt
    if (mVerboseOutput)
      std::cout << "Eta L" << l << "\n\n";

    effEtaAllPt[l] = new TCanvas(Form("effEtaAllPt_L%d", l));

    mEffEtaGoodAllPt[l] = std::make_unique<TEfficiency>(*mNGoodMatchesEtaAllPt[l], *mDuplicatedEtaAllPt[l]);
    stileEfficiencyGraph(mEffEtaGoodAllPt[l], Form("mEffEtaGoodAllPt_L%d", l), Form("L%d;#eta;Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesEtaAllPt[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesEtaAllPt[l]->GetBinContent(ibin) > mDuplicatedEtaAllPt[l]->GetBinContent(ibin)) {
        if (mVerboseOutput)
          std::cout << "--- EtaAllPt: Npass = " << mNFakeMatchesEtaAllPt[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedEtaAllPt[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
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
      effEta[l][ipt] = new TCanvas(Form("effEta_L%d_pt%d", l, ipt));

      mEffEtaGood[l][ipt] = std::make_unique<TEfficiency>(*mNGoodMatchesEta[l][ipt], *mDuplicatedEta[l][ipt]);
      stileEfficiencyGraph(mEffEtaGood[l][ipt], Form("mEffEtaGood_L%d_pt%d", l, ipt), Form("L%d     %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#eta;Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

      for (int ibin = 1; ibin <= mNFakeMatchesEta[l][ipt]->GetNbinsX(); ibin++) {
        if (mNFakeMatchesEta[l][ipt]->GetBinContent(ibin) > mDuplicatedEta[l][ipt]->GetBinContent(ibin)) {
          if (mVerboseOutput)
            std::cout << "--- Eta : Npass = " << mNFakeMatchesEta[l][ipt]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedEta[l][ipt]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
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
      effPhi[l][ipt] = new TCanvas(Form("effPhi_L%d_pt%d", l, ipt));

      for (int ibin = 1; ibin <= mNGoodMatchesPhi[l][ipt]->GetNbinsX(); ibin++) {
        if (mNGoodMatchesPhi[l][ipt]->GetBinContent(ibin) > mDuplicatedPhi[l][ipt]->GetBinContent(ibin)) {
          if (mVerboseOutput)
            std::cout << "--- Phi L: Npass = " << mNGoodMatchesPhi[l][ipt]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhi[l][ipt]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
          mNGoodMatchesPhi[l][ipt]->SetBinContent(ibin, 0);
        }
      }

      mEffPhiGood[l][ipt] = std::make_unique<TEfficiency>(*mNGoodMatchesPhi[l][ipt], *mDuplicatedPhi[l][ipt]);
      stileEfficiencyGraph(mEffPhiGood[l][ipt], Form("mEffPhiGood_L%d_pt%d", l, ipt), Form("L%d     %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#phi (deg);Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

      for (int ibin = 1; ibin <= mNFakeMatchesPhi[l][ipt]->GetNbinsX(); ibin++) {
        if (mNFakeMatchesPhi[l][ipt]->GetBinContent(ibin) > mDuplicatedPhi[l][ipt]->GetBinContent(ibin)) {
          if (mVerboseOutput)
            std::cout << "--- Phi L: Npass = " << mNFakeMatchesPhi[l][ipt]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhi[l][ipt]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
          mNFakeMatchesPhi[l][ipt]->SetBinContent(ibin, mDuplicatedPhi[l][ipt]->GetBinContent(ibin));
        }
      }

      mEffPhiFake[l][ipt] = std::make_unique<TEfficiency>(*mNFakeMatchesPhi[l][ipt], *mDuplicatedPhi[l][ipt]);
      stileEfficiencyGraph(mEffPhiFake[l][ipt], Form("mEffPhiFake_L%d_pt%d", l, ipt), Form("L%d    %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#phi (deg);Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]), false, kFullDiamond, 1, kRed + 1, kRed + 1);

      axphi[ipt]->SetTitle(Form("L%d     %.1f #leq #it{p}_{T} < %.1f GeV/#it{c};#phi (deg);Efficiency", l, mrangesPt[ipt][0], mrangesPt[ipt][1]));
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
    if (mVerboseOutput)
      std::cout << "Phi L" << l << "\n\n";

    effPhiAllPt[l] = new TCanvas(Form("effPhiAllPt_L%d", l));

    for (int ibin = 1; ibin <= mNGoodMatchesPhiAllPt[l]->GetNbinsX(); ibin++) {
      if (mNGoodMatchesPhiAllPt[l]->GetBinContent(ibin) > mDuplicatedPhiAllPt[l]->GetBinContent(ibin)) {
        if (mVerboseOutput)
          std::cout << "--- phi all good Npass = " << mNGoodMatchesPhiAllPt[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhiAllPt[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
        mNGoodMatchesPhiAllPt[l]->SetBinContent(ibin, 0);
      }
    }

    mEffPhiGoodAllPt[l] = std::make_unique<TEfficiency>(*mNGoodMatchesPhiAllPt[l], *mDuplicatedPhiAllPt[l]);
    stileEfficiencyGraph(mEffPhiGoodAllPt[l], Form("mEffPhiGoodAllPt_L%d", l), Form("L%d;#phi;Efficiency", l), false, kFullDiamond, 1, kGreen + 3, kGreen + 3);

    for (int ibin = 1; ibin <= mNFakeMatchesPhiAllPt[l]->GetNbinsX(); ibin++) {
      if (mNFakeMatchesPhiAllPt[l]->GetBinContent(ibin) > mDuplicatedPhiAllPt[l]->GetBinContent(ibin)) {
        if (mVerboseOutput)
          std::cout << "--- phi all fake Npass = " << mNFakeMatchesPhiAllPt[l]->GetBinContent(ibin) << ",  Nall = " << mDuplicatedPhiAllPt[l]->GetBinContent(ibin) << " for ibin = " << ibin << std::endl;
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
}

void EfficiencyStudy::saveDataInfo()
{
  // save histograms for data (phi, eta, pt,...)
  LOGP(info, "--------------- saveDataInfo");

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;
  unsigned int totClus = 0;

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

      if (ncl < 7)
        continue;
      float ip[2];
      track.getImpactParams(0, 0, 0, 0, ip);

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();

      float phiTrack = trackParCov.getPhi() * 180 / M_PI;

      // if (pt < mPtCuts[0] || pt > mPtCuts[1]) continue;
      // if (eta < mEtaCuts[0] || eta > mEtaCuts[1]) continue;

      float phioriginal = 0;
      float phiduplicated = 0;

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]]; // cluster spacepoint in the tracking system
        auto staveOriginal = mGeometry->getStave(clusOriginal.getSensorID());
        auto chipOriginal = mGeometry->getChipIdInStave(clusOriginal.getSensorID());

        auto layer = mGeometry->getLayer(clusOriginal.getSensorID());
        if (layer >= NLAYERS)
          continue; // checking only selected layers

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;

        phioriginal = clusOriginalPointGlob.phi() * 180 / M_PI;

        mPhiOriginal[layer]->Fill(phioriginal);
        mPhiTrackOriginal[layer]->Fill(phiTrack);
        mPtOriginal[layer]->Fill(pt);
        mEtaOriginal[layer]->Fill(eta);
        m3DClusterPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y(), clusOriginalPointGlob.z());
        m2DClusterOriginalPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y());
      } // end loop on clusters
      totClus += ncl;
    } // end loop on tracks per ROF
  }   // end loop on ROFRecords array
  LOGP(info, "Total number of clusters: {} ", totClus);
}

void EfficiencyStudy::getEfficiency(bool isMC)
{
  // Extract the efficiency for the IB, exploiting the staves overlaps and the duplicated clusters for the tracks passing through the overlaps
  // The denominator for the efficiency calculation will be the number of tracks per layer fulfilling some cuts (DCA, phi, eta, pt)
  // The numerator will be the number of duplicated clusters for the tracks passing through the overlaps

  LOGP(info, "--------------- getEfficiency");

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  o2::gpu::gpustd::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
  auto propagator = o2::base::Propagator::Instance();

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;
  int nLabels = 0;
  unsigned int totClus = 0;

  int nbPt = 75;
  double xbins[nbPt + 1], ptcutl = 0.05, ptcuth = 7.5;
  double a = std::log(ptcuth / ptcutl) / nbPt;
  for (int i = 0; i <= nbPt; i++)
    xbins[i] = ptcutl * std::exp(i * a);

  int totNClusters;
  int nDuplClusters;

  // denominator fot the efficiency calculation
  for (unsigned int iROF = 0; iROF < mTracksROFRecords.size(); iROF++) { // loop on ROFRecords array

    rofIndexTrack = mTracksROFRecords[iROF].getFirstEntry();
    rofNEntriesTrack = mTracksROFRecords[iROF].getNEntries();

    rofIndexClus = mClustersROFRecords[iROF].getFirstEntry();
    rofNEntriesClus = mClustersROFRecords[iROF].getNEntries();

    ////// imposing cuts on the tracks = collecting tracks for the denominator
    for (unsigned int iTrack = rofIndexTrack; iTrack < rofIndexTrack + rofNEntriesTrack; iTrack++) { // loop on tracks per ROF
      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();
      float phi = -999.;
      float phiOriginal = -999.;

      float chi2 = track.getChi2();

      float ip[2];
      track.getImpactParams(0, 0, 0, 0, ip);

      float phiTrack = trackParCov.getPhi() * 180 / M_PI;

      // applying the cuts on the track - only pt and eta, and chi2 cuts since for phi(cluster) the layer is needed
      if (pt < mPtCuts[0] || pt > mPtCuts[1])
        continue;
      if (eta < mEtaCuts[0] || eta > mEtaCuts[1])
        continue;
      if (chi2 > mChi2cut)
        continue;

      /// the cut on phi, since it is layer-dependent, can be applied only after finding the cluster and then the layer

      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7)
        continue;

      o2::MCCompLabel tracklab;
      if (isMC) {
        tracklab = mTracksMCLabels[iTrack];
        if (tracklab.isFake())
          continue;
      }

      if (mVerboseOutput && isMC) {
        LOGP(info, "--------- track Label: ");
        tracklab.print();
      }

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track to extract layer, stave and chip to restrict the possible matches to be searched with the DCA cut
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]];
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());

        UShort_t rowOriginal = clusOriginal.getRow();

        if (layerOriginal >= NLAYERS) {
          continue;
        }

        IPOriginalxy[layerOriginal]->Fill(ip[0]);
        IPOriginalz[layerOriginal]->Fill(ip[1]);

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;
        // phiOriginal = std::(clusOriginalPointGlob.y(), clusOriginalPointGlob.x()) * 180 / M_PI + 180;
        phiOriginal = clusOriginalPointGlob.phi() * 180 / M_PI;

        mXoriginal->Fill(clusOriginalPointGlob.x());
        mYoriginal->Fill(clusOriginalPointGlob.y());
        mZoriginal->Fill(clusOriginalPointGlob.z());

        // std::cout<<" Layer: "<<layerOriginal<<" chipid: "<<clusOriginal.getChipID()<<" x: "<<clusOriginalPointGlob.x()<<" y: "<<clusOriginalPointGlob.y()<<" z: "<<clusOriginalPointGlob.z()<<std::endl;

        m2DClusterOriginalPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y());
        m3DClusterPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y(), clusOriginalPointGlob.z());

        /// applying the cuts on the phi of the original cluster
        bool keepTrack = false; /// wether or not a cluster is found in an eligible track in the corresponding layer
        if (layerOriginal == 0) {

          for (int i = 0; i < 10; i++) {
            if ((phiOriginal >= mPhiCutsL0[i][0] && phiOriginal <= mPhiCutsL0[i][1])) {
              keepTrack = true;
            }
          }
        }
        if (layerOriginal == 1) {
          for (int i = 0; i < 12; i++) {
            if ((phiOriginal >= mPhiCutsL1[i][0] && phiOriginal <= mPhiCutsL1[i][1])) {
              keepTrack = true;
            }
          }
        }
        if (layerOriginal == 2) {
          for (int i = 0; i < 17; i++) {
            if ((phiOriginal >= mPhiCutsL2[i][0] && phiOriginal <= mPhiCutsL2[i][1])) {
              keepTrack = true;
            }
          }
        }

        /////////////////////////////////////
        if (!(keepTrack))
          continue; /// if the track (cluster) is not eligible for any layer, go to the next one
        else {      /// fill the den and go ahead
          chi2trackAccepted->Fill(chi2);
          denPt[layerOriginal]->Fill(pt);
          denPhi[layerOriginal]->Fill(phiOriginal);
          denEta[layerOriginal]->Fill(eta);
          nTracksSelected[layerOriginal]++;
        }

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
        /// for each original cluster iterate over all the possible duplicated clusters to see first wether increment or not the denominator (if a track has a possible duplicated cluster in the selected phi region)
        /// then if the phi is within the cuts, select the "adjacent" clusters (stave +-1, chip =,+-1) and calculate the DCA with the track. Then choose the closest one.
        // std::cout<<"Loop on clusters 2"<<std::endl;
        for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF
          auto clusDuplicated = mClusters[iClus];

          auto clusDuplicatedPoint = mITSClustersArray[iClus];

          o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
          o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
          phi = clusDuplicatedPointGlob.phi() * 180 / M_PI;

          //// applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
          if (clusDuplicated.getSensorID() == clusOriginal.getSensorID())
            continue;
          auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
          if (layerDuplicated != layerOriginal)
            continue;
          auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
          if (abs(staveDuplicated - staveOriginal) != 1)
            continue;
          auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
          if (abs(chipDuplicated - chipOriginal) > 1)
            continue;

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
          // std::cout<<"DCA: "<<clusDuplicatedDCA[0]<<"  "<<clusDuplicatedDCA[1]<<"  (should be within ["<<mDCACutsXY[layerDuplicated][0]<<","<<mDCACutsXY[layerDuplicated][1]<<"] and ["<<mDCACutsZ[layerDuplicated][0]<<","<<mDCACutsZ[layerDuplicated][1]<<"])"<<std::endl;
          // std::cout<<"Point Duplicated (x,y,z): "<<clusDuplicatedPointGlob.x()<<"  "<<clusDuplicatedPointGlob.y()<<"  "<<clusDuplicatedPointGlob.z()<<std::endl;
          // std::cout<<"Point Original (x,y,z): "<<clusOriginalPointGlob.x()<<"  "<<clusOriginalPointGlob.y()<<"  "<<clusOriginalPointGlob.z()<<std::endl;
          // std::cout<<"Layer, chipid, stave : "<<layerDuplicated<<"  "<<chipDuplicated<<"  "<<staveDuplicated<<std::endl;
          // std::cout<<"Track position: "<<trackParCov.getX()<<"  "<<trackParCov.getY()<<"  "<<trackParCov.getZ()<<std::endl;
          DistanceClustersX[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.x() - clusOriginalPointGlob.x()));
          DistanceClustersY[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.y() - clusOriginalPointGlob.y()));
          DistanceClustersZ[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.z() - clusOriginalPointGlob.z()));

          // Imposing that the distance between the duplicated cluster and the track is less than x sigma
          if (!(clusDuplicatedDCA[0] > mDCACutsXY[layerDuplicated][0] && clusDuplicatedDCA[0] < mDCACutsXY[layerDuplicated][1] && clusDuplicatedDCA[1] > mDCACutsZ[layerDuplicated][0] && clusDuplicatedDCA[1] < mDCACutsZ[layerDuplicated][1])) {
            DCAxyRejected[layerDuplicated]->Fill(clusDuplicatedDCA[0]);
            DCAzRejected[layerDuplicated]->Fill(clusDuplicatedDCA[1]);
            continue;
          }

          m2DClusterDuplicatedPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y());
          m3DDuplicatedClusterPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y(), clusDuplicatedPointGlob.z());

          mXduplicated->Fill(clusDuplicatedPointGlob.x());
          mYduplicated->Fill(clusDuplicatedPointGlob.y());
          mZduplicated->Fill(clusDuplicatedPointGlob.z());

          IPOriginalifDuplicatedxy[layerOriginal]->Fill(ip[0]);
          IPOriginalifDuplicatedz[layerOriginal]->Fill(ip[1]);

          DistanceClustersXAftercuts[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.x() - clusOriginalPointGlob.x()));
          DistanceClustersYAftercuts[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.y() - clusOriginalPointGlob.y()));
          DistanceClustersZAftercuts[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.z() - clusOriginalPointGlob.z()));

          if (mVerboseOutput)
            LOGP(info, "Propagation ok");
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
          }
          adjacentFound = 1;
        } // end loop on all the clusters in the rof -> at this point we have the information on the closest cluster (if there is one)

        // here clusID_rDCA_label is updated with the closest cluster to the track other than the original one

        if (!adjacentFound)
          continue;
        nDuplClusters++;
        nDuplicatedClusters[layerOriginal]++;
        numPt[layerOriginal]->Fill(ptDuplicated);
        numPhi[layerOriginal]->Fill(phiDuplicated);
        numEta[layerOriginal]->Fill(etaDuplicated);
        mZvsPhiDUplicated[layerOriginal]->Fill(clusZ, phiDuplicated);

        // checking if it is a good or fake match looking at the labels (only if isMC)
        if (isMC) {
          bool isGood = false;
          for (auto lab : std::get<2>(clusID_rDCA_label)) {
            if (lab == tracklab) {
              isGood = true;
              numPtGood[layerOriginal]->Fill(ptDuplicated);
              numPhiGood[layerOriginal]->Fill(phiDuplicated);
              numEtaGood[layerOriginal]->Fill(etaDuplicated);
              continue;
            }
          }
          if (!isGood) {
            numPtFake[layerOriginal]->Fill(ptDuplicated);
            numPhiFake[layerOriginal]->Fill(phiDuplicated);
            numEtaFake[layerOriginal]->Fill(etaDuplicated);
          }
        }
      } // end loop on clusters associated to the track
      totNClusters += NLAYERS;
    } // end loop on tracks per ROF
  }   // end loop on ROFRecords array

  std::cout << " Num of duplicated clusters L0: " << nDuplicatedClusters[0] << " N tracks selected: " << nTracksSelected[0] << std::endl;
  std::cout << " Num of duplicated clusters L1: " << nDuplicatedClusters[1] << " N tracks selected: " << nTracksSelected[1] << std::endl;
  std::cout << " Num of duplicated clusters L2: " << nDuplicatedClusters[2] << " N tracks selected: " << nTracksSelected[2] << std::endl;

  std::cout << " --------- N total clusters: " << totNClusters << std::endl;
  std::cout << " --------- N duplicated clusters: " << nDuplClusters << std::endl;
}

void EfficiencyStudy::getEfficiencyAndTrackInfo(bool isMC)
{
  // Extract the efficiency for the IB, exploiting the staves overlaps and the duplicated clusters for the tracks passing through the overlaps
  // The denominator for the efficiency calculation will be the number of tracks per layer fulfilling some cuts (DCA, phi, eta, pt)
  // The numerator will be the number of duplicated clusters for the tracks passing through the overlaps
  // additionally, print/save info (to be used in MC)

  LOGP(info, "--------------- getEfficiency");

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  o2::gpu::gpustd::array<float, 2> clusOriginalDCA, clusDuplicatedDCA;
  auto propagator = o2::base::Propagator::Instance();

  unsigned int rofIndexTrack = 0;
  unsigned int rofNEntriesTrack = 0;
  unsigned int rofIndexClus = 0;
  unsigned int rofNEntriesClus = 0;
  int nLabels = 0;
  unsigned int totClus = 0;

  int nbPt = 75;
  double xbins[nbPt + 1], ptcutl = 0.05, ptcuth = 7.5;
  double a = std::log(ptcuth / ptcutl) / nbPt;
  for (int i = 0; i <= nbPt; i++)
    xbins[i] = ptcutl * std::exp(i * a);

  int totNClusters;
  int nDuplClusters;

  // denominator fot the efficiency calculation
  for (unsigned int iROF = 0; iROF < mTracksROFRecords.size(); iROF++) { // loop on ROFRecords array

    rofIndexTrack = mTracksROFRecords[iROF].getFirstEntry();
    rofNEntriesTrack = mTracksROFRecords[iROF].getNEntries();

    rofIndexClus = mClustersROFRecords[iROF].getFirstEntry();
    rofNEntriesClus = mClustersROFRecords[iROF].getNEntries();

    ////// imposing cuts on the tracks = collecting tracks for the denominator
    for (unsigned int iTrack = rofIndexTrack; iTrack < rofIndexTrack + rofNEntriesTrack; iTrack++) { // loop on tracks per ROF
      auto track = mTracks[iTrack];
      o2::track::TrackParCov trackParCov = mTracks[iTrack];

      auto pt = trackParCov.getPt();
      auto eta = trackParCov.getEta();
      float phi = -999.;
      float phiOriginal = -999.;

      float chi2 = track.getChi2();

      chi2track->Fill(chi2);

      float phiTrack = trackParCov.getPhi() * 180 / M_PI;

      // applying the cuts on the track - only pt and eta cuts since for phi(cluster) the layer is needed
      if (pt < mPtCuts[0] || pt > mPtCuts[1])
        continue;
      if (eta < mEtaCuts[0] || eta > mEtaCuts[1])
        continue;
      if (chi2 > mChi2cut)
        continue;
      /// the cut on phi, since it is layer-dependent, can be applied only after finding the cluster and then the layer

      int firstClus = track.getFirstClusterEntry(); // get the first cluster of the track
      int ncl = track.getNumberOfClusters();        // get the number of clusters of the track

      if (ncl < 7)
        continue;

      o2::MCCompLabel tracklab;
      if (isMC) {
        tracklab = mTracksMCLabels[iTrack];
        if (tracklab.isFake())
          continue;
      }

      if (mVerboseOutput && isMC) {
        LOGP(info, "--------- track Label: ");
        tracklab.print();
      }

      for (int iclTrack = firstClus; iclTrack < firstClus + ncl; iclTrack++) { // loop on clusters associated to the track to extract layer, stave and chip to restrict the possible matches to be searched with the DCA cut
        auto& clusOriginal = mClusters[mInputITSidxs[iclTrack]];
        auto clusOriginalPoint = mITSClustersArray[mInputITSidxs[iclTrack]];
        auto layerOriginal = mGeometry->getLayer(clusOriginal.getSensorID());

        UShort_t rowOriginal = clusOriginal.getRow();

        if (layerOriginal >= NLAYERS) {
          continue;
        }

        o2::math_utils::Point3D<float> clusOriginalPointTrack = {clusOriginalPoint.getX(), clusOriginalPoint.getY(), clusOriginalPoint.getZ()};
        o2::math_utils::Point3D<float> clusOriginalPointGlob = mGeometry->getMatrixT2G(clusOriginal.getSensorID()) * clusOriginalPointTrack;
        phiOriginal = clusOriginalPointGlob.phi() * 180 / M_PI;

        mXoriginal->Fill(clusOriginalPointGlob.x());
        mYoriginal->Fill(clusOriginalPointGlob.y());
        mZoriginal->Fill(clusOriginalPointGlob.z());

        m2DClusterOriginalPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y());
        m3DClusterPositions->Fill(clusOriginalPointGlob.x(), clusOriginalPointGlob.y(), clusOriginalPointGlob.z());

        /// applying the cuts on the phi of the original cluster
        bool keepTrack = false; /// wether or not a cluster is found in an eligible track in the corresponding layer

        if (layerOriginal == 0) {
          for (int i = 0; i < 10; i++) {
            if ((phiOriginal >= mPhiCutsL0[i][0] && phiOriginal <= mPhiCutsL0[i][1])) {
              keepTrack = true;
            }
          }
        }
        if (layerOriginal == 1) {
          for (int i = 0; i < 12; i++) {
            if ((phiOriginal >= mPhiCutsL1[i][0] && phiOriginal <= mPhiCutsL1[i][1])) {
              keepTrack = true;
            }
          }
        }
        if (layerOriginal == 2) {
          for (int i = 0; i < 17; i++) {
            if ((phiOriginal >= mPhiCutsL2[i][0] && phiOriginal <= mPhiCutsL2[i][1])) {
              keepTrack = true;
            }
          }
        }
        if (!(keepTrack))
          continue; /// if the track (cluster) is not eligible for any layer, go to the next one
        else {      /// fill the den and go ahead
          chi2trackAccepted->Fill(chi2);
          denPt[layerOriginal]->Fill(pt);
          denPhi[layerOriginal]->Fill(phiOriginal);
          denEta[layerOriginal]->Fill(eta);
          nTracksSelected[layerOriginal]++;
        }
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

        o2::MCCompLabel labelCandidateDuplicated;
        bool duplExists = false;

        /// for each original cluster iterate over all the possible duplicated clusters to see first wether increment or not the denominator (if a track has a possible duplicated cluster in the selected phi region)
        /// then if the phi is within the cuts, select the "adjacent" clusters (stave +-1, chip =,+-1) and calculate the DCA with the track. Then choose the closest one.
        for (unsigned int iClus = rofIndexClus; iClus < rofIndexClus + rofNEntriesClus; iClus++) { // iteration over ALL the clusters in the ROF
          auto clusDuplicated = mClusters[iClus];

          auto clusDuplicatedPoint = mITSClustersArray[iClus];

          o2::math_utils::Point3D<float> clusDuplicatedPointTrack = {clusDuplicatedPoint.getX(), clusDuplicatedPoint.getY(), clusDuplicatedPoint.getZ()};
          o2::math_utils::Point3D<float> clusDuplicatedPointGlob = mGeometry->getMatrixT2G(clusDuplicated.getSensorID()) * clusDuplicatedPointTrack;
          phi = clusDuplicatedPointGlob.phi() * 180 / M_PI;

          //// applying constraints: the cluster should be on the same layer, should be on an adjacent stave and on the same or adjacent chip position
          if (clusDuplicated.getSensorID() == clusOriginal.getSensorID())
            continue;
          auto layerDuplicated = mGeometry->getLayer(clusDuplicated.getSensorID());
          if (layerDuplicated != layerOriginal)
            continue;
          labelCandidateDuplicated = mClustersMCLCont->getLabels(iClus)[0];
          if (labelCandidateDuplicated == tracklab) {
            duplExists = true;
            std::cout << "Duplicated should exist with label: " << labelCandidateDuplicated.asString() << "  , phi = " << phi << " and be: ";
            clusDuplicated.print();
          }
          auto staveDuplicated = mGeometry->getStave(clusDuplicated.getSensorID());
          if (abs(staveDuplicated - staveOriginal) != 1)
            continue;
          auto chipDuplicated = mGeometry->getChipIdInStave(clusDuplicated.getSensorID());
          if (abs(chipDuplicated - chipOriginal) > 1)
            continue;

          std::cout << "checks passed" << std::endl;

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

          std::cout << "dca calculated: " << clusDuplicatedDCA[0] << "  " << clusDuplicatedDCA[1] << std::endl;

          DCAxyData[layerDuplicated]->Fill(clusDuplicatedDCA[0]);
          DCAzData[layerDuplicated]->Fill(clusDuplicatedDCA[1]);
          DistanceClustersX[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.x() - clusOriginalPointGlob.x()));
          DistanceClustersY[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.y() - clusOriginalPointGlob.y()));
          DistanceClustersZ[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.z() - clusOriginalPointGlob.z()));

          // Imposing that the distance between the duplicated cluster and the track is less than x sigma
          if (!(clusDuplicatedDCA[0] > mDCACutsXY[layerDuplicated][0] && clusDuplicatedDCA[0] < mDCACutsXY[layerDuplicated][1] && clusDuplicatedDCA[1] > mDCACutsZ[layerDuplicated][0] && clusDuplicatedDCA[1] < mDCACutsZ[layerDuplicated][1])) {
            DCAxyRejected[layerDuplicated]->Fill(clusDuplicatedDCA[0]);
            DCAzRejected[layerDuplicated]->Fill(clusDuplicatedDCA[1]);
            continue;
          }
          m2DClusterDuplicatedPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y());
          m3DDuplicatedClusterPositions->Fill(clusDuplicatedPointGlob.x(), clusDuplicatedPointGlob.y(), clusDuplicatedPointGlob.z());
          mXduplicated->Fill(clusDuplicatedPointGlob.x());
          mYduplicated->Fill(clusDuplicatedPointGlob.y());
          mZduplicated->Fill(clusDuplicatedPointGlob.z());

          DistanceClustersXAftercuts[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.x() - clusOriginalPointGlob.x()));
          DistanceClustersYAftercuts[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.y() - clusOriginalPointGlob.y()));
          DistanceClustersZAftercuts[layerDuplicated]->Fill(abs(clusDuplicatedPointGlob.z() - clusOriginalPointGlob.z()));

          if (mVerboseOutput)
            LOGP(info, "Propagation ok");
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
          }
          adjacentFound = 1;
          std::cout << "Duplicated found with label: " << labsDuplicated[0] << " and phi: " << phiDuplicated << std::endl;
          clusDuplicated.print();
          std::cout << "-----" << std::endl;
        } // end loop on all the clusters in the rof -> at this point we have the information on the closest cluster (if there is one)

        // here clusID_rDCA_label is updated with the closest cluster to the track other than the original one
        // checking if it is a good or fake match looking at the labels (only if isMC)
        if (!adjacentFound) {
          if (duplExists) {
            std::cout << "No duplicated found but should exist" << std::endl;
            std::cout << "DCA cuts were: xy-> " << mDCACutsXY[layerOriginal][0] << " to " << mDCACutsXY[layerOriginal][1] << " and z-> " << mDCACutsZ[layerOriginal][0] << " to " << mDCACutsZ[layerOriginal][1] << "\n-----" << std::endl;
          } else {
            std::cout << "No duplicated found and does not exist" << std::endl;
          }
          continue;
        }
        std::cout << "-----" << std::endl;
        nDuplClusters++;
        nDuplicatedClusters[layerOriginal]++;
        numPt[layerOriginal]->Fill(ptDuplicated);
        numPhi[layerOriginal]->Fill(phiDuplicated);
        numEta[layerOriginal]->Fill(etaDuplicated);
        mZvsPhiDUplicated[layerOriginal]->Fill(clusZ, phiDuplicated);

        if (isMC) {
          bool isGood = false;
          for (auto lab : std::get<2>(clusID_rDCA_label)) {
            if (lab == tracklab) {
              isGood = true;
              numPtGood[layerOriginal]->Fill(ptDuplicated);
              numPhiGood[layerOriginal]->Fill(phiDuplicated);
              numEtaGood[layerOriginal]->Fill(etaDuplicated);
              continue;
            }
          }
          if (!isGood) {
            numPtFake[layerOriginal]->Fill(ptDuplicated);
            numPhiFake[layerOriginal]->Fill(phiDuplicated);
            numEtaFake[layerOriginal]->Fill(etaDuplicated);
          }
        }
      } // end loop on clusters associated to the track
      totNClusters += NLAYERS;
    } // end loop on tracks per ROF
  }   // end loop on ROFRecords array

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
    // studyDCAcutsMC();
    // studyClusterSelectionMC();
    // getEfficiencyAndTrackInfo(mUseMC);
    // countDuplicatedAfterCuts();
  } else if (!mUseMC) {
    // saveDataInfo();
  }

  getEfficiency(mUseMC);

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

  mOutFile->mkdir("DistanceClusters/");
  mOutFile->mkdir("DCA/");
  mOutFile->mkdir("Pt_Eta_Phi/");

  if (mUseMC) {

    mOutFile->cd("DistanceClusters");
    for (int i = 0; i < NLAYERS; i++) {
      mDistanceClustersX[i]->Write();
      mDistanceClustersY[i]->Write();
      mDistanceClustersZ[i]->Write();
      mDistanceClusters[i]->Write();
    }

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
      mPhiOriginal[i]->Write();
      mPhiTrackOriginal[i]->Write();
      mDuplicatedPhiAllPt[i]->Write();
      mPtOriginal[i]->Write();
      mPtDuplicated[i]->Write();
      mEtaDuplicated[i]->Write();
      mPhiDuplicated[i]->Write();
      mPhiTrackDuplicated[i]->Write();
      mPhiTrackDuplicatedvsphiDuplicated[i]->Write();
      mPhiTrackoriginalvsphioriginal[i]->Write();
      mPhiOriginalIfDuplicated[i]->Write();
      mDuplicatedPt[i]->Write();
      mDuplicatedPtEta[i]->Write();
      mDuplicatedPtPhi[i]->Write();
      mDuplicatedEtaPhi[i]->Write();
      mEtaOriginal[i]->Write();
      mDuplicatedEtaAllPt[i]->Write();
      mDuplicatedRow[i]->Write();

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
    mPhiTrackOriginal[i]->Write();
    mPtOriginal[i]->Write();
    mEtaOriginal[i]->Write();
    mZvsPhiDUplicated[i]->Write();
    chipRowDuplicated[i]->Write();
    chipRowOriginalIfDuplicated[i]->Write();
  }

  mOutFile->mkdir("chi2");
  mOutFile->cd("chi2/");

  chi2track->Write();
  chi2trackAccepted->Write();

  mOutFile->cd("EfficiencyFinal/");

  for (int l = 0; l < NLAYERS; l++) {

    TEfficiency* effPt = new TEfficiency(*numPt[l], *denPt[l]);
    effPt->SetName(Form("effPt_layer%d", l));
    effPt->SetTitle(Form("L%d;p_{T} (GeV/c);Efficiency", l));
    TEfficiency* effPtGood = new TEfficiency(*numPtGood[l], *denPt[l]);
    effPtGood->SetName(Form("effPtGood_layer%d", l));
    effPtGood->SetTitle(Form("L%d;p_{T} (GeV/c);Efficiency Good Matches", l));
    TEfficiency* effPtFake = new TEfficiency(*numPtFake[l], *denPt[l]);
    effPtFake->SetName(Form("effPtFake_layer%d", l));
    effPtFake->SetTitle(Form("L%d;p_{T} (GeV/c);Efficiency Fake Matches", l));
    effPt->Write();
    effPtGood->Write();
    effPtFake->Write();

    TEfficiency* effPhi = new TEfficiency(*numPhi[l], *denPhi[l]);
    effPhi->SetName(Form("effPhi_layer%d", l));
    effPhi->SetTitle(Form("L%d;#phi;Efficiency", l));
    TEfficiency* effPhiGood = new TEfficiency(*numPhiGood[l], *denPhi[l]);
    effPhiGood->SetName(Form("effPhiGood_layer%d", l));
    effPhiGood->SetTitle(Form("L%d;#phi;Efficiency Good Matches", l));
    TEfficiency* effPhiFake = new TEfficiency(*numPhiFake[l], *denPhi[l]);
    effPhiFake->SetName(Form("effPhiFake_layer%d", l));
    effPhiFake->SetTitle(Form("L%d;#phi;Efficiency Fake Matches", l));
    effPhi->Write();
    effPhiGood->Write();
    effPhiFake->Write();

    TEfficiency* effEta = new TEfficiency(*numEta[l], *denEta[l]);
    effEta->SetName(Form("effEta_layer%d", l));
    effEta->SetTitle(Form("L%d;#eta;Efficiency", l));
    TEfficiency* effEtaGood = new TEfficiency(*numEtaGood[l], *denEta[l]);
    effEtaGood->SetName(Form("effEtaGood_layer%d", l));
    effEtaGood->SetTitle(Form("L%d;#eta;Efficiency Good Matches", l));
    TEfficiency* effEtaFake = new TEfficiency(*numEtaFake[l], *denEta[l]);
    effEtaFake->SetName(Form("effEtaFake_layer%d", l));
    effEtaFake->SetTitle(Form("L%d;#eta;Efficiency Fake Matches", l));
    effEta->Write();
    effEtaGood->Write();
    effEtaFake->Write();

    numPhi[l]->Write();
    denPhi[l]->Write();
    numPt[l]->Write();
    denPt[l]->Write();
    numEta[l]->Write();
    denEta[l]->Write();
  }

  mOutFile->cd("DCAFinal/");

  for (int l = 0; l < NLAYERS; l++) {
    DCAxyData[l]->Write();
    DCAzData[l]->Write();
    DistanceClustersX[l]->Write();
    DistanceClustersY[l]->Write();
    DistanceClustersZ[l]->Write();
    DistanceClustersXAftercuts[l]->Write();
    DistanceClustersYAftercuts[l]->Write();
    DistanceClustersZAftercuts[l]->Write();
    DCAxyRejected[l]->Write();
    DCAzRejected[l]->Write();
  }

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