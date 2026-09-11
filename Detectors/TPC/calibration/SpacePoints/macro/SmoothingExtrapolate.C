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

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <algorithm>
#include <filesystem>
#include <string>
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TF1.h"
#include "TGraph.h"
#include "TProfile.h"
#include "Math/MinimizerOptions.h"

#include "Algorithm/RangeTokenizer.h"
#include "SpacePoints/TrackResiduals.h"
#include "TPCCalibration/TPCFastSpaceChargeCorrectionHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "TPCCalibration/TPCScaler.h"
#if __has_include("TPCBaseRecSim/CDBTypes.h")
#include "TPCBaseRecSim/CDBTypes.h"
#else
#include "TPCBase/CDBTypes.h"
#endif

#endif

using namespace o2::tpc;
using namespace o2::gpu;

//------------------------------------------------------------------------------------------------------------
static const Float_t RowX[153] =
  {
    85.225, 85.975, 86.725, 87.475, 88.225, 88.975, 89.725, 90.475, 91.225, 91.975, 92.725, 93.475, 94.225, 94.975, 95.725, 96.475,
    97.225, 97.975, 98.725, 99.475, 100.225, 100.975, 101.725, 102.475, 103.225, 103.975, 104.725, 105.475, 106.225, 106.975, 107.725,
    108.475, 109.225, 109.975, 110.725, 111.475, 112.225, 112.975, 113.725, 114.475, 115.225, 115.975, 116.725, 117.475, 118.225, 118.975,
    119.725, 120.475, 121.225, 121.975, 122.725, 123.475, 124.225, 124.975, 125.725, 126.475, 127.225, 127.975, 128.725, 129.475, 130.225,
    130.975, 131.725, 135.200, 136.200, 137.200, 138.200, 139.200, 140.200, 141.200, 142.200, 143.200, 144.200, 145.200, 146.200, 147.200,
    148.200, 149.200, 150.200, 151.200, 152.200, 153.200, 154.200, 155.200, 156.200, 157.200, 158.200, 159.200, 160.200, 161.200, 162.200,
    163.200, 164.200, 165.200, 166.200, 167.200, 168.200, 171.400, 172.600, 173.800, 175.000, 176.200, 177.400, 178.600, 179.800, 181.000,
    182.200, 183.400, 184.600, 185.800, 187.000, 188.200, 189.400, 190.600, 191.800, 193.000, 194.200, 195.400, 196.600, 197.800, 199.000,
    200.200, 201.400, 202.600, 203.800, 205.000, 206.200, 209.650, 211.150, 212.650, 214.150, 215.650, 217.150, 218.650, 220.150, 221.650,
    223.150, 224.650, 226.150, 227.650, 229.150, 230.650, 232.150, 233.650, 235.150, 236.650, 238.150, 239.650, 241.150, 242.650, 244.150,
    245.650, 246.650}; // last value added
//------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
Double_t PolyFitFunc(Double_t* x_val, Double_t* par)
{
  Double_t x, y, par0, par1, par2, par3, par4, par5;
  par0 = par[0];
  par1 = par[1];
  par2 = par[2];
  par3 = par[3];
  par4 = par[4];
  par5 = par[5];
  x = x_val[0];
  y = par0 + par1 * x + par2 * x * x + par3 * x * x * x + par4 * x * x * x * x + par5 * x * x * x * x * x;
  return y;
}
//----------------------------------------------------------------------------------------

// Function to create Gaussian filter
void vec_FilterCreation(std::vector<std::vector<std::vector<Double_t>>>& vec_GKernel, Int_t Delta_X, Int_t Delta_Y, Int_t Delta_Z, Double_t sigma)
{
  // initialising standard deviation to 1.0
  // double sigma = 1.0;
  double r, s = 2.0 * sigma * sigma;

  // sum is for normalization
  double sum = 0.0;

  // generating 5x5 kernel
  for (int x = -Delta_X; x <= Delta_X; x++) {
    for (int y = -Delta_Y; y <= Delta_Y; y++) {
      for (int z = -Delta_Z; z <= Delta_Z; z++) {
        r = sqrt(x * x + y * y + z * z);
        vec_GKernel[x + Delta_X][y + Delta_Y][z + Delta_Z] = (exp(-(r * r) / s)) / (M_PI * s);
        sum += vec_GKernel[x + Delta_X][y + Delta_Y][z + Delta_Z];
      }
    }
  }

  // normalising the Kernel
  for (int i = 0; i < (Delta_X * 2 + 1); ++i) {
    for (int j = 0; j < (Delta_Y * 2 + 1); ++j) {
      for (int k = 0; k < (Delta_Z * 2 + 1); ++k) {
        vec_GKernel[i][j][k] /= sum;
      }
    }
  }
}

// Mean TPC scaler (IDC) values for one timestamp, from the standard CCDB CalScaler/CalScalerWeights
// objects. Used for the offline IDC join below -- see the "Offline IDC join" block in
// SmoothingExtrapolate() for why this is done here rather than during map creation.
bool getScalerValues(o2::ccdb::BasicCCDBManager& ccdbmgr, long tfTimeInMS, float& scA, float& scC)
{
  auto* scalerTree = ccdbmgr.getForTimeStamp<TTree>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalScaler), long(std::ceil(tfTimeInMS)));
  auto* scalerWeights = ccdbmgr.getForTimeStamp<TPCScalerWeights>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalScalerWeights), long(std::ceil(tfTimeInMS)));

  if (!scalerTree) {
    LOGP(error, "Could not get 'TPC/Calib/Scaler' for time stamp {}", tfTimeInMS);
    return false;
  }
  // The caller sets setFatalWhenNull(false), so a missing object comes back as nullptr rather than
  // aborting -- the weights have to be checked before being dereferenced below.
  if (!scalerWeights) {
    LOGP(error, "Could not get 'TPC/Calib/ScalerWeights' for time stamp {}", tfTimeInMS);
    return false;
  }

  o2::tpc::TPCScaler scaler;
  scaler.setFromTree(*(scalerTree));
  scaler.setScalerWeights(*scalerWeights);
  scaler.useWeights(true);
  scaler.setIonDriftTimeMS(500);

  static bool defaultScalerReported = false;
  static bool badScalerValueReported = false;
  if (scaler.getRun() == 0) {
    if (!defaultScalerReported) {
      LOGP(error, "Retrieved default scaler entry 'TPC/Calib/Scaler' for time stamp {}", tfTimeInMS);
      defaultScalerReported = true;
    }
    return false;
  }

  scA = scaler.getMeanScaler(tfTimeInMS, o2::tpc::Side::A);
  scC = scaler.getMeanScaler(tfTimeInMS, o2::tpc::Side::C);

  if ((scA <= 0) || (scC <= 0)) {
    if (!badScalerValueReported) {
      LOGP(error, "Bad scaler value, first seen for time stamp {}, scA: {}, scC: {}", tfTimeInMS, scA, scC);
      badScalerValueReported = true;
    }
    scA = 0;
    scC = 0;
    return false;
  }

  return true;
}

void SmoothingExtrapolate(const char* fileName = "debugVoxRes.root", TString fileOutName = "SmoothVoxRes.root",
                          int do_smoothing = 1, int do_extrapolation = 2,
                          int A11maxZ2X = -1, bool maskIA11 = false,
                          Int_t N_bins_X_GF = 1, Int_t N_bins_Y_GF = 2,
                          Int_t N_bins_Z_GF = 0, Float_t sigma_GF = 1.2)
{

  // do_extrapolation:
  // 0 -> no extrapolation to low radii done
  // 1 -> only smoothed values are extrapolated
  // 2 -> smoothed and raw values are extrapolated
  // 3 -> only raw values are extrapolated

  // Example, smoothing and extrapolating both smoothed and raw values:
  // SmoothingExtrapolate("voxRes.<run>_<from>_<to>.it0.root", "voxRes.<run>_smooth.root", 1, 2, -1, false, 1, 2, 0, 1.2)

  const float maxDeltaCut = 25.0; // maximum value for any Delta to be accepted
  const float min_statistics = 20;
  const float max_extrapolation_value = 20.0; // maximum value for extrapolation in DX, DY, DZ
  //----------------------------------------------------------------
  // input
  if (gSystem->AccessPathName(fileName)) {
    LOGP(error, "input file {} does not exist", fileName);
    return;
  }

  auto file = std::unique_ptr<TFile>(TFile::Open(fileName, "READ"));
  if (!file || !file->IsOpen()) {
    LOGP(error, "input file {} does not exist", fileName);
    return;
  }

  TTree* voxResTree = nullptr;
  file->cd();
  gDirectory->GetObject("voxResTree", voxResTree);
  if (!voxResTree) {
    LOGP(error, "tree voxResTree does not exist in {}", fileName);
    return;
  }

  o2::tpc::TrackResiduals::VoxRes* voxRes_map = nullptr;
  Long64_t entries_input_map = voxResTree->GetEntries();
  LOGP(info, "entries_input_map: {}", entries_input_map);
  voxResTree->SetBranchAddress("voxRes", &voxRes_map);

  // required for the binning that was used
  auto userInfo = voxResTree->GetUserInfo();
  if (!userInfo->FindObject("y2xBinning") || !userInfo->FindObject("z2xBinning")) {
    LOGP(error, "'y2xBinning' or 'z2xBinning' not found in UserInfo, but required to get the correct binning");
    return;
  }

  // Obtain configuration
  const SpacePointsCalibConfParam& params = SpacePointsCalibConfParam::Instance();
  if (std::filesystem::exists("scdconfig.ini")) {
    params.updateFromFile("scdconfig.ini");
  }
  // TrackResiduals::setZ2XBinning() (called below) reads scdcalib.maxZ2X directly and uses it to scale
  // the physical z/x bin boundaries -- it is baked into what each z2x voxel index in the input tree
  // actually means, not a cosmetic knob. There is no scdconfig.ini on the GRID, so without this the
  // code default (1.0) would silently apply instead of whatever stage 1 actually used (production 1.4),
  // misaligning this macro's re-derived binning against the tree's real geometry. Must happen BEFORE
  // setZ2XBinning() below. See staticMapCreatorCPM.C's UserInfo::Add("maxZ2X", ...) for where this comes
  // from.
  if (auto* maxZ2XObj = userInfo->FindObject("maxZ2X")) {
    const std::string maxZ2XStr = maxZ2XObj->GetTitle();
    o2::conf::ConfigurableParam::setValue("scdcalib.maxZ2X", maxZ2XStr);
    LOGP(info, "Set scdcalib.maxZ2X = {} from input UserInfo (matches stage 1)", maxZ2XStr);
  } else {
    LOGP(warning,
         "'maxZ2X' not found in input UserInfo (older input file?) -- using scdcalib.maxZ2X = {} "
         "(scdconfig.ini/code default), which may NOT match the value stage 1 actually used to "
         "build this tree's z2x binning!",
         params.maxZ2X);
  }

  LOGP(info, "----- Dumping configuration values START -----");
  params.printKeyValues();
  LOGP(info, "----- Dumping configuration values END -----");

  LOGP(info, "Get binning from userInfo");
  o2::tpc::TrackResiduals trackResiduals;
  auto y2xBins = o2::RangeTokenizer::tokenize<float>(userInfo->FindObject("y2xBinning")->GetTitle());
  auto z2xBins = o2::RangeTokenizer::tokenize<float>(userInfo->FindObject("z2xBinning")->GetTitle());
  trackResiduals.setY2XBinning(y2xBins);
  trackResiduals.setZ2XBinning(z2xBins);
  trackResiduals.init();
  LOGP(info, "trackResiduals initialized");
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // Offline IDC join: staticMapCreatorCPM.C on the GRID intentionally does not access IDCs
  // live (useCTPLumi=2 -- CCDB access from inside the hot per-TF loop was slow/unreliable), so
  // meanIDC/medianIDC in the input's UserInfo are placeholder 0. Per-TF timestamps are recorded
  // regardless (the "OrbitLumiInfo" tree's timeMSsel branch) specifically so this can be joined back
  // offline, with a properly-cached CCDB client. Timestamps are sorted before querying:
  // TPCScaler/TPCScalerWeights CCDB objects are valid over a time range, and BasicCCDBManager's cache
  // (setCaching(true)) only hits when consecutive queries land in the same validity window --
  // unsorted access would bounce between windows and force a real CCDB fetch almost every call.
  float meanIDCReal = 0.f;
  float medianIDCReal = 0.f;
  {
    TTree* orbitLumiTree = nullptr;
    file->GetObject("OrbitLumiInfo", orbitLumiTree);
    if (!orbitLumiTree || orbitLumiTree->GetEntries() == 0) {
      LOGP(warning, "Offline IDC join: no 'OrbitLumiInfo' tree (or it's empty) in the input file -- meanIDC/medianIDC stay 0");
    } else {
      std::vector<long>* timeMSselPtr = nullptr;
      orbitLumiTree->SetBranchAddress("timeMSsel", &timeMSselPtr);
      orbitLumiTree->GetEntry(0);
      if (!timeMSselPtr || timeMSselPtr->empty()) {
        LOGP(warning, "Offline IDC join: 'timeMSsel' branch missing or empty -- meanIDC/medianIDC stay 0");
      } else {
        std::vector<long> sortedTimes(*timeMSselPtr);
        std::sort(sortedTimes.begin(), sortedTimes.end());

        auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
        ccdbmgr.setCaching(true);
        ccdbmgr.setFatalWhenNull(false);
        ccdbmgr.setURL("http://alice-ccdb.cern.ch");

        std::vector<float> averageIDCs;
        averageIDCs.reserve(sortedTimes.size());
        for (long tfTimeInMS : sortedTimes) {
          float scA = 0.f, scC = 0.f;
          if (getScalerValues(ccdbmgr, tfTimeInMS, scA, scC)) {
            averageIDCs.emplace_back((scA + scC) / 2.f);
          }
        }
        if (averageIDCs.empty()) {
          LOGP(warning, "Offline IDC join: no valid scaler values found for any of {} TFs -- meanIDC/medianIDC stay 0", sortedTimes.size());
        } else {
          double sum = 0.0;
          for (float v : averageIDCs) {
            sum += v;
          }
          meanIDCReal = static_cast<float>(sum / averageIDCs.size());
          medianIDCReal = static_cast<float>(TMath::Median(static_cast<Long64_t>(averageIDCs.size()), averageIDCs.data()));
          LOGP(info, "Offline IDC join: {} of {} TFs gave a valid scaler value, meanIDC={}, medianIDC={}",
               averageIDCs.size(), sortedTimes.size(), meanIDCReal, medianIDCReal);
        }
      }
    }
  }
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // Get voxel map binning from map

  const int nXBins = trackResiduals.getNXBins();
  const int nY2XBins = trackResiduals.getNY2XBins();
  const int nZ2XBins = trackResiduals.getNZ2XBins();
  LOGP(info, "binning X,Y2X,Z2X: {}, {}, {}", nXBins, nY2XBins, nZ2XBins);
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // output
  // Create a new file + a clone of old tree in new file
  TFile* outputfile = new TFile(fileOutName.Data(), "RECREATE");
  LOGP(info, "Output file: {} created", fileOutName.Data());

  o2::tpc::TrackResiduals::VoxRes mVoxelResultsOut{};                      ///< the results from mVoxelResults are copied in here to be able to stream them
  o2::tpc::TrackResiduals::VoxRes* mVoxelResultsOutPtr{&mVoxelResultsOut}; ///< pointer to set the branch address to for the output
  std::unique_ptr<TTree> mTreeOut;
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("GSLSimAn");

  TF1* func_PolyFitFunc = new TF1("func_PolyFitFunc", PolyFitFunc, 0, 150, 6);
  TF1* func_PolyFitFunc_raw = new TF1("func_PolyFitFunc_raw", PolyFitFunc, 0, 150, 6);
  TProfile* tp_DX_vs_X_raw = new TProfile("tp_DX_vs_X_raw", "tp_DX_vs_X_raw", 250, 0, 250);
  TProfile* tp_Stat_vs_X = new TProfile("tp_Stat_vs_X", "tp_Stat_vs_X;row;<entries>", 250, 0, 250);
  TProfile* tp_Stat_vs_X_single = new TProfile("tp_Stat_vs_X_single", "tp_Stat_vs_X;row;<entries>", 250, 0, 250);
  TProfile* tp_DX_vs_X_single = new TProfile("tp_DX_vs_X_single", "tp_DX_vs_X;row;<entries>", 250, 0, 250);
  TGraph* tg_Stat_vs_X_slice = new TGraph();
  TProfile* tp_DX_vs_Row_raw = new TProfile("tp_DX_vs_Row_raw", "tp_DX_vs_Row_raw", 250, 0, 250);
  TProfile* tp_DX_vs_X_smooth = new TProfile("tp_DX_vs_X_smooth", "tp_DX_vs_X_smooth", 250, 0, 250);
  TProfile* tp_DX_vs_X_smooth_extr = new TProfile("tp_DX_vs_X_smooth_extr", "tp_DX_vs_X_smooth_extr", 250, 0, 250);
  TProfile* tp_DY_vs_X_smooth_extr = new TProfile("tp_DY_vs_X_smooth_extr", "tp_DY_vs_X_smooth_extr", 250, 0, 250);
  TProfile* tp_DZ_vs_X_smooth_extr_A = new TProfile("tp_DZ_vs_X_smooth_extr_A", "tp_DZ_vs_X_smooth_extr_A", 250, 0, 250);
  TProfile* tp_DZ_vs_X_smooth_extr_C = new TProfile("tp_DZ_vs_X_smooth_extr_C", "tp_DZ_vs_X_smooth_extr_C", 250, 0, 250);
  TProfile* tp_DZ_vs_X_smooth_A = new TProfile("tp_DZ_vs_X_smooth_A", "tp_DZ_vs_X_smooth_A", 250, 0, 250);
  TProfile* tp_DZ_vs_X_smooth_C = new TProfile("tp_DZ_vs_X_smooth_C", "tp_DZ_vs_X_smooth_C", 250, 0, 250);
  TProfile* tp_DZ_vs_X_raw_A = new TProfile("tp_DZ_vs_X_raw_A", "tp_DZ_vs_X_raw_A", 250, 0, 250);
  TProfile* tp_DZ_vs_X_raw_C = new TProfile("tp_DZ_vs_X_raw_C", "tp_DZ_vs_X_raw_C", 250, 0, 250);
  TProfile* tp_Stat_vs_row = new TProfile("tp_Stat_vs_row", "tp_Stat_vs_row;row;<entries>", 152, 0, 152);
  int n_bins_z_phi_sector = 36 * nY2XBins * nZ2XBins;
  TH1D* h_x_start_fit_vs_z_phi_sector = new TH1D("h_x_start_fit_vs_z_phi_sector", "h_x_start_fit_vs_z_phi_sector", n_bins_z_phi_sector, 0, n_bins_z_phi_sector);
  //----------------------------------------------------------------

  //--------------------------------------------------------------------------
  // Prepare output data
  std::vector<std::vector<std::vector<std::vector<std::vector<Float_t>>>>> vec_DXYZ_vox;
  std::vector<std::vector<std::vector<std::vector<std::vector<Float_t>>>>> vec_DXYZ_vox_GF;
  vec_DXYZ_vox.resize(6);
  vec_DXYZ_vox_GF.resize(6);
  for (Int_t i_xyz = 0; i_xyz < 6; i_xyz++) {
    vec_DXYZ_vox[i_xyz].resize(36);
    vec_DXYZ_vox_GF[i_xyz].resize(36);
    for (Int_t i_sector = 0; i_sector < 36; i_sector++) {
      vec_DXYZ_vox[i_xyz][i_sector].resize(152);
      vec_DXYZ_vox_GF[i_xyz][i_sector].resize(152);
      for (Int_t voxX = 0; voxX < 152; voxX++) {
        vec_DXYZ_vox[i_xyz][i_sector][voxX].resize((nY2XBins));
        vec_DXYZ_vox_GF[i_xyz][i_sector][voxX].resize((nY2XBins));
        for (Int_t voxY = 0; voxY < (nY2XBins); voxY++) {
          vec_DXYZ_vox[i_xyz][i_sector][voxX][voxY].resize((nZ2XBins));
          vec_DXYZ_vox_GF[i_xyz][i_sector][voxX][voxY].resize((nZ2XBins));
          for (Int_t voxZ = 0; voxZ < (nZ2XBins); voxZ++) {
            vec_DXYZ_vox[i_xyz][i_sector][voxX][voxY][voxZ] = 0.0;
            vec_DXYZ_vox_GF[i_xyz][i_sector][voxX][voxY][voxZ] = 0.0;
          }
        }
      }
    }
  }
  std::vector<std::vector<std::vector<Float_t>>> vec_max_X_fit;
  vec_max_X_fit.resize(36);
  for (Int_t i_sector = 0; i_sector < 36; i_sector++) {
    vec_max_X_fit[i_sector].resize(nY2XBins);
    for (Int_t voxY = 0; voxY < (nY2XBins); voxY++) {
      vec_max_X_fit[i_sector][voxY].resize(nZ2XBins);
      for (Int_t voxZ = 0; voxZ < (nZ2XBins); voxZ++) {
        vec_max_X_fit[i_sector][voxY][voxZ] = 0.0;
      }
    }
  }

  for (Long64_t jentry = 0; jentry < entries_input_map; jentry++) {
    voxResTree->GetEntry(jentry);

    const auto bvox_X = voxRes_map->bvox[o2::tpc::TrackResiduals::VoxX]; // bin number in x (= pad row)
    const auto bvox_F = voxRes_map->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
    const auto bvox_Z = voxRes_map->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4
    const int sector = (int)voxRes_map->bsec;
    const float xAV = voxRes_map->stat[o2::tpc::TrackResiduals::VoxX];
    const float z2xAV = voxRes_map->stat[o2::tpc::TrackResiduals::VoxZ];
    const float zAV = z2xAV * xAV;

    vec_DXYZ_vox[0][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->D[0];    // dX
    vec_DXYZ_vox[1][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->D[1];    // dY
    vec_DXYZ_vox[2][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->D[2];    // dZ
    vec_DXYZ_vox[3][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->stat[3]; // #entries
    vec_DXYZ_vox[4][sector][bvox_X][bvox_F][bvox_Z] = xAV;                 // xAV
    vec_DXYZ_vox[5][sector][bvox_X][bvox_F][bvox_Z] = zAV;                 // zAV

    vec_DXYZ_vox_GF[0][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->DS[0];   // dXS
    vec_DXYZ_vox_GF[1][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->DS[1];   // dYS
    vec_DXYZ_vox_GF[2][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->DS[2];   // dZS
    vec_DXYZ_vox_GF[3][sector][bvox_X][bvox_F][bvox_Z] = voxRes_map->stat[3]; // #entries
    vec_DXYZ_vox_GF[4][sector][bvox_X][bvox_F][bvox_Z] = xAV;                 // xAV
    vec_DXYZ_vox_GF[5][sector][bvox_X][bvox_F][bvox_Z] = zAV;                 // zAV

    tp_Stat_vs_row->Fill(bvox_X, voxRes_map->stat[3]);
    float voxX_pos = RowX[bvox_X];
    if (fabs(bvox_F - (int)(nY2XBins / 2)) <= 1) {
      tp_Stat_vs_X->Fill(voxX_pos, voxRes_map->stat[3]);
    }
    if (bvox_Z == 0) {
      // if(voxX_pos < (85.0+32.0))
      {
        tp_DX_vs_X_raw->Fill(voxX_pos, voxRes_map->D[0]);
        tp_DX_vs_Row_raw->Fill(bvox_X, voxRes_map->D[0]);
        if (sector < 18) {
          tp_DZ_vs_X_raw_A->Fill(voxX_pos, voxRes_map->D[2]);
        } else {
          tp_DZ_vs_X_raw_C->Fill(voxX_pos, voxRes_map->D[2]);
        }
      }
    }
  }

  //--------------------------------------------------------------------------------
  tp_Stat_vs_row->GetXaxis()->SetRangeUser(20, 62);
  const float meanEntries = tp_Stat_vs_row->GetMean(2);
  tp_Stat_vs_row->GetXaxis()->SetRangeUser(2, 1);
  const int startRowGoodEntries = tp_Stat_vs_row->FindFirstBinAbove(meanEntries * 0.7) - 1;

  // don't trust bins with too low statistics
  tp_DX_vs_X_raw->GetXaxis()->SetRangeUser(RowX[startRowGoodEntries], RowX[63]);
  // const float max_DX   = tp_DX_vs_X_raw ->GetBinContent(tp_DX_vs_X_raw->GetMaximumBin());
  // const float max_X_DX = tp_DX_vs_X_raw ->GetBinCenter(tp_DX_vs_X_raw->GetMaximumBin());
  tp_DX_vs_Row_raw->GetXaxis()->SetRangeUser(startRowGoodEntries, 63);
  // const int max_Row_DX = tp_DX_vs_Row_raw ->GetBinCenter(tp_DX_vs_Row_raw->GetMaximumBin());

  float max_X_stat = 0.0;
  for (int ibin = (tp_Stat_vs_X->GetNbinsX() - 3); ibin >= 0; ibin--) {
    float X_val = tp_Stat_vs_X->GetBinCenter(ibin);
    float stat = tp_Stat_vs_X->GetBinContent(ibin);
    float DX = tp_DX_vs_X_raw->GetBinContent(ibin);
    if (X_val < (85.0 + 32.0)) {
      Double_t stat_previous[3] = {tp_Stat_vs_X->GetBinContent(ibin + 1), tp_Stat_vs_X->GetBinContent(ibin + 2), tp_Stat_vs_X->GetBinContent(ibin + 3)};
      Double_t Xpos_previous[3] = {tp_Stat_vs_X->GetBinCenter(ibin + 1), tp_Stat_vs_X->GetBinCenter(ibin + 2), tp_Stat_vs_X->GetBinCenter(ibin + 3)};
      if ((stat - stat_previous[0]) < 0.0 && (stat - stat_previous[1]) < 0.0 && (stat - stat_previous[2]) < 0.0 && stat > 0.0) {
        Double_t ratio_stat[3] = {stat / stat_previous[0], stat / stat_previous[1], stat / stat_previous[2]};
        if (ratio_stat[0] < 0.8 && ratio_stat[1] < 0.5 && ratio_stat[2] < 0.5) {
          max_X_stat = Xpos_previous[2];
          break;
        }
      }
      if (fabs(stat < 0.1)) {
        max_X_stat = Xpos_previous[2];
        break;
      }
    }
  }

  const float max_X_DX = max_X_stat;
  int max_Row_DX = 0;

  // find corresponding row
  for (int i = 0; i < 50; ++i) {
    if (RowX[i] > max_X_DX) {
      break;
    }
    max_Row_DX = i;
  }
  const float max_DX = tp_DX_vs_X_raw->GetBinContent(tp_DX_vs_X_raw->FindBin(max_X_DX));

  LOGP(info, "max DX value: {:.3f}, X-position of max DX value: {:.3f} (row: {}), first row checked: {}, mean entries: {:.2f}, max_X_stat: {:.3f}", max_DX, max_X_DX, max_Row_DX, startRowGoodEntries, meanEntries, max_X_stat);
  //--------------------------------------------------------------------------------

  //--------------------------------------------------------------------------------
  LOGP(info, "Calculating extrapolation fit start values for every phi slice");
  for (Int_t i_sector = 0; i_sector < 36; i_sector++) {
    for (Int_t voxY = 0; voxY < (nY2XBins); voxY++) {
      for (Int_t voxZ = 0; voxZ < (nZ2XBins); voxZ++) {
        // fill the TGraph used for fitting
        int ipoint = 0;
        for (Int_t voxX = 0; voxX <= 151; voxX++) {
          float voxX_pos = RowX[voxX];
          float statistics = vec_DXYZ_vox_GF[3][i_sector][voxX][voxY][voxZ];
          float DX = vec_DXYZ_vox_GF[0][i_sector][voxX][voxY][voxZ];
          tg_Stat_vs_X_slice->SetPoint(ipoint, voxX_pos, statistics);
          ipoint++;

          if (i_sector == 11 && voxY == 10 && voxZ == 27) {
            tp_Stat_vs_X_single->Fill(voxX_pos, statistics);
            tp_DX_vs_X_single->Fill(voxX_pos, DX);
          }
        }

        float max_X_stat = 0.0;
        for (int ibin = (tg_Stat_vs_X_slice->GetN() - 3); ibin >= 0; ibin--) {
          double X_val = 0.0;
          double stat = 0.0;
          tg_Stat_vs_X_slice->GetPoint(ibin, X_val, stat);
          if (stat < min_statistics)
            continue;
          if (X_val < (85.0 + 32.0)) {
            Double_t stat_previous[3] = {0.0, 0.0, 0.0};
            Double_t Xpos_previous[3] = {0.0, 0.0, 0.0};

            tg_Stat_vs_X_slice->GetPoint(ibin + 1, Xpos_previous[0], stat_previous[0]);
            tg_Stat_vs_X_slice->GetPoint(ibin + 2, Xpos_previous[1], stat_previous[1]);
            tg_Stat_vs_X_slice->GetPoint(ibin + 3, Xpos_previous[2], stat_previous[2]);

            if ((stat - stat_previous[0]) < 0.0 && (stat - stat_previous[1]) < 0.0 && (stat - stat_previous[2]) < 0.0 && stat > 0.0) {
              Double_t ratio_stat[3] = {stat / stat_previous[0], stat / stat_previous[1], stat / stat_previous[2]};
              // if(i_sector == 9 && voxY == 19 && voxZ == 27)
              //{
              // }
              if (ratio_stat[0] < 0.8 && ratio_stat[1] < 0.5 && ratio_stat[2] < 0.5) {
                // first check if there aren't any bins at lower radii with statistics
                int flag_low_bin = 0;
                for (int ibinB = ibin - 1; ibinB >= 0; ibinB--) {
                  double X_valB = 0.0;
                  double statB = 0.0;
                  tg_Stat_vs_X_slice->GetPoint(ibinB, X_valB, statB);
                  if (statB > 0.0 && statB / stat_previous[0] > 0.8) {
                    ibin = ibinB;
                    flag_low_bin = 1;
                    break;
                  }
                }
                if (!flag_low_bin) {
                  max_X_stat = Xpos_previous[2];
                  break;
                }
              }
            }
            if (fabs(stat < 0.1)) {
              // first check if there aren't any bins at lower radii with statistics
              int flag_low_bin = 0;
              for (int ibinB = ibin - 1; ibinB >= 0; ibinB--) {
                double X_valB = 0.0;
                double statB = 0.0;
                tg_Stat_vs_X_slice->GetPoint(ibinB, X_valB, statB);
                if (statB > 0.0) {
                  ibin = ibinB;
                  flag_low_bin = 1;
                  break;
                }
              }
              if (!flag_low_bin) {
                max_X_stat = Xpos_previous[2];
                break;
              }
            }
          }
        }

        if (max_X_stat < 85.0) {
          max_X_stat = max_X_DX; // set to average value
        }
        int i_bin_z_phi_sector = voxZ * 36 * nY2XBins + i_sector * nY2XBins + voxY;
        h_x_start_fit_vs_z_phi_sector->SetBinContent(i_bin_z_phi_sector, max_X_stat);
        tg_Stat_vs_X_slice->Set(0);

        vec_max_X_fit[i_sector][voxY][voxZ] = max_X_stat;
      } // end of Z loop
    } // end of Y loop
  } // end of sector loop
  LOGP(info, "Done calculating extrapolation fit start values for every phi slice");
  //--------------------------------------------------------------------------------

  // =========================================================================
  // treat acceptance edge in z-direction
  // replace values very close to or beyond the pad plane by a value a bit further away
  const float maxZ = 242.f;
  for (Int_t i_sector = 0; i_sector < 36; i_sector++) {
    for (Int_t voxX = 0; voxX < nXBins; voxX++) {
      for (Int_t voxY = 0; voxY < nY2XBins; voxY++) {
        float lastValue[4] = {0.f, 0.f, 0.f, 0.f};
        float lastValueS[4] = {0.f, 0.f, 0.f, 0.f};
        float lastValueA11[4] = {0.f, 0.f, 0.f, 0.f};
        float lastValueSA11[4] = {0.f, 0.f, 0.f, 0.f};
        for (Int_t voxZ = 0; voxZ < nZ2XBins; voxZ++) {
          const float absZ = std::abs(vec_DXYZ_vox[5][i_sector][voxX][voxY][voxZ]);
          // if (i_sector==0&&voxX>149&&voxY==5) {
          //}
          if (absZ < maxZ) {
            for (int i = 0; i < 4; ++i) {
              lastValue[i] = vec_DXYZ_vox[i][i_sector][voxX][voxY][voxZ];
              lastValueS[i] = vec_DXYZ_vox_GF[i][i_sector][voxX][voxY][voxZ];
            }
          } else {
            for (int i = 0; i < 4; ++i) {
              vec_DXYZ_vox[i][i_sector][voxX][voxY][voxZ] = lastValue[i];
              vec_DXYZ_vox_GF[i][i_sector][voxX][voxY][voxZ] = lastValueS[i];
            }
          }

          if (i_sector == 11 && A11maxZ2X > -1) {
            if (voxZ <= A11maxZ2X) {
              for (int i = 0; i < 4; ++i) {
                lastValueA11[i] = vec_DXYZ_vox[i][i_sector][voxX][voxY][voxZ];
                lastValueSA11[i] =
                  vec_DXYZ_vox_GF[i][i_sector][voxX][voxY][voxZ];
              }
            } else {
              for (int i = 0; i < 4; ++i) {
                vec_DXYZ_vox[i][i_sector][voxX][voxY][voxZ] = lastValueA11[i];
                vec_DXYZ_vox_GF[i][i_sector][voxX][voxY][voxZ] =
                  lastValueSA11[i];
              }
            }
          }
        }
      }
    }
  }

  //----------------------------------------------------------------
  // Gaussian filtering

  if (do_smoothing) {
    LOGP(info, "Gaussian filtering started");
    std::vector<std::vector<std::vector<Double_t>>> vec_GKernel;
    vec_GKernel.resize(N_bins_X_GF * 2 + 1);
    for (Int_t i_X = 0; i_X < (Int_t)vec_GKernel.size(); i_X++) {
      vec_GKernel[i_X].resize(N_bins_Y_GF * 2 + 1);
      for (Int_t i_Y = 0; i_Y < (Int_t)vec_GKernel[i_X].size(); i_Y++) {
        vec_GKernel[i_X][i_Y].resize(N_bins_Z_GF * 2 + 1);
      }
    }
    vec_FilterCreation(vec_GKernel, N_bins_X_GF, N_bins_Y_GF, N_bins_Z_GF, sigma_GF);

    for (Int_t i_sector = 0; i_sector < 36; i_sector++) {
      std::vector<std::vector<std::vector<std::vector<Float_t>>>> arr_values;
      std::vector<std::vector<std::vector<std::vector<Float_t>>>> arr_values_used;
      arr_values.resize(3);
      arr_values_used.resize(3);

      for (Int_t i_xyz = 0; i_xyz < 3; i_xyz++) {
        arr_values[i_xyz].resize(N_bins_X_GF * 2 + 1);
        arr_values_used[i_xyz].resize(N_bins_X_GF * 2 + 1);
        for (Int_t i_X = 0; i_X < (Int_t)arr_values[i_xyz].size(); i_X++) {
          arr_values[i_xyz][i_X].resize(N_bins_Y_GF * 2 + 1);
          arr_values_used[i_xyz][i_X].resize(N_bins_Y_GF * 2 + 1);
          for (Int_t i_Y = 0; i_Y < (Int_t)arr_values[i_xyz][i_X].size(); i_Y++) {
            arr_values[i_xyz][i_X][i_Y].resize(N_bins_Z_GF * 2 + 1);
            arr_values_used[i_xyz][i_X][i_Y].resize(N_bins_Z_GF * 2 + 1);
            for (Int_t i_Z = 0; i_Z < (Int_t)arr_values[i_xyz][i_X][i_Y].size(); i_Z++) {
              arr_values[i_xyz][i_X][i_Y][i_Z] = 0.0;
              arr_values_used[i_xyz][i_X][i_Y][i_Z] = 0.0;
            }
          }
        }
      }

      // CRU 0 : 000 - 016 (IROC)
      // CRU 1 : 017 - 031 (IROC)
      // CRU 2 : 032 - 047 (IROC)
      // CRU 3 : 048 - 062 (IROC)

      // CRU 4 : 063 - 080 (OROC 1)
      // CRU 5 : 081 - 096 (OROC 1)

      // CRU 6 : 097 - 112 (OROC 2)
      // CRU 7 : 113 - 126 (OROC 2)

      // CRU 8 : 127 - 139 (OROC 3)
      // CRU 9 : 140 - 151 (OROC 3)

      std::vector<std::vector<int>> vec_ROC_row;
      vec_ROC_row.resize(4);
      for (int iRoc = 0; iRoc < 4; iRoc++) {
        vec_ROC_row[iRoc].resize(2);
      }
      vec_ROC_row[0][0] = 0;
      vec_ROC_row[0][1] = 62;
      vec_ROC_row[1][0] = 63;
      vec_ROC_row[1][1] = 96;
      vec_ROC_row[2][0] = 97;
      vec_ROC_row[2][1] = 126;
      vec_ROC_row[3][0] = 127;
      vec_ROC_row[3][1] = 151;

      for (int iRoc = 0; iRoc < 4; iRoc++) {
        int minRow = vec_ROC_row[iRoc][0];
        if (do_extrapolation && (iRoc == 0)) {
          minRow = max_Row_DX + 2; // don't smooth over low radii large distortions
        }
        for (Int_t voxX = minRow; voxX <= vec_ROC_row[iRoc][1]; voxX++) {
          for (Int_t voxY = 0; voxY < (nY2XBins); voxY++) {
            for (Int_t voxZ = 0; voxZ < (nZ2XBins); voxZ++) {
              Float_t sum_weight[3] = {0.0};
              Float_t sum_values[3] = {0.0};
              for (Int_t index_voxXB = -N_bins_X_GF; index_voxXB <= N_bins_X_GF; index_voxXB++) {
                Int_t voxXB = voxX + index_voxXB;
                if (voxXB < vec_ROC_row[iRoc][0])
                  continue;
                if (voxXB > vec_ROC_row[iRoc][1])
                  continue;
                for (Int_t index_voxYB = -N_bins_Y_GF; index_voxYB <= N_bins_Y_GF; index_voxYB++) {
                  Int_t voxYB = voxY + index_voxYB;
                  if (voxYB < 0)
                    continue;
                  if (voxYB >= nY2XBins)
                    continue;
                  for (Int_t index_voxZB = -N_bins_Z_GF; index_voxZB <= N_bins_Z_GF; index_voxZB++) {
                    Int_t voxZB = voxZ + index_voxZB;
                    if (voxZB < 0)
                      continue;
                    if (voxZB >= (nZ2XBins))
                      continue;
                    float statistics = vec_DXYZ_vox[3][i_sector][voxXB][voxYB][voxZB];
                    if ((int)statistics == 0)
                      continue;
                    if (TMath::IsNaN(vec_DXYZ_vox[0][i_sector][voxXB][voxYB][voxZB]))
                      continue; // NaN check
                    if (TMath::IsNaN(vec_DXYZ_vox[1][i_sector][voxXB][voxYB][voxZB]))
                      continue; // NaN check
                    if (TMath::IsNaN(vec_DXYZ_vox[2][i_sector][voxXB][voxYB][voxZB]))
                      continue; // NaN check
                    if (fabs(vec_DXYZ_vox[0][i_sector][voxXB][voxYB][voxZB]) > maxDeltaCut)
                      continue;
                    if (fabs(vec_DXYZ_vox[1][i_sector][voxXB][voxYB][voxZB]) > maxDeltaCut)
                      continue;
                    if (fabs(vec_DXYZ_vox[2][i_sector][voxXB][voxYB][voxZB]) > maxDeltaCut)
                      continue;
                    for (Int_t i_xyz = 0; i_xyz < 3; i_xyz++) {
                      arr_values_used[i_xyz][index_voxXB + N_bins_X_GF][index_voxYB + N_bins_Y_GF][index_voxZB + N_bins_Z_GF] = 1.0;
                      arr_values[i_xyz][index_voxXB + N_bins_X_GF][index_voxYB + N_bins_Y_GF][index_voxZB + N_bins_Z_GF] = vec_GKernel[index_voxXB + N_bins_X_GF][index_voxYB + N_bins_Y_GF][index_voxZB + N_bins_Z_GF] * vec_DXYZ_vox[i_xyz][i_sector][voxXB][voxYB][voxZB];
                      sum_weight[i_xyz] += vec_GKernel[index_voxXB + N_bins_X_GF][index_voxYB + N_bins_Y_GF][index_voxZB + N_bins_Z_GF];
                      sum_values[i_xyz] += arr_values[i_xyz][index_voxXB + N_bins_X_GF][index_voxYB + N_bins_Y_GF][index_voxZB + N_bins_Z_GF];
                      if (TMath::IsNaN(vec_DXYZ_vox[i_xyz][i_sector][voxXB][voxYB][voxZB])) {
                        LOGP(error, "NaN vec_DXYZ_vox detected in xyz {}, sec {}, voxX {}, voxY {}, voxZ {}", i_xyz, i_sector, voxXB, voxYB, voxZB);
                      }
                      if (TMath::IsNaN(sum_values[i_xyz])) {
                        LOGP(error, "NaN sum_values detected in xyz {}, sec {}, voxX {}, voxY {}, voxZ {}", i_xyz, i_sector, voxXB, voxYB, voxZB);
                      }
                    }
                  }
                }
              }

              for (Int_t i_xyz = 0; i_xyz < 3; i_xyz++) {
                if (sum_weight[i_xyz] > 0.0) {
                  sum_values[i_xyz] /= sum_weight[i_xyz];
                  vec_DXYZ_vox_GF[i_xyz][i_sector][voxX][voxY][voxZ] = sum_values[i_xyz];
                  if (TMath::IsNaN(vec_DXYZ_vox_GF[i_xyz][i_sector][voxX][voxY][voxZ])) {
                    LOGP(error, "NaN detected during smoothing process in xyz {}, sec {}, voxX {}, voxY {}, voxZ {}", i_xyz, i_sector, voxX, voxY, voxZ);
                  }

                  float voxX_pos = RowX[voxX];
                  if (voxZ == 0) {
                    if (i_xyz == 0)
                      tp_DX_vs_X_smooth->Fill(voxX_pos, sum_values[i_xyz]);
                    if (i_sector < 18) {
                      if (i_xyz == 2)
                        tp_DZ_vs_X_smooth_A->Fill(voxX_pos, sum_values[i_xyz]);
                    } else {
                      if (i_xyz == 2)
                        tp_DZ_vs_X_smooth_C->Fill(voxX_pos, sum_values[i_xyz]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } // end loop over sectors
  } // do_smoothing
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // Do extrapolation to low radii
  if (do_extrapolation > 0) {
    LOGP(info, "Starting extrapolation to small radii");
    // const float start_fit = max_X_DX + 0.0;
    // const float stop_fit  = max_X_DX + 10.0;
    TGraph* tg_data_for_fit = new TGraph();
    TGraph* tg_data_for_fit_raw = new TGraph();
    for (Int_t i_sector = 0; i_sector < 36; i_sector++) {
      for (Int_t voxY = 0; voxY < (nY2XBins); voxY++) {
        for (Int_t voxZ = 0; voxZ < (nZ2XBins); voxZ++) {
          const float start_fit = vec_max_X_fit[i_sector][voxY][voxZ] + 0.0;
          const float stop_fit = vec_max_X_fit[i_sector][voxY][voxZ] + 10.0;
          // find maximum dX
          // float maxXdX = 0;
          // float maxdX = 0;
          // for(Int_t voxX = startRowGoodEntries; voxX < 63; voxX++) // enough to search in IROC
          //{
          // const float dX  = vec_DXYZ_vox[0][i_sector][voxX][voxY][voxZ];
          // if (dX > maxdX) {
          // maxdX = dX;
          // maxXdX = RowX[voxX];
          //}
          //}
          // const float start_fit = maxXdX + 1.0;
          // const float stop_fit  = maxXdX + 10.0;

          for (Int_t i_xyz = 0; i_xyz < 3; i_xyz++) {
            // fill the TGraph used for fitting
            int ipoint = 0;
            int meanStat = 1; // statistics to use in extrapolation region
            for (Int_t voxX = 0; voxX <= 151; voxX++) {
              // float voxX_pos = vec_DXYZ_vox_GF[4][i_sector][voxX][voxY][voxZ];
              float voxX_pos = RowX[voxX];
              if (voxX_pos < start_fit)
                continue;
              if (voxX_pos > stop_fit)
                break;
              float statistics = vec_DXYZ_vox_GF[3][i_sector][voxX][voxY][voxZ];
              if (statistics < min_statistics)
                continue;
              float DXYZval = vec_DXYZ_vox_GF[i_xyz][i_sector][voxX][voxY][voxZ];
              tg_data_for_fit->SetPoint(ipoint, voxX_pos, DXYZval);
              float DXYZval_raw = vec_DXYZ_vox[i_xyz][i_sector][voxX][voxY][voxZ];
              tg_data_for_fit_raw->SetPoint(ipoint, voxX_pos, DXYZval_raw);
              ipoint++;
            }
            if (ipoint < 5)
              continue;

            // fit the TGraph
            for (Int_t i = 0; i < 6; i++) {
              func_PolyFitFunc->SetParameter(i, 0.0);
              func_PolyFitFunc->SetParError(i, 0.0);
              func_PolyFitFunc_raw->SetParameter(i, 0.0);
              func_PolyFitFunc_raw->SetParError(i, 0.0);
              if (i > 2) {
                func_PolyFitFunc->FixParameter(i, 0.0);
                func_PolyFitFunc_raw->FixParameter(i, 0.0);
              }
            }
            func_PolyFitFunc->SetParameter(0, 0.2);
            func_PolyFitFunc->SetParameter(1, 0.3);
            func_PolyFitFunc->SetParameter(2, 0.4);
            func_PolyFitFunc->SetRange(start_fit, stop_fit);
            tg_data_for_fit->Fit("func_PolyFitFunc", "QWMN", "", start_fit, stop_fit);

            func_PolyFitFunc_raw->SetParameter(0, 0.2);
            func_PolyFitFunc_raw->SetParameter(1, 0.3);
            func_PolyFitFunc_raw->SetParameter(2, 0.4);
            func_PolyFitFunc_raw->SetRange(start_fit, stop_fit);
            tg_data_for_fit_raw->Fit("func_PolyFitFunc_raw", "QWMN", "", start_fit, stop_fit);

            // if (ipoint>0) {
            //   meanStat /= ipoint;
            // }

            // do the low radii extrapolation
            for (Int_t voxX = 0; voxX <= 151; voxX++) {
              // float voxX_pos = vec_DXYZ_vox_GF[4][i_sector][voxX][voxY][voxZ];
              float voxX_pos = RowX[voxX];
              if (voxX_pos > start_fit) {
                break;
              }
              double extrapolation_value = func_PolyFitFunc->Eval(voxX_pos);
              if (fabs(extrapolation_value) > max_extrapolation_value)
                extrapolation_value = TMath::Sign(1, extrapolation_value) * max_extrapolation_value;
              vec_DXYZ_vox_GF[i_xyz][i_sector][voxX][voxY][voxZ] = extrapolation_value;
              vec_DXYZ_vox_GF[3][i_sector][voxX][voxY][voxZ] = meanStat; // set statistics -> important for spline creation

              double extrapolation_value_raw = func_PolyFitFunc_raw->Eval(voxX_pos);
              if (fabs(extrapolation_value_raw) > max_extrapolation_value)
                extrapolation_value_raw = TMath::Sign(1, extrapolation_value_raw) * max_extrapolation_value;
              vec_DXYZ_vox[i_xyz][i_sector][voxX][voxY][voxZ] = extrapolation_value_raw;
              vec_DXYZ_vox[3][i_sector][voxX][voxY][voxZ] = meanStat; // set statistics -> important for spline creation
            }

            tg_data_for_fit->Set(0);
            tg_data_for_fit_raw->Set(0);
          } // end of xyz

          for (Int_t voxX = 0; voxX <= 151; voxX++) {
            if (voxZ == 0) // for QA
            {
              float voxX_pos = RowX[voxX];
              tp_DX_vs_X_smooth_extr->Fill(voxX_pos, vec_DXYZ_vox_GF[0][i_sector][voxX][voxY][voxZ]);
              if (i_sector < 18) {
                tp_DZ_vs_X_smooth_extr_A->Fill(voxX_pos, vec_DXYZ_vox_GF[2][i_sector][voxX][voxY][voxZ]);
              } else {
                tp_DZ_vs_X_smooth_extr_C->Fill(voxX_pos, vec_DXYZ_vox_GF[2][i_sector][voxX][voxY][voxZ]);
              }
            }
          }
        }
      }
    }
  }
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // IROC A11 maskign
  if (maskIA11) {
    int i_sector = 11;
    for (Int_t voxY = 0; voxY < (nY2XBins); voxY++) {
      for (Int_t voxZ = 0; voxZ < (nZ2XBins); voxZ++) {
        for (Int_t voxX = 0; voxX <= 62; voxX++) {
          for (Int_t i_xyz = 0; i_xyz < 4; i_xyz++) {
            vec_DXYZ_vox_GF[i_xyz][i_sector][voxX][voxY][voxZ] = 0;
            vec_DXYZ_vox[i_xyz][i_sector][voxX][voxY][voxZ] = 0;
          }
        }
      }
    }
  }

  //----------------------------------------------------------------
  mTreeOut = std::make_unique<TTree>("voxResTree", "Voxel results and statistics");
  mTreeOut->Branch("voxRes", &mVoxelResultsOutPtr);
  // copy user info
  auto userInfoOut = mTreeOut->GetUserInfo();
  for (auto o : *userInfo) {
    userInfoOut->Add(o->Clone());
  }
  // Overwrite the placeholder meanIDC/medianIDC just cloned above with the real offline-joined
  // values computed in the "Offline IDC join" block earlier in this function.
  if (auto* stale = userInfoOut->FindObject("meanIDC")) {
    userInfoOut->Remove(stale);
    delete stale;
  }
  if (auto* stale = userInfoOut->FindObject("medianIDC")) {
    userInfoOut->Remove(stale);
    delete stale;
  }
  userInfoOut->Add(new TNamed("meanIDC", std::to_string(meanIDCReal).data()));
  userInfoOut->Add(new TNamed("medianIDC", std::to_string(medianIDCReal).data()));
  userInfoOut->Add(new TNamed("startRowGoodEntries", std::to_string(startRowGoodEntries).data()));
  userInfoOut->Add(new TNamed("maxDX", std::to_string(max_DX).data()));
  userInfoOut->Add(new TNamed("maxDX_lx", std::to_string(max_X_DX).data()));
  userInfoOut->Add(new TNamed("maxDX_row", std::to_string(max_Row_DX).data()));

  // copy aliases
  if (voxResTree->GetListOfAliases()) {
    for (auto o : *voxResTree->GetListOfAliases()) {
      mTreeOut->SetAlias(o->GetName(), o->GetTitle());
    }
  }

  for (Long64_t jentry = 0; jentry < entries_input_map; jentry++) {
    voxResTree->GetEntry(jentry);

    auto bvox_X = voxRes_map->bvox[o2::tpc::TrackResiduals::VoxX]; // bin number in x (= pad row)
    auto bvox_F = voxRes_map->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
    auto bvox_Z = voxRes_map->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4
    int sector = (int)voxRes_map->bsec;
    // Int_t index_map  = bvox_X+152*bvox_F+152*(nY2XBins)*bvox_Z+152*(nY2XBins)*(nZ2XBins)*sector;

    mVoxelResultsOut = *voxRes_map; // copy the entry from the input map

    for (int ixyz = 0; ixyz < 3; ixyz++) {
      if (do_extrapolation == 1 || do_extrapolation == 2)
        mVoxelResultsOut.DS[ixyz] = (float)vec_DXYZ_vox_GF[ixyz][sector][bvox_X][bvox_F][bvox_Z]; // overwrite the smoothed values
      if (do_extrapolation == 2 || do_extrapolation == 3)
        mVoxelResultsOut.D[ixyz] = (float)vec_DXYZ_vox[ixyz][sector][bvox_X][bvox_F][bvox_Z]; // overwrite the raw values
    }
    if (do_extrapolation == 1 || do_extrapolation == 2)
      mVoxelResultsOut.stat[3] = (float)vec_DXYZ_vox_GF[3][sector][bvox_X][bvox_F][bvox_Z];
    if (do_extrapolation == 2 || do_extrapolation == 3)
      mVoxelResultsOut.stat[3] = (float)vec_DXYZ_vox[3][sector][bvox_X][bvox_F][bvox_Z];
    for (int ixyz = 0; ixyz < 3; ixyz++) {
      if (TMath::IsNaN(mVoxelResultsOut.DS[ixyz])) // NaN
      {
        LOGP(error, "NaN detected in smoothed value xyz {}, sec {}, voxX {}, voxY {}, voxZ {}", ixyz, sector, bvox_X, bvox_F, bvox_Z);
        mVoxelResultsOut.DS[ixyz] = 0.0;
        mVoxelResultsOut.stat[3] = 0;
        mVoxelResultsOut.flags |= TrackResiduals::Masked;
      } else {
        mVoxelResultsOut.flags |= TrackResiduals::SmoothDone;
      }
      if (TMath::IsNaN(mVoxelResultsOut.D[ixyz])) // NaN
      {
        LOGP(error, "NaN detected in raw value xyz {}, sec {}, voxX {}, voxY {}, voxZ {}", ixyz, sector, bvox_X, bvox_F, bvox_Z);
        mVoxelResultsOut.D[ixyz] = 0.0;
      }
    }
    mTreeOut->Fill();
  }
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  outputfile->cd();
  mTreeOut->Write();
  mTreeOut.release();
  tp_DX_vs_X_raw->Write();
  tp_DX_vs_X_smooth->Write();
  tp_DX_vs_X_smooth_extr->Write();
  tp_DZ_vs_X_raw_A->Write();
  tp_DZ_vs_X_raw_C->Write();
  tp_DZ_vs_X_smooth_A->Write();
  tp_DZ_vs_X_smooth_C->Write();
  tp_DZ_vs_X_smooth_extr_A->Write();
  tp_DZ_vs_X_smooth_extr_C->Write();
  tp_Stat_vs_X->Write();
  tp_Stat_vs_row->Write();
  h_x_start_fit_vs_z_phi_sector->Write();
  tp_Stat_vs_X_single->Write();
  tp_DX_vs_X_single->Write();
  outputfile->Close();
  //----------------------------------------------------------------
}
