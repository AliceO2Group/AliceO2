// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastTransformInitCPM.C
/// \brief A macro for generating TPC fast transformation
///        out of set of space charge correction voxels
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>
///

/// how to run the macro:
///
/// root -l TPCFastTransformInitCPM.C'("debugVoxRes.root")'
///

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <filesystem>
#include <string>
#include <vector>
#include <regex>
#include <array>
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TNtuple.h"
#include "TMath.h"
#include "Riostream.h"

#include "CommonUtils/TreeStreamRedirector.h"
#include "MathUtils/fit.h"
#include "Algorithm/RangeTokenizer.h"
#include "Framework/Logger.h"
#include "GPU/TPCFastTransform.h"
#include "SpacePoints/TrackResiduals.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TPCCalibration/TPCFastSpaceChargeCorrectionHelper.h"

#endif

struct sMean {
  float mean;
  float rms;
  float meanErr;
  float rmsErr;

  ClassDef(sMean, 1);
};

#pragma link C++ class sMean + ;
#pragma link C++ class std::vector < sMean> + ;

using namespace o2::tpc;
using namespace o2::gpu;
namespace mu = o2::math_utils;

void createFastTransform(std::string outFileName, TTree* voxResTree, o2::tpc::TrackResiduals& trackResiduals, bool useSmoothed, bool invertSigns, float meanIDC = 0.f, float meanCTP = 0.f, int debug = 0, int useCTPLumi = 0);

void TPCFastTransformInitCPM(const char* fileName = "debugVoxRes.root",
                             const char* outFileName = "TPCFastTransform_VoxRes.root",
                             bool useSmoothed = false,
                             int useCTPLumi = 0,
                             bool invertSigns = true,
                             float meanIDC = 0.f,
                             float meanCTP = 0.f,
                             int debug = 0,
                             int nThreads = 8)
{

  // Initialise TPCFastTransform object from "voxRes" tree of
  // o2::tpc::TrackResiduals::VoxRes track residual voxels
  //

  /*
    To visiualise the results:

    root -l transformDebug.root
    corr->Draw("cx:y:z","iRoc==0&&iRow==10","")
    grid->Draw("cx:y:z","iRoc==0&&iRow==10","same")
    vox->Draw("vx:y:z","iRoc==0&&iRow==10","same")
  */

  o2::tpc::TPCFastSpaceChargeCorrectionHelper::instance()->setNthreads(nThreads);

  if (gSystem->AccessPathName(fileName)) {
    LOGP(error, "input file {} does not exist!", fileName);
    return;
  }

  auto file = std::unique_ptr<TFile>(TFile::Open(fileName, "READ"));
  if (!file || !file->IsOpen()) {
    LOGP(error, "could not open input file {}", fileName);
    return;
  }

  TTree* voxResTree = nullptr;
  file->cd();
  gDirectory->GetObject("voxResTree", voxResTree);
  if (!voxResTree) {
    LOGP(error, "tree voxResTree does not exist!");
    return;
  }
  auto userInfo = voxResTree->GetUserInfo();
  if (!userInfo->FindObject("y2xBinning") || !userInfo->FindObject("z2xBinning")) {
    LOGP(error, "'y2xBinning' or 'z2xBinning' not found in UserInfo, but required to get the correct binning");
    return;
  }

  // Obtain configuration
  const SpacePointsCalibConfParam& params = SpacePointsCalibConfParam::Instance();
  if (!std::filesystem::exists("scdconfig.ini")) {
    LOGP(warning, "Did not find configuration file. Using default parameters and storing them in scdconfig.ini");
    params.writeINI("scdconfig.ini", "scdcalib"); // to write default parameters to a file
  } else {
    params.updateFromFile("scdconfig.ini");
  }
  // TrackResiduals::setZ2XBinning() (called below) reads scdcalib.maxZ2X directly and uses it to scale
  // the physical z/x bin boundaries -- baked into what each z2x voxel index in the input tree actually
  // means, not a cosmetic knob. There is no scdconfig.ini on the GRID, so without this the code default
  // (1.0) would silently apply instead of whatever stage 1 actually used (production 1.4), misaligning
  // this macro's re-derived binning against the tree's real geometry. Must happen BEFORE setZ2XBinning()
  // below. See staticMapCreatorCPM.C's UserInfo::Add("maxZ2X", ...) for where this comes from --
  // SmoothingExtrapolate.C clones the whole input UserInfo onto its own output, so it survives into this
  // macro's input unchanged.
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

  // required for the binning that was used
  o2::tpc::TrackResiduals trackResiduals;
  auto y2xBins = o2::RangeTokenizer::tokenize<float>(userInfo->FindObject("y2xBinning")->GetTitle());
  const std::string z2xStr = userInfo->FindObject("z2xBinning")->GetTitle();
  auto z2xBins = o2::RangeTokenizer::tokenize<float>(z2xStr);
  LOGP(info, "z2xBins: {}", z2xStr);
  trackResiduals.setY2XBinning(y2xBins);
  trackResiduals.setZ2XBinning(z2xBins);
  trackResiduals.init();

  auto getFromUserInfo = [userInfo](std::string value, float& valueF) {
    if (valueF != 0) {
      LOGP(info, "{} set to {} via command line, not reading it from userInfo", value, valueF);
      return;
    }

    if (!userInfo || !userInfo->FindObject(value.data())) {
      LOGP(error, "Could not find value for {} in userInfo", value);
      valueF = 0.f;
      return;
    }
    valueF = std::atof(userInfo->FindObject(value.data())->GetTitle());
    LOGP(info, "Found {} = {} in userInfo", value, valueF);
  };

  getFromUserInfo("meanIDC", meanIDC);
  getFromUserInfo("meanCTP", meanCTP);

  if ((useCTPLumi != 2 && meanIDC == 0) || meanCTP == 0) {
    LOGP(fatal, "meanCTP ({}) or meanIDC ({}) not set!", meanCTP, meanIDC);
  }
  if (useCTPLumi == 2) {
    LOGP(warning, "Explicitly disabled IDCs!");
  }

  createFastTransform(outFileName, voxResTree, trackResiduals, useSmoothed, invertSigns, meanIDC, meanCTP, debug, useCTPLumi);
}

void createFastTransform(std::string outFileName, TTree* voxResTree, o2::tpc::TrackResiduals& trackResiduals, bool useSmoothed, bool invertSigns, float meanIDC, float meanCTP, int debug, int useCTPLumi)
{
  LOGP(info, "create fast transformation ... ");
  std::regex reg(".*FT_voxRes\\.residuals\\.([0-9]{6})_([0-9]{13})_([0-9]{13})_([0-9]+)_([0-9]+).*\\.root");
  std::smatch base_match;
  int run = -1;
  long validFrom = -1;
  long validUntil = -1;
  int firstTF = -1;
  int lastTF = -1;
  if (std::regex_match(outFileName, base_match, reg)) {
    run = std::stoi(base_match[1].str());
    validFrom = std::stol(base_match[2].str());
    validUntil = std::stol(base_match[3].str());
    firstTF = std::stol(base_match[4].str());
    lastTF = std::stol(base_match[5].str());
    LOGP(info, "Found run {}, validFrom {}, validUntil {}, firstTF {}, lastTF {}", run, validFrom, validUntil, firstTF, lastTF);
  }

  auto* helper = o2::tpc::TPCFastTransformHelperO2::instance();

  o2::tpc::TPCFastSpaceChargeCorrectionHelper* corrHelper = o2::tpc::TPCFastSpaceChargeCorrectionHelper::instance();

#if __has_include("TPCFastTransformPOD.h")
  TTree* voxResTreeInverse = nullptr;
  o2::gpu::TPCFastSpaceChargeCorrectionMap mapDirect(0, 0), mapInverse(0, 0);
  auto corrPtr = corrHelper->createFromTrackResiduals(trackResiduals, voxResTree, voxResTreeInverse, useSmoothed, invertSigns, &mapDirect, &mapInverse);
#else
  auto corrPtr = corrHelper->createFromTrackResiduals(trackResiduals, voxResTree, useSmoothed, invertSigns);
#endif

  std::unique_ptr<o2::gpu::TPCFastTransform> fastTransform(helper->create(0, *corrPtr));
  fastTransform->setLumi(meanCTP);
  fastTransform->setIDC(meanIDC); // for SW version with IDC in FastTransfrom

  o2::gpu::TPCFastSpaceChargeCorrection& corr = fastTransform->getCorrection();

  LOGP(info, "... create fast transformation completed");

  if (!outFileName.empty()) {
    fastTransform->writeToFile(outFileName.data(), "ccdb_object");
  }

  LOGP(info, "verify the results ...");

  // the difference

  double maxDiff[3] = {0., 0., 0.};
  int maxDiffRoc[3] = {0, 0, 0};
  int maxDiffRow[3] = {0, 0, 0};

  double sumDiff[3] = {0., 0., 0.};
  long nDiff = 0;

  // a debug file with some NTuples

  TDirectory* currDir = gDirectory;

  const std::filesystem::path pFileOutput(outFileName);
  std::string outPath(pFileOutput.parent_path().c_str());
  if (outPath.empty()) {
    outPath = ".";
  }
  const std::string fileOutputDebug = fmt::format("{}/{}.debug.root", outPath, pFileOutput.stem().c_str());
  const std::string fileOutputSummary = fmt::format("{}/{}.summary.root", outPath, pFileOutput.stem().c_str());

  o2::utils::TreeStreamRedirector summary(fileOutputSummary.data(), "recreate");

  TFile* debugFile = new TFile(fileOutputDebug.data(), "RECREATE");
  debugFile->cd();

  // ntuple with the input data: voxel corrections
  debugFile->cd();
  TNtuple* debugVox = new TNtuple("vox", "vox", "iRoc:iRow:y2xbin:z2xbin:x:y:z:vx:vy:vz:cx:cy:cz");

  debugVox->SetMarkerStyle(8);
  debugVox->SetMarkerSize(0.8);
  debugVox->SetMarkerColor(kBlue);

  currDir->cd();

  // check the difference in voxels and fill corresp. debug ntuple

  LOGP(info, "verify the results ...");

  const o2::gpu::TPCFastTransformGeo& geo = helper->getGeometry();

  o2::tpc::TrackResiduals::VoxRes* v = nullptr;
  TBranch* branch = voxResTree->GetBranch("voxRes");
  branch->SetAddress(&v);
  branch->SetAutoDelete(kTRUE);

  int nNaNdXV = 0;
  int nNaNdYV = 0;
  int nNaNdZV = 0;
  int nNaNdXC = 0;
  int nNaNdYC = 0;
  int nNaNdZC = 0;

  int lastSector = -1;

  std::vector<float> statsPerSecMean(36);
  std::vector<float> statsPerSecStdDev(36);
  std::vector<float> statsPerSecMedian(36);

  std::vector<float> maxDiffPerSec[3];

  std::vector<sMean> deviationPerSecLTM95[3];
  std::vector<float> deviationPerSecMedian[3];

  std::vector<float> entriesStats;
  std::vector<float> deviations[3];
  entriesStats.reserve(152 * trackResiduals.getNY2XBins() * trackResiduals.getNZ2XBins());
  for (int i = 0; i < 3; ++i) {
    deviations[i].reserve(152 * trackResiduals.getNY2XBins() * trackResiduals.getNY2XBins());
    maxDiffPerSec[i].resize(36);
    deviationPerSecLTM95[i].resize(36);
    deviationPerSecMedian[i].resize(36);
  }

  // retrieve infos from UserInfo
  auto getFromUserInfo = [](TList* u, const char* name, int defVal = -1) {
    const auto o = u->FindObject(name);
    return o ? std::atoi(o->GetTitle()) : defVal;
  };

  auto userInfo = voxResTree->GetUserInfo();
  const int nSlicesPhiZ = getFromUserInfo(userInfo, "nSlicesPhiZ");
  const int maxTracks = getFromUserInfo(userInfo, "maxTracks");
  const int minTracks = getFromUserInfo(userInfo, "minTracks");
  const int nTracksProcessed = getFromUserInfo(userInfo, "nTracksProcessed");
  const bool badCalib = static_cast<bool>(getFromUserInfo(userInfo, "badCalib", 0));

  for (int iVox = 0; iVox < voxResTree->GetEntriesFast(); iVox++) {

    voxResTree->GetEntry(iVox);

    const float voxEntries = v->stat[o2::tpc::TrackResiduals::VoxV];
    const int xBin = v->bvox[o2::tpc::TrackResiduals::VoxX];   // bin number in x (= pad row)
    const int y2xBin = v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
    const int z2xBin = v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4
    const int iRoc = (int)v->bsec;
    const int iRow = (int)xBin;

    const float x = trackResiduals.getX(xBin);             // radius of the pad row
    const float y2x = trackResiduals.getY2X(xBin, y2xBin); // y/x coordinate of the bin ~-0.15 ... 0.15
    const float z2x = trackResiduals.getZ2X(z2xBin);       // z/x coordinate of the bin 0.1 .. 0.9
    const float y = x * y2x;
#if __has_include("TPCFastTransformPOD.h")
    const float z = x * z2x * ((iRoc >= geo.getNumberOfSectorsA()) ? -1.f : 1.f);
#else
    const float z = x * z2x * ((iRoc >= geo.getNumberOfSlicesA()) ? -1.f : 1.f);
#endif

    float correctionX = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResX] : v->D[o2::tpc::TrackResiduals::ResX];
    float correctionY = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResY] : v->D[o2::tpc::TrackResiduals::ResY];
    float correctionZ = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResZ] : v->D[o2::tpc::TrackResiduals::ResZ];

    if (invertSigns) {
      correctionX *= -1.;
      correctionY *= -1.;
      correctionZ *= -1.;
    }

    entriesStats.emplace_back(voxEntries);
    statsPerSecMean[iRoc] += voxEntries;
    statsPerSecStdDev[iRoc] += voxEntries * voxEntries;

    nNaNdXV += TMath::IsNaN(correctionX);
    nNaNdYV += TMath::IsNaN(correctionY);
    nNaNdZV += TMath::IsNaN(correctionZ);

#if __has_include("TPCFastTransformPOD.h")
    float cx, cy, cz;
    corr.getCorrectionLocal(iRoc, iRow, y, z, cx, cy, cz);
#else
    float u, v, cx, cu, cv, cy, cz;
    geo.convLocalToUV(iRoc, y, z, u, v);
    corr.getCorrection(iRoc, iRow, u, v, cx, cu, cv);
    geo.convUVtoLocal(iRoc, u + cu, v + cv, cy, cz);
    cy -= y;
    cz -= z;
#endif

    nNaNdXC += TMath::IsNaN(cx);
    nNaNdYC += TMath::IsNaN(cy);
    nNaNdZC += TMath::IsNaN(cz);

    const float d[3] = {cx - correctionX, cy - correctionY, cz - correctionZ};
    for (int i = 0; i < 3; i++) {
      const float dAbs = std::abs(d[i]);
      maxDiffPerSec[i][iRoc] = std::max(maxDiffPerSec[i][iRoc], dAbs);
      deviations[i].emplace_back(dAbs);
      if (std::abs(maxDiff[i]) < dAbs) {
        maxDiff[i] = d[i];
        maxDiffRoc[i] = iRoc;
        maxDiffRow[i] = iRow;
        LOGP(info, "roc {} row {} xyz {} diff {}", iRoc, iRow, i, d[i]);
      }
      sumDiff[i] += d[i] * d[i];
    }
    nDiff++;

    debugVox->Fill(iRoc, iRow, y2xBin, z2xBin, x, y, z, correctionX, correctionY, correctionZ, cx, cy, cz);

    if (lastSector > -1 && lastSector != iRoc) {
      if (entriesStats.size() > 0) {
        statsPerSecMean[lastSector] /= entriesStats.size();
        statsPerSecStdDev[lastSector] /= entriesStats.size();
        statsPerSecStdDev[lastSector] = (std::sqrt(std::abs(statsPerSecStdDev[lastSector] - statsPerSecMean[lastSector] * statsPerSecMean[lastSector])));
        statsPerSecMedian[lastSector] = mu::median<float>(entriesStats);
      }
      entriesStats.clear();

      static std::vector<size_t> indexDev;
      indexDev.resize(deviations[0].size());
      for (int i = 0; i < 3; i++) {
        std::array<float, 7> fitRes;
        mu::LTMUnbinned(deviations[i], indexDev, fitRes, 0.95);
        deviationPerSecLTM95[i][lastSector].mean = fitRes[1];
        deviationPerSecLTM95[i][lastSector].rms = fitRes[2];
        deviationPerSecLTM95[i][lastSector].meanErr = fitRes[3];
        deviationPerSecLTM95[i][lastSector].rmsErr = fitRes[4];
        deviationPerSecMedian[i][lastSector] = mu::median<float>(deviations[i]);
        deviations[i].clear();
      }
    }
    lastSector = iRoc;
  }
  // last sector
  if (lastSector > -1 && entriesStats.size() > 0) {
    statsPerSecMean[lastSector] /= entriesStats.size();
    statsPerSecStdDev[lastSector] /= entriesStats.size();
    statsPerSecStdDev[lastSector] /= std::sqrt(std::abs(statsPerSecStdDev[lastSector] - statsPerSecMean[lastSector] * statsPerSecMean[lastSector]));
    statsPerSecMedian[lastSector] = mu::median<float>(entriesStats);
    entriesStats.clear();

    std::vector<size_t> indexDev;
    indexDev.resize(deviations[0].size());
    for (int i = 0; i < 3; i++) {
      std::array<float, 7> fitRes;
      mu::LTMUnbinned(deviations[i], indexDev, fitRes, 0.95);
      deviationPerSecLTM95[i][lastSector].mean = fitRes[1];
      deviationPerSecLTM95[i][lastSector].rms = fitRes[2];
      deviationPerSecLTM95[i][lastSector].meanErr = fitRes[3];
      deviationPerSecLTM95[i][lastSector].rmsErr = fitRes[4];
      deviationPerSecMedian[i][lastSector] = mu::median<float>(deviations[i]);
      deviations[i].clear();
    }
  }

  const int nNaNV = nNaNdXV + nNaNdYV + nNaNdZV;
  const int nNaNC = nNaNdXC + nNaNdYC + nNaNdZC;
  const auto sNaNV = fmt::format("NaNV: {} {} {} {}", nNaNV, nNaNdXV, nNaNdYV, nNaNdZV);
  const auto sNaNC = fmt::format("NaNC: {} {} {} {}", nNaNC, nNaNdXC, nNaNdYC, nNaNdZC);

  if (nNaNV > 0) {
    LOGP(error, "{}", sNaNV);
  } else {
    LOGP(info, "{}", sNaNV);
  }
  if (nNaNC > 0) {
    LOGP(error, "{}", sNaNC);
  } else {
    LOGP(info, "{}", sNaNC);
  }

  summary << "summary"
          << "file=" << outFileName
          //
          << "meanIDC=" << meanIDC
          << "meanCTP=" << meanCTP
          << "run=" << run
          << "validFrom=" << validFrom
          << "validUntil=" << validUntil
          << "firstTF=" << firstTF
          << "lastTF =" << lastTF
          //
          << "nNaNdXV=" << nNaNdXV
          << "nNaNdYV=" << nNaNdYV
          << "nNaNdZV=" << nNaNdZV
          << "nNaNdXC=" << nNaNdXC
          << "nNaNdYC=" << nNaNdYC
          << "nNaNdZC=" << nNaNdZC
          //
          << "statsMean=" << statsPerSecMean
          << "statsStdDev=" << statsPerSecStdDev
          << "statsMedian=" << statsPerSecMedian
          //
          << "DdXLTM95=" << deviationPerSecLTM95[0]
          << "DdYLTM95=" << deviationPerSecLTM95[1]
          << "DdZLTM95=" << deviationPerSecLTM95[2]
          << "DdXMedian=" << deviationPerSecMedian[0]
          << "DdYMedian=" << deviationPerSecMedian[1]
          << "DdZMedian=" << deviationPerSecMedian[2]
          << "DdXMax=" << maxDiffPerSec[0]
          << "DdYMax=" << maxDiffPerSec[1]
          << "DdZMax=" << maxDiffPerSec[2]
          //
          << "nSlicesPhiZ=" << nSlicesPhiZ
          << "maxTracks=" << maxTracks
          << "minTracks=" << minTracks
          << "nTracksProcessed=" << nTracksProcessed
          << "badCalib=" << badCalib
          << "\n";

  summary.Close();

#if __has_include("TPCFastTransformPOD.h")
  if (debug > 0) {
    debugFile->cd();
    TNtuple* ntAll = new TNtuple("all", "all", "sec:row:x:y:z:cx:cy:cz:ix:iy:iz");
    ntAll->SetMarkerStyle(8);
    ntAll->SetMarkerSize(0.1);
    ntAll->SetMarkerColor(kBlack);

    debugFile->cd();
    TNtuple* ntGrid = new TNtuple("grid", "grid", "sec:row:x:y:z:cx:cy:cz:ix:iy:iz");
    ntGrid->SetMarkerStyle(8);
    ntGrid->SetMarkerSize(1.2);
    ntGrid->SetMarkerColor(kBlack);

    debugFile->cd();
    TNtuple* ntFitPoints = new TNtuple("fitpoints", "fit points", "sec:row:x:y:z:px:py:pz:cx:cy:cz");
    ntFitPoints->SetMarkerStyle(8);
    ntFitPoints->SetMarkerSize(0.4);
    ntFitPoints->SetMarkerColor(kRed);

    currDir->cd();

    auto getInvCorrections = [&](int iSector, int iRow, float realY, float realZ, float& ix, float& iy, float& iz) {
      ix = corr.getCorrectionXatRealYZ(iSector, iRow, realY, realZ);
      corr.getCorrectionYZatRealYZ(iSector, iRow, realY, realZ, iy, iz);
    };

    auto getAllCorrections = [&](int iSector, int iRow, float y, float z, float& cx, float& cy, float& cz, float& ix, float& iy, float& iz) {
      corr.getCorrectionLocal(iSector, iRow, y, z, cx, cy, cz);
      getInvCorrections(iSector, iRow, y + cy, z + cz, ix, iy, iz);
    };

    LOGP(info, "create debug ntuples at spline grid points and high granular ...");

    for (int32_t iSector = 0; iSector < geo.getNumberOfSectors(); iSector++) {
      LOGP(info, "debug ntuples for sector {}", iSector);

      for (int32_t iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {

        double x = geo.getRowInfo(iRow).x;

        const auto& gridY = corr.getSplineForRow(iRow).getGridX1();
        const auto& gridZ = corr.getSplineForRow(iRow).getGridX2();

        {
          std::vector<double> points[2], knots[2];
          auto [yMin, yMax] = geo.getRowInfo(iRow).getYrange();
          auto [zMin, zMax] = geo.getZrange(iSector);

          for (int32_t iu = 0; iu < gridY.getNumberOfKnots(); iu++) {
            float y, z;
            corr.convGridToLocal(iSector, iRow, gridY.getKnot(iu).getU(), 0., y, z);
            knots[0].push_back(y);
            points[0].push_back(y);
          }
          for (int32_t iv = 0; iv < gridZ.getNumberOfKnots(); iv++) {
            float y, z;
            corr.convGridToLocal(iSector, iRow, 0., gridZ.getKnot(iv).getU(), y, z);
            knots[1].push_back(z);
            points[1].push_back(z);
          }

          for (int32_t iyz = 0; iyz <= 1; iyz++) {
            std::sort(knots[iyz].begin(), knots[iyz].end());
            std::sort(points[iyz].begin(), points[iyz].end());
            int32_t n = points[iyz].size();
            int nsteps = (iyz == 0) ? 10 : 5;
            for (int32_t i = 0; i < n - 1; i++) {
              double d = (points[iyz][i + 1] - points[iyz][i]) / nsteps;
              for (int32_t ii = 1; ii < nsteps; ii++) {
                points[iyz].push_back(points[iyz][i] + d * ii);
              }
            }
          }
          points[0].push_back(yMin);
          points[0].push_back(yMax);
          points[1].push_back(zMin);
          points[1].push_back(zMax);
          for (int32_t iyz = 0; iyz <= 1; iyz++) {
            std::sort(points[iyz].begin(), points[iyz].end());
          }

          for (int32_t iter = 0; iter < 2; iter++) {
            std::vector<double>& py = ((iter == 0) ? knots[0] : points[0]);
            std::vector<double>& pz = ((iter == 0) ? knots[1] : points[1]);
            for (uint32_t iu = 0; iu < py.size(); iu++) {
              for (uint32_t iv = 0; iv < pz.size(); iv++) {
                float y = py[iu];
                float z = pz[iv];
                float cx{0}, cy{0}, cz{0}, ix{0}, iy{0}, iz{0};
                getAllCorrections(iSector, iRow, y, z, cx, cy, cz, ix, iy, iz);
                if (iter == 0) {
                  ntGrid->Fill(iSector, iRow, x, y, z, cx, cy, cz, ix, iy, iz);
                } else {
                  ntAll->Fill(iSector, iRow, x, y, z, cx, cy, cz, ix, iy, iz);
                }
              }
            }
          }
        }

        // the data points used in spline fit
        auto& fitPoints = mapDirect.getPoints(iSector, iRow);
        for (uint32_t ip = 0; ip < fitPoints.size(); ip++) {
          auto point = fitPoints[ip];
          float y = point.mY;
          float z = point.mZ;
          float correctionX = point.mDx;
          float correctionY = point.mDy;
          float correctionZ = point.mDz;
          float cx, cy, cz;
          corr.getCorrectionLocal(iSector, iRow, y, z, cx, cy, cz);
          ntFitPoints->Fill(iSector, iRow, x, y, z, correctionX, correctionY, correctionZ, cx, cy, cz);
        }
      }
    }

    debugFile->cd();
    ntAll->Write();
    ntGrid->Write();
    ntFitPoints->Write();
  }
#else
  if (debug > 0) {
    // ntuple with spline grid points
    debugFile->cd();
    // ntuple with created TPC corrections
    TNtuple* debugCorr = new TNtuple("corr", "corr", "iRoc:iRow:x:y:z:cx:cy:cz");

    debugCorr->SetMarkerStyle(8);
    debugCorr->SetMarkerSize(0.1);
    debugCorr->SetMarkerColor(kBlack);

    TNtuple* debugGrid = new TNtuple("grid", "grid", "iRoc:iRow:x:y:z:cx:cy:cz");

    debugGrid->SetMarkerStyle(8);
    debugGrid->SetMarkerSize(1.2);
    debugGrid->SetMarkerColor(kBlack);

    // ntuple with data points created from voxels (with data smearing and
    // extension to the edges)
    TNtuple* debugPoints = new TNtuple("points", "points", "iRoc:iRow:x:y:z:px:py:pz:cx:cy:cz");

    debugPoints->SetMarkerStyle(8);
    debugPoints->SetMarkerSize(0.4);
    debugPoints->SetMarkerColor(kRed);

    currDir->cd();

    LOGP(info, "create debug ntuples at spline grid points and high granular ...");

    for (int iRoc = 0; iRoc < geo.getNumberOfSlices(); iRoc++) {
      LOGP(info, "debug ntuples for roc {}", iRoc);
      for (int iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {

        double x = geo.getRowInfo(iRow).x;

        // the correction

        for (double su = 0.; su <= 1.0001; su += 0.01) {
          for (double sv = 0.; sv <= 1.0001; sv += 0.1) {
            float u, v;
            geo.convScaledUVtoUV(iRoc, iRow, su, sv, u, v);
            float y, z;
            geo.convUVtoLocal(iRoc, u, v, y, z);
            float cx, cu, cv;
            corr.getCorrection(iRoc, iRow, u, v, cx, cu, cv);
            float cy, cz;
            geo.convUVtoLocal(iRoc, u + cu, v + cv, cy, cz);
            cy -= y;
            cz -= z;
            debugCorr->Fill(iRoc, iRow, x, y, z, cx, cy, cz);
          }
        }

        // the spline grid

        const auto& gridU = corr.getSpline(iRoc, iRow).getGridX1();
        const auto& gridV = corr.getSpline(iRoc, iRow).getGridX2();
        for (int iu = 0; iu < gridU.getNumberOfKnots(); iu++) {
          // double su = gridU.convUtoX(gridU.getKnot(iu).getU());
          for (int iv = 0; iv < gridV.getNumberOfKnots(); iv++) {
            // double sv = gridV.convUtoX(gridV.getKnot(iv).getU());
            float u, v;
            corr.convGridToUV(iRoc, iRow, iu, iv, u, v);
            float y, z;
            geo.convUVtoLocal(iRoc, u, v, y, z);
            float cx, cu, cv;
            corr.getCorrection(iRoc, iRow, u, v, cx, cu, cv);
            float cy, cz;
            geo.convUVtoLocal(iRoc, u + cu, v + cv, cy, cz);
            cy -= y;
            cz -= z;
            debugGrid->Fill(iRoc, iRow, x, y, z, cx, cy, cz);
          }
        }

        // the data points used in spline fit
        // (they are kept in
        // TPCFastTransformHelperO2::instance()->getCorrectionMap() )

        o2::gpu::TPCFastSpaceChargeCorrectionMap& map = corrHelper->getCorrectionMap();
        auto& points = map.getPoints(iRoc, iRow);

        for (unsigned int ip = 0; ip < points.size(); ip++) {
          auto point = points[ip];
          float y = point.mY;
          float z = point.mZ;
          float correctionX = point.mDx;
          float correctionY = point.mDy;
          float correctionZ = point.mDz;

          float u, v, cx, cu, cv, cy, cz;
          geo.convLocalToUV(iRoc, y, z, u, v);
          corr.getCorrection(iRoc, iRow, u, v, cx, cu, cv);
          geo.convUVtoLocal(iRoc, u + cu, v + cv, cy, cz);
          cy -= y;
          cz -= z;

          debugPoints->Fill(iRoc, iRow, x, y, z, correctionX, correctionY, correctionZ, cx, cy, cz);
        }
      }
    }

    debugFile->cd();
    debugCorr->Write();
    debugGrid->Write();
    debugPoints->Write();
  }
#endif

  for (int i = 0; i < 3; i++) {
    sumDiff[i] = sqrt(sumDiff[i]) / nDiff;
  }

  LOGP(info, "Max difference in x :  {} at ROC {} row {}", maxDiff[0], maxDiffRoc[0], maxDiffRow[0]);
  LOGP(info, "Max difference in y :  {} at ROC {} row {}", maxDiff[1], maxDiffRoc[1], maxDiffRow[1]);
  LOGP(info, "Max difference in z :  {} at ROC {} row {}", maxDiff[2], maxDiffRoc[2], maxDiffRow[2]);
  LOGP(info, "Mean difference in x,y,z : {} {} {}", sumDiff[0], sumDiff[1], sumDiff[2]);

  corr.testInverse(0);

  debugFile->cd();
  debugVox->Write();
  debugFile->Close();
}
