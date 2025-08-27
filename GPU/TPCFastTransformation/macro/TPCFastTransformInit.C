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

/// \file  generateTPCCorrection.C
/// \brief A macro for generating TPC fast transformation
///        out of set of space charge correction voxels
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>
///

/// how to run the macro:
///
/// root -l TPCFastTransformInit.C'("debugVoxRes.root")'
///

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <filesystem>
#include <string>
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TNtuple.h"
#include "Riostream.h"

#include "Algorithm/RangeTokenizer.h"
#include "Framework/Logger.h"
#include "GPU/TPCFastTransform.h"
#include "SpacePoints/TrackResiduals.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TPCCalibration/TPCFastSpaceChargeCorrectionHelper.h"
#endif

#include "Algorithm/RangeTokenizer.h"

using namespace o2::tpc;
using namespace o2::gpu;

void TPCFastTransformInit(const char* fileName = "debugVoxRes.root", const char* fileNameInv = "debugVoxResInv.root",
                          const char* outFileName = "TPCFastTransform_VoxRes.root", bool useSmoothed = false, bool invertSigns = false)
{

  // Initialise TPCFastTransform object from "voxRes" tree of
  // o2::tpc::TrackResiduals::VoxRes track residual voxels
  //

  /*
    To visiualise the results:

    root -l transformDebug.root
    all->Draw("cx:y:z","sec==0&&iRow==10","")
    grid->Draw("cx:y:z","sec==0&&iRow==10","same")
    vox->Draw("vx:y:z","sec==0&&iRow==10","same")
    points->Draw("px:y:z","sec==0&&row==10","same")
  */

  const bool debugMirrorAdata2C = 0;

  if (gSystem->AccessPathName(fileName)) {
    std::cout << " input file " << fileName << " does not exist!" << std::endl;
    return;
  }

  auto file = std::unique_ptr<TFile>(TFile::Open(fileName, "READ"));
  if (!file || !file->IsOpen()) {
    std::cout << " input file " << fileName << " does not exist!" << std::endl;
    return;
  }

  TTree* voxResTree = nullptr;
  file->cd();
  gDirectory->GetObject("voxResTree", voxResTree);
  if (!voxResTree) {
    std::cout << "tree voxResTree does not exist!" << std::endl;
    return;
  }

  TTree* voxResTreeInverse = nullptr;
  std::unique_ptr<TFile> fileInv;
  if (fileNameInv && !std::string(fileNameInv).empty()) {
    fileInv = std::unique_ptr<TFile>(TFile::Open(fileNameInv, "READ"));
    if (!fileInv || !fileInv->IsOpen()) {
      std::cout << " input file " << fileNameInv << " does not exist!" << std::endl;
      return;
    }
    fileInv->cd();
    gDirectory->GetObject("voxResTree", voxResTreeInverse);
  }

  auto userInfo = voxResTree->GetUserInfo();

  if (!userInfo->FindObject("y2xBinning") || !userInfo->FindObject("z2xBinning")) {
    std::cout << "'y2xBinning' or 'z2xBinning' not found in UserInfo, but required to get the correct binning" << std::endl;
    return;
  }

  userInfo->Print();

  // required for the binning that was used
  o2::tpc::TrackResiduals trackResiduals;
  auto y2xBins = o2::RangeTokenizer::tokenize<float>(userInfo->FindObject("y2xBinning")->GetTitle());
  auto z2xBins = o2::RangeTokenizer::tokenize<float>(userInfo->FindObject("z2xBinning")->GetTitle());
  trackResiduals.setY2XBinning(y2xBins);
  trackResiduals.setZ2XBinning(z2xBins);
  trackResiduals.init();

  { // debug output

    std::cout << " ===== input track residuals ==== " << std::endl;
    std::cout << "voxel tree y2xBins: " << y2xBins.size() << std::endl;

    for (auto y2x : y2xBins) {
      std::cout << " y2x: " << y2x << std::endl;
    }
    std::cout << std::endl;

    int32_t nY2Xbins = trackResiduals.getNY2XBins();

    std::cout << " TrackResiduals y2x bins: " << nY2Xbins << std::endl;
    for (int32_t i = 0; i < nY2Xbins; i++) {
      std::cout << "scaled getY2X(bin) : " << trackResiduals.getY2X(0, i) / trackResiduals.getMaxY2X(0) << std::endl;
    }

    std::cout << "voxel tree z2xBins: " << z2xBins.size() << std::endl;

    for (auto z2x : z2xBins) {
      std::cout << "z2x: " << z2x << std::endl;
    }
    std::cout << std::endl;

    int32_t nZ2Xbins = trackResiduals.getNZ2XBins();
    std::cout << " TrackResiduals z2x bins: " << nZ2Xbins << std::endl;
    for (int32_t i = 0; i < nZ2Xbins; i++) {
      std::cout << "getZ2X(bin) : " << trackResiduals.getZ2X(i) << std::endl;
    }
    std::cout << " ==================================== " << std::endl;
  }

  std::cout << "create fast transformation ... " << std::endl;

  auto* helper = o2::tpc::TPCFastTransformHelperO2::instance();

  o2::tpc::TPCFastSpaceChargeCorrectionHelper* corrHelper = o2::tpc::TPCFastSpaceChargeCorrectionHelper::instance();

  corrHelper->setNthreadsToMaximum();
  corrHelper->setNthreads(1);

  if (debugMirrorAdata2C) {
    corrHelper->setDebugMirrorAdata2C();
  }
  // corrHelper->setDebugUseVoxelCenters();

  o2::gpu::TPCFastSpaceChargeCorrectionMap mapDirect(0, 0), mapInverse(0, 0);

  auto corrPtr = corrHelper->createFromTrackResiduals(trackResiduals, voxResTree, voxResTreeInverse, useSmoothed, invertSigns,
                                                      &mapDirect, &mapInverse);

  std::unique_ptr<o2::gpu::TPCFastTransform> fastTransform(
    helper->create(0, *corrPtr));

  std::cout << "... create fast transformation completed " << std::endl;

  if (*outFileName) {
    fastTransform->writeToFile(outFileName, "ccdb_object");
  }

  if (1) { // read transformation from the file

    // const char* fileName = "master/out.root";

    const char* fileName = outFileName;

    std::cout << "load corrections from file " << fileName << std::endl;

    fastTransform->cloneFromObject(*TPCFastTransform::loadFromFile(fileName, "ccdb_object"), nullptr);

    o2::gpu::TPCFastSpaceChargeCorrection& corr = fastTransform->getCorrection();

    if (0) {
      std::cout << "check the loaded correction ..." << std::endl;

      const o2::gpu::TPCFastTransformGeo& geo = helper->getGeometry();

      // for (int32_t iSector = 0; iSector < geo.getNumberOfSectors(); iSector++) {
      for (int32_t iSector = 0; iSector < 1; iSector++) {
        for (int32_t iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {
          auto& info = corr.getSectorRowInfo(iSector, iRow);
          std::cout << "sector " << iSector << " row " << iRow
                    << " gridY0 " << info.gridMeasured.getY0() << " gridZ0 " << info.gridMeasured.getZ0()
                    << " scaleYtoGrid " << info.gridMeasured.getYscale() << " scaleLtoGrid " << info.gridMeasured.getZscale()
                    << " gridRealY0 " << info.gridReal.getY0() << " gridRealZ0 " << info.gridReal.getZ0()
                    << " scaleRealYtoGrid " << info.gridReal.getYscale() << " scaleRealLtoGrid " << info.gridReal.getZscale()
                    << std::endl;
        }
      }
    }
  }

  std::cout << "verify the results ..." << std::endl;

  o2::gpu::TPCFastSpaceChargeCorrection& corr = fastTransform->getCorrection();

  // a debug file with some NTuples

  TDirectory* currDir = gDirectory;

  TFile* debugFile = new TFile("transformDebug.root", "RECREATE");
  debugFile->cd();

  // debug ntuple with created TPC corrections
  //
  // measured x,y,z; corrections cx,cy,cz from the measured to the real x,y,z;
  // inverse corrections ix,iy,iz at the real position (x+cx,y+cy,z+cz)
  // ideally, ix = cx, iy = cy, iz = cz
  TNtuple* ntAll = new TNtuple("all", "all",
                               debugMirrorAdata2C ? "sec:row:x:y:z:cx:cy:cz:ix:iy:iz:cxC:cyC:czC:ixC:iyC:izC"
                                                  : "sec:row:x:y:z:cx:cy:cz:ix:iy:iz");

  ntAll->SetMarkerStyle(8);
  ntAll->SetMarkerSize(0.1);
  ntAll->SetMarkerColor(kBlack);

  debugFile->cd();
  TNtuple* ntInvAll = new TNtuple("invall", "invall",
                                  debugMirrorAdata2C ? "sec:row:x:y:z:cx:cy:cz:cxC:cyC:czC"
                                                     : "sec:row:x:y:z:cx:cy:cz");

  ntInvAll->SetMarkerStyle(8);
  ntInvAll->SetMarkerSize(0.1);
  ntInvAll->SetMarkerColor(kBlack);

  // duplicate of debugVox + the spline data at voxels in a different color
  debugFile->cd();
  TNtuple* ntVox =
    new TNtuple("vox", "vox",
                debugMirrorAdata2C ? "sec:row:n:x:y:z:vx:vy:vz:cx:cy:cz:ix:iy:iz:cxC:cyC:czC:ixC:iyC:izC"
                                   : "sec:row:n:x:y:z:vx:vy:vz:cx:cy:cz:ix:iy:iz");

  ntVox->SetMarkerStyle(8);
  ntVox->SetMarkerSize(0.8);
  ntVox->SetMarkerColor(kMagenta);

  // duplicate of debugVox + the spline data at voxels in a different color
  debugFile->cd();
  TNtuple* ntInvVox =
    new TNtuple("invvox", "invvox",
                debugMirrorAdata2C ? "sec:row:n:x:y:z:vx:vy:vz:cx:cy:cz:cxC:cyC:czC"
                                   : "sec:row:n:x:y:z:vx:vy:vz:cx:cy:cz");

  ntInvVox->SetMarkerStyle(8);
  ntInvVox->SetMarkerSize(0.8);
  ntInvVox->SetMarkerColor(kMagenta);

  // corrections at the spline grid points
  debugFile->cd();
  TNtuple* ntGrid = new TNtuple("grid", "grid", "sec:row:x:y:z:cx:cy:cz:ix:iy:iz");

  ntGrid->SetMarkerStyle(8);
  ntGrid->SetMarkerSize(1.2);
  ntGrid->SetMarkerColor(kBlack);

  // corrections at the spline grid points
  debugFile->cd();
  TNtuple* ntInvGrid = new TNtuple("invgrid", "invgrid", "sec:row:x:y:z:cx:cy:cz");

  ntInvGrid->SetMarkerStyle(8);
  ntInvGrid->SetMarkerSize(1.2);
  ntGrid->SetMarkerColor(kBlack);

  // ntuple with data points created from voxels (with the data smearing, extension to the edges etc.)
  debugFile->cd();
  TNtuple* ntFitPoints =
    new TNtuple("fitpoints", "fit points", "sec:row:x:y:z:px:py:pz:cx:cy:cz");

  ntFitPoints->SetMarkerStyle(8);
  ntFitPoints->SetMarkerSize(0.4);
  ntFitPoints->SetMarkerColor(kRed);

  debugFile->cd();
  TNtuple* ntInvFitPoints =
    new TNtuple("invfitpoints", "fit points", "sec:row:x:y:z:px:py:pz:cx:cy:cz");

  ntInvFitPoints->SetMarkerStyle(8);
  ntInvFitPoints->SetMarkerSize(0.4);
  ntInvFitPoints->SetMarkerColor(kRed);

  currDir->cd();

  // check the difference in voxels and fill corresp. debug ntuple

  std::cout << "verify the results ..." << std::endl;

  const o2::gpu::TPCFastTransformGeo& geo = helper->getGeometry();

  auto getInvCorrections = [&](int iSector, int iRow, float realY, float realZ, float& ix, float& iy, float& iz) {
    // get the inverse corrections ix, iy, iz at x,y,z
    ix = corr.getCorrectionXatRealYZ(iSector, iRow, realY, realZ);
    const auto c = corr.getCorrectionYZatRealYZ(iSector, iRow, realY, realZ);
    iy = c[0];
    iz = c[1];
  };

  auto getAllCorrections = [&](int iSector, int iRow, float y, float z, float& cx, float& cy, float& cz, float& ix, float& iy, float& iz) {
    // get the corrections cx,cy,cz at x,y,z
    const auto c = corr.getCorrectionLocal(iSector, iRow, y, z);
    cx = c[0];
    cy = c[1];
    cz = c[2];
    getInvCorrections(iSector, iRow, y + cy, z + cz, ix, iy, iz);
  };

  for (int direction = 0; direction < 2; direction++) { // 0 - normal, 1 - inverse

    TTree* currentTree = (direction == 0) ? voxResTree : voxResTreeInverse;
    if (!currentTree) {
      std::cout << "tree voxResTree does not exist!" << std::endl;
      return;
    }

    o2::tpc::TrackResiduals::VoxRes* v = nullptr;
    TBranch* branch = currentTree->GetBranch("voxRes");
    branch->SetAddress(&v);
    branch->SetAutoDelete(kTRUE);

    int32_t iSectorLast = -1;
    int32_t iRowLast = -1;

    // the difference

    double maxDiff[3] = {0., 0., 0.};
    int32_t maxDiffSector[3] = {0, 0, 0};
    int32_t maxDiffRow[3] = {0, 0, 0};

    double sumDiff[3] = {0., 0., 0.};
    int64_t nDiff = 0;

    std::cout << "fill debug ntuples at voxels ..." << std::endl;

    for (int32_t iVox = 0; iVox < currentTree->GetEntriesFast(); iVox++) {

      currentTree->GetEntry(iVox);

      float voxEntries = v->stat[o2::tpc::TrackResiduals::VoxV];

      int32_t xBin =
        v->bvox[o2::tpc::TrackResiduals::VoxX]; // bin number in x (= pad row)

      int32_t y2xBin =
        v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14

      int32_t z2xBin =
        v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4

      int32_t iSector = (int32_t)v->bsec;
      int32_t iRow = (int32_t)xBin;

      iSectorLast = iSector;
      iRowLast = iRow;

      double x = trackResiduals.getX(xBin); // radius of the pad row

      double y2x = trackResiduals.getY2X(
        xBin, y2xBin); // y/x coordinate of the bin ~-0.15 ... 0.15

      double z2x =
        trackResiduals.getZ2X(z2xBin); // z/x coordinate of the bin 0.1 .. 0.9

      double y = x * y2x;
      double z = x * z2x;

      double correctionX = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResX] : v->D[o2::tpc::TrackResiduals::ResX];
      double correctionY = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResY] : v->D[o2::tpc::TrackResiduals::ResY];
      double correctionZ = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResZ] : v->D[o2::tpc::TrackResiduals::ResZ];

      double voxelSizeY = x / trackResiduals.getDY2XI(xBin, y2xBin);
      double voxelSizeZ = x * trackResiduals.getDZ2X(z2xBin);

      if (invertSigns) {
        correctionX *= -1.;
        correctionY *= -1.;
        correctionZ *= -1.;
      }

      if (!corrHelper->isDebugUseVoxelCenters()) {
        if (voxEntries > 0.) {
          // use mean statistical positions instead of the bin centers, unless they are wrong
          double yFit = x * v->stat[o2::tpc::TrackResiduals::VoxF];
          if (fabs(yFit - y) <= corrHelper->getVoxelMeanValidityRange() * voxelSizeY / 2.) {
            y = yFit;
          }
          double zFit = x * v->stat[o2::tpc::TrackResiduals::VoxZ];
          if (fabs(zFit - z) <= corrHelper->getVoxelMeanValidityRange() * voxelSizeZ / 2.) {
            z = zFit;
          }
        }
      }

      int mirrorSector = iSector + geo.getNumberOfSectorsA();

      if (iSector >= geo.getNumberOfSectorsA()) {
        z = -z;
        mirrorSector = iSector - geo.getNumberOfSectorsA();
      }

      float cx{0.f}, cy{0.f}, cz{0.f}, ix{0.f}, iy{0.f}, iz{0.f};
      float cxC{0.f}, cyC{0.f}, czC{0.f}, ixC{0.f}, iyC{0.f}, izC{0.f};
      if (direction == 0) {
        getAllCorrections(iSector, iRow, y, z, cx, cy, cz, ix, iy, iz);
        if (debugMirrorAdata2C) {
          getAllCorrections(mirrorSector, iRow, y, -z, cxC, cyC, czC, ixC, iyC, izC);
        }
        float ntEntry[] = {(float)iSector, (float)iRow, voxEntries,
                           (float)x, (float)y, (float)z,
                           (float)correctionX, (float)correctionY, (float)correctionZ,
                           (float)cx, (float)cy, (float)cz,
                           (float)ix, (float)iy, (float)iz,
                           (float)cxC, (float)cyC, (float)czC, (float)ixC, (float)iyC, (float)izC};

        // fill the ntuple with the correction at the voxel
        ntVox->Fill(ntEntry);
      } else {
        getInvCorrections(iSector, iRow, y, z, cx, cy, cz);
        if (debugMirrorAdata2C) {
          getInvCorrections(mirrorSector, iRow, y, -z, cxC, cyC, czC);
        }
        float ntEntry[] = {(float)iSector, (float)iRow, voxEntries,
                           (float)x, (float)y, (float)z,
                           (float)correctionX, (float)correctionY, (float)correctionZ,
                           (float)cx, (float)cy, (float)cz,
                           (float)cxC, (float)cyC, (float)czC};
        // fill the ntuple with the correction at the voxel
        ntInvVox->Fill(ntEntry);
      }

      if (voxEntries >= 1.) {
        double d[3] = {cx - correctionX, cy - correctionY, cz - correctionZ};

        for (int32_t i = 0; i < 3; i++) {
          if (fabs(maxDiff[i]) < fabs(d[i])) {
            maxDiff[i] = d[i];
            maxDiffSector[i] = iSector;
            maxDiffRow[i] = iRow;
            // std::cout << " sector " << iSector << " row " << iRow << " xyz " << i
            //  << " diff " << d[i] << " entries " << voxEntries << " y " << y2xBin << " z " << z2xBin << std::endl;
          }
          sumDiff[i] += d[i] * d[i];
        }
        nDiff++;
      }
    }

    std::cout
      << "fill debug ntuples everywhere .." << std::endl;

    for (int32_t iSector = 0; iSector < geo.getNumberOfSectors(); iSector++) {
      // for (int32_t iSector = 0; iSector < 1; iSector++) {
      std::cout << "debug ntules for sector " << iSector << std::endl;

      int mirrorSector = (iSector >= geo.getNumberOfSectorsA()) ? iSector - geo.getNumberOfSectorsA() : iSector + geo.getNumberOfSectorsA();

      for (int32_t iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {

        double x = geo.getRowInfo(iRow).x;

        // the spline grid

        const auto& gridY = corr.getSpline(iSector, iRow).getGridX1();
        const auto& gridZ = corr.getSpline(iSector, iRow).getGridX2();
        if (iSector == 0 && iRow == 0) {
          std::cout << "spline scenario " << corr.getSectorRowInfo(iSector, iRow).splineScenarioID << std::endl;
          std::cout << "spline grid Y: u = " << 0 << ".." << gridY.getUmax() << ", x = " << gridY.getXmin() << ".." << gridY.getXmax() << std::endl;
          std::cout << "spline grid Z: u = " << 0 << ".." << gridZ.getUmax() << ", x = " << gridZ.getXmin() << ".." << gridZ.getXmax() << std::endl;
        }

        // the correction
        {
          std::vector<double> points[2], knots[2];

          auto [yMin, yMax] = geo.getRowInfo(iRow).getYrange();
          auto [zMin, zMax] = geo.getZrange(iSector);

          points[0].push_back(yMin);
          points[0].push_back(yMax);
          points[1].push_back(zMin);
          points[1].push_back(zMax);

          for (int32_t iu = 0; iu < gridY.getNumberOfKnots(); iu++) {
            auto [y, z] = corr.convGridToLocal(iSector, iRow, gridY.getKnot(iu).getU(), 0.);
            knots[0].push_back(y);
            points[0].push_back(y);
          }
          for (int32_t iv = 0; iv < gridZ.getNumberOfKnots(); iv++) {
            auto [y, z] = corr.convGridToLocal(iSector, iRow, 0., gridZ.getKnot(iv).getU());
            knots[1].push_back(z);
            points[1].push_back(z);
          }

          for (int32_t iyz = 0; iyz <= 1; iyz++) {
            std::sort(knots[iyz].begin(), knots[iyz].end());
            std::sort(points[iyz].begin(), points[iyz].end());
            int32_t n = points[iyz].size();
            for (int32_t i = 0; i < n - 1; i++) {
              double d = (points[iyz][i + 1] - points[iyz][i]) / 10.;
              for (int32_t ii = 1; ii < 10; ii++) {
                points[iyz].push_back(points[iyz][i] + d * ii);
              }
            }
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
                float cxC{0}, cyC{0}, czC{0}, ixC{0}, iyC{0}, izC{0};
                if (direction == 0) {
                  getAllCorrections(iSector, iRow, y, z, cx, cy, cz, ix, iy, iz);
                  if (debugMirrorAdata2C) {
                    getAllCorrections(mirrorSector, iRow, y, -z, cxC, cyC, czC, ixC, iyC, izC);
                  }
                  if (iter == 0) {
                    ntGrid->Fill(iSector, iRow, x, y, z, cx, cy, cz, ix, iy, iz);
                  } else {
                    float ntEntry[] = {(float)iSector, (float)iRow, (float)x, y, z,
                                       cx, cy, cz, ix, iy, iz,
                                       cxC, cyC, czC, ixC, iyC, izC};
                    ntAll->Fill(ntEntry);
                  }
                } else {
                  getInvCorrections(iSector, iRow, y, z, cx, cy, cz);
                  if (debugMirrorAdata2C) {
                    getInvCorrections(mirrorSector, iRow, y, -z, cxC, cyC, czC);
                  }
                  if (iter == 0) {
                    ntInvGrid->Fill(iSector, iRow, x, y, z, cx, cy, cz);
                  } else {
                    float ntEntry[] = {(float)iSector, (float)iRow, (float)x, y, z,
                                       cx, cy, cz,
                                       cxC, cyC, czC};
                    ntInvAll->Fill(ntEntry);
                  }
                }
              }
            }
          }
        }

        // the data points used in spline fit
        // (they are kept in
        // TPCFastTransformHelperO2::instance()->getCorrectionMap() )

        o2::gpu::TPCFastSpaceChargeCorrectionMap& map = (direction == 0 ? mapDirect : mapInverse);

        auto& points = map.getPoints(iSector, iRow);

        for (uint32_t ip = 0; ip < points.size(); ip++) {
          auto point = points[ip];
          float y = point.mY;
          float z = point.mZ;
          float correctionX = point.mDx;
          float correctionY = point.mDy;
          float correctionZ = point.mDz;
          if (direction == 0) {
            auto [cx, cy, cz] =
              corr.getCorrectionLocal(iSector, iRow, y, z);
            ntFitPoints->Fill(iSector, iRow, x, y, z, correctionX, correctionY,
                              correctionZ, cx, cy, cz);
          } else {
            float cx =
              corr.getCorrectionXatRealYZ(iSector, iRow, y, z);
            auto [cy, cz] =
              corr.getCorrectionYZatRealYZ(iSector, iRow, y, z);
            ntInvFitPoints->Fill(iSector, iRow, x, y, z, correctionX, correctionY,
                                 correctionZ, cx, cy, cz);
          }
        }
      }
    }

    for (int32_t i = 0; i < 3; i++) {
      sumDiff[i] = sqrt(sumDiff[i]) / nDiff;
    }

    std::cout << "Max difference in x :  " << maxDiff[0] << " at Sector "
              << maxDiffSector[0] << " row " << maxDiffRow[0] << std::endl;

    std::cout << "Max difference in y :  " << maxDiff[1] << " at Sector "
              << maxDiffSector[1] << " row " << maxDiffRow[1] << std::endl;

    std::cout << "Max difference in z :  " << maxDiff[2] << " at Sector "
              << maxDiffSector[2] << " row " << maxDiffRow[2] << std::endl;

    std::cout << "Mean difference in x,y,z : " << sumDiff[0] << " " << sumDiff[1]
              << " " << sumDiff[2] << std::endl;
  } // direction

  corr.testInverse(true);

  debugFile->cd();
  ntAll->Write();
  ntVox->Write();
  ntGrid->Write();
  ntFitPoints->Write();
  ntInvAll->Write();
  ntInvVox->Write();
  ntInvGrid->Write();
  ntInvFitPoints->Write();

  debugFile->Close();
}
