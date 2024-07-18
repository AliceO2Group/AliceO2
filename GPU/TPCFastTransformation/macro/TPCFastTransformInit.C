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

void TPCFastTransformInit(const char* fileName = "debugVoxRes.root",
                          const char* outFileName = "TPCFastTransform_VoxRes.root", bool useSmoothed = false, bool invertSigns = false)
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
    corrvox->Draw("cx:y:z","iRoc==0&&iRow==10","same")
    points->Draw("px:y:z","iRoc==0&&iRow==10","same")
  */

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

    int nY2Xbins = trackResiduals.getNY2XBins();

    std::cout << " TrackResiduals y2x bins: " << nY2Xbins << std::endl;
    for (int i = 0; i < nY2Xbins; i++) {
      std::cout << "scaled getY2X(bin) : " << trackResiduals.getY2X(0, i) / trackResiduals.getMaxY2X(0) << std::endl;
    }

    std::cout << "voxel tree z2xBins: " << z2xBins.size() << std::endl;

    for (auto z2x : z2xBins) {
      std::cout << "z2x: " << z2x << std::endl;
    }
    std::cout << std::endl;

    int nZ2Xbins = trackResiduals.getNZ2XBins();
    std::cout << " TrackResiduals z2x bins: " << nZ2Xbins << std::endl;
    for (int i = 0; i < nZ2Xbins; i++) {
      std::cout << "getZ2X(bin) : " << trackResiduals.getZ2X(i) << std::endl;
    }
    std::cout << " ==================================== " << std::endl;
  }

  std::cout << "create fast transformation ... " << std::endl;

  auto* helper = o2::tpc::TPCFastTransformHelperO2::instance();

  o2::tpc::TPCFastSpaceChargeCorrectionHelper* corrHelper = o2::tpc::TPCFastSpaceChargeCorrectionHelper::instance();

  corrHelper->setNthreadsToMaximum();
  // corrHelper->setNthreads(1);

  auto corrPtr = corrHelper->createFromTrackResiduals(trackResiduals, voxResTree, useSmoothed, invertSigns);

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

      // for (int iRoc = 0; iRoc < geo.getNumberOfSlices(); iRoc++) {
      for (int iRoc = 0; iRoc < 1; iRoc++) {
        for (int iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {
          auto& info = corr.getSliceRowInfo(iRoc, iRow);
          std::cout << "roc " << iRoc << " row " << iRow
                    << " gridV0 " << info.gridV0 << " gridCorrU0 " << info.gridCorrU0 << " gridCorrV0 " << info.gridCorrV0
                    << " scaleCorrUtoGrid " << info.scaleCorrUtoGrid << " scaleCorrVtoGrid " << info.scaleCorrVtoGrid
                    << " gridU0 " << info.gridU0 << " scaleUtoGrid " << info.scaleUtoGrid << " scaleVtoGrid " << info.scaleVtoGrid
                    << std::endl;
        }
      }
    }
  }

  std::cout << "verify the results ..." << std::endl;

  o2::gpu::TPCFastSpaceChargeCorrection& corr = fastTransform->getCorrection();

  // the difference

  double maxDiff[3] = {0., 0., 0.};
  int maxDiffRoc[3] = {0, 0, 0};
  int maxDiffRow[3] = {0, 0, 0};

  double sumDiff[3] = {0., 0., 0.};
  long nDiff = 0;

  // a debug file with some NTuples

  TDirectory* currDir = gDirectory;

  TFile* debugFile = new TFile("transformDebug.root", "RECREATE");
  debugFile->cd();

  // ntuple with created TPC corrections
  TNtuple* debugCorr = new TNtuple("corr", "corr", "iRoc:iRow:x:y:z:cx:cy:cz");

  debugCorr->SetMarkerStyle(8);
  debugCorr->SetMarkerSize(0.1);
  debugCorr->SetMarkerColor(kBlack);

  // ntuple with the input data: voxel corrections
  debugFile->cd();
  TNtuple* debugVox =
    new TNtuple("vox", "vox", "iRoc:iRow:n:x:y:z:vx:vy:vz:cx:cy:cz");

  debugVox->SetMarkerStyle(8);
  debugVox->SetMarkerSize(0.8);
  debugVox->SetMarkerColor(kBlue);

  // duplicate of debugVox
  debugFile->cd();
  TNtuple* debugCorrVox =
    new TNtuple("corrvox", "corrvox", "iRoc:iRow:n:x:y:z:vx:vy:vz:cx:cy:cz");

  debugCorrVox->SetMarkerStyle(8);
  debugCorrVox->SetMarkerSize(0.8);
  debugCorrVox->SetMarkerColor(kMagenta);

  // ntuple with spline grid points
  debugFile->cd();
  TNtuple* debugGrid = new TNtuple("grid", "grid", "iRoc:iRow:x:y:z:cx:cy:cz");

  debugGrid->SetMarkerStyle(8);
  debugGrid->SetMarkerSize(1.2);
  debugGrid->SetMarkerColor(kBlack);

  // ntuple with data points created from voxels (with data smearing and
  // extension to the edges)
  debugFile->cd();
  TNtuple* debugPoints =
    new TNtuple("points", "points", "iRoc:iRow:x:y:z:px:py:pz:cx:cy:cz");

  debugPoints->SetMarkerStyle(8);
  debugPoints->SetMarkerSize(0.4);
  debugPoints->SetMarkerColor(kRed);

  currDir->cd();

  // check the difference in voxels and fill corresp. debug ntuple

  std::cout << "verify the results ..." << std::endl;

  const o2::gpu::TPCFastTransformGeo& geo = helper->getGeometry();

  o2::tpc::TrackResiduals::VoxRes* v = nullptr;
  TBranch* branch = voxResTree->GetBranch("voxRes");
  branch->SetAddress(&v);
  branch->SetAutoDelete(kTRUE);

  int iRocLast = -1;
  int iRowLast = -1;

  for (int iVox = 0; iVox < voxResTree->GetEntriesFast(); iVox++) {

    voxResTree->GetEntry(iVox);

    float voxEntries = v->stat[o2::tpc::TrackResiduals::VoxV];

    int xBin =
      v->bvox[o2::tpc::TrackResiduals::VoxX]; // bin number in x (= pad row)

    int y2xBin =
      v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14

    int z2xBin =
      v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4

    int iRoc = (int)v->bsec;
    int iRow = (int)xBin;

    iRocLast = iRoc;
    iRowLast = iRow;

    double x = trackResiduals.getX(xBin); // radius of the pad row

    double y2x = trackResiduals.getY2X(
      xBin, y2xBin); // y/x coordinate of the bin ~-0.15 ... 0.15

    double z2x =
      trackResiduals.getZ2X(z2xBin); // z/x coordinate of the bin 0.1 .. 0.9

    double y = x * y2x;
    double z = x * z2x;

    if (iRoc >= geo.getNumberOfSlicesA()) {
      z = -z;
    }

    double correctionX = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResX] : v->D[o2::tpc::TrackResiduals::ResX];
    double correctionY = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResY] : v->D[o2::tpc::TrackResiduals::ResY];
    double correctionZ = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResZ] : v->D[o2::tpc::TrackResiduals::ResZ];

    if (invertSigns) {
      correctionX *= -1.;
      correctionY *= -1.;
      correctionZ *= -1.;
    }

    if (voxEntries > 0.) { // use mean statistical positions instead of the bin centers:
      y = x * v->stat[o2::tpc::TrackResiduals::VoxF];
      z = x * v->stat[o2::tpc::TrackResiduals::VoxZ];
    }

    float u, v, cx, cu, cv, cy, cz;
    geo.convLocalToUV(iRoc, y, z, u, v);
    corr.getCorrection(iRoc, iRow, u, v, cx, cu, cv);
    geo.convUVtoLocal(iRoc, u + cu, v + cv, cy, cz);
    cy -= y;
    cz -= z;

    double d[3] = {cx - correctionX, cy - correctionY, cz - correctionZ};
    if (voxEntries >= 1.) {
      for (int i = 0; i < 3; i++) {
        if (fabs(maxDiff[i]) < fabs(d[i])) {
          maxDiff[i] = d[i];
          maxDiffRoc[i] = iRoc;
          maxDiffRow[i] = iRow;
          // std::cout << " roc " << iRoc << " row " << iRow << " xyz " << i
          //  << " diff " << d[i] << " entries " << voxEntries << " y " << y2xBin << " z " << z2xBin << std::endl;
        }
        sumDiff[i] += d[i] * d[i];
      }
      nDiff++;
    }

    debugVox->Fill(iRoc, iRow, voxEntries, x, y, z, correctionX, correctionY, correctionZ,
                   cx, cy, cz);
    debugCorrVox->Fill(iRoc, iRow, voxEntries, x, y, z, correctionX, correctionY, correctionZ,
                       cx, cy, cz);
  }

  std::cout << "create debug ntuples ..." << std::endl;

  for (int iRoc = 0; iRoc < geo.getNumberOfSlices(); iRoc++) {
    // for (int iRoc = 0; iRoc < 1; iRoc++) {
    std::cout << "debug ntules for roc " << iRoc << std::endl;
    for (int iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {

      double x = geo.getRowInfo(iRow).x;

      // the spline grid

      const auto& gridU = corr.getSpline(iRoc, iRow).getGridX1();
      const auto& gridV = corr.getSpline(iRoc, iRow).getGridX2();
      if (iRoc == 0 && iRow == 0) {
        std::cout << "spline scenario " << corr.getRowInfo(iRow).splineScenarioID << std::endl;
        std::cout << "spline grid U: u = " << 0 << ".." << gridU.getUmax() << ", x = " << gridU.getXmin() << ".." << gridU.getXmax() << std::endl;
        std::cout << "spline grid V: u = " << 0 << ".." << gridV.getUmax() << ", x = " << gridV.getXmin() << ".." << gridV.getXmax() << std::endl;
      }

      // the correction
      {
        std::vector<double> p[2], g[2];

        p[0].push_back(geo.getRowInfo(iRow).getUmin());
        for (int iu = 0; iu < gridU.getNumberOfKnots(); iu++) {
          float u, v;
          corr.convGridToUV(iRoc, iRow, gridU.getKnot(iu).getU(), 0., u, v);
          g[0].push_back(u);
          p[0].push_back(u);
        }
        p[0].push_back(geo.getRowInfo(iRow).getUmax());

        p[1].push_back(0.);
        for (int iv = 0; iv < gridV.getNumberOfKnots(); iv++) {
          float u, v;
          corr.convGridToUV(iRoc, iRow, 0., gridV.getKnot(iv).getU(), u, v);
          g[1].push_back(v);
          p[1].push_back(v);
        }
        p[1].push_back(geo.getTPCzLength(iRoc));

        for (int iuv = 0; iuv < 2; iuv++) {
          int n = p[iuv].size();
          for (unsigned int i = 0; i < n - 1; i++) {
            double d = (p[iuv][i + 1] - p[iuv][i]) / 10.;
            for (int ii = 1; ii < 10; ii++) {
              p[iuv].push_back(p[iuv][i] + d * ii);
            }
          }
          std::sort(p[iuv].begin(), p[iuv].end());
        }

        for (int iter = 0; iter < 2; iter++) {
          std::vector<double>& pu = ((iter == 0) ? g[0] : p[0]);
          std::vector<double>& pv = ((iter == 0) ? g[1] : p[1]);
          for (unsigned int iu = 0; iu < pu.size(); iu++) {
            for (unsigned int iv = 0; iv < pv.size(); iv++) {
              float u = pu[iu];
              float v = pv[iv];
              float x, y, z;
              geo.convUVtoLocal(iRoc, u, v, y, z);
              float cx, cu, cv;
              corr.getCorrection(iRoc, iRow, u, v, cx, cu, cv);
              float cy, cz;
              geo.convUVtoLocal(iRoc, u + cu, v + cv, cy, cz);
              cy -= y;
              cz -= z;
              if (iter == 0) {
                debugGrid->Fill(iRoc, iRow, x, y, z, cx, cy, cz);
              } else {
                debugCorr->Fill(iRoc, iRow, x, y, z, cx, cy, cz);
              }
            }
          }
        }
      }

      // the data points used in spline fit
      // (they are kept in
      // TPCFastTransformHelperO2::instance()->getCorrectionMap() )

      o2::gpu::TPCFastSpaceChargeCorrectionMap& map =
        corrHelper->getCorrectionMap();
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

        debugPoints->Fill(iRoc, iRow, x, y, z, correctionX, correctionY,
                          correctionZ, cx, cy, cz);
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    sumDiff[i] = sqrt(sumDiff[i]) / nDiff;
  }

  std::cout << "Max difference in x :  " << maxDiff[0] << " at ROC "
            << maxDiffRoc[0] << " row " << maxDiffRow[0] << std::endl;

  std::cout << "Max difference in y :  " << maxDiff[1] << " at ROC "
            << maxDiffRoc[1] << " row " << maxDiffRow[1] << std::endl;

  std::cout << "Max difference in z :  " << maxDiff[2] << " at ROC "
            << maxDiffRoc[2] << " row " << maxDiffRow[2] << std::endl;

  std::cout << "Mean difference in x,y,z : " << sumDiff[0] << " " << sumDiff[1]
            << " " << sumDiff[2] << std::endl;

  corr.testInverse(0);

  debugFile->cd();
  debugCorr->Write();
  debugVox->Write();
  debugCorrVox->Write();
  debugGrid->Write();
  debugPoints->Write();
  debugFile->Close();
}
