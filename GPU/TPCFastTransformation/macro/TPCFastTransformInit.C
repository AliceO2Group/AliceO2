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

#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"

#include "GPU/TPCFastTransform.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "SpacePoints/TrackResiduals.h"

#include "Riostream.h"

#endif

using namespace o2::tpc;
using namespace o2::gpu;

void TPCFastTransformInit(const char* fileName = "debugVoxRes.root")
{
  // Initialise TPCFastTransform object from "voxRes" tree of
  // o2::tpc::TrackResiduals::VoxRes track residual voxels
  //

  if (gSystem->AccessPathName(fileName)) {
    std::cout << " input file " << fileName << " does not exist!" << std::endl;
    return;
  }

  auto file = std::unique_ptr<TFile>(TFile::Open(fileName, "READ"));
  if (!file || !file->IsOpen()) {
    std::cout << " input file " << fileName << " does not exist!" << std::endl;
    return;
  }

  TTree* tree = nullptr;
  file->cd();
  gDirectory->GetObject("voxResTree", tree);
  if (!tree) {
    std::cout << "tree voxResTree does not exist!" << std::endl;
    return;
  }

  o2::tpc::TrackResiduals::VoxRes* v;
  tree->SetBranchAddress("voxRes", &v);

  o2::tpc::TrackResiduals trackResiduals;
  trackResiduals.init(); // also initializes the default binning which was used

  auto* helper = o2::tpc::TPCFastTransformHelperO2::instance();
  const o2::gpu::TPCFastTransformGeo& geo = helper->getGeometry();
  o2::gpu::TPCFastSpaceChargeCorrectionMap& map = helper->getCorrectionMap();
  map.init(geo.getNumberOfSlices(), geo.getNumberOfRows());

  for (int iVox = 0; iVox < tree->GetEntriesFast(); iVox++) {
    tree->GetEntry(iVox);
    auto xBin = v->bvox[o2::tpc::TrackResiduals::VoxX];   // bin number in x (= pad row)
    auto y2xBin = v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
    auto z2xBin = v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4

    auto x = trackResiduals.getX(xBin);             // radius of the pad row
    auto y2x = trackResiduals.getY2X(xBin, y2xBin); // y/x coordinate of the bin ~-0.15 ... 0.15
    auto z2x = trackResiduals.getZ2X(z2xBin);       // z/x coordinate of the bin 0.1 .. 0.9
    float y = x * y2x;
    float z = x * z2x;

    float correctionX = v->D[o2::tpc::TrackResiduals::ResX];
    float correctionY = v->D[o2::tpc::TrackResiduals::ResY];
    float correctionZ = v->D[o2::tpc::TrackResiduals::ResZ];
    int iRoc = (int)v->bsec;
    int iRow = (int)xBin;

    map.addCorrectionPoint(iRoc, iRow, y, z, correctionX, correctionY, correctionZ);
    // cout << iVox << ": sec " << iRoc << " row " << (int)xBin << " x y z " << x << " " << y << " " << z
    //    << " dx dy dz " << correctionX << " " << correctionY << " " << correctionZ << endl;
  }

  std::unique_ptr<o2::gpu::TPCFastTransform> fastTransform(helper->create(0));
  o2::gpu::TPCFastSpaceChargeCorrection& corr = fastTransform->getCorrection();

  // check the difference

  double maxDiff[3] = {0., 0., 0.};
  double sumDiff[3] = {0., 0., 0.};
  long nDiff = 0;

  for (int iVox = 0; iVox < tree->GetEntriesFast(); iVox++) {
    tree->GetEntry(iVox);

    auto xBin = v->bvox[o2::tpc::TrackResiduals::VoxX];   // bin number in x (= pad row)
    auto y2xBin = v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
    auto z2xBin = v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4

    auto x = trackResiduals.getX(xBin);             // radius of the pad row
    auto y2x = trackResiduals.getY2X(xBin, y2xBin); // y/x coordinate of the bin ~-0.15 ... 0.15
    auto z2x = trackResiduals.getZ2X(z2xBin);       // z/x coordinate of the bin 0.1 .. 0.9
    float y = x * y2x;
    float z = x * z2x;

    float correctionX = v->D[o2::tpc::TrackResiduals::ResX];
    float correctionY = v->D[o2::tpc::TrackResiduals::ResY];
    float correctionZ = v->D[o2::tpc::TrackResiduals::ResZ];

    int iRoc = (int)v->bsec;
    int iRow = (int)xBin;

    float u, v, cx, cu, cv, cy, cz;
    geo.convLocalToUV(iRoc, y, z, u, v);
    corr.getCorrection(iRoc, iRow, u, v, cx, cu, cv);
    geo.convUVtoLocal(iRoc, u + cu, v + cv, cy, cz);
    cy -= y;
    cz -= z;
    double d[3] = {cx - correctionX,
                   cy - correctionY,
                   cz - correctionZ};
    for (int i = 0; i < 3; i++) {
      if (fabs(maxDiff[i]) < fabs(d[i])) {
        maxDiff[i] = d[i];
      }
      sumDiff[i] += d[i] * d[i];
    }
    nDiff++;
  }
  for (int i = 0; i < 3; i++) {
    sumDiff[i] = sqrt(sumDiff[i]) / nDiff;
  }

  std::cout << "Max difference in x,y,z :  " << maxDiff[0] << " " << maxDiff[1] << " " << maxDiff[2] << endl;
  std::cout << "Mean difference in x,y,z : " << sumDiff[0] << " " << sumDiff[1] << " " << sumDiff[2] << endl;

  file->Close();
}
