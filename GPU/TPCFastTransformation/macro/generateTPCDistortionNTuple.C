// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  generateTPCDistortion.C
/// \brief A macro for generating TPC distortion ntuple to test the IrregularSpline2D3DCalibrator class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>
///
///  Load the macro:
///  .L generateTPCDistortion.C++
///

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "TFile.h"
#include "TNtuple.h"
#include "TH3.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCSimulation/SpaceCharge.h"
#include "GPU/TPCFastTransform.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"

#endif

o2::tpc::SpaceCharge* sc = 0;
using namespace o2::tpc;
using namespace o2::gpu;

void generateTPCDistortionNTuple()
{
  // open file with space-charge density histograms
  // location: alien:///alice/cern.ch/user/e/ehellbar/TPCSpaceCharge/RUN3/ SCDensityHistograms/InputSCDensityHistograms.root
  // The file contains several sc densities for epsilon=20 scenario
  // Naming convention: inputSCDensity3D<nPileUpEvents>
  //          - nPileUpEvents = 0,2,4,6,8,10,12
  //                - number of events (in units of 1000) contributing to the ion density. 0 = 130000 events, 2 = 2000 events, 4 = 4000 events, etc.
  //                - expected default map with 8000 pileup events: inputSCDensity3D8
  auto mFileSCDensity = std::unique_ptr<TFile>(TFile::Open("InputSCDensityHistograms.root"));
  if (!mFileSCDensity || !mFileSCDensity->IsOpen()) {
    std::cout << " input file does not exist!" << std::endl;
    return;
  }
  // get the histogram with the sc density
  std::unique_ptr<TH3> mHisSCDensity3D = std::unique_ptr<TH3>((TH3*)mFileSCDensity->Get("inputSCDensity3D8"));

  if (!mHisSCDensity3D) {
    std::cout << "inputSCDensity3D8 histogramm does not exist!" << std::endl;
    return;
  }

  // define number of bins in r, rphi, z for the lookup tables
  // nR annd nZ have to be 2^n + 1, nPhi is arbitrary
  // expected default for desired precision: 129, 129, 144 (takes a lot of time, you can also us 65 in r and z)
  int mNRRows = 129;
  int mNZCols = 129;
  int mNPhiSlices = 180;

  // create space-charge object
  sc = new o2::tpc::SpaceCharge(mNRRows, mNPhiSlices, mNZCols);
  sc->setInitialSpaceChargeDensity((TH3*)mHisSCDensity3D.get());
  // select constant distortions (over time), realistic distortions changing in time not yet pushed to official code
  sc->setSCDistortionType(o2::tpc::SpaceCharge::SCDistortionType::SCDistortionsConstant);
  // gas parameters nor Ne-CO2-N2 90-10-5
  sc->setOmegaTauT1T2(0.32, 1, 1);
  // start calculation of lookup tables (takes some time)
  sc->calculateLookupTables();

  // create TPC transformation to get the TPC geometry

  std::unique_ptr<o2::gpu::TPCFastTransform> fastTransform(o2::tpc::TPCFastTransformHelperO2::instance()->create(0));

  o2::gpu::TPCDistortionIRS& dist = fastTransform->getDistortionNonConst();
  const o2::gpu::TPCFastTransformGeo& geo = fastTransform->getGeometry();

  TFile* f = new TFile("tpcDistortion.root", "RECREATE");
  TNtuple* nt = new TNtuple("dist", "dist", "slice:row:su:sv:dx:du:dv");

  int nSlices = 1; //fastTransform->getNumberOfSlices();
  //for( int slice=0; slice<nSlices; slice++){
  for (int slice = 0; slice < 1; slice++) {
    const o2::gpu::TPCFastTransformGeo::SliceInfo& sliceInfo = geo.getSliceInfo(slice);

    for (int row = 0; row < geo.getNumberOfRows(); row++) {

      float x = geo.getRowInfo(row).x;

      for (float su = 0.; su <= 1.; su += 0.01) {
        for (float sv = 0.; sv <= 1.; sv += 0.01) {
          float u, v, y = 0, z = 0;
          geo.convScaledUVtoUV(slice, row, su, sv, u, v);
          geo.convUVtoLocal(slice, u, v, y, z);

          // local 2 global
          float gx, gy, gz;
          geo.convLocalToGlobal(slice, x, y, z, gx, gy, gz);

          o2::tpc::GlobalPosition3D positionCorrected(gx, gy, gz);
          sc->correctElectron(positionCorrected);
          gx = positionCorrected.x();
          gy = positionCorrected.y();
          gz = positionCorrected.z();

          // global to local
          float x1, y1, z1;
          geo.convGlobalToLocal(slice, gx, gy, gz, x1, y1, z1);
          float u1 = 0, v1 = 0;
          geo.convLocalToUV(slice, y1, z1, u1, v1);

          float dx = x1 - x;
          float du = u1 - u;
          float dv = v1 - v;
          cout << slice << " " << row << " " << su << " " << sv << " " << dx << " " << du << " " << dv << endl;
          nt->Fill(slice, row, su, sv, dx, du, dv);
        }
      }
    }
  }
  nt->Write();
  f->Write();
}
