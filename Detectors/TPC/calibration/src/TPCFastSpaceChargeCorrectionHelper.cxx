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

/// \file TPCFastSpaceChargeCorrectionHelper.cxx
/// \author Sergey Gorbunov

#include "TPCCalibration/TPCFastSpaceChargeCorrectionHelper.h"

#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"
#include "SpacePoints/TrackResiduals.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCFastTransform.h"
#include "Spline2DHelper.h"
#include "Riostream.h"
#include <fairlogger/Logger.h>

using namespace o2::gpu;

namespace o2
{
namespace tpc
{

TPCFastSpaceChargeCorrectionHelper* TPCFastSpaceChargeCorrectionHelper::sInstance = nullptr;

TPCFastSpaceChargeCorrectionHelper* TPCFastSpaceChargeCorrectionHelper::instance()
{
  // returns TPCFastSpaceChargeCorrectionHelper instance (singleton)
  if (!sInstance) {
    sInstance = new TPCFastSpaceChargeCorrectionHelper();
    sInstance->initGeometry();
  }
  return sInstance;
}

void TPCFastSpaceChargeCorrectionHelper::initGeometry()
{
  // initialize geometry

  const Mapper& mapper = Mapper::instance();

  const int nRows = mapper.getNumberOfRows();

  mGeo.startConstruction(nRows);

  auto& detParam = ParameterDetector::Instance();
  float tpcZlengthSideA = detParam.TPClength;
  float tpcZlengthSideC = detParam.TPClength;

  mGeo.setTPCzLength(tpcZlengthSideA, tpcZlengthSideC);

  mGeo.setTPCalignmentZ(0.);

  for (int iRow = 0; iRow < mGeo.getNumberOfRows(); iRow++) {
    Sector sector = 0;
    int regionNumber = 0;
    while (iRow >= mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber)) {
      regionNumber++;
    }

    const PadRegionInfo& region = mapper.getPadRegionInfo(regionNumber);

    int nPads = mapper.getNumberOfPadsInRowSector(iRow);
    float padWidth = region.getPadWidth();

    const GlobalPadNumber pad = mapper.globalPadNumber(PadPos(iRow, nPads / 2));
    const PadCentre& padCentre = mapper.padCentre(pad);
    float xRow = padCentre.X();

    mGeo.setTPCrow(iRow, xRow, nPads, padWidth);
  }

  mGeo.finishConstruction();

  // check if calculated pad geometry is consistent with the map
  testGeometry(mGeo);

  mIsInitialized = 1;
}

std::unique_ptr<TPCFastSpaceChargeCorrection> TPCFastSpaceChargeCorrectionHelper::create(const int nKnotsY, const int nKnotsZ)
{
  // create the correction map

  if (!mIsInitialized) {
    initGeometry();
  }

  std::unique_ptr<TPCFastSpaceChargeCorrection> correctionPtr(new TPCFastSpaceChargeCorrection);

  TPCFastSpaceChargeCorrection& correction = *correctionPtr;

  const int nRows = mGeo.getNumberOfRows();
  const int nCorrectionScenarios = nRows / 10 + 1;

  correction.startConstruction(mGeo, nCorrectionScenarios);

  // assign spline type for TPC rows
  for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
    int scenario = row / 10;
    if (scenario >= nCorrectionScenarios) {
      scenario = nCorrectionScenarios - 1;
    }
    correction.setRowScenarioID(row, scenario);
  }

  // adjust the number of knots and the knot positions for the TPC correction splines

  /*
     TODO: update the calibrator
    IrregularSpline2D3DCalibrator calibrator;
    calibrator.setRasterSize(41, 41);
    calibrator.setMaxNKnots(21, 21);
    calibrator.setMaximalDeviation(0.01);

    IrregularSpline2D3D raster;
    raster.constructRegular(101, 101);
    std::vector<float> rasterData(3 * raster.getNumberOfKnots());
    */

  for (int scenario = 0; scenario < nCorrectionScenarios; scenario++) {
    int row = scenario * 10;
    TPCFastSpaceChargeCorrection::SplineType spline;
    if (!mCorrectionMap.isInitialized() || row >= nRows) {
      spline.recreate(nKnotsY, nKnotsZ);
    } else {
      // TODO: update the calibrator
      spline.recreate(nKnotsY, nKnotsZ);
      /*
        // create the input function
        for (int knot = 0; knot < raster.getNumberOfKnots(); knot++) {
          float su = 0.f, sv = 0.f;
          raster.getKnotUV(knot, su, sv);
          float dx = 0.f, du = 0.f, dv = 0.f;
          const int slice = 0;
          getSpaceChargeCorrection(slice, row, su, sv, dx, du, dv);
          rasterData[3 * knot + 0] = dx;
          rasterData[3 * knot + 1] = du;
          rasterData[3 * knot + 2] = dv;
        }
        raster.correctEdges(rasterData.data());
        std::function<void(float, float, float&, float&, float&)> f;
        f = [&raster, &rasterData](float su, float sv, float& dx, float& du, float& dv) {
          raster.getSpline(rasterData.data(), su, sv, dx, du, dv);
        };

        calibrator.calibrateSpline(spline, f);
        std::cout << "calibrated spline for scenario " << scenario << ", TPC row " << row << ": knots u "
                  << spline.getGridU().getNumberOfKnots() << ", v "
                  << spline.getGridV().getNumberOfKnots() << std::endl;
                  */
    }
    correction.setSplineScenario(scenario, spline);
  }
  correction.finishConstruction();

  fillSpaceChargeCorrectionFromMap(correction);

  return std::move(correctionPtr);
}

void TPCFastSpaceChargeCorrectionHelper::fillSpaceChargeCorrectionFromMap(TPCFastSpaceChargeCorrection& correction)
{
  // calculate correction map: dx,du,dv = ( origTransform() -> x,u,v) - fastTransformNominal:x,u,v
  // for the future: switch TOF correction off for a while

  if (!mIsInitialized) {
    initGeometry();
  }

  if (!mCorrectionMap.isInitialized()) {
    correction.setNoCorrection();
    return;
  }

  for (int slice = 0; slice < correction.getGeometry().getNumberOfSlices(); slice++) {
    for (int row = 0; row < correction.getGeometry().getNumberOfRows(); row++) {
      TPCFastSpaceChargeCorrection::SplineType& spline = correction.getSpline(slice, row);
      Spline2DHelper<float> helper;
      float* splineParameters = correction.getSplineData(slice, row);
      /* old style

        helper.setSpline(spline, 3, 3);
        auto F = [&](double su, double sv, double dxuv[3]) {
          getSpaceChargeCorrection(slice, row, su, sv, dxuv[0], dxuv[1], dxuv[2]);
        };
        helper.approximateFunction(splineParameters, 0., 1., 0., 1., F);
        */
      const std::vector<o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint>& data = mCorrectionMap.getPoints(slice, row);
      int nDataPoints = data.size();
      if (nDataPoints >= 4) {
        std::vector<double> pointSU(nDataPoints);
        std::vector<double> pointSV(nDataPoints);
        std::vector<double> pointCorr(3 * nDataPoints); // 3 dimensions
        for (int i = 0; i < nDataPoints; ++i) {
          double su, sv, dx, du, dv;
          getSpaceChargeCorrection(correction, slice, row, data[i], su, sv, dx, du, dv);
          pointSU[i] = su;
          pointSV[i] = sv;
          pointCorr[3 * i + 0] = dx;
          pointCorr[3 * i + 1] = du;
          pointCorr[3 * i + 2] = dv;
        }
        helper.approximateDataPoints(spline, splineParameters, 0., spline.getGridX1().getNumberOfKnots() - 1, 0., spline.getGridX2().getNumberOfKnots() - 1, &pointSU[0],
                                     &pointSV[0], &pointCorr[0], nDataPoints);
      } else {
        for (int i = 0; i < spline.getNumberOfParameters(); i++) {
          splineParameters[i] = 0.f;
        }
      }
    } // row
  }   // slice
  correction.initInverse();
}

void TPCFastSpaceChargeCorrectionHelper::getSpaceChargeCorrection(const TPCFastSpaceChargeCorrection& correction, int slice, int row, o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint p,
                                                                  double& su, double& sv, double& dx, double& du, double& dv)
{
  // get space charge correction in internal TPCFastTransform coordinates su,sv->dx,du,dv

  if (!mIsInitialized) {
    initGeometry();
  }

  // not corrected coordinates in u,v
  float u = 0.f, v = 0.f, fsu = 0.f, fsv = 0.f;
  mGeo.convLocalToUV(slice, p.mY, p.mZ, u, v);
  correction.convUVtoGrid(slice, row, u, v, fsu, fsv);
  // mGeo.convUVtoScaledUV(slice, row, u, v, fsu, fsv);
  su = fsu;
  sv = fsv;
  // corrected coordinates in u,v
  float u1 = 0.f, v1 = 0.f;
  mGeo.convLocalToUV(slice, p.mY + p.mDy, p.mZ + p.mDz, u1, v1);

  dx = p.mDx;
  du = u1 - u;
  dv = v1 - v;
}

void TPCFastSpaceChargeCorrectionHelper::setGlobalSpaceChargeCorrection(
  std::function<void(int roc, double gx, double gy, double gz,
                     double& dgx, double& dgy, double& dgz)>
    correctionGlobal)
{
  auto correctionLocal = [&](int roc, int irow, double ly, double lz,
                             double& dlx, double& dly, double& dlz) {
    double lx = mGeo.getRowInfo(irow).x;
    float gx, gy, gz;
    mGeo.convLocalToGlobal(roc, lx, ly, lz, gx, gy, gz);
    double dgx, dgy, dgz;
    correctionGlobal(roc, gx, gy, gz, dgx, dgy, dgz);
    float lx1, ly1, lz1;
    mGeo.convGlobalToLocal(roc, gx + dgx, gy + dgy, gz + dgz, lx1, ly1, lz1);
    dlx = lx1 - lx;
    dly = ly1 - ly;
    dlz = lz1 - lz;
  };
  setLocalSpaceChargeCorrection(correctionLocal);
}

void TPCFastSpaceChargeCorrectionHelper::setLocalSpaceChargeCorrection(
  std::function<void(int roc, int irow, double y, double z,
                     double& dx, double& dy, double& dz)>
    correctionLocal)
{
  /// set space charge correction in the local coordinates
  /// as a continious function

  int nRocs = mGeo.getNumberOfSlices();
  int nRows = mGeo.getNumberOfRows();
  mCorrectionMap.init(nRocs, nRows);

  for (int iRoc = 0; iRoc < nRocs; iRoc++) {
    for (int iRow = 0; iRow < nRows; iRow++) {
      double dsu = 1. / (3 * 8 - 3);
      double dsv = 1. / (3 * 20 - 3);
      for (double su = 0.f; su < 1.f + .5 * dsu; su += dsv) {
        for (double sv = 0.f; sv < 1.f + .5 * dsv; sv += dsv) {
          float ly, lz;
          mGeo.convScaledUVtoLocal(iRoc, iRow, su, sv, ly, lz);
          double dx, dy, dz;
          correctionLocal(iRoc, iRow, ly, lz, dx, dy, dz);
          mCorrectionMap.addCorrectionPoint(iRoc, iRow,
                                            ly, lz, dx, dy, dz);
        }
      }
    } // row
  }   // roc
}

void TPCFastSpaceChargeCorrectionHelper::testGeometry(const TPCFastTransformGeo& geo) const
{
  const Mapper& mapper = Mapper::instance();

  if (geo.getNumberOfSlices() != Sector::MAXSECTOR) {
    LOG(fatal) << "Wrong number of sectors :" << geo.getNumberOfSlices() << " instead of " << Sector::MAXSECTOR << std::endl;
  }

  if (geo.getNumberOfRows() != mapper.getNumberOfRows()) {
    LOG(fatal) << "Wrong number of rows :" << geo.getNumberOfRows() << " instead of " << mapper.getNumberOfRows() << std::endl;
  }

  double maxDx = 0, maxDy = 0;

  for (int row = 0; row < geo.getNumberOfRows(); row++) {

    const int nPads = geo.getRowInfo(row).maxPad + 1;

    if (nPads != mapper.getNumberOfPadsInRowSector(row)) {
      LOG(fatal) << "Wrong number of pads :" << nPads << " instead of " << mapper.getNumberOfPadsInRowSector(row) << std::endl;
    }

    const double x = geo.getRowInfo(row).x;

    // check if calculated pad positions are equal to the real ones

    for (int pad = 0; pad < nPads; pad++) {
      const GlobalPadNumber p = mapper.globalPadNumber(PadPos(row, pad));
      const PadCentre& c = mapper.padCentre(p);
      double u = geo.convPadToU(row, pad);

      const double dx = x - c.X();
      const double dy = u - (-c.Y()); // diferent sign convention for Y coordinate in the map

      if (fabs(dx) >= 1.e-6 || fabs(dy) >= 1.e-5) {
        LOG(warning) << "wrong calculated pad position:"
                     << " row " << row << " pad " << pad << " x calc " << x << " x in map " << c.X() << " dx " << (x - c.X())
                     << " y calc " << u << " y in map " << -c.Y() << " dy " << dy << std::endl;
      }
      if (fabs(maxDx) < fabs(dx)) {
        maxDx = dx;
      }
      if (fabs(maxDy) < fabs(dy)) {
        maxDy = dy;
      }
    }
  }

  if (fabs(maxDx) >= 1.e-4 || fabs(maxDy) >= 1.e-4) {
    LOG(fatal) << "wrong calculated pad position:"
               << " max Dx " << maxDx << " max Dy " << maxDy << std::endl;
  }
}

std::unique_ptr<o2::gpu::TPCFastSpaceChargeCorrection> TPCFastSpaceChargeCorrectionHelper::createSpaceChargeCorrection(
  const o2::tpc::TrackResiduals& trackResiduals, TTree* voxResTree)
{
  // create o2::gpu::TPCFastSpaceChargeCorrection  from o2::tpc::TrackResiduals::VoxRes voxel tree

  std::unique_ptr<o2::gpu::TPCFastSpaceChargeCorrection> correctionPtr(new o2::gpu::TPCFastSpaceChargeCorrection);

  o2::gpu::TPCFastSpaceChargeCorrection& correction = *correctionPtr;

  // o2::tpc::TrackResiduals::VoxRes* v = nullptr;
  // voxResTree->SetBranchAddress("voxRes", &v);

  o2::tpc::TrackResiduals::VoxRes* v = nullptr;
  TBranch* branch = voxResTree->GetBranch("voxRes");
  branch->SetAddress(&v);
  branch->SetAutoDelete(kTRUE);

  auto* helper = o2::tpc::TPCFastSpaceChargeCorrectionHelper::instance();
  const o2::gpu::TPCFastTransformGeo& geo = helper->getGeometry();

  o2::gpu::TPCFastSpaceChargeCorrectionMap& map = helper->getCorrectionMap();
  map.init(geo.getNumberOfSlices(), geo.getNumberOfRows());

  int nY2Xbins = trackResiduals.getNY2XBins();
  int nZ2Xbins = trackResiduals.getNZ2XBins();

  int nKnotsY = nY2Xbins / 2;
  int nKnotsZ = nZ2Xbins / 2;

  if (nKnotsY < 2) {
    nKnotsY = 2;
  }

  if (nKnotsZ < 2) {
    nKnotsZ = 2;
  }

  // std::cout << "n knots Y: " << nKnotsY << std::endl;
  // std::cout << "n knots Z: " << nKnotsZ << std::endl;

  { // create the correction object

    const int nRows = geo.getNumberOfRows();
    const int nCorrectionScenarios = 1;

    correction.startConstruction(geo, nCorrectionScenarios);

    // init rows
    for (int row = 0; row < geo.getNumberOfRows(); row++) {
      correction.setRowScenarioID(row, 0);
    }
    { // init spline scenario
      TPCFastSpaceChargeCorrection::SplineType spline;
      spline.recreate(nKnotsY, nKnotsZ);
      correction.setSplineScenario(0, spline);
    }
    correction.finishConstruction();
  } // .. create the correction object

  // set the grid borders in Z to Z/X==1
  for (int iRoc = 0; iRoc < geo.getNumberOfSlices(); iRoc++) {
    for (int iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {
      auto rowInfo = geo.getRowInfo(iRow);
      o2::gpu::TPCFastSpaceChargeCorrection::SliceRowInfo& info = correction.getSliceRowInfo(iRoc, iRow);
      double len = 0.;
      if (iRoc < geo.getNumberOfSlicesA()) {
        len = geo.getTPCzLengthA();
      } else {
        len = geo.getTPCzLengthC();
      }
      info.gridV0 = len - rowInfo.x;
      if (info.gridV0 < 0.) {
        info.gridV0 = 0.;
      }
    }
  }

  for (int iVox = 0; iVox < voxResTree->GetEntriesFast(); iVox++) {

    voxResTree->GetEntry(iVox);
    auto xBin =
      v->bvox[o2::tpc::TrackResiduals::VoxX]; // bin number in x (= pad row)
    auto y2xBin =
      v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
    auto z2xBin =
      v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4

    int iRoc = (int)v->bsec;
    int iRow = (int)xBin;

    // x,y,z of the voxel in local TPC coordinates

    double x = trackResiduals.getX(xBin); // radius of the pad row
    double y2x = trackResiduals.getY2X(
      xBin, y2xBin); // y/x coordinate of the bin ~-0.15 ... 0.15
    double z2x =
      trackResiduals.getZ2X(z2xBin); // z/x coordinate of the bin 0.1 .. 0.9
    double y = x * y2x;
    double z = x * z2x;

    if (iRoc >= geo.getNumberOfSlicesA()) {
      z = -z;
      // y = -y;
    }

    {
      float sx, sy, sz;
      trackResiduals.getVoxelCoordinates(iRoc, xBin, y2xBin, z2xBin, sx, sy, sz);
      sy *= x;
      sz *= x;
      if (fabs(sx - x) + fabs(sy - y) + fabs(sz - z) > 1.e-4) {
        std::cout << "wrong coordinates: " << x << " " << y << " " << z << " / " << sx << " " << sy << " " << sz << std::endl;
      }
    }

    // TODO: skip empty voxels?
    // float voxEntries = v->stat[o2::tpc::TrackResiduals::VoxV];
    // if (voxEntries < 1.) { // no statistics
    // continue;
    //}

    // double statX = v->stat[o2::tpc::TrackResiduals::VoxX]; // weight
    // double statY = v->stat[o2::tpc::TrackResiduals::VoxF]; // weight
    // double statZ = v->stat[o2::tpc::TrackResiduals::VoxZ]; // weight

    // double dx = 1. / trackResiduals.getDXI(xBin);
    double dy = x / trackResiduals.getDY2XI(xBin, y2xBin);
    double dz = x * trackResiduals.getDZ2X(z2xBin);

    double correctionX = v->D[o2::tpc::TrackResiduals::ResX];
    double correctionY = v->D[o2::tpc::TrackResiduals::ResY];
    double correctionZ = v->D[o2::tpc::TrackResiduals::ResZ];
    double correctionD = v->D[o2::tpc::TrackResiduals::ResD];

    // add one point per voxel

    // map.addCorrectionPoint(iRoc, iRow, y, z, correctionX, correctionY,
    //                     correctionZ);

    // add several points per voxel,
    // extend values of the edge voxels to the edges of the TPC row
    //

    double yFirst = y - dy / 2.;
    double yLast = y + dy / 2.;

    if (y2xBin == 0) { // extend value of the first Y bin to the row edge
      float u, v;
      if (iRoc < geo.getNumberOfSlicesA()) {
        geo.convScaledUVtoUV(iRoc, iRow, 0., 0., u, v);
      } else {
        geo.convScaledUVtoUV(iRoc, iRow, 1., 0., u, v);
      }
      float py, pz;
      geo.convUVtoLocal(iRoc, u, v, py, pz);
      yFirst = py;
    }

    if (y2xBin == trackResiduals.getNY2XBins() - 1) { // extend value of the last Y bin to the row edge
      float u, v;
      if (iRoc < geo.getNumberOfSlicesA()) {
        geo.convScaledUVtoUV(iRoc, iRow, 1., 0., u, v);
      } else {
        geo.convScaledUVtoUV(iRoc, iRow, 0., 0., u, v);
      }
      float py, pz;
      geo.convUVtoLocal(iRoc, u, v, py, pz);
      yLast = py;
    }

    double z0 = 0.;
    if (iRoc < geo.getNumberOfSlicesA()) {
      z0 = geo.getTPCzLengthA();
    } else {
      z0 = -geo.getTPCzLengthC();
    }

    double yStep = (yLast - yFirst) / 2;

    for (double py = yFirst; py <= yLast + yStep / 2.; py += yStep) {

      for (double pz = z - dz / 2.; pz <= z + dz / 2. + 1.e-4; pz += dz / 2.) {
        map.addCorrectionPoint(iRoc, iRow, py, pz, correctionX, correctionY,
                               correctionZ);
      }

      if (z2xBin == trackResiduals.getNZ2XBins() - 1) {
        // extend value of the first Z bin to the readout, linear decrease of all values to 0.
        int nZsteps = 3;
        for (int is = 0; is < nZsteps; is++) {
          double pz = z + (z0 - z) * (is + 1.) / nZsteps;
          double s = (nZsteps - 1. - is) / nZsteps;
          map.addCorrectionPoint(iRoc, iRow, py, pz, s * correctionX,
                                 s * correctionY, s * correctionZ);
        }
      }
    }
  }
  helper->fillSpaceChargeCorrectionFromMap(correction);
  return std::move(correctionPtr);
}

} // namespace tpc
} // namespace o2
