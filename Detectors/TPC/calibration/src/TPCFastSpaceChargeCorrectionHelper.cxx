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
#include "ChebyshevFit1D.h"
#include "Spline2DHelper.h"
#include "Riostream.h"
#include <fairlogger/Logger.h>
#include <thread>
#include "TStopwatch.h"

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

void TPCFastSpaceChargeCorrectionHelper::setNthreads(int n)
{
  LOG(info) << "fast space charge correction helper: use " << n << ((n > 1) ? " cpu threads" : " cpu thread");
  mNthreads = (n > 0) ? n : 1;
}

void TPCFastSpaceChargeCorrectionHelper::setNthreadsToMaximum()
{
  /// sets number of threads to N cpu cores

  mNthreads = std::thread::hardware_concurrency();

  LOG(info) << "fast space charge correction helper: use " << mNthreads << ((mNthreads > 1) ? " cpu threads" : " cpu thread");

  if (mNthreads < 1) {
    mNthreads = 1;
  }
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

  LOG(info) << "fast space charge correction helper: init from data points";

  for (int slice = 0; slice < correction.getGeometry().getNumberOfSlices(); slice++) {

    auto myThread = [&](int iThread) {
      for (int row = iThread; row < correction.getGeometry().getNumberOfRows(); row += mNthreads) {

        TPCFastSpaceChargeCorrection::SplineType& spline = correction.getSpline(slice, row);
        Spline2DHelper<float> helper;
        float* splineParameters = correction.getSplineData(slice, row);
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
    };  // thread

    std::vector<std::thread> threads(mNthreads);

    // run n threads
    for (int i = 0; i < mNthreads; i++) {
      threads[i] = std::thread(myThread, i);
    }

    // wait for the threads to finish
    for (auto& th : threads) {
      th.join();
    }

  } // slice

  initInverse(correction, 0);
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

std::unique_ptr<TPCFastSpaceChargeCorrection> TPCFastSpaceChargeCorrectionHelper::createFromGlobalCorrection(
  std::function<void(int roc, double gx, double gy, double gz,
                     double& dgx, double& dgy, double& dgz)>
    correctionGlobal,
  const int nKnotsY, const int nKnotsZ)
{
  /// creates TPCFastSpaceChargeCorrection object from a continious space charge correction in global coordinates

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
  return std::move(createFromLocalCorrection(correctionLocal, nKnotsY, nKnotsZ));
}

std::unique_ptr<TPCFastSpaceChargeCorrection> TPCFastSpaceChargeCorrectionHelper::createFromLocalCorrection(
  std::function<void(int roc, int irow, double y, double z, double& dx, double& dy, double& dz)> correctionLocal,
  const int nKnotsY, const int nKnotsZ)
{
  /// creates TPCFastSpaceChargeCorrection object from a continious space charge correction in local coordinates

  LOG(info) << "fast space charge correction helper: create correction using " << mNthreads << " threads";

  std::unique_ptr<TPCFastSpaceChargeCorrection> correctionPtr(new TPCFastSpaceChargeCorrection);
  TPCFastSpaceChargeCorrection& correction = *correctionPtr;

  { // create a correction map

    if (!mIsInitialized) {
      initGeometry();
    }

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

    for (int scenario = 0; scenario < nCorrectionScenarios; scenario++) {
      int row = scenario * 10;
      TPCFastSpaceChargeCorrection::SplineType spline;
      spline.recreate(nKnotsY, nKnotsZ);
      correction.setSplineScenario(scenario, spline);
    }
    correction.finishConstruction();
  }

  LOG(info) << "fast space charge correction helper: fill data points from an input SP correction function";

  {
    /// set space charge correction in the local coordinates
    /// as a continious function

    int nRocs = mGeo.getNumberOfSlices();
    int nRows = mGeo.getNumberOfRows();
    mCorrectionMap.init(nRocs, nRows);

    for (int iRoc = 0; iRoc < nRocs; iRoc++) {

      auto myThread = [&](int iThread) {
        for (int iRow = iThread; iRow < nRows; iRow += mNthreads) {
          const auto& info = mGeo.getRowInfo(iRow);
          double vMax = mGeo.getTPCzLength(iRoc);
          double dv = vMax / (6. * (nKnotsZ - 1));

          double dpad = info.maxPad / (6. * (nKnotsY - 1));
          for (double pad = 0; pad < info.maxPad + .5 * dpad; pad += dpad) {
            float u = mGeo.convPadToU(iRow, pad);
            for (double v = 0.; v < vMax + .5 * dv; v += dv) {
              float ly, lz;
              mGeo.convUVtoLocal(iRoc, u, v, ly, lz);
              double dx, dy, dz;
              correctionLocal(iRoc, iRow, ly, lz, dx, dy, dz);
              mCorrectionMap.addCorrectionPoint(iRoc, iRow,
                                                ly, lz, dx, dy, dz);
            }
          }
        } // row
      };  // thread

      std::vector<std::thread> threads(mNthreads);

      // run n threads
      for (int i = 0; i < mNthreads; i++) {
        threads[i] = std::thread(myThread, i);
      }

      // wait for the threads to finish
      for (auto& th : threads) {
        th.join();
      }

    } // roc

    fillSpaceChargeCorrectionFromMap(correction);
  }

  return std::move(correctionPtr);
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

std::unique_ptr<o2::gpu::TPCFastSpaceChargeCorrection> TPCFastSpaceChargeCorrectionHelper::createFromTrackResiduals(
  const o2::tpc::TrackResiduals& trackResiduals, TTree* voxResTree, bool useSmoothed, bool invertSigns)
{
  // create o2::gpu::TPCFastSpaceChargeCorrection  from o2::tpc::TrackResiduals::VoxRes voxel tree

  LOG(info) << "fast space charge correction helper: create correction using " << mNthreads << " threads";

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
      double len = geo.getTPCzLength(iRoc);
      info.gridV0 = len - rowInfo.x;
      if (info.gridV0 < 0.) {
        info.gridV0 = 0.;
      }
    }
  }

  LOG(info) << "fast space charge correction helper: fill data points from track residuals";

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

    // skip empty voxels
    float voxEntries = v->stat[o2::tpc::TrackResiduals::VoxV];
    if (voxEntries < 1.) { // no statistics
      continue;
    }

    // double statX = v->stat[o2::tpc::TrackResiduals::VoxX]; // weight
    // double statY = v->stat[o2::tpc::TrackResiduals::VoxF]; // weight
    // double statZ = v->stat[o2::tpc::TrackResiduals::VoxZ]; // weight

    // double dx = 1. / trackResiduals.getDXI(xBin);
    double dy = x / trackResiduals.getDY2XI(xBin, y2xBin);
    double dz = x * trackResiduals.getDZ2X(z2xBin);

    double correctionX = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResX] : v->D[o2::tpc::TrackResiduals::ResX];
    double correctionY = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResY] : v->D[o2::tpc::TrackResiduals::ResY];
    double correctionZ = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResZ] : v->D[o2::tpc::TrackResiduals::ResZ];
    if (invertSigns) {
      correctionX *= -1.;
      correctionY *= -1.;
      correctionZ *= -1.;
    }
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

void TPCFastSpaceChargeCorrectionHelper::initMaxDriftLength(o2::gpu::TPCFastSpaceChargeCorrection& correction, bool prn)
{
  /// initialise max drift length

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  ChebyshevFit1D chebFitter;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    if (prn) {
      LOG(info) << "init MaxDriftLength for slice " << slice;
    }
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    TPCFastSpaceChargeCorrection::SliceInfo& sliceInfo = correction.getSliceInfo(slice);
    sliceInfo.vMax = 0.f;

    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      TPCFastSpaceChargeCorrection::RowActiveArea& area = correction.getSliceRowInfo(slice, row).activeArea;
      area.cvMax = 0;
      area.vMax = 0;
      area.cuMin = mGeo.convPadToU(row, 0.f);
      area.cuMax = -area.cuMin;
      chebFitter.reset(4, 0., mGeo.getRowInfo(row).maxPad);
      double x = mGeo.getRowInfo(row).x;
      for (int pad = 0; pad < mGeo.getRowInfo(row).maxPad; pad++) {
        float u = mGeo.convPadToU(row, (float)pad);
        float v0 = 0;
        float v1 = 1.1 * vLength;
        float vLastValid = -1;
        float cvLastValid = -1;
        while (v1 - v0 > 0.1) {
          float v = 0.5 * (v0 + v1);
          float dx, du, dv;
          correction.getCorrection(slice, row, u, v, dx, du, dv);
          double cx = x + dx;
          double cu = u + du;
          double cv = v + dv;
          double r2 = cx * cx + cu * cu;
          if (cv < 0) {
            v0 = v;
          } else if (cv <= vLength && r2 >= tpcR2min && r2 <= tpcR2max) {
            v0 = v;
            vLastValid = v;
            cvLastValid = cv;
          } else {
            v1 = v;
          }
        }
        if (vLastValid > 0.) {
          chebFitter.addMeasurement(pad, vLastValid);
        }
        if (area.vMax < vLastValid) {
          area.vMax = vLastValid;
        }
        if (area.cvMax < cvLastValid) {
          area.cvMax = cvLastValid;
        }
      }
      chebFitter.fit();
      for (int i = 0; i < 5; i++) {
        area.maxDriftLengthCheb[i] = chebFitter.getCoefficients()[i];
      }
      if (sliceInfo.vMax < area.vMax) {
        sliceInfo.vMax = area.vMax;
      }
    } // row
  }   // slice
}

void TPCFastSpaceChargeCorrectionHelper::initInverse(o2::gpu::TPCFastSpaceChargeCorrection& correction, bool prn)
{
  std::vector<o2::gpu::TPCFastSpaceChargeCorrection*> corr{&correction};
  initInverse(corr, std::vector<float>{1}, prn);
}

void TPCFastSpaceChargeCorrectionHelper::initInverse(std::vector<o2::gpu::TPCFastSpaceChargeCorrection*>& corrections, const std::vector<float>& scaling, bool prn)
{
  /// initialise inverse transformation
  TStopwatch watch;
  LOG(info) << "fast space charge correction helper: init inverse";

  if (corrections.size() != scaling.size()) {
    LOGP(error, "Input corrections and scaling values have different size");
    return;
  }

  auto& correction = *(corrections.front());
  initMaxDriftLength(correction, prn);

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    // LOG(info) << "inverse transform for slice " << slice ;
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();

    auto myThread = [&](int iThread) {
      Spline2DHelper<float> helper;
      std::vector<float> splineParameters;
      ChebyshevFit1D chebFitterX, chebFitterU, chebFitterV;

      for (int row = iThread; row < mGeo.getNumberOfRows(); row += mNthreads) {
        TPCFastSpaceChargeCorrection::SplineType spline = correction.getSpline(slice, row);
        helper.setSpline(spline, 10, 10);
        std::vector<double> dataPointCU, dataPointCV, dataPointF;

        float u0, u1, v0, v1;
        mGeo.convScaledUVtoUV(slice, row, 0., 0., u0, v0);
        mGeo.convScaledUVtoUV(slice, row, 1., 1., u1, v1);

        double x = mGeo.getRowInfo(row).x;
        int nPointsU = (spline.getGridX1().getNumberOfKnots() - 1) * 10;
        int nPointsV = (spline.getGridX2().getNumberOfKnots() - 1) * 10;

        double stepU = (u1 - u0) / (nPointsU - 1);
        double stepV = (v1 - v0) / (nPointsV - 1);

        if (prn) {
          LOG(info) << "u0 " << u0 << " u1 " << u1 << " v0 " << v0 << " v1 " << v1;
        }
        TPCFastSpaceChargeCorrection::RowActiveArea& area = correction.getSliceRowInfo(slice, row).activeArea;
        area.cuMin = 1.e10;
        area.cuMax = -1.e10;

        /*
        v1 = area.vMax;
        stepV = (v1 - v0) / (nPointsU - 1);
        if (stepV < 1.f) {
          stepV = 1.f;
        }
        */

        for (double u = u0; u < u1 + stepU; u += stepU) {
          for (double v = v0; v < v1 + stepV; v += stepV) {
            float dx, du, dv;
            correction.getCorrection(slice, row, u, v, dx, du, dv);
            dx *= scaling[0];
            du *= scaling[0];
            dv *= scaling[0];
            // add remaining corrections
            for (int i = 1; i < corrections.size(); ++i) {
              float dxTmp, duTmp, dvTmp;
              corrections[i]->getCorrection(slice, row, u, v, dxTmp, duTmp, dvTmp);
              dx += dxTmp * scaling[i];
              du += duTmp * scaling[i];
              dv += dvTmp * scaling[i];
            }
            double cx = x + dx;
            double cu = u + du;
            double cv = v + dv;
            if (cu < area.cuMin) {
              area.cuMin = cu;
            }
            if (cu > area.cuMax) {
              area.cuMax = cu;
            }

            dataPointCU.push_back(cu);
            dataPointCV.push_back(cv);
            dataPointF.push_back(dx);
            dataPointF.push_back(du);
            dataPointF.push_back(dv);

            if (prn) {
              LOG(info) << "measurement cu " << cu << " cv " << cv << " dx " << dx << " du " << du << " dv " << dv;
            }
          } // v
        }   // u

        if (area.cuMax - area.cuMin < 0.2) {
          area.cuMax = .1;
          area.cuMin = -.1;
        }
        if (area.cvMax < 0.1) {
          area.cvMax = .1;
        }
        if (prn) {
          LOG(info) << "slice " << slice << " row " << row << " max drift L = " << correction.getMaxDriftLength(slice, row)
                    << " active area: cuMin " << area.cuMin << " cuMax " << area.cuMax << " vMax " << area.vMax << " cvMax " << area.cvMax;
        }

        TPCFastSpaceChargeCorrection::SliceRowInfo& info = correction.getSliceRowInfo(slice, row);
        info.gridCorrU0 = area.cuMin;
        info.scaleCorrUtoGrid = spline.getGridX1().getUmax() / (area.cuMax - area.cuMin);
        info.scaleCorrVtoGrid = spline.getGridX2().getUmax() / area.cvMax;

        info.gridCorrU0 = u0;
        info.gridCorrV0 = info.gridV0;
        info.scaleCorrUtoGrid = spline.getGridX1().getUmax() / (u1 - info.gridCorrU0);
        info.scaleCorrVtoGrid = spline.getGridX2().getUmax() / (v1 - info.gridCorrV0);

        int nDataPoints = dataPointCU.size();
        for (int i = 0; i < nDataPoints; i++) {
          dataPointCU[i] = (dataPointCU[i] - info.gridCorrU0) * info.scaleCorrUtoGrid;
          dataPointCV[i] = (dataPointCV[i] - info.gridCorrV0) * info.scaleCorrVtoGrid;
        }

        splineParameters.resize(spline.getNumberOfParameters());

        helper.approximateDataPoints(spline, splineParameters.data(), 0., spline.getGridX1().getUmax(),
                                     0., spline.getGridX2().getUmax(),
                                     dataPointCU.data(), dataPointCV.data(),
                                     dataPointF.data(), dataPointCU.size());

        float* splineX = correction.getSplineData(slice, row, 1);
        float* splineUV = correction.getSplineData(slice, row, 2);
        for (int i = 0; i < spline.getNumberOfParameters() / 3; i++) {
          splineX[i] = splineParameters[3 * i + 0];
          splineUV[2 * i + 0] = splineParameters[3 * i + 1];
          splineUV[2 * i + 1] = splineParameters[3 * i + 2];
        }
      } // row
    };  // thread

    std::vector<std::thread> threads(mNthreads);

    // run n threads
    for (int i = 0; i < mNthreads; i++) {
      threads[i] = std::thread(myThread, i);
    }

    // wait for the threads to finish
    for (auto& th : threads) {
      th.join();
    }

  } // slice
  float duration = watch.RealTime();
  LOGP(info, "Inverse took: {}s", duration);
}

} // namespace tpc
} // namespace o2
