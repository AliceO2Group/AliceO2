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
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "ROOT/TTreeProcessorMT.hxx"
#include <algorithm>
#include <sstream>

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

  mGeo.setTPCzLength(detParam.TPClength);

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

void TPCFastSpaceChargeCorrectionHelper::fillSpaceChargeCorrectionFromMap(TPCFastSpaceChargeCorrection& correction, bool processingInverseCorrection)
{
  // calculate correction map: dx,du,dv = ( origTransform() -> x,u,v) - fastTransformNominal:x,u,v
  // for the future: switch TOF correction off for a while

  TStopwatch watch;

  if (!mIsInitialized) {
    initGeometry();
  }

  if (!mCorrectionMap.isInitialized()) {
    correction.setNoCorrection();
    return;
  }

  LOG(info) << "fast space charge correction helper: init from data points";

  for (int sector = 0; sector < correction.getGeometry().getNumberOfSectors(); sector++) {

    auto myThread = [&](int iThread) {
      for (int row = iThread; row < correction.getGeometry().getNumberOfRows(); row += mNthreads) {

        TPCFastSpaceChargeCorrection::SplineType& spline = correction.getSpline(sector, row);
        Spline2DHelper<float> helper;
        std::vector<float> splineParameters;
        splineParameters.resize(spline.getNumberOfParameters());

        const std::vector<o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint>& data = mCorrectionMap.getPoints(sector, row);
        int nDataPoints = data.size();
        auto& info = correction.getSectorRowInfo(sector, row);
        if (!processingInverseCorrection) {
          info.resetMaxValues();
        }
        info.updateMaxValues(1., 1., 1.);
        info.updateMaxValues(-1., -1., -1.);

        if (nDataPoints >= 4) {
          std::vector<double> pointGU(nDataPoints);
          std::vector<double> pointGV(nDataPoints);
          std::vector<double> pointWeight(nDataPoints);
          std::vector<double> pointCorr(3 * nDataPoints); // 3 dimensions
          for (int i = 0; i < nDataPoints; ++i) {
            o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint p = data[i];
            // not corrected grid coordinates
            auto [gu, gv, scale] = correction.convLocalToGrid(sector, row, p.mY, p.mZ);
            if (scale - 1.f > 1.e-6) { // point is outside the grid
              continue;
            }
            pointGU[i] = gu;
            pointGV[i] = gv;
            pointWeight[i] = p.mWeight;
            pointCorr[3 * i + 0] = p.mDx;
            pointCorr[3 * i + 1] = p.mDy;
            pointCorr[3 * i + 2] = p.mDz;
            info.updateMaxValues(5. * p.mDx, 5. * p.mDy, 5. * p.mDz);
          }
          helper.approximateDataPoints(spline, splineParameters.data(), 0., spline.getGridX1().getUmax(), 0., spline.getGridX2().getUmax(), pointGU.data(),
                                       pointGV.data(), pointCorr.data(), pointWeight.data(), nDataPoints);
        } else {
          for (int i = 0; i < spline.getNumberOfParameters(); i++) {
            splineParameters[i] = 0.f;
          }
        }

        if (processingInverseCorrection) {
          float* splineX = correction.getCorrectionDataInvX(sector, row);
          float* splineYZ = correction.getCorrectionDataInvYZ(sector, row);
          for (int i = 0; i < spline.getNumberOfParameters() / 3; i++) {
            splineX[i] = splineParameters[3 * i + 0];
            splineYZ[2 * i + 0] = splineParameters[3 * i + 1];
            splineYZ[2 * i + 1] = splineParameters[3 * i + 2];
          }
        } else {
          float* splineXYZ = correction.getCorrectionData(sector, row);
          for (int i = 0; i < spline.getNumberOfParameters(); i++) {
            splineXYZ[i] = splineParameters[i];
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

  } // sector

  watch.Stop();

  LOGP(info, "Space charge correction tooks: {}s", watch.RealTime());
} // fillSpaceChargeCorrectionFromMap

std::unique_ptr<TPCFastSpaceChargeCorrection> TPCFastSpaceChargeCorrectionHelper::createFromGlobalCorrection(
  std::function<void(int sector, double gx, double gy, double gz,
                     double& dgx, double& dgy, double& dgz)>
    correctionGlobal,
  const int nKnotsY, const int nKnotsZ)
{
  /// creates TPCFastSpaceChargeCorrection object from a continious space charge correction in global coordinates

  auto correctionLocal = [&](int sector, int irow, double ly, double lz,
                             double& dlx, double& dly, double& dlz) {
    double lx = mGeo.getRowInfo(irow).x;
    float gx, gy, gz;
    mGeo.convLocalToGlobal(sector, lx, ly, lz, gx, gy, gz);
    double dgx, dgy, dgz;
    correctionGlobal(sector, gx, gy, gz, dgx, dgy, dgz);
    float lx1, ly1, lz1;
    mGeo.convGlobalToLocal(sector, gx + dgx, gy + dgy, gz + dgz, lx1, ly1, lz1);
    dlx = lx1 - lx;
    dly = ly1 - ly;
    dlz = lz1 - lz;
  };
  return std::move(createFromLocalCorrection(correctionLocal, nKnotsY, nKnotsZ));
}

std::unique_ptr<TPCFastSpaceChargeCorrection> TPCFastSpaceChargeCorrectionHelper::createFromLocalCorrection(
  std::function<void(int sector, int irow, double y, double z, double& dx, double& dy, double& dz)> correctionLocal,
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
    for (int sector = 0; sector < mGeo.getNumberOfSectors(); sector++) {
      for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
        int scenario = row / 10;
        if (scenario >= nCorrectionScenarios) {
          scenario = nCorrectionScenarios - 1;
        }
        correction.setRowScenarioID(sector, row, scenario);
      }
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

    int nSectors = mGeo.getNumberOfSectors();
    int nRows = mGeo.getNumberOfRows();
    mCorrectionMap.init(nSectors, nRows);

    for (int iSector = 0; iSector < nSectors; iSector++) {

      auto myThread = [&](int iThread) {
        for (int iRow = iThread; iRow < nRows; iRow += mNthreads) {
          const auto& info = mGeo.getRowInfo(iRow);
          double dl = mGeo.getTPCzLength() / (6. * (nKnotsZ - 1));
          double dpad = info.maxPad / (6. * (nKnotsY - 1));
          for (double pad = 0; pad < info.maxPad + .5 * dpad; pad += dpad) {
            for (double l = 0.; l < mGeo.getTPCzLength() + .5 * dl; l += dl) {
              auto [y, z] = mGeo.convPadDriftLengthToLocal(iSector, iRow, pad, l);
              double dx, dy, dz;
              correctionLocal(iSector, iRow, y, z, dx, dy, dz);
              mCorrectionMap.addCorrectionPoint(iSector, iRow,
                                                y, z, dx, dy, dz, 1.);
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

    } // sector

    fillSpaceChargeCorrectionFromMap(correction, false);
    initInverse(correction, false);
  }

  return std::move(correctionPtr);
} // createFromLocalCorrection

void TPCFastSpaceChargeCorrectionHelper::testGeometry(const TPCFastTransformGeo& geo) const
{
  const Mapper& mapper = Mapper::instance();

  if (geo.getNumberOfSectors() != Sector::MAXSECTOR) {
    LOG(fatal) << "Wrong number of sectors :" << geo.getNumberOfSectors() << " instead of " << Sector::MAXSECTOR << std::endl;
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
      auto [y, z] = geo.convPadDriftLengthToLocal(0, row, pad, 0.);
      const double dx = x - c.X();
      const double dy = y - (-c.Y()); // diferent sign convention for Y coordinate in the map

      if (fabs(dx) >= 1.e-6 || fabs(dy) >= 1.e-5) {
        LOG(warning) << "wrong calculated pad position:"
                     << " row " << row << " pad " << pad << " x calc " << x << " x in map " << c.X() << " dx " << (x - c.X())
                     << " y calc " << y << " y in map " << -c.Y() << " dy " << dy << std::endl;
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
  const o2::tpc::TrackResiduals& trackResiduals, TTree* voxResTree, TTree* voxResTreeInverse, bool useSmoothed, bool invertSigns,
  TPCFastSpaceChargeCorrectionMap* fitPointsDirect,
  TPCFastSpaceChargeCorrectionMap* fitPointsInverse)
{
  // create o2::gpu::TPCFastSpaceChargeCorrection  from o2::tpc::TrackResiduals::VoxRes voxel tree

  LOG(info) << "fast space charge correction helper: create correction from track residuals using " << mNthreads << " threads";

  TStopwatch watch, watch1;

  std::unique_ptr<o2::gpu::TPCFastSpaceChargeCorrection> correctionPtr(new o2::gpu::TPCFastSpaceChargeCorrection);

  o2::gpu::TPCFastSpaceChargeCorrection& correction = *correctionPtr;

  auto* helper = o2::tpc::TPCFastSpaceChargeCorrectionHelper::instance();
  const o2::gpu::TPCFastTransformGeo& geo = helper->getGeometry();

  int nY2Xbins = trackResiduals.getNY2XBins();
  int nZ2Xbins = trackResiduals.getNZ2XBins();

  std::vector<double> knotsDouble[3];

  knotsDouble[0].reserve(nY2Xbins);
  knotsDouble[1].reserve(nZ2Xbins);
  knotsDouble[2].reserve(nZ2Xbins);

  // to get enouth measurements, make a spline knot at every second bin. Boundary bins are always included.

  for (int i = 0, j = nY2Xbins - 1; i <= j; i += 2, j -= 2) {
    knotsDouble[0].push_back(trackResiduals.getY2X(0, i));
    if (j >= i + 1) {
      knotsDouble[0].push_back(trackResiduals.getY2X(0, j));
    }
  }

  for (int i = 0, j = nZ2Xbins - 1; i <= j; i += 2, j -= 2) {
    knotsDouble[1].push_back(trackResiduals.getZ2X(i));
    knotsDouble[2].push_back(-trackResiduals.getZ2X(i));
    if (j >= i + 1) {
      knotsDouble[1].push_back(trackResiduals.getZ2X(j));
      knotsDouble[2].push_back(-trackResiduals.getZ2X(j));
    }
  }

  std::vector<int> knotsInt[3];

  for (int dim = 0; dim < 3; dim++) {
    auto& knotsD = knotsDouble[dim];
    std::sort(knotsD.begin(), knotsD.end());

    double pitch = knotsD[1] - knotsD[0]; // min distance between the knots
    for (int i = 2; i < knotsD.size(); i++) {
      double d = knotsD[i] - knotsD[i - 1];
      if (d < pitch) {
        pitch = d;
      }
    }
    // spline knots must be positioned on the grid with an integer internal coordinate
    // we set the knot positioning accuracy to 0.1*pitch
    pitch = 0.1 * pitch;
    auto& knotsI = knotsInt[dim];
    knotsI.reserve(knotsD.size());
    double u0 = knotsD[0];
    double u1 = knotsD[knotsD.size() - 1];
    for (auto& u : knotsD) {
      u -= u0;
      int iu = int(u / pitch + 0.5);
      knotsI.push_back(iu);
      // debug printout: corrected vs original knot positions, scaled to [-1,1] interval
      double uorig = u / (u1 - u0) * 2 - 1.;
      u = (iu * pitch) / (u1 - u0) * 2 - 1.;
      LOG(info) << "TPC SC splines: convert " << (dim == 0 ? "y" : (dim == 1 ? "z" : "-z")) << " bin to the knot: " << uorig << " -> " << u << " -> " << iu;
    }

    if (knotsI.size() < 2) { // minimum 2 knots
      knotsI.clear();
      knotsI.push_back(0);
      knotsI.push_back(1);
    }
  }

  auto& yKnotsInt = knotsInt[0];
  auto& zKnotsIntA = knotsInt[1];
  auto& zKnotsIntC = knotsInt[2];

  int nKnotsY = yKnotsInt.size();
  int nKnotsZA = zKnotsIntA.size();
  int nKnotsZC = zKnotsIntC.size();

  // std::cout << "n knots Y: " << nKnotsY << std::endl;
  // std::cout << "n knots Z: " << nKnotsZA << ",  " << nKnotsZC << std::endl;

  const int nRows = geo.getNumberOfRows();
  const int nSectors = geo.getNumberOfSectors();

  { // create the correction object

    const int nCorrectionScenarios = 2; // different grids for TPC A and TPC C sides

    correction.startConstruction(geo, nCorrectionScenarios);

    // init rows
    for (int iSector = 0; iSector < nSectors; iSector++) {
      int id = iSector < geo.getNumberOfSectorsA() ? 0 : 1;
      for (int row = 0; row < geo.getNumberOfRows(); row++) {
        correction.setRowScenarioID(iSector, row, id);
      }
    }
    { // init spline scenario
      TPCFastSpaceChargeCorrection::SplineType spline;
      spline.recreate(nKnotsY, &yKnotsInt[0], nKnotsZA, &zKnotsIntA[0]);
      correction.setSplineScenario(0, spline);
      spline.recreate(nKnotsY, &yKnotsInt[0], nKnotsZC, &zKnotsIntC[0]);
      correction.setSplineScenario(1, spline);
    }
    correction.finishConstruction();
  } // .. create the correction object

  // set the grid borders
  for (int iSector = 0; iSector < geo.getNumberOfSectors(); iSector++) {
    for (int iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {
      auto& info = correction.getSectorRowInfo(iSector, iRow);
      const auto& spline = correction.getSpline(iSector, iRow);
      double rowX = geo.getRowInfo(iRow).x;
      double yMin = rowX * trackResiduals.getY2X(iRow, 0);
      double yMax = rowX * trackResiduals.getY2X(iRow, trackResiduals.getNY2XBins() - 1);
      double zMin = rowX * trackResiduals.getZ2X(0);
      double zMax = rowX * trackResiduals.getZ2X(trackResiduals.getNZ2XBins() - 1);
      double zOut = zMax;
      if (iSector >= geo.getNumberOfSectorsA()) {
        // TPC C side
        zOut = -zOut;
        zMax = -zMin;
        zMin = zOut;
      }
      info.gridMeasured.set(yMin, spline.getGridX1().getUmax() / (yMax - yMin), // y
                            zMin, spline.getGridX2().getUmax() / (zMax - zMin), // z
                            zOut, geo.getZreadout(iSector));                    // correction scaling region

      info.gridReal = info.gridMeasured;

      // std::cout << " iSector " << iSector << " iRow " << iRow << " uMin: " << uMin << " uMax: " << uMax << " vMin: " << vMin << " vMax: " << vMax
      //<< " grid scale u "<< info.scaleUtoGrid << " grid scale v "<< info.scaleVtoGrid<< std::endl;
    }
  }

  LOG(info) << "fast space charge correction helper: preparation took " << watch1.RealTime() << "s";

  for (int processingInverseCorrection = 0; processingInverseCorrection <= 1; processingInverseCorrection++) {

    TTree* currentTree = (processingInverseCorrection) ? voxResTreeInverse : voxResTree;

    if (!currentTree) {
      continue;
    }
    const char* directionName = (processingInverseCorrection) ? "inverse" : "direct";
    LOG(info) << "\n fast space charge correction helper: Process " << directionName
              << " correction: fill data points from track residuals.. ";

    TStopwatch watch3;
    o2::gpu::TPCFastSpaceChargeCorrectionMap& map = helper->getCorrectionMap();
    map.init(geo.getNumberOfSectors(), geo.getNumberOfRows());

    // read the data Sector by Sector

    // data in the tree is not sorted by row
    // first find which data belong to which row

    struct VoxelData {
      int mNentries{0};    // number of entries
      float mX, mY, mZ;    // mean position in the local coordinates
      float mCx, mCy, mCz; // corrections to the local coordinates
    };

    std::vector<VoxelData> vSectorData[nRows * nSectors];
    for (int ir = 0; ir < nRows * nSectors; ir++) {
      vSectorData[ir].resize(nY2Xbins * nZ2Xbins);
    }

    { // read data from the tree to vSectorData

      ROOT::TTreeProcessorMT processor(*currentTree, mNthreads);
      std::string errMsg = std::string("Error reading ") + directionName + " track residuals: ";
      auto myThread = [&](TTreeReader& readerSubRange) {
        TTreeReaderValue<o2::tpc::TrackResiduals::VoxRes> v(readerSubRange, "voxRes");
        while (readerSubRange.Next()) {
          int iSector = (int)v->bsec;
          if (iSector < 0 || iSector >= nSectors) {
            LOG(fatal) << errMsg << "Sector number " << iSector << " is out of range";
            continue;
          }
          int iRow = (int)v->bvox[o2::tpc::TrackResiduals::VoxX]; // bin number in x (= pad row)
          if (iRow < 0 || iRow >= nRows) {
            LOG(fatal) << errMsg << "Row number " << iRow << " is out of range";
          }
          double rowX = trackResiduals.getX(iRow);         // X of the pad row
          int iy = v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
          int iz = v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4
          auto& data = vSectorData[iSector * nRows + iRow][iy * nZ2Xbins + iz];
          data.mNentries = (int)v->stat[o2::tpc::TrackResiduals::VoxV];
          data.mX = v->stat[o2::tpc::TrackResiduals::VoxX];
          data.mY = v->stat[o2::tpc::TrackResiduals::VoxF] * rowX;
          data.mZ = v->stat[o2::tpc::TrackResiduals::VoxZ] * rowX;
          data.mCx = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResX] : v->D[o2::tpc::TrackResiduals::ResX];
          data.mCy = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResY] : v->D[o2::tpc::TrackResiduals::ResY];
          data.mCz = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResZ] : v->D[o2::tpc::TrackResiduals::ResZ];
          if (invertSigns) {
            data.mCx *= -1.;
            data.mCy *= -1.;
            data.mCz *= -1.;
          }
        }
      };
      processor.Process(myThread);
    }

    // debug: mirror the data for TPC C side

    if (mDebugMirrorAdata2C) {
      for (int iSector = 0; iSector < geo.getNumberOfSectorsA(); iSector++) {
        for (int iRow = 0; iRow < nRows; iRow++) {
          for (int iy = 0; iy < nY2Xbins; iy++) {
            for (int iz = 0; iz < nZ2Xbins; iz++) {
              auto& dataA = vSectorData[iSector * nRows + iRow][iy * nZ2Xbins + iz];
              auto& dataC = vSectorData[(iSector + geo.getNumberOfSectorsA()) * nRows + iRow][iy * nZ2Xbins + iz];
              dataC = dataA;          // copy the data
              dataC.mZ = -dataC.mZ;   // mirror the Z coordinate
              dataC.mCz = -dataC.mCz; // mirror the Z correction
            }
          }
        }
      }
    }

    double maxError[3] = {0., 0., 0.};
    int nErrors = 0;

    for (int iSector = 0; iSector < nSectors; iSector++) {

      // now process the data row-by-row

      auto myThread = [&](int iThread, int nTreads) {
        struct Voxel {
          float mY, mZ;            // non-distorted local coordinates
          float mDy, mDz;          // voxel size
          int mSmoothingStep{100}; // is the voxel data original or smoothed at this step
        };

        std::vector<Voxel> vRowVoxels(nY2Xbins * nZ2Xbins);

        for (int iRow = iThread; iRow < nRows; iRow += nTreads) {
          // LOG(info) << "Processing Sector " << iSector << " row " << iRow;

          // complete the voxel data
          {
            int xBin = iRow;
            double x = trackResiduals.getX(xBin); // radius of the pad row
            double dx = 1. / trackResiduals.getDXI(xBin);
            bool isDataFound = false;
            for (int iy = 0; iy < nY2Xbins; iy++) {
              for (int iz = 0; iz < nZ2Xbins; iz++) {
                auto& data = vSectorData[iSector * nRows + iRow][iy * nZ2Xbins + iz];
                auto& vox = vRowVoxels[iy * nZ2Xbins + iz];
                // y/x coordinate of the bin ~-0.15 ... 0.15
                double y2x = trackResiduals.getY2X(xBin, iy);
                // z/x coordinate of the bin 0.1 .. 0.9
                double z2x = trackResiduals.getZ2X(iz);
                vox.mY = x * y2x;
                vox.mZ = x * z2x;
                vox.mDy = x / trackResiduals.getDY2XI(xBin, iy);
                vox.mDz = x * trackResiduals.getDZ2X(iz);
                if (iSector >= geo.getNumberOfSectorsA()) {
                  vox.mZ = -vox.mZ;
                }
                if (data.mNentries > 0) { // voxel contains data
                  vox.mSmoothingStep = 0; // take original data
                  isDataFound = true;

                  // correct the mean position if it is outside the voxel
                  std::stringstream msg;
                  if (fabs(x - data.mX) > mVoxelMeanValidityRange * dx / 2.) {
                    msg << "\n         x: center " << x << " dx " << data.mX - x << " half bin size " << dx / 2;
                  }

                  if (fabs(vox.mY - data.mY) > mVoxelMeanValidityRange * vox.mDy / 2.) {
                    msg << "\n         y: center " << vox.mY << " dy " << data.mY - vox.mY << " half bin size " << vox.mDy / 2;
                    data.mY = vox.mY;
                  }

                  if (fabs(vox.mZ - data.mZ) > mVoxelMeanValidityRange * vox.mDz / 2.) {
                    msg << "\n         z: center " << vox.mZ << " dz " << data.mZ - vox.mZ << " half bin size " << vox.mDz / 2;
                    data.mZ = vox.mZ;
                  }

                  if (!msg.str().empty()) {
                    bool isMaxErrorExceeded = (fabs(data.mX - x) / dx > maxError[0]) ||
                                              (fabs(data.mY - vox.mY) / vox.mDy > maxError[1]) ||
                                              (fabs(data.mZ - vox.mZ) / vox.mDz > maxError[2]);
                    static std::mutex mutex;
                    mutex.lock();
                    nErrors++;
                    if (nErrors < 20 || isMaxErrorExceeded) {
                      LOG(warning) << directionName << " correction: error N " << nErrors << "fitted voxel position is outside the voxel: "
                                   << " sector " << iSector << " row " << iRow << " bin y " << iy << " bin z " << iz
                                   << msg.str();
                      maxError[0] = GPUCommonMath::Max<double>(maxError[0], fabs(data.mX - x) / dx);
                      maxError[1] = GPUCommonMath::Max<double>(maxError[1], fabs(data.mY - vox.mY) / vox.mDy);
                      maxError[2] = GPUCommonMath::Max<double>(maxError[2], fabs(data.mZ - vox.mZ) / vox.mDz);
                    }
                    mutex.unlock();
                  }

                } else { // no data, take voxel center position
                  data.mCx = 0.;
                  data.mCy = 0.;
                  data.mCz = 0.;
                  data.mX = x;
                  data.mY = vox.mY;
                  data.mZ = vox.mZ;
                  vox.mSmoothingStep = 100; // fill this data point with smoothed values from the neighbours
                }
                if (mDebugUseVoxelCenters) { // debug: always use voxel center instead of the mean position
                  data.mY = vox.mY;
                  data.mZ = vox.mZ;
                }
              }
            }

            if (!isDataFound) { // fill everything with 0
              for (int iy = 0; iy < nY2Xbins; iy++) {
                for (int iz = 0; iz < nZ2Xbins; iz++) {
                  vRowVoxels[iy * nZ2Xbins + iz].mSmoothingStep = 0;
                }
              }
            }
          } // complete the voxel data

          // repare the voxel data: fill empty voxels

          int nRepairs = 0;

          for (int ismooth = 1; ismooth <= 2; ismooth++) {
            for (int iy = 0; iy < nY2Xbins; iy++) {
              for (int iz = 0; iz < nZ2Xbins; iz++) {
                auto& data = vSectorData[iSector * nRows + iRow][iy * nZ2Xbins + iz];
                auto& vox = vRowVoxels[iy * nZ2Xbins + iz];
                if (vox.mSmoothingStep <= ismooth) { // already filled
                  continue;
                }
                nRepairs++;
                data.mCx = 0.;
                data.mCy = 0.;
                data.mCz = 0.;
                double w = 0.;
                bool filled = false;
                auto update = [&](int iy1, int iz1) {
                  auto& data1 = vSectorData[iSector * nRows + iRow][iy1 * nZ2Xbins + iz1];
                  auto& vox1 = vRowVoxels[iy1 * nZ2Xbins + iz1];
                  if (vox1.mSmoothingStep >= ismooth) {
                    return false;
                  }
                  double w1 = 1. / (abs(iy - iy1) + abs(iz - iz1) + 1);
                  data.mCx += w1 * data1.mCx;
                  data.mCy += w1 * data1.mCy;
                  data.mCz += w1 * data1.mCz;
                  w += w1;
                  filled = true;
                  return true;
                };

                for (int iy1 = iy - 1; iy1 >= 0 && !update(iy1, iz); iy1--) {
                }
                for (int iy1 = iy + 1; iy1 < nY2Xbins && !update(iy1, iz); iy1++) {
                }
                for (int iz1 = iz - 1; iz1 >= 0 && !update(iy, iz1); iz1--) {
                }
                for (int iz1 = iz + 1; iz1 < nZ2Xbins && !update(iy, iz1); iz1++) {
                }

                if (filled) {
                  data.mCx /= w;
                  data.mCy /= w;
                  data.mCz /= w;
                  vox.mSmoothingStep = ismooth;
                }
              } // iz
            } // iy
          } // ismooth

          if (nRepairs > 0) {
            LOG(debug) << "Sector " << iSector << " row " << iRow << ": " << nRepairs << " voxel repairs for " << nY2Xbins * nZ2Xbins << " voxels";
          }

          // feed the row data to the helper

          auto& info = correction.getSectorRowInfo(iSector, iRow);
          const auto& spline = correction.getSpline(iSector, iRow);

          auto addVoxel = [&](int iy, int iz, double weight) {
            auto& vox = vRowVoxels[iy * nZ2Xbins + iz];
            if (vox.mSmoothingStep > 2) {
              LOG(fatal) << "empty voxel is not repared: y " << iy << " z " << iz;
            }
            auto& data = vSectorData[iSector * nRows + iRow][iy * nZ2Xbins + iz];
            map.addCorrectionPoint(iSector, iRow, data.mY, data.mZ, data.mCx, data.mCy, data.mCz, weight);
          };

          auto addEdge = [&](int iy1, int iz1, int iy2, int iz2, int nPoints) {
            // add n points on the edge between two voxels excluding the voxel points
            if (nPoints < 1) {
              return;
            }
            if (iy1 < 0 || iy1 >= nY2Xbins || iz1 < 0 || iz1 >= nZ2Xbins) {
              return;
            }
            if (iy2 < 0 || iy2 >= nY2Xbins || iz2 < 0 || iz2 >= nZ2Xbins) {
              return;
            }
            auto& data1 = vSectorData[iSector * nRows + iRow][iy1 * nZ2Xbins + iz1];
            auto& vox1 = vRowVoxels[iy1 * nZ2Xbins + iz1];
            auto& data2 = vSectorData[iSector * nRows + iRow][iy2 * nZ2Xbins + iz2];
            auto& vox2 = vRowVoxels[iy2 * nZ2Xbins + iz2];
            double y1 = data1.mY;
            double z1 = data1.mZ;
            double cx1 = data1.mCx;
            double cy1 = data1.mCy;
            double cz1 = data1.mCz;
            double y2 = data2.mY;
            double z2 = data2.mZ;
            double cx2 = data2.mCx;
            double cy2 = data2.mCy;
            double cz2 = data2.mCz;

            for (int is = 1; is <= nPoints; is++) {
              double s2 = is / (double)(nPoints + 1);
              double s1 = 1. - s2;
              double y = s1 * y1 + s2 * y2;
              double z = s1 * z1 + s2 * z2;
              double cx = s1 * cx1 + s2 * cx2;
              double cy = s1 * cy1 + s2 * cy2;
              double cz = s1 * cz1 + s2 * cz2;
              map.addCorrectionPoint(iSector, iRow, y, z, cx, cy, cz, 1.);
            }
          };

          // original measurements weighted by 8 at each voxel and 8 additional artificial measurements around each voxel
          //
          // (y+1, z) 8 1 1 8 (y+1, z+1)
          //          1 1 1 1 1
          //          1 1 1 1 1
          //    (y,z) 8 1 1 8 1
          //          1 1 1 1 1

          for (int iy = 0; iy < nY2Xbins; iy++) {
            for (int iz = 0; iz < nZ2Xbins; iz++) {
              addVoxel(iy, iz, 8);
              addEdge(iy, iz, iy, iz + 1, 2);
              addEdge(iy, iz, iy + 1, iz, 2);
              addEdge(iy, iz, iy + 1, iz + 1, 2);
              addEdge(iy + 1, iz, iy, iz + 1, 2);
            }
          }

        } // iRow
      }; // myThread

      // run n threads

      int nThreads = mNthreads;
      // nThreads = 1;

      std::vector<std::thread> threads(nThreads);

      for (int i = 0; i < nThreads; i++) {
        threads[i] = std::thread(myThread, i, nThreads);
      }

      // wait for the threads to finish
      for (auto& th : threads) {
        th.join();
      }
    } // iSector

    LOGP(info, "Reading & reparing of the track residuals tooks: {}s", watch3.RealTime());

    LOG(info) << "fast space charge correction helper: create space charge from the map of data points..";

    TStopwatch watch4;

    if (!processingInverseCorrection && fitPointsDirect) {
      *fitPointsDirect = helper->getCorrectionMap();
    }
    if (processingInverseCorrection && fitPointsInverse) {
      *fitPointsInverse = helper->getCorrectionMap();
    }

    helper->fillSpaceChargeCorrectionFromMap(correction, processingInverseCorrection);

    LOG(info) << "fast space charge correction helper: creation from the data map took " << watch4.RealTime() << "s";

  } // processingInverseCorrection

  if (voxResTree && !voxResTreeInverse) {
    LOG(info) << "fast space charge correction helper: init inverse correction from direct correction..";
    TStopwatch watch4;
    helper->initInverse(correction, false);
    LOG(info) << "fast space charge correction helper: init inverse correction took " << watch4.RealTime() << "s";
  }

  LOGP(info, "Creation from track residuals tooks in total: {}s", watch.RealTime());

  return std::move(correctionPtr);

} // createFromTrackResiduals

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

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSectorsA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  for (int sector = 0; sector < mGeo.getNumberOfSectors(); sector++) {
    // LOG(info) << "inverse transform for sector " << sector ;

    auto myThread = [&](int iThread) {
      Spline2DHelper<float> helper;
      std::vector<float> splineParameters;

      for (int row = iThread; row < mGeo.getNumberOfRows(); row += mNthreads) {
        auto& sectorRowInfo = correction.getSectorRowInfo(sector, row);
        sectorRowInfo.gridReal = sectorRowInfo.gridMeasured;

        TPCFastSpaceChargeCorrection::SplineType spline = correction.getSpline(sector, row);
        helper.setSpline(spline, 10, 10);

        std::vector<double> gridU;
        {
          const auto& grid = spline.getGridX1();
          for (int i = 0; i < grid.getNumberOfKnots(); i++) {
            if (i == grid.getNumberOfKnots() - 1) {
              gridU.push_back(grid.getKnot(i).u);
              break;
            }
            for (double s = 1.; s > 0.; s -= 0.1) {
              gridU.push_back(s * grid.getKnot(i).u + (1. - s) * grid.getKnot(i + 1).u);
            }
          }
        }
        std::vector<double> gridV;
        {
          const auto& grid = spline.getGridX2();
          for (int i = 0; i < grid.getNumberOfKnots(); i++) {
            if (i == grid.getNumberOfKnots() - 1) {
              gridV.push_back(grid.getKnot(i).u);
              break;
            }
            for (double s = 1.; s > 0.; s -= 0.1) {
              gridV.push_back(s * grid.getKnot(i).u + (1. - s) * grid.getKnot(i + 1).u);
            }
          }
        }

        std::vector<double> dataPointGridU, dataPointGridV, dataPointF, dataPointWeight;
        dataPointGridU.reserve(gridU.size() * gridV.size());
        dataPointGridV.reserve(gridU.size() * gridV.size());
        dataPointF.reserve(3 * gridU.size() * gridV.size());
        dataPointWeight.reserve(gridU.size() * gridV.size());

        for (int iu = 0; iu < gridU.size(); iu++) {
          for (int iv = 0; iv < gridV.size(); iv++) {

            auto [y, z] = correction.convGridToLocal(sector, row, gridU[iu], gridV[iv]);
            double dx = 0, dy = 0, dz = 0;

            // add corrections
            for (int i = 0; i < corrections.size(); ++i) {
              auto [dxTmp, dyTmp, dzTmp] = corrections[i]->getCorrectionLocal(sector, row, y, z);
              dx += dxTmp * scaling[i];
              dy += dyTmp * scaling[i];
              dz += dzTmp * scaling[i];
            }
            auto [gridU, gridV, scale] = correction.convRealLocalToGrid(sector, row, y + dy, z + dz);
            dataPointGridU.push_back(gridU);
            dataPointGridV.push_back(gridV);
            dataPointF.push_back(scale * dx);
            dataPointF.push_back(scale * dy);
            dataPointF.push_back(scale * dz);
            dataPointWeight.push_back(1.);
          }
        }

        int nDataPoints = dataPointGridU.size();
        splineParameters.resize(spline.getNumberOfParameters());

        helper.approximateDataPoints(spline, splineParameters.data(), 0., spline.getGridX1().getUmax(),
                                     0., spline.getGridX2().getUmax(),
                                     dataPointGridU.data(), dataPointGridV.data(),
                                     dataPointF.data(), dataPointWeight.data(), nDataPoints);

        float* splineX = correction.getCorrectionDataInvX(sector, row);
        float* splineUV = correction.getCorrectionDataInvYZ(sector, row);
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

  } // sector
  float duration = watch.RealTime();
  LOGP(info, "Inverse tooks: {}s", duration);
}

void TPCFastSpaceChargeCorrectionHelper::mergeCorrections(
  o2::gpu::TPCFastSpaceChargeCorrection& mainCorrection, float mainScale,
  const std::vector<std::pair<const o2::gpu::TPCFastSpaceChargeCorrection*, float>>& additionalCorrections, bool /*prn*/)
{
  /// merge several corrections

  TStopwatch watch;
  LOG(info) << "fast space charge correction helper: Merge corrections";

  const auto& geo = mainCorrection.getGeometry();

  for (int sector = 0; sector < geo.getNumberOfSectors(); sector++) {

    auto myThread = [&](int iThread) {
      for (int row = iThread; row < geo.getNumberOfRows(); row += mNthreads) {
        const auto& spline = mainCorrection.getSpline(sector, row);

        float* splineParameters = mainCorrection.getCorrectionData(sector, row);
        float* splineParametersInvX = mainCorrection.getCorrectionDataInvX(sector, row);
        float* splineParametersInvYZ = mainCorrection.getCorrectionDataInvYZ(sector, row);

        auto& secRowInfo = mainCorrection.getSectorRowInfo(sector, row);

        constexpr int nKnotPar1d = 4;
        constexpr int nKnotPar2d = nKnotPar1d * 2;
        constexpr int nKnotPar3d = nKnotPar1d * 3;

        { // scale the main correction
          for (int i = 0; i < 3; i++) {
            secRowInfo.maxCorr[i] *= mainScale;
            secRowInfo.minCorr[i] *= mainScale;
          }
          double parscale[4] = {mainScale, mainScale, mainScale, mainScale * mainScale};
          for (int iknot = 0, ind = 0; iknot < spline.getNumberOfKnots(); iknot++) {
            for (int ipar = 0; ipar < nKnotPar1d; ++ipar) {
              for (int idim = 0; idim < 3; idim++, ind++) {
                splineParameters[ind] *= parscale[ipar];
              }
            }
          }
          for (int iknot = 0, ind = 0; iknot < spline.getNumberOfKnots(); iknot++) {
            for (int ipar = 0; ipar < nKnotPar1d; ++ipar) {
              for (int idim = 0; idim < 1; idim++, ind++) {
                splineParametersInvX[ind] *= parscale[ipar];
              }
            }
          }
          for (int iknot = 0, ind = 0; iknot < spline.getNumberOfKnots(); iknot++) {
            for (int ipar = 0; ipar < nKnotPar1d; ++ipar) {
              for (int idim = 0; idim < 2; idim++, ind++) {
                splineParametersInvYZ[ind] *= parscale[ipar];
              }
            }
          }
        }

        // add the other corrections

        const auto& gridU = spline.getGridX1();
        const auto& gridV = spline.getGridX2();

        for (int icorr = 0; icorr < additionalCorrections.size(); ++icorr) {
          const auto& corr = *(additionalCorrections[icorr].first);
          double scale = additionalCorrections[icorr].second;
          auto& linfo = corr.getSectorRowInfo(sector, row);
          secRowInfo.updateMaxValues(linfo.getMaxValues(), scale);
          secRowInfo.updateMaxValues(linfo.getMinValues(), scale);

          double scaleU = secRowInfo.gridMeasured.getYscale() / linfo.gridMeasured.getYscale();
          double scaleV = secRowInfo.gridMeasured.getZscale() / linfo.gridMeasured.getZscale();
          double scaleRealU = secRowInfo.gridReal.getYscale() / linfo.gridReal.getYscale();
          double scaleRealV = secRowInfo.gridReal.getZscale() / linfo.gridReal.getZscale();

          for (int iu = 0; iu < gridU.getNumberOfKnots(); iu++) {
            double u = gridU.getKnot(iu).u;
            for (int iv = 0; iv < gridV.getNumberOfKnots(); iv++) {
              double v = gridV.getKnot(iv).u;
              int knotIndex = spline.getKnotIndex(iu, iv);
              float P[nKnotPar3d];

              { // direct correction
                auto [y, z] = mainCorrection.convGridToLocal(sector, row, u, v);
                // return values: u, v, scaling factor
                auto [lu, lv, ls] = corr.convLocalToGrid(sector, row, y, z);
                ls *= scale;
                double parscale[4] = {ls, ls * scaleU, ls * scaleV, ls * ls * scaleU * scaleV};
                const auto& spl = corr.getSpline(sector, row);
                spl.interpolateParametersAtU(corr.getCorrectionData(sector, row), lu, lv, P);
                for (int ipar = 0, ind = 0; ipar < nKnotPar1d; ++ipar) {
                  for (int idim = 0; idim < 3; idim++, ind++) {
                    splineParameters[knotIndex * nKnotPar3d + ind] += parscale[ipar] * P[ind];
                  }
                }
              }

              auto [y, z] = mainCorrection.convGridToRealLocal(sector, row, u, v);
              // return values: u, v, scaling factor
              auto [lu, lv, ls] = corr.convRealLocalToGrid(sector, row, y, z);
              ls *= scale;
              double parscale[4] = {ls, ls * scaleRealU, ls * scaleRealV, ls * ls * scaleRealU * scaleRealV};

              { // inverse X correction
                corr.getSplineInvX(sector, row).interpolateParametersAtU(corr.getCorrectionDataInvX(sector, row), lu, lv, P);
                for (int ipar = 0, ind = 0; ipar < nKnotPar1d; ++ipar) {
                  for (int idim = 0; idim < 1; idim++, ind++) {
                    splineParametersInvX[knotIndex * nKnotPar1d + ind] += parscale[ipar] * P[ind];
                  }
                }
              }

              { // inverse YZ correction
                corr.getSplineInvYZ(sector, row).interpolateParametersAtU(corr.getCorrectionDataInvYZ(sector, row), lu, lv, P);
                for (int ipar = 0, ind = 0; ipar < nKnotPar1d; ++ipar) {
                  for (int idim = 0; idim < 2; idim++, ind++) {
                    splineParametersInvYZ[knotIndex * nKnotPar2d + ind] += parscale[ipar] * P[ind];
                  }
                }
              }

            } // iv
          } // iu
        } // corrections

      } // row
    }; // thread

    std::vector<std::thread> threads(mNthreads);

    // run n threads
    for (int i = 0; i < mNthreads; i++) {
      threads[i] = std::thread(myThread, i);
    }

    // wait for the threads to finish
    for (auto& th : threads) {
      th.join();
    }

  } // sector
  float duration = watch.RealTime();
  LOGP(info, "Merge of corrections tooks: {}s", duration);
}

void TPCFastSpaceChargeCorrectionHelper::setDebugUseVoxelCenters()
{
  LOG(info) << "fast space charge correction helper: use voxel centers for correction";
  mDebugUseVoxelCenters = true;
}

void TPCFastSpaceChargeCorrectionHelper::setDebugMirrorAdata2C()
{
  LOG(info) << "fast space charge correction helper: mirror A data to C data";
  mDebugMirrorAdata2C = true;
}

} // namespace tpc
} // namespace o2
