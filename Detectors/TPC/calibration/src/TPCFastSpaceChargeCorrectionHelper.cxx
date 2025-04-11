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
        if (nDataPoints >= 4) {
          std::vector<double> pointGU(nDataPoints);
          std::vector<double> pointGV(nDataPoints);
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
            pointCorr[3 * i + 0] = p.mDx;
            pointCorr[3 * i + 1] = p.mDy;
            pointCorr[3 * i + 2] = p.mDz;
            if (!processingInverseCorrection) {
              info.updateMaxValues(20. * p.mDx, 20. * p.mDy, 20. * p.mDz);
            }
          }
          helper.approximateDataPoints(spline, splineParameters.data(), 0., spline.getGridX1().getUmax(), 0., spline.getGridX2().getUmax(), &pointGU[0],
                                       &pointGV[0], &pointCorr[0], nDataPoints);
        } else {
          for (int i = 0; i < spline.getNumberOfParameters(); i++) {
            splineParameters[i] = 0.f;
          }
        }

        if (processingInverseCorrection) {
          float* splineX = correction.getSplineDataInvX(sector, row);
          float* splineYZ = correction.getSplineDataInvYZ(sector, row);
          for (int i = 0; i < spline.getNumberOfParameters() / 3; i++) {
            splineX[i] = splineParameters[3 * i + 0];
            splineYZ[2 * i + 0] = splineParameters[3 * i + 1];
            splineYZ[2 * i + 1] = splineParameters[3 * i + 2];
          }
        } else {
          float* splineXYZ = correction.getSplineData(sector, row);
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
                                                y, z, dx, dy, dz);
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
  const o2::tpc::TrackResiduals& trackResiduals, TTree* voxResTree, TTree* voxResTreeInverse, bool useSmoothed, bool invertSigns)
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

  std::vector<double> uvBinsDouble[2];

  uvBinsDouble[0].reserve(nY2Xbins);
  uvBinsDouble[1].reserve(nZ2Xbins);

  for (int i = 0, j = nY2Xbins - 1; i <= j; i += 2, j -= 2) {
    uvBinsDouble[0].push_back(trackResiduals.getY2X(0, i));
    if (j >= i + 1) {
      uvBinsDouble[0].push_back(trackResiduals.getY2X(0, j));
    }
  }

  for (int i = 0, j = nZ2Xbins - 1; i <= j; i += 2, j -= 2) {
    uvBinsDouble[1].push_back(-trackResiduals.getZ2X(i));
    if (j >= i + 1) {
      uvBinsDouble[1].push_back(-trackResiduals.getZ2X(j));
    }
  }

  std::vector<int> uvBinsInt[2];

  for (int iuv = 0; iuv < 2; iuv++) {
    auto& bins = uvBinsDouble[iuv];
    std::sort(bins.begin(), bins.end());

    auto& binsInt = uvBinsInt[iuv];
    binsInt.reserve(bins.size());

    double dy = bins[1] - bins[0];
    for (int i = 2; i < bins.size(); i++) {
      double dd = bins[i] - bins[i - 1];
      if (dd < dy) {
        dy = dd;
      }
    }
    // spline knots must be positioned on the grid with integer internal coordinate
    // take the knot position accuracy of 0.1*dy
    dy = dy / 10.;
    double y0 = bins[0];
    double y1 = bins[bins.size() - 1];
    for (auto& y : bins) {
      y -= y0;
      int iy = int(y / dy + 0.5);
      binsInt.push_back(iy);
      double yold = y / (y1 - y0) * 2 - 1.;
      y = iy * dy;
      y = y / (y1 - y0) * 2 - 1.;
      if (iuv == 0) {
        LOG(info) << "TPC SC splines: convert y bin: " << yold << " -> " << y << " -> " << iy;
      } else {
        LOG(info) << "TPC SC splines: convert z bin: " << yold << " -> " << y << " -> " << iy;
      }
    }

    if (binsInt.size() < 2) {
      binsInt.clear();
      binsInt.push_back(0);
      binsInt.push_back(1);
    }
  }

  auto& yBinsInt = uvBinsInt[0];
  auto& zBinsInt = uvBinsInt[1];

  int nKnotsY = yBinsInt.size();
  int nKnotsZ = zBinsInt.size();

  // std::cout << "n knots Y: " << nKnotsY << std::endl;
  // std::cout << "n knots Z: " << nKnotsZ << std::endl;

  const int nRows = geo.getNumberOfRows();
  const int nSectors = geo.getNumberOfSectors();

  { // create the correction object

    const int nCorrectionScenarios = 1;

    correction.startConstruction(geo, nCorrectionScenarios);

    // init rows
    for (int row = 0; row < geo.getNumberOfRows(); row++) {
      correction.setRowScenarioID(row, 0);
    }
    { // init spline scenario
      TPCFastSpaceChargeCorrection::SplineType spline;
      spline.recreate(nKnotsY, &yBinsInt[0], nKnotsZ, &zBinsInt[0]);
      correction.setSplineScenario(0, spline);
    }
    correction.finishConstruction();
  } // .. create the correction object

  // set the grid borders
  for (int iSector = 0; iSector < geo.getNumberOfSectors(); iSector++) {
    for (int iRow = 0; iRow < geo.getNumberOfRows(); iRow++) {
      const auto& rowInfo = geo.getRowInfo(iRow);
      auto& info = correction.getSectorRowInfo(iSector, iRow);
      const auto& spline = correction.getSpline(iSector, iRow);
      double yMin = rowInfo.x * trackResiduals.getY2X(iRow, 0);
      double yMax = rowInfo.x * trackResiduals.getY2X(iRow, trackResiduals.getNY2XBins() - 1);
      double zMin = rowInfo.x * trackResiduals.getZ2X(0);
      double zMax = rowInfo.x * trackResiduals.getZ2X(trackResiduals.getNZ2XBins() - 1);
      double uMin = yMin;
      double uMax = yMax;
      double vMin = geo.getTPCzLength() - zMax;
      double vMax = geo.getTPCzLength() - zMin;
      info.gridU0 = uMin;
      info.scaleUtoGrid = spline.getGridX1().getUmax() / (uMax - uMin);
      info.gridV0 = vMin;
      info.scaleVtoGrid = spline.getGridX2().getUmax() / (vMax - vMin);
      info.gridCorrU0 = info.gridU0;
      info.gridCorrV0 = info.gridV0;
      info.scaleCorrUtoGrid = info.scaleUtoGrid;
      info.scaleCorrVtoGrid = info.scaleVtoGrid;

      // std::cout << " iSector " << iSector << " iRow " << iRow << " uMin: " << uMin << " uMax: " << uMax << " vMin: " << vMin << " vMax: " << vMax
      //<< " grid scale u "<< info.scaleUtoGrid << " grid scale v "<< info.scaleVtoGrid<< std::endl;
    }
  }

  LOG(info) << "fast space charge correction helper: preparation took " << watch1.RealTime() << "s";

  for (int processingInverseCorrection = 0; processingInverseCorrection < 2; processingInverseCorrection++) {

    TTree* currentTree = (processingInverseCorrection) ? voxResTreeInverse : voxResTree;

    if (!currentTree) {
      continue;
    }

    LOG(info) << "fast space charge correction helper: " << ((processingInverseCorrection) ? "inverse" : "direct")
              << " : fill data points from track residuals.. ";

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

      auto myThread = [&](TTreeReader& readerSubRange) {
        TTreeReaderValue<o2::tpc::TrackResiduals::VoxRes> v(readerSubRange, "voxRes");
        while (readerSubRange.Next()) {
          int iSector = (int)v->bsec;
          if (iSector < 0 || iSector >= nSectors) {
            LOG(fatal) << "Error reading voxels: voxel Sector number " << iSector << " is out of range";
            continue;
          }
          int iRow = (int)v->bvox[o2::tpc::TrackResiduals::VoxX]; // bin number in x (= pad row)
          if (iRow < 0 || iRow >= nRows) {
            LOG(fatal) << "Row number " << iRow << " is out of range";
          }
          int iy = v->bvox[o2::tpc::TrackResiduals::VoxF]; // bin number in y/x 0..14
          int iz = v->bvox[o2::tpc::TrackResiduals::VoxZ]; // bin number in z/x 0..4
          auto& data = vSectorData[iSector * nRows + iRow][iy * nZ2Xbins + iz];
          data.mNentries = (int)v->stat[o2::tpc::TrackResiduals::VoxV];
          data.mX = v->stat[o2::tpc::TrackResiduals::VoxX];
          data.mY = v->stat[o2::tpc::TrackResiduals::VoxF];
          data.mZ = v->stat[o2::tpc::TrackResiduals::VoxZ];
          data.mCx = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResX] : v->D[o2::tpc::TrackResiduals::ResX];
          data.mCy = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResY] : v->D[o2::tpc::TrackResiduals::ResY];
          data.mCz = useSmoothed ? v->DS[o2::tpc::TrackResiduals::ResZ] : v->D[o2::tpc::TrackResiduals::ResZ];
          if (0 && data.mNentries < 1) {
            data.mCx = 0.;
            data.mCy = 0.;
            data.mCz = 0.;
            data.mNentries = 1;
          }
        }
      };
      processor.Process(myThread);
    }

    for (int iSector = 0; iSector < nSectors; iSector++) {

      // now process the data row-by-row

      auto myThread = [&](int iThread, int nTreads) {
        struct Voxel {
          float mY, mZ;            // not-distorted local coordinates
          float mDy, mDz;          // bin size
          int mSmoothingStep{100}; // is the voxel data original or smoothed at this step
        };

        std::vector<Voxel> vRowVoxels(nY2Xbins * nZ2Xbins);

        for (int iRow = iThread; iRow < nRows; iRow += nTreads) {
          // LOG(info) << "Processing Sector " << iSector << " row " << iRow;

          // complete the voxel data

          {
            int xBin = iRow;
            double x = trackResiduals.getX(xBin); // radius of the pad row
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
                data.mY *= x;
                data.mZ *= x;
                /*
                if ( fabs(x - data.mX) > 0.01 || fabs(vox.mY - data.mY) > 5. || fabs(vox.mZ - data.mZ) > 5.) {
                  std::cout
                    << " sector " << iSector << " row " << iRow
                    << " voxel x " << x << " y " << vox.mY << " z " << vox.mZ
                    << " data x " << data.mX << " y " << data.mY << " z " << data.mZ
                    << std::endl;
                }
                */
                if (0) { // debug: always use voxel center instead of the mean position
                  data.mY = vox.mY;
                  data.mZ = vox.mZ;
                }
                if (data.mNentries < 1) { // no data
                  data.mCx = 0.;
                  data.mCy = 0.;
                  data.mCz = 0.;
                  data.mY = vox.mY;
                  data.mZ = vox.mZ;
                  vox.mSmoothingStep = 100;
                } else { // voxel contains data
                  if (invertSigns) {
                    data.mCx *= -1.;
                    data.mCy *= -1.;
                    data.mCz *= -1.;
                  }
                  vox.mSmoothingStep = 0; // original data
                  isDataFound = true;
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

          auto addEdge = [&](int iy1, int iz1, int iy2, int iz2, int nSteps) {
            auto& data1 = vSectorData[iSector * nRows + iRow][iy1 * nZ2Xbins + iz1];
            auto& vox1 = vRowVoxels[iy1 * nZ2Xbins + iz1];
            auto& data2 = vSectorData[iSector * nRows + iRow][iy2 * nZ2Xbins + iz2];
            auto& vox2 = vRowVoxels[iy2 * nZ2Xbins + iz2];
            if (vox1.mSmoothingStep > 2) {
              LOG(fatal) << "empty voxel is not repared: y " << iy1 << " z " << iz1;
            }
            if (vox2.mSmoothingStep > 2) {
              LOG(fatal) << "empty voxel is not repared: y " << iy2 << " z " << iz2;
            }
            double y1 = vox1.mY;
            double z1 = vox1.mZ;
            double cx1 = data1.mCx;
            double cy1 = data1.mCy;
            double cz1 = data1.mCz;
            double y2 = vox2.mY;
            double z2 = vox2.mZ;
            double cx2 = data2.mCx;
            double cy2 = data2.mCy;
            double cz2 = data2.mCz;

            for (int is = 0; is < nSteps; is++) {
              double s2 = is / (double)nSteps;
              double s1 = 1. - s2;
              double y = s1 * y1 + s2 * y2;
              double z = s1 * z1 + s2 * z2;
              double cx = s1 * cx1 + s2 * cx2;
              double cy = s1 * cy1 + s2 * cy2;
              double cz = s1 * cz1 + s2 * cz2;
              map.addCorrectionPoint(iSector, iRow, y, z, cx, cy, cz);
            }
          };

          for (int iy = 0; iy < nY2Xbins; iy++) {
            for (int iz = 0; iz < nZ2Xbins - 1; iz++) {
              addEdge(iy, iz, iy, iz + 1, 3);
            }
            addEdge(iy, nZ2Xbins - 1, iy, nZ2Xbins - 1, 1);
          }

          for (int iz = 0; iz < nZ2Xbins; iz++) {
            for (int iy = 0; iy < nY2Xbins - 1; iy++) {
              addEdge(iy, iz, iy + 1, iz, 3);
            }
            addEdge(nY2Xbins - 1, iz, nY2Xbins - 1, iz, 1);
          } // iy

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

        std::vector<double> dataPointGridU, dataPointGridV, dataPointF;
        dataPointGridU.reserve(gridU.size() * gridV.size());
        dataPointGridV.reserve(gridU.size() * gridV.size());
        dataPointF.reserve(3 * gridU.size() * gridV.size());

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

            double realY = y + dy;
            double realZ = z + dz;
            float realU, realV;
            mGeo.convLocalToUV1(sector, realY, realZ, realU, realV);

            dataPointGridU.push_back(realU);
            dataPointGridV.push_back(realV);
            dataPointF.push_back(dx);
            dataPointF.push_back(dy);
            dataPointF.push_back(dz);
          }
        }

        // define the grid for the inverse correction

        auto& sectorRowInfo = correction.getSectorRowInfo(sector, row);

        sectorRowInfo.gridCorrU0 = sectorRowInfo.gridU0;
        sectorRowInfo.gridCorrV0 = sectorRowInfo.gridV0;
        sectorRowInfo.scaleCorrUtoGrid = sectorRowInfo.scaleUtoGrid;
        sectorRowInfo.scaleCorrVtoGrid = sectorRowInfo.scaleVtoGrid;

        int nDataPoints = dataPointGridU.size();

        // convert real Y,Z to grid U,V
        for (int i = 0; i < nDataPoints; i++) {
          dataPointGridU[i] = (dataPointGridU[i] - sectorRowInfo.gridCorrU0) * sectorRowInfo.scaleCorrUtoGrid;
          dataPointGridV[i] = (dataPointGridV[i] - sectorRowInfo.gridCorrV0) * sectorRowInfo.scaleCorrVtoGrid;
        }

        splineParameters.resize(spline.getNumberOfParameters());

        helper.approximateDataPoints(spline, splineParameters.data(), 0., spline.getGridX1().getUmax(),
                                     0., spline.getGridX2().getUmax(),
                                     dataPointGridU.data(), dataPointGridV.data(),
                                     dataPointF.data(), nDataPoints);

        float* splineX = correction.getSplineDataInvX(sector, row);
        float* splineUV = correction.getSplineDataInvYZ(sector, row);
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

void TPCFastSpaceChargeCorrectionHelper::MergeCorrections(
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

        float* splineParameters = mainCorrection.getSplineData(sector, row);
        float* splineParametersInvX = mainCorrection.getSplineDataInvX(sector, row);
        float* splineParametersInvYZ = mainCorrection.getSplineDataInvYZ(sector, row);

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

          double scaleU = secRowInfo.scaleUtoGrid / linfo.scaleUtoGrid;
          double scaleV = secRowInfo.scaleVtoGrid / linfo.scaleVtoGrid;

          for (int iu = 0; iu < gridU.getNumberOfKnots(); iu++) {
            double u = gridU.getKnot(iu).u;
            for (int iv = 0; iv < gridV.getNumberOfKnots(); iv++) {
              double v = gridV.getKnot(iu).u;
              int knotIndex = spline.getKnotIndex(iu, iv);
              float P[nKnotPar3d];

              { // direct correction
                auto [y, z] = mainCorrection.convGridToLocal(sector, row, u, v);
                // return values: u, v, scaling factor
                auto [lu, lv, ls] = corr.convLocalToGrid(sector, row, y, z);
                ls *= scale;
                double parscale[4] = {ls, ls * scaleU, ls * scaleV, ls * ls * scaleU * scaleV};
                const auto& spl = corr.getSpline(sector, row);
                spl.interpolateParametersAtU(corr.getSplineData(sector, row), lu, lv, P);
                for (int ipar = 0, ind = 0; ipar < nKnotPar1d; ++ipar) {
                  for (int idim = 0; idim < 3; idim++, ind++) {
                    splineParameters[knotIndex * nKnotPar3d + ind] += parscale[ipar] * P[ind];
                  }
                }
              }

              auto [y, z] = mainCorrection.convGridToCorrectedLocal(sector, row, u, v);
              // return values: u, v, scaling factor
              auto [lu, lv, ls] = corr.convCorrectedLocalToGrid(sector, row, y, z);
              ls *= scale;
              double parscale[4] = {ls, ls * scaleU, ls * scaleV, ls * ls * scaleU * scaleV};

              { // inverse X correction
                corr.getSplineInvX(sector, row).interpolateParametersAtU(corr.getSplineDataInvX(sector, row), lu, lv, P);
                for (int ipar = 0, ind = 0; ipar < nKnotPar1d; ++ipar) {
                  for (int idim = 0; idim < 1; idim++, ind++) {
                    splineParametersInvX[knotIndex * nKnotPar1d + ind] += parscale[ipar] * P[ind];
                  }
                }
              }

              { // inverse YZ correction
                corr.getSplineInvYZ(sector, row).interpolateParametersAtU(corr.getSplineDataInvYZ(sector, row), lu, lv, P);
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

} // namespace tpc
} // namespace o2
