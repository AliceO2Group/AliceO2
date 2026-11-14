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

/// \file testO2TPCSpaceCharge.cxx
/// \brief closure tests of the global distortions and corrections calculated from analytical electric fields, reading of stored objects

#define BOOST_TEST_MODULE Test TPC SpaceCharge class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSpaceCharge/SpaceCharge.h"
#include "TPCSpaceCharge/SpaceChargeHelpers.h"
#include "TPCSpaceCharge/PoissonSolver.h"
#include "TMemFile.h"
#include "TH3F.h"
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <array>
#include <vector>

namespace o2
{
namespace tpc
{

using DataT = double;
using SC = SpaceCharge<DataT>;

namespace
{
struct Residuals {
  DataT rms[3]{}; ///< rms of the residuals in r, rphi, z
  DataT max[3]{}; ///< maximum absolute residual in r, rphi, z
};

/// distort random points with the global distortions and correct them with the global corrections: the original point should be recovered
Residuals closure(const SC& sc, const Side side)
{
  std::mt19937 gen(42);
  std::uniform_real_distribution<DataT> distR(95, 240);
  std::uniform_real_distribution<DataT> distPhi(0, 2 * M_PI);
  std::uniform_real_distribution<DataT> distZ(5, 245);
  const int nPoints = 20000;
  Residuals res;
  for (int i = 0; i < nPoints; ++i) {
    const DataT r = distR(gen);
    const DataT phi = distPhi(gen);
    const DataT z = (side == Side::A) ? distZ(gen) : -distZ(gen);
    DataT dZ, dR, dRPhi;
    sc.getDistortionsCyl(z, r, phi, side, dZ, dR, dRPhi);
    const DataT rDist = r + dR;
    const DataT phiDist = phi + dRPhi / r;
    const DataT zDist = z + dZ;
    DataT cZ, cR, cRPhi;
    sc.getCorrectionsCyl(zDist, rDist, phiDist, side, cZ, cR, cRPhi);
    const DataT diff[3]{rDist + cR - r, (phiDist + cRPhi / rDist - phi) * r, zDist + cZ - z};
    for (int k = 0; k < 3; ++k) {
      res.rms[k] += diff[k] * diff[k];
      res.max[k] = std::max(res.max[k], std::abs(diff[k]));
    }
  }
  for (int k = 0; k < 3; ++k) {
    res.rms[k] = std::sqrt(res.rms[k] / nPoints);
  }
  return res;
}

/// \return number of vertices with a different density (both sides)
int getNDifferentDensity(const SC& sc1, const SC& sc2)
{
  int nDiff = 0;
  for (const Side side : {Side::A, Side::C}) {
    for (size_t iPhi = 0; iPhi < sc1.getNPhiVertices(); ++iPhi) {
      for (size_t iR = 0; iR < sc1.getNRVertices(); ++iR) {
        for (size_t iZ = 0; iZ < sc1.getNZVertices(); ++iZ) {
          nDiff += (sc1.getDensity(iZ, iR, iPhi, side) != sc2.getDensity(iZ, iR, iPhi, side));
        }
      }
    }
  }
  return nDiff;
}

/// \return maximum absolute density of both sides
DataT getMaxDensity(const SC& sc)
{
  DataT maxDensity = 0;
  for (const Side side : {Side::A, Side::C}) {
    for (size_t iPhi = 0; iPhi < sc.getNPhiVertices(); ++iPhi) {
      for (size_t iR = 0; iR < sc.getNRVertices(); ++iR) {
        for (size_t iZ = 0; iZ < sc.getNZVertices(); ++iZ) {
          maxDensity = std::max(maxDensity, std::abs(sc.getDensity(iZ, iR, iPhi, side)));
        }
      }
    }
  }
  return maxDensity;
}

/// fill histograms of the A side (z in [0, zMax]) and of the C side (z in [-zMax, 0]) with random entries
void fillDensityHistos(TH3& hA, TH3& hC)
{
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> distPhi(0, 2 * M_PI);
  std::uniform_real_distribution<double> distR(hA.GetYaxis()->GetXmin(), hA.GetYaxis()->GetXmax());
  std::uniform_real_distribution<double> distZ(hA.GetZaxis()->GetXmin(), hA.GetZaxis()->GetXmax());
  std::uniform_real_distribution<double> distW(0.5, 2);
  for (int i = 0; i < 5000; ++i) {
    const double phi = distPhi(gen);
    const double r = distR(gen);
    const double z = distZ(gen);
    hA.Fill(phi, r, z, distW(gen));
    hC.Fill(phi, r, -z, distW(gen));
  }
}

constexpr double ZMAXHIST = 250;
constexpr double RMINHIST = 83.5;
constexpr double RMAXHIST = 254.5;

struct GlobalDist {
  std::vector<DataT> r, rphi, z;
};

GlobalDist getGlobalDistortions(const SC& sc, const Side side, const int nZ, const int nR, const int nPhi)
{
  GlobalDist dist;
  for (int iPhi = 0; iPhi < nPhi; ++iPhi) {
    for (int iR = 0; iR < nR; ++iR) {
      for (int iZ = 0; iZ < nZ; ++iZ) {
        dist.r.emplace_back(sc.getGlobalDistR(iZ, iR, iPhi, side));
        dist.rphi.emplace_back(sc.getGlobalDistRPhi(iZ, iR, iPhi, side));
        dist.z.emplace_back(sc.getGlobalDistZ(iZ, iR, iPhi, side));
      }
    }
  }
  return dist;
}
} // namespace

/// global distortions with the standard method (integration of local distortions) and with the fast method (inversion of the global corrections)
/// have to be consistent with the global corrections and with each other
BOOST_AUTO_TEST_CASE(SpaceCharge_closure_test)
{
  const int nZ = 65;
  const int nR = 65;
  const int nPhi = 90;
  SC::setNThreads(4);
  for (const Side side : {Side::A, Side::C}) {
    SC sc(0, nZ, nR, nPhi);
    AnalyticalFields<DataT> fields(side);
    fields.setParameters(5e-7, 0.5, 1e-4); // global distortions of O(cm)

    sc.calcLocalDistortionsCorrections(SC::Type::Distortions, fields);
    sc.calcLocalDistortionsCorrections(SC::Type::Corrections, fields);
    sc.calcGlobalCorrections(sc.getLocalCorrInterpolator(side));

    // standard method
    sc.calcGlobalDistortions(sc.getLocalDistInterpolator(side));
    const GlobalDist distStd = getGlobalDistortions(sc, side, nZ, nR, nPhi);
    const Residuals resStd = closure(sc, side);

    // fast method
    sc.calcGlobalDistWithGlobalCorrIterative(sc.getGlobalCorrInterpolator(side));
    const GlobalDist distFast = getGlobalDistortions(sc, side, nZ, nR, nPhi);
    const Residuals resFast = closure(sc, side);

    // magnitude of the distortions and difference between the methods (the first z vertex of the fast method is extrapolated)
    DataT maxDist[3]{};
    DataT meanDiff[3]{};
    DataT maxDiff[3]{};
    int nDiff = 0;
    for (size_t i = 0; i < distStd.z.size(); ++i) {
      const DataT valStd[3]{distStd.r[i], distStd.rphi[i], distStd.z[i]};
      const DataT valFast[3]{distFast.r[i], distFast.rphi[i], distFast.z[i]};
      if ((i % nZ) == 0) {
        continue;
      }
      ++nDiff;
      for (int k = 0; k < 3; ++k) {
        maxDist[k] = std::max(maxDist[k], std::abs(valStd[k]));
        meanDiff[k] += valStd[k] - valFast[k];
        maxDiff[k] = std::max(maxDiff[k], std::abs(valStd[k] - valFast[k]));
      }
    }
    for (int k = 0; k < 3; ++k) {
      meanDiff[k] /= nDiff;
    }

    BOOST_TEST_MESSAGE("side " << static_cast<int>(side) << " max global distortion (r, rphi, z) [cm]: " << maxDist[0] << " " << maxDist[1] << " " << maxDist[2]);
    BOOST_TEST_MESSAGE("  closure standard rms: " << resStd.rms[0] << " " << resStd.rms[1] << " " << resStd.rms[2] << " max: " << resStd.max[0] << " " << resStd.max[1] << " " << resStd.max[2]);
    BOOST_TEST_MESSAGE("  closure fast     rms: " << resFast.rms[0] << " " << resFast.rms[1] << " " << resFast.rms[2] << " max: " << resFast.max[0] << " " << resFast.max[1] << " " << resFast.max[2]);
    BOOST_TEST_MESSAGE("  standard - fast mean: " << meanDiff[0] << " " << meanDiff[1] << " " << meanDiff[2] << " max: " << maxDiff[0] << " " << maxDiff[1] << " " << maxDiff[2]);

    // the distortions must be large enough for a meaningful test
    BOOST_CHECK(maxDist[0] > 0.5);
    BOOST_CHECK(maxDist[2] > 0.1);
    // closure and agreement of the methods: O(0.01 um) for these smooth fields, tolerance 0.2 um (rms) and 0.1 um (mean difference)
    for (int k = 0; k < 3; ++k) {
      BOOST_CHECK_SMALL(resStd.rms[k], 2e-5);
      BOOST_CHECK_SMALL(resFast.rms[k], 2e-5);
      BOOST_CHECK_SMALL(meanDiff[k], 1e-5);
    }
  }
}

/// unphysically large fields: the electrons drift out of the grid during the calculation of the global corrections.
/// The calculation must not access memory outside of the grid, invalid positions give NaN values
BOOST_AUTO_TEST_CASE(SpaceCharge_largeFields_test)
{
  const int nZ = 33;
  const int nR = 33;
  const int nPhi = 36;
  SC::setNThreads(4);
  const Side side = Side::A;
  SC sc(0, nZ, nR, nPhi);
  AnalyticalFields<DataT> fields(side);
  fields.setParameters(5e-5, 0.5, 1e-4); // local corrections of several cm per z bin
  sc.calcLocalDistortionsCorrections(SC::Type::Corrections, fields);
  sc.calcGlobalCorrections(sc.getLocalCorrInterpolator(side));
  sc.calcGlobalDistWithGlobalCorrIterative(sc.getGlobalCorrInterpolator(side));

  // query at a non-finite position
  DataT cZ, cR, cRPhi;
  sc.getCorrectionsCyl(std::numeric_limits<DataT>::quiet_NaN(), 150, 1, side, cZ, cR, cRPhi);
  BOOST_CHECK(std::isnan(cZ) && std::isnan(cR) && std::isnan(cRPhi));
  sc.getCorrectionsCyl(100, 150, std::numeric_limits<DataT>::infinity(), side, cZ, cR, cRPhi);
  BOOST_CHECK(std::isnan(cZ) && std::isnan(cR) && std::isnan(cRPhi));
  // query at a phi far outside of the grid (periodic): has to terminate and give the same value as the phi in the grid
  DataT cZFar, cRFar, cRPhiFar;
  sc.getCorrectionsCyl(100, 150, 1e300, side, cZFar, cRFar, cRPhiFar);
  const DataT periodPhi = nPhi * sc.getGridSpacingPhi(side); // period of the grid (not exactly 2 pi in double precision)
  sc.getCorrectionsCyl(100, 150, 1 + 1000 * periodPhi, side, cZFar, cRFar, cRPhiFar);
  sc.getCorrectionsCyl(100, 150, 1, side, cZ, cR, cRPhi);
  BOOST_CHECK_CLOSE(cRFar, cR, 1e-6);
}

/// the stopping criterion set with PoissonSolver::setConvergenceError() (here the previous absolute criterion) is used by SpaceCharge::poissonSolver()
/// and not overwritten by a default value. An explicitly given stopping criterion is set for the PoissonSolver
BOOST_AUTO_TEST_CASE(SpaceCharge_convergenceError_test)
{
  const int nZ = 33;
  const int nR = 33;
  const int nPhi = 36;
  SC::setNThreads(4);
  const Side side = Side::A;
  SC sc(0, nZ, nR, nPhi);
  AnalyticalFields<DataT> fields(side);
  sc.setChargeDensityFromFormula(fields);
  sc.setPotentialBoundaryFromFormula(fields);

  // reference with the default stopping criterion
  sc.poissonSolver(side);
  std::vector<DataT> potRef;
  for (int iPhi = 0; iPhi < nPhi; ++iPhi) {
    for (int iR = 0; iR < nR; ++iR) {
      for (int iZ = 0; iZ < nZ; ++iZ) {
        potRef.emplace_back(sc.getPotential(iZ, iR, iPhi, side));
      }
    }
  }

  // previous absolute stopping criterion
  MGParameters::relativeConvergence = false;
  PoissonSolver<DataT>::setConvergenceError(1e-6);
  sc.poissonSolver(side);
  BOOST_CHECK_EQUAL(PoissonSolver<DataT>::getConvergenceError(), DataT(1e-6));
  DataT maxDiff = 0;
  DataT maxPot = 0;
  size_t index = 0;
  for (int iPhi = 0; iPhi < nPhi; ++iPhi) {
    for (int iR = 0; iR < nR; ++iR) {
      for (int iZ = 0; iZ < nZ; ++iZ) {
        maxDiff = std::max(maxDiff, std::abs(sc.getPotential(iZ, iR, iPhi, side) - potRef[index]));
        maxPot = std::max(maxPot, std::abs(potRef[index]));
        ++index;
      }
    }
  }
  BOOST_TEST_MESSAGE("max |V_abs - V_default| / max |V|: " << maxDiff / maxPot);
  BOOST_CHECK_SMALL(maxDiff / maxPot, DataT(1e-6));

  // explicitly given stopping criterion
  MGParameters::relativeConvergence = true;
  sc.poissonSolver(side, 1e-9);
  BOOST_CHECK_EQUAL(PoissonSolver<DataT>::getConvergenceError(), DataT(1e-9));

  // restore the default for the other tests
  PoissonSolver<DataT>::setConvergenceError(PoissonSolver<DataT>::getDefaultConvergenceError());
}

/// density from histograms with a pending fill buffer (TH1::SetBuffer()): has to be the same as from the histograms without buffer
BOOST_AUTO_TEST_CASE(SpaceCharge_densityFromBufferedHisto_test)
{
  SC::setNThreads(4);
  TH3F hA("hA", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 32, 0, ZMAXHIST);
  TH3F hC("hC", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 32, -ZMAXHIST, 0);
  fillDensityHistos(hA, hC);
  TH3F hABuffer("hABuffer", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 32, 0, ZMAXHIST);
  TH3F hCBuffer("hCBuffer", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 32, -ZMAXHIST, 0);
  hABuffer.SetBuffer(10000);
  hCBuffer.SetBuffer(10000);
  fillDensityHistos(hABuffer, hCBuffer);
  BOOST_REQUIRE(hABuffer.GetBuffer() && (hABuffer.GetBuffer()[0] > 0)); // the entries are still in the buffer

  SC sc(0, 17, 17, 18);
  sc.fillChargeDensityFromHisto(hA, hC);
  SC scBuffer(0, 17, 17, 18);
  scBuffer.fillChargeDensityFromHisto(hABuffer, hCBuffer);
  BOOST_CHECK(getMaxDensity(sc) > 0);
  BOOST_CHECK_EQUAL(getNDifferentDensity(sc, scBuffer), 0);

  // one histogram for both sides
  TH3F hAC("hAC", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 64, -ZMAXHIST, ZMAXHIST);
  TH3F hACBuffer("hACBuffer", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 64, -ZMAXHIST, ZMAXHIST);
  hACBuffer.SetBuffer(20000);
  fillDensityHistos(hAC, hAC);
  fillDensityHistos(hACBuffer, hACBuffer);
  BOOST_REQUIRE(hACBuffer.GetBuffer() && (hACBuffer.GetBuffer()[0] > 0));
  sc.fillChargeDensityFromHisto(hAC);
  scBuffer.fillChargeDensityFromHisto(hACBuffer);
  BOOST_CHECK(getMaxDensity(sc) > 0);
  BOOST_CHECK_EQUAL(getNDifferentDensity(sc, scBuffer), 0);
}

/// density from A-side and C-side histograms with a different binning: the histograms must not be read outside of their bin arrays.
/// A different number of bins is an error (the density is not changed and false is returned), a different range is read with the bin numbers of the A side (as before)
BOOST_AUTO_TEST_CASE(SpaceCharge_densityFromHistoBinning_test)
{
  SC::setNThreads(4);
  TH3F hA("hA", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 32, 0, ZMAXHIST);
  TH3F hC("hC", "", 36, 0, 2 * M_PI, 34, RMINHIST, RMAXHIST, 32, -ZMAXHIST, 0);
  fillDensityHistos(hA, hC);
  SC scRef(0, 17, 17, 18);
  BOOST_CHECK(scRef.fillChargeDensityFromHisto(hA, hC));
  BOOST_CHECK(getMaxDensity(scRef) > 0);

  // different number of bins of the C-side histogram
  for (const auto& nBins : {std::array<int, 3>{18, 34, 32}, std::array<int, 3>{36, 17, 32}, std::array<int, 3>{36, 34, 16}, std::array<int, 3>{18, 17, 16}, std::array<int, 3>{72, 34, 32}}) {
    TH3F hCOther("hCOther", "", nBins[0], 0, 2 * M_PI, nBins[1], RMINHIST, RMAXHIST, nBins[2], -ZMAXHIST, 0);
    fillDensityHistos(hCOther, hCOther);
    SC sc(0, 17, 17, 18);
    BOOST_CHECK(sc.fillChargeDensityFromHisto(hA, hC));
    BOOST_CHECK(!sc.fillChargeDensityFromHisto(hA, hCOther));
    BOOST_CHECK_EQUAL(getNDifferentDensity(sc, scRef), 0);
  }

  // same number of bins but different range of the C-side histogram
  TH3F hCShifted("hCShifted", "", 36, 0, 2 * M_PI, 34, RMINHIST + 10, RMAXHIST + 10, 32, 0, ZMAXHIST);
  for (int bin = 0; bin < hC.GetNcells(); ++bin) {
    hCShifted.SetBinContent(bin, hC.GetBinContent(bin));
  }
  SC scShifted(0, 17, 17, 18);
  BOOST_CHECK(scShifted.fillChargeDensityFromHisto(hA, hCShifted));
  BOOST_CHECK_EQUAL(getNDifferentDensity(scShifted, scRef), 0);
}

/// write the object to a file and read it back: the grid (transient) has to be set up from the stored number of vertices
/// without calling initAfterReadingFromFile() (read rule in the LinkDef)
template <typename T>
void checkReadFromFile()
{
  const int nZ = 33;
  const int nR = 65;
  const int nPhi = 90; // not the default number of vertices
  const Side side = Side::A;
  SpaceCharge<T> sc(0, nZ, nR, nPhi);
  AnalyticalFields<T> fields(side);
  fields.setParameters(5e-7, 0.5, 1e-4);
  sc.calcLocalDistortionsCorrections(SpaceCharge<T>::Type::Corrections, fields);
  sc.calcGlobalCorrections(sc.getLocalCorrInterpolator(side));

  TMemFile file("sc.root", "RECREATE");
  file.WriteObject(&sc, "sc");
  std::unique_ptr<SpaceCharge<T>> scRead(file.Get<SpaceCharge<T>>("sc"));
  BOOST_REQUIRE(scRead);
  BOOST_CHECK_EQUAL(scRead->getNPhiVertices(), nPhi);
  BOOST_CHECK_EQUAL(scRead->getGridSpacingPhi(side), sc.getGridSpacingPhi(side));
  BOOST_CHECK_EQUAL(scRead->getGridSpacingR(side), sc.getGridSpacingR(side));
  BOOST_CHECK_EQUAL(scRead->getGridSpacingZ(side), sc.getGridSpacingZ(side));
  for (const T phi : {0.1, 1.3, 4.7}) {
    T cZ, cR, cRPhi, cZRead, cRRead, cRPhiRead;
    sc.getCorrectionsCyl(120, 150, phi, side, cZ, cR, cRPhi);
    scRead->getCorrectionsCyl(120, 150, phi, side, cZRead, cRRead, cRPhiRead);
    BOOST_CHECK_EQUAL(cRRead, cR);
    BOOST_CHECK_EQUAL(cRPhiRead, cRPhi);
    BOOST_CHECK_EQUAL(cZRead, cZ);
  }
}

BOOST_AUTO_TEST_CASE(SpaceCharge_readFromFile_test)
{
  SC::setNThreads(4);
  SpaceCharge<float>::setNThreads(4);
  checkReadFromFile<double>();
  checkReadFromFile<float>();
}

/// distortions scaled with the global corrections of a derivative map (scaleGlobalDistWithDerivativeCorr) have to be consistent with the scaled
/// corrections C(y) + s * C_der(y) at the distorted point y (in cartesian coordinates, as in the reconstruction) and agree with the inversion of the
/// scaled corrections (calcGlobalDistWithGlobalCorrIterativeLinearCartesian), here with exact corrections of the average map
BOOST_AUTO_TEST_CASE(SpaceCharge_scaleWithDerivative_test)
{
  const int nZ = 65;
  const int nR = 65;
  const int nPhi = 90;
  SC::setNThreads(4);
  for (const Side side : {Side::A, Side::C}) {
    // global corrections from analytical fields, global distortions by inversion
    auto makeMap = [&](const DataT parA, const DataT parB, const DataT parC) {
      auto sc = std::make_unique<SC>(0, nZ, nR, nPhi);
      AnalyticalFields<DataT> fields(side);
      fields.setParameters(parA, parB, parC);
      sc->calcLocalDistortionsCorrections(SC::Type::Corrections, fields);
      sc->calcGlobalCorrections(sc->getLocalCorrInterpolator(side));
      sc->calcGlobalDistWithGlobalCorrIterative(sc->getGlobalCorrInterpolator(side));
      return sc;
    };
    // derivative map: smaller than the average map and with a different shape
    const auto der = makeMap(1e-7, 1.5, 3e-4);

    // without global distortions of the object or without global corrections of the derivative map nothing is scaled, which is returned to the caller
    {
      SC scEmpty(0, nZ, nR, nPhi);
      BOOST_CHECK(!scEmpty.scaleGlobalDistWithDerivativeCorr(side, *der, 1.f));
      auto sc = makeMap(5e-7, 0.5, 1e-4);
      const GlobalDist distBefore = getGlobalDistortions(*sc, side, nZ, nR, nPhi);
      BOOST_CHECK(!sc->scaleGlobalDistWithDerivativeCorr(side, scEmpty, 1.f));
      const GlobalDist distAfter = getGlobalDistortions(*sc, side, nZ, nR, nPhi);
      BOOST_CHECK(distAfter.r == distBefore.r);
      BOOST_CHECK(distAfter.rphi == distBefore.rphi);
      BOOST_CHECK(distAfter.z == distBefore.z);
    }

    for (const float scale : {-3.f, -1.f, 2.f}) {
      auto sc = makeMap(5e-7, 0.5, 1e-4);
      auto ref = makeMap(5e-7, 0.5, 1e-4);
      BOOST_CHECK(sc->scaleGlobalDistWithDerivativeCorr(side, *der, scale));
      ref->calcGlobalDistWithGlobalCorrIterativeLinearCartesian(side, der.get(), scale);

      // closure at random points p: y = p + D(p), y + C(y) + scale * C_der(y) - p
      std::mt19937 gen(42);
      std::uniform_real_distribution<DataT> distR(95, 240);
      std::uniform_real_distribution<DataT> distPhi(0, 2 * M_PI);
      std::uniform_real_distribution<DataT> distZ(5, 245);
      const int nPoints = 20000;
      DataT rms[3]{};
      DataT maxRes[3]{};
      DataT maxDer = 0;
      for (int i = 0; i < nPoints; ++i) {
        const DataT r = distR(gen);
        const DataT phi = distPhi(gen);
        const DataT p[3]{r * std::cos(phi), r * std::sin(phi), (side == Side::A) ? distZ(gen) : -distZ(gen)};
        DataT d[3];
        sc->getDistortions(p[0], p[1], p[2], side, d[0], d[1], d[2]);
        const DataT y[3]{p[0] + d[0], p[1] + d[1], p[2] + d[2]};
        DataT c[3], cDer[3];
        sc->getCorrections(y[0], y[1], y[2], side, c[0], c[1], c[2]);
        der->getCorrections(y[0], y[1], y[2], side, cDer[0], cDer[1], cDer[2]);
        for (int k = 0; k < 3; ++k) {
          const DataT res = y[k] + c[k] + scale * cDer[k] - p[k];
          rms[k] += res * res;
          maxRes[k] = std::max(maxRes[k], std::abs(res));
          maxDer = std::max(maxDer, std::abs(scale * cDer[k]));
        }
      }
      for (int k = 0; k < 3; ++k) {
        rms[k] = std::sqrt(rms[k] / nPoints);
      }

      // difference to the inversion of the scaled corrections at the vertices (the first z vertex of the inversion is extrapolated)
      const GlobalDist dist = getGlobalDistortions(*sc, side, nZ, nR, nPhi);
      const GlobalDist distRef = getGlobalDistortions(*ref, side, nZ, nR, nPhi);
      DataT maxDiff[3]{};
      for (size_t i = 0; i < dist.z.size(); ++i) {
        if ((i % nZ) == 0) {
          continue;
        }
        maxDiff[0] = std::max(maxDiff[0], std::abs(dist.r[i] - distRef.r[i]));
        maxDiff[1] = std::max(maxDiff[1], std::abs(dist.rphi[i] - distRef.rphi[i]));
        maxDiff[2] = std::max(maxDiff[2], std::abs(dist.z[i] - distRef.z[i]));
      }

      BOOST_TEST_MESSAGE("side " << static_cast<int>(side) << " scale " << scale << " max |scale * C_der| [cm]: " << maxDer);
      BOOST_TEST_MESSAGE("  closure with the scaled corrections (x, y, z) rms: " << rms[0] << " " << rms[1] << " " << rms[2] << " max: " << maxRes[0] << " " << maxRes[1] << " " << maxRes[2]);
      BOOST_TEST_MESSAGE("  max difference to the inversion (r, rphi, z): " << maxDiff[0] << " " << maxDiff[1] << " " << maxDiff[2]);

      // the scaled derivative must be large enough for a meaningful test
      BOOST_CHECK(maxDer > 0.1);
      // closure O(0.2 um) rms, O(1 um) max, difference to the inversion O(0.5 um) for these smooth fields: tolerance 0.5 um (rms) and 2 um
      for (int k = 0; k < 3; ++k) {
        BOOST_CHECK_SMALL(rms[k], 5e-5);
        BOOST_CHECK_SMALL(maxRes[k], 5e-4);
        BOOST_CHECK_SMALL(maxDiff[k], 2e-4);
      }
    }
  }
}

/// scaleGlobalDistWithDerivativeCorr in single precision (as used by the Digitizer): for radial distortions and radial derivative corrections the scaled
/// distortions are radial. The r-phi distortion at the vertices has to be zero within the rounding of the positions, also for the vertices with phi > pi
/// (no difference of two angles close to 2 pi and no wrapping with the 2 pi in single precision, which is larger than 2 pi by 1.7e-7)
BOOST_AUTO_TEST_CASE(SpaceCharge_scaleWithDerivativeFloat_test)
{
  using SCFloat = SpaceCharge<float>;
  const int nZ = 17;
  const int nR = 33;
  const int nPhi = 90;
  SCFloat::setNThreads(4);
  const float distR = 0.5f;     // radial distortion (cm)
  const float corrRDer = -0.1f; // radial correction of the derivative map (cm)
  const float scale = 2;
  const auto radial = [](const float val) {
    return [val](int, float gx, float gy, float, float& gCx, float& gCy, float& gCz) {
      const float r = std::hypot(gx, gy);
      gCx = val * gx / r;
      gCy = val * gy / r;
      gCz = 0;
    };
  };
  for (const Side side : {Side::A, Side::C}) {
    SCFloat sc(0, nZ, nR, nPhi);
    sc.setGlobalDistortions(radial(distR), side);
    SCFloat der(0, nZ, nR, nPhi);
    der.setGlobalCorrections(radial(corrRDer), side);
    BOOST_REQUIRE(sc.scaleGlobalDistWithDerivativeCorr(side, der, scale));

    // r-phi distortion of the vertices with phi <= pi (index 0) and phi > pi (index 1) and difference of the radial distortion to distR - scale * corrRDer.
    // Without the first and the last radius: their start points are outside of the grid
    double mean[2]{};
    double rms[2]{};
    double maxDiffR = 0;
    int nVertices[2]{};
    for (int iPhi = 0; iPhi < nPhi; ++iPhi) {
      const int half = (sc.getPhiVertex(iPhi, side) > M_PI) ? 1 : 0;
      for (int iR = 1; iR < nR - 1; ++iR) {
        for (int iZ = 0; iZ < nZ; ++iZ) {
          const double distRPhi = sc.getGlobalDistRPhi(iZ, iR, iPhi, side);
          mean[half] += distRPhi;
          rms[half] += distRPhi * distRPhi;
          ++nVertices[half];
          const double diffR = sc.getGlobalDistR(iZ, iR, iPhi, side) - (distR - scale * corrRDer);
          maxDiffR = std::max(maxDiffR, std::abs(diffR));
        }
      }
    }
    BOOST_TEST_MESSAGE("side " << static_cast<int>(side) << " single precision: max difference of the radial distortion [cm]: " << maxDiffR);
    BOOST_CHECK_SMALL(maxDiffR, 3e-4);
    for (int half = 0; half < 2; ++half) {
      mean[half] /= nVertices[half];
      rms[half] = std::sqrt(rms[half] / nVertices[half]);
      BOOST_TEST_MESSAGE("  r-phi distortion [cm] for phi " << (half ? "> " : "<=") << " pi: mean " << mean[half] << " rms " << rms[half]);
      // rounding of the positions in the maps and in the cartesian look ups: no offset (< 0.01 um) and O(0.1 um) rms for phi <= pi, O(0.2 um) rms for phi > pi
      // (the phi of the vertices with phi <= pi is the same number in [-pi, pi], for phi > pi it is rounded).
      // With the difference of the angles wrapped with 2 pi: offset 0.4 um and 0.6 um rms for phi > pi
      BOOST_CHECK_SMALL(std::abs(mean[half]), 1e-5);
      BOOST_CHECK_SMALL(rms[half], 4e-5);
    }
  }
}

} // namespace tpc
} // namespace o2
