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

/// \file  SpaceChargeIO.cxx
/// \brief RDataFrame-based file IO utilities of SpaceCharge / DataContainer3D,
///        split out so libO2TPCSpaceCharge does not depend on ROOTDataFrame.

#include "TPCSpaceCharge/SpaceCharge.h"
#include "TPCSpaceCharge/DataContainer3D.h"
#include "TPCSpaceCharge/RegularGrid3D.h"
#include "TPCSpaceCharge/TriCubic.h"
#include "TPCSpaceCharge/PoissonSolverHelpers.h"
#include "TPCBase/Mapper.h"
#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonConstants/LHCConstants.h"
#include "MathUtils/Utils.h"
#include "Framework/Logger.h"
#include "fmt/core.h"
#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"
#include "ROOT/RDataFrame.hxx"
#include <numeric>
#include <memory>
#include <algorithm>
#include <random>
#include <vector>

using namespace o2::tpc;

template <typename DataT>
auto DataContainer3D<DataT>::getDataSlice(const std::vector<DataT>& data, size_t entries, const size_t values_per_entry, ULong64_t entry)
{
  const long indStart = entry * values_per_entry;
  if (entry < (entries - 1)) {
    return std::pair(indStart, std::vector<float>(data.begin() + indStart, data.begin() + indStart + values_per_entry));
  } else if (entry == (entries - 1)) {
    // last entry might have different number of values. just copy the rest...
    return std::pair(indStart, std::vector<float>(data.begin() + indStart, data.end()));
  }
  return std::pair(indStart, std::vector<float>());
};

template <typename DataT>
int DataContainer3D<DataT>::writeToFile(std::string_view file, std::string_view option, std::string_view name, const int nthreads) const
{
  // max number of floats per Entry
  const size_t maxvalues = sizeof(float) * 1024 * 1024;

  // total number of values to be stored
  const size_t nsize = getNDataPoints();

  // calculate number of entries in the tree and restrict if the number of values per threads exceeds max size
  size_t entries = ((nsize / nthreads) > maxvalues) ? (nsize / maxvalues) : nthreads;

  if (entries > nsize) {
    entries = nsize;
  }

  // calculate numbers to store per entry
  const size_t values_per_entry = nsize / entries;

  // in case of remainder add additonal entry
  const size_t values_lastEntry = nsize % entries;
  if (values_lastEntry) {
    entries += 1;
  }

  // in case EnableImplicitMT was already called with different number of threads, perform reset
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);

  // define dataframe which will be stored in the TTree
  ROOT::RDataFrame dFrame(entries);

  // define function which is used to fill the data frame
  auto dfStore = dFrame.DefineSlotEntry(name, [&data = std::as_const(mData), entries, values_per_entry](unsigned int, ULong64_t entry) { return DataContainer3D<DataT>::getDataSlice(data, entries, values_per_entry, entry); });
  dfStore = dfStore.Define("nz", [mZVertices = mZVertices]() { return mZVertices; });
  dfStore = dfStore.Define("nr", [mRVertices = mRVertices]() { return mRVertices; });
  dfStore = dfStore.Define("nphi", [mPhiVertices = mPhiVertices]() { return mPhiVertices; });

  // define options of TFile
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;
  opt.fOverwriteIfExists = true; // overwrite if already exists

  TStopwatch timer;
  // note: first call has some overhead (~2s)
  dfStore.Snapshot(name, file, {name.data(), "nz", "nr", "nphi"}, opt);
  timer.Print("u");
  return 0;
}

template <typename DataT>
bool DataContainer3D<DataT>::initFromFile(std::string_view file, std::string_view name, const int nthreads)
{
  // in case EnableImplicitMT was already called with different number of threads, perform reset
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);

  // compare first the meta data (is the number of vertices the same)
  // define data frame from imput file
  ROOT::RDataFrame dFrame(name, file);

  // compare vertices
  auto comp = [mZVertices = mZVertices, mRVertices = mRVertices, mPhiVertices = mPhiVertices](const unsigned short nz, const unsigned short nr, const unsigned short nphi) {
    if ((nz == mZVertices) && (nr == mRVertices) && (nphi == mPhiVertices)) {
      return false;
    }
    return true;
  };

  auto count = dFrame.Filter(comp, {"nz", "nr", "nphi"}).Count();
  if (*count != 0) {
    LOGP(error, "Data from input file has different number of vertices! Found {} same vertices", *count);
    return false;
  }

  // define lambda function which is used to copy the data
  auto readData = [&mData = mData](const std::pair<long, std::vector<float>>& data) {
    std::copy(data.second.begin(), data.second.end(), mData.begin() + data.first);
  };

  LOGP(info, "Reading {} from file {}", name, file);

  // fill data from RDataFrame
  TStopwatch timer;
  dFrame.Foreach(readData, {name.data()});
  timer.Print("u");
  return true;
}

template <typename DataT>
void DataContainer3D<DataT>::dumpSlice(std::string_view treename, std::string_view fileIn, std::string_view fileOut, std::string_view option, std::pair<unsigned short, unsigned short> rangeiR, std::pair<unsigned short, unsigned short> rangeiZ, std::pair<unsigned short, unsigned short> rangeiPhi, const int nthreads)
{
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);
  ROOT::RDataFrame dFrame(treename, fileIn);

  auto df = dFrame.Define("slice", [rangeiZ, rangeiR, rangeiPhi](const std::pair<long, std::vector<float>>& values, unsigned short nz, unsigned short nr, unsigned short nphi) {
    std::vector<size_t> ir;
    std::vector<size_t> iphi;
    std::vector<size_t> iz;
    std::vector<float> r;
    std::vector<float> phi;
    std::vector<float> z;
    std::vector<float> vals;
    std::vector<size_t> globalIdx;
    std::vector<LocalPosition3D> lPos;
    const auto nvalues = values.second.size();
    ir.reserve(nvalues);
    iphi.reserve(nvalues);
    iz.reserve(nvalues);
    r.reserve(nvalues);
    phi.reserve(nvalues);
    z.reserve(nvalues);
    vals.reserve(nvalues);
    lPos.reserve(nvalues);
    globalIdx.reserve(nvalues);
    for (size_t i = 0; i < nvalues; ++i) {
      const size_t idx = values.first + i;
      const auto iZTmp = o2::tpc::DataContainer3D<float>::getIndexZ(idx, nz, nr, nphi);
      if ((rangeiZ.first < rangeiZ.second) && ((iZTmp < rangeiZ.first) || (iZTmp > rangeiZ.second))) {
        continue;
      }

      const auto iRTmp = o2::tpc::DataContainer3D<float>::getIndexR(idx, nz, nr, nphi);
      if ((rangeiR.first < rangeiR.second) && ((iRTmp < rangeiR.first) || (iRTmp > rangeiR.second))) {
        continue;
      }

      const auto iPhiTmp = o2::tpc::DataContainer3D<float>::getIndexPhi(idx, nz, nr, nphi);
      if ((rangeiPhi.first < rangeiPhi.second) && ((iPhiTmp < rangeiPhi.first) || (iPhiTmp > rangeiPhi.second))) {
        continue;
      }

      const float rTmp = o2::tpc::GridProperties<float>::getRMin() + o2::tpc::GridProperties<float>::getGridSpacingR(nr) * iRTmp;
      const float zTmp = o2::tpc::GridProperties<float>::getZMin() + o2::tpc::GridProperties<float>::getGridSpacingZ(nz) * iZTmp;
      const float phiTmp = o2::tpc::GridProperties<float>::getPhiMin() + o2::tpc::GridProperties<float>::getGridSpacingPhi(nphi) * (MGParameters::normalizeGridToNSector / double(SECTORSPERSIDE)) * iPhiTmp;

      const float x = rTmp * std::cos(phiTmp);
      const float y = rTmp * std::sin(phiTmp);
      const LocalPosition3D pos(x, y, zTmp);
      unsigned char secNum = std::floor(phiTmp / SECPHIWIDTH);
      Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
      LocalPosition3D lPosTmp = Mapper::GlobalToLocal(pos, sector);

      lPos.emplace_back(lPosTmp);
      ir.emplace_back(iRTmp);
      iphi.emplace_back(iPhiTmp);
      iz.emplace_back(iZTmp);
      r.emplace_back(rTmp);
      phi.emplace_back(phiTmp);
      z.emplace_back(zTmp);
      vals.emplace_back(values.second[i]);
      globalIdx.emplace_back(idx);
    }
    return std::make_tuple(vals, iz, ir, iphi, z, r, phi, lPos, globalIdx);
  },
                          {treename.data(), "nz", "nr", "nphi"});

  // define options of TFile
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;
  df.Snapshot(treename, fileOut, {"slice"}, opt);
}

template <typename DataT>
void DataContainer3D<DataT>::dumpInterpolation(std::string_view treename, std::string_view fileIn, std::string_view fileOut, std::string_view option, std::pair<float, float> rangeR, std::pair<float, float> rangeZ, std::pair<float, float> rangePhi, const int nR, const int nZ, const int nPhi, const int nthreads)
{
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);
  ROOT::RDataFrame dFrame(nPhi);

  // get vertices for input TTree which is needed to define the grid for interpolation
  unsigned short nr, nz, nphi;
  if (!getVertices(treename, fileIn, nr, nz, nphi)) {
    return;
  }

  // load data from input TTree
  DataContainer3D<DataT> data;
  data.setGrid(nz, nr, nphi, true);
  data.initFromFile(fileIn, treename, nthreads);

  // define grid for interpolation
  using GridProp = GridProperties<DataT>;
  const RegularGrid3D<DataT> mGrid3D(GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, GridProp::getGridSpacingZ(nz), GridProp::getGridSpacingR(nr), o2::tpc::GridProperties<float>::getGridSpacingPhi(nphi) * (MGParameters::normalizeGridToNSector / double(SECTORSPERSIDE)), ParamSpaceCharge{nr, nz, nphi});

  auto interpolate = [&mGrid3D = std::as_const(mGrid3D), &data = std::as_const(data), rangeR, rangeZ, rangePhi, nR, nZ, nPhi](unsigned int, ULong64_t iPhi) {
    std::vector<size_t> ir;
    std::vector<size_t> iphi;
    std::vector<size_t> iz;
    std::vector<float> r;
    std::vector<float> phi;
    std::vector<float> z;
    std::vector<float> vals;
    std::vector<size_t> globalIdx;
    std::vector<LocalPosition3D> lPos;
    const auto nvalues = nR * nZ;
    ir.reserve(nvalues);
    iphi.reserve(nvalues);
    iz.reserve(nvalues);
    r.reserve(nvalues);
    phi.reserve(nvalues);
    z.reserve(nvalues);
    vals.reserve(nvalues);
    lPos.reserve(nvalues);
    globalIdx.reserve(nvalues);

    const float rSpacing = (rangeR.second - rangeR.first) / (nR - 1);
    const float zSpacing = (rangeZ.second - rangeZ.first) / (nZ - 1);
    const float phiSpacing = (rangePhi.second - rangePhi.first) / (nPhi - 1);
    const DataT phiPos = rangePhi.first + iPhi * phiSpacing;
    // loop over grid and interpolate values
    for (int iR = 0; iR < nR; ++iR) {
      const DataT rPos = rangeR.first + iR * rSpacing;
      for (int iZ = 0; iZ < nZ; ++iZ) {
        const size_t idx = (iZ + nZ * (iR + iPhi * nR)); // unique index to Build index with other friend TTrees
        const DataT zPos = rangeZ.first + iZ * zSpacing;
        ir.emplace_back(iR);
        iphi.emplace_back(iPhi);
        iz.emplace_back(iZ);
        r.emplace_back(rPos);
        phi.emplace_back(phiPos);
        z.emplace_back(zPos);
        vals.emplace_back(data.interpolate(zPos, rPos, phiPos, mGrid3D)); // interpolated values
        globalIdx.emplace_back(idx);
        const float x = rPos * std::cos(phiPos);
        const float y = rPos * std::sin(phiPos);
        const LocalPosition3D pos(x, y, zPos);
        unsigned char secNum = std::floor(phiPos / SECPHIWIDTH); // TODO CHECK THIS
        Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
        LocalPosition3D lPosTmp = Mapper::GlobalToLocal(pos, sector);
        lPos.emplace_back(lPosTmp);
      }
    }
    return std::make_tuple(vals, iz, ir, iphi, z, r, phi, lPos, globalIdx);
  };

  // define RDataFrame entry
  auto dfStore = dFrame.DefineSlotEntry(treename, interpolate);

  // define options of TFile
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;

  TStopwatch timer;
  // note: first call has some overhead (~2s)
  dfStore.Snapshot(treename, fileOut, {treename.data()}, opt);
  timer.Print("u");
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToTree(const char* outFileName, const Side side, const int nZPoints, const int nRPoints, const int nPhiPoints, const bool randomize) const
{
  const DataT phiSpacing = GridProp::getGridSpacingPhi(nPhiPoints) * (MGParameters::normalizeGridToNSector / double(SECTORSPERSIDE));
  const DataT rSpacing = GridProp::getGridSpacingR(nRPoints);
  const DataT zSpacing = side == Side::A ? GridProp::getGridSpacingZ(nZPoints) : -GridProp::getGridSpacingZ(nZPoints);

  std::uniform_real_distribution<DataT> uniR(-rSpacing / 2, rSpacing / 2);
  std::uniform_real_distribution<DataT> uniPhi(-phiSpacing / 2, phiSpacing / 2);

  std::vector<std::vector<float>> phiPosOut(nPhiPoints);
  std::vector<std::vector<float>> rPosOut(nPhiPoints);
  std::vector<std::vector<float>> zPosOut(nPhiPoints);
  std::vector<std::vector<int>> iPhiOut(nPhiPoints);
  std::vector<std::vector<int>> iROut(nPhiPoints);
  std::vector<std::vector<int>> iZOut(nPhiPoints);
  std::vector<std::vector<float>> densityOut(nPhiPoints);
  std::vector<std::vector<float>> potentialOut(nPhiPoints);
  std::vector<std::vector<float>> eZOut(nPhiPoints);
  std::vector<std::vector<float>> eROut(nPhiPoints);
  std::vector<std::vector<float>> ePhiOut(nPhiPoints);
  std::vector<std::vector<float>> distZOut(nPhiPoints);
  std::vector<std::vector<float>> distROut(nPhiPoints);
  std::vector<std::vector<float>> distRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> corrZOut(nPhiPoints);
  std::vector<std::vector<float>> corrROut(nPhiPoints);
  std::vector<std::vector<float>> corrRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> lcorrZOut(nPhiPoints);
  std::vector<std::vector<float>> lcorrROut(nPhiPoints);
  std::vector<std::vector<float>> lcorrRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> ldistZOut(nPhiPoints);
  std::vector<std::vector<float>> ldistROut(nPhiPoints);
  std::vector<std::vector<float>> ldistRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> xOut(nPhiPoints);
  std::vector<std::vector<float>> yOut(nPhiPoints);
  std::vector<std::vector<float>> bROut(nPhiPoints);
  std::vector<std::vector<float>> bZOut(nPhiPoints);
  std::vector<std::vector<float>> bPhiOut(nPhiPoints);
  std::vector<std::vector<LocalPosition3D>> lPosOut(nPhiPoints);
  std::vector<std::vector<int>> sectorOut(nPhiPoints);
  std::vector<std::vector<size_t>> globalIdxOut(nPhiPoints);
  std::vector<std::vector<bool>> isOnPadPlane(nPhiPoints);

#pragma omp parallel for num_threads(sNThreads)
  for (int iPhi = 0; iPhi < nPhiPoints; ++iPhi) {
    const int nPoints = nZPoints * nRPoints;
    phiPosOut[iPhi].reserve(nPoints);
    rPosOut[iPhi].reserve(nPoints);
    zPosOut[iPhi].reserve(nPoints);
    iPhiOut[iPhi].reserve(nPoints);
    iROut[iPhi].reserve(nPoints);
    iZOut[iPhi].reserve(nPoints);
    densityOut[iPhi].reserve(nPoints);
    potentialOut[iPhi].reserve(nPoints);
    eZOut[iPhi].reserve(nPoints);
    eROut[iPhi].reserve(nPoints);
    ePhiOut[iPhi].reserve(nPoints);
    distZOut[iPhi].reserve(nPoints);
    distROut[iPhi].reserve(nPoints);
    distRPhiOut[iPhi].reserve(nPoints);
    corrZOut[iPhi].reserve(nPoints);
    corrROut[iPhi].reserve(nPoints);
    corrRPhiOut[iPhi].reserve(nPoints);
    lcorrZOut[iPhi].reserve(nPoints);
    lcorrROut[iPhi].reserve(nPoints);
    lcorrRPhiOut[iPhi].reserve(nPoints);
    ldistZOut[iPhi].reserve(nPoints);
    ldistROut[iPhi].reserve(nPoints);
    ldistRPhiOut[iPhi].reserve(nPoints);
    xOut[iPhi].reserve(nPoints);
    yOut[iPhi].reserve(nPoints);
    bROut[iPhi].reserve(nPoints);
    bZOut[iPhi].reserve(nPoints);
    bPhiOut[iPhi].reserve(nPoints);
    lPosOut[iPhi].reserve(nPoints);
    sectorOut[iPhi].reserve(nPoints);
    globalIdxOut[iPhi].reserve(nPoints);
    isOnPadPlane[iPhi].reserve(nPoints);

    std::mt19937 rng(std::random_device{}());
    DataT phiPos = iPhi * phiSpacing;
    for (int iR = 0; iR < nRPoints; ++iR) {
      DataT rPos = getRMin(side) + iR * rSpacing;
      for (int iZ = 0; iZ < nZPoints; ++iZ) {
        DataT zPos = getZMin(side) + iZ * zSpacing;
        if (randomize) {
          phiPos += uniPhi(rng);
          o2::math_utils::detail::bringTo02PiGen<DataT>(phiPos);
          rPos += uniR(rng);
        }

        DataT density = getDensityCyl(zPos, rPos, phiPos, side);
        DataT potential = getPotentialCyl(zPos, rPos, phiPos, side);

        DataT distZ{};
        DataT distR{};
        DataT distRPhi{};
        getDistortionsCyl(zPos, rPos, phiPos, side, distZ, distR, distRPhi);

        DataT ldistZ{};
        DataT ldistR{};
        DataT ldistRPhi{};
        getLocalDistortionsCyl(zPos, rPos, phiPos, side, ldistZ, ldistR, ldistRPhi);

        // get average distortions
        DataT corrZ{};
        DataT corrR{};
        DataT corrRPhi{};
        // getCorrectionsCyl(zPos, rPos, phiPos, side, corrZ, corrR, corrRPhi);

        const DataT zDistorted = zPos + distZ;
        const DataT radiusDistorted = rPos + distR;
        const DataT phiDistorted = regulatePhi(phiPos + distRPhi / rPos, side);
        getCorrectionsCyl(zDistorted, radiusDistorted, phiDistorted, side, corrZ, corrR, corrRPhi);
        corrRPhi *= rPos / radiusDistorted;

        DataT lcorrZ{};
        DataT lcorrR{};
        DataT lcorrRPhi{};
        getLocalCorrectionsCyl(zPos, rPos, phiPos, side, lcorrZ, lcorrR, lcorrRPhi);

        // get average distortions
        DataT eZ{};
        DataT eR{};
        DataT ePhi{};
        getElectricFieldsCyl(zPos, rPos, phiPos, side, eZ, eR, ePhi);

        // global coordinates
        const float x = getXFromPolar(rPos, phiPos);
        const float y = getYFromPolar(rPos, phiPos);

        // b field
        const float bR = mBField.evalFieldR(zPos, rPos, phiPos);
        const float bZ = mBField.evalFieldZ(zPos, rPos, phiPos);
        const float bPhi = mBField.evalFieldPhi(zPos, rPos, phiPos);

        const LocalPosition3D pos(x, y, zPos);
        unsigned char secNum = std::floor(phiPos / SECPHIWIDTH);
        Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
        LocalPosition3D lPos = Mapper::GlobalToLocal(pos, sector);

        phiPosOut[iPhi].emplace_back(phiPos);
        rPosOut[iPhi].emplace_back(rPos);
        zPosOut[iPhi].emplace_back(zPos);
        iPhiOut[iPhi].emplace_back(iPhi);
        iROut[iPhi].emplace_back(iR);
        iZOut[iPhi].emplace_back(iZ);
        if (mDensity[side].getNDataPoints()) {
          densityOut[iPhi].emplace_back(density);
        }
        if (mPotential[side].getNDataPoints()) {
          potentialOut[iPhi].emplace_back(potential);
        }
        if (mElectricFieldEr[side].getNDataPoints()) {
          eZOut[iPhi].emplace_back(eZ);
          eROut[iPhi].emplace_back(eR);
          ePhiOut[iPhi].emplace_back(ePhi);
        }
        if (mGlobalDistdR[side].getNDataPoints()) {
          distZOut[iPhi].emplace_back(distZ);
          distROut[iPhi].emplace_back(distR);
          distRPhiOut[iPhi].emplace_back(distRPhi);
        }
        if (mGlobalCorrdR[side].getNDataPoints()) {
          corrZOut[iPhi].emplace_back(corrZ);
          corrROut[iPhi].emplace_back(corrR);
          corrRPhiOut[iPhi].emplace_back(corrRPhi);
        }
        if (mLocalCorrdR[side].getNDataPoints()) {
          lcorrZOut[iPhi].emplace_back(lcorrZ);
          lcorrROut[iPhi].emplace_back(lcorrR);
          lcorrRPhiOut[iPhi].emplace_back(lcorrRPhi);
        }
        if (mLocalDistdR[side].getNDataPoints()) {
          ldistZOut[iPhi].emplace_back(ldistZ);
          ldistROut[iPhi].emplace_back(ldistR);
          ldistRPhiOut[iPhi].emplace_back(ldistRPhi);
        }
        xOut[iPhi].emplace_back(x);
        yOut[iPhi].emplace_back(y);
        bROut[iPhi].emplace_back(bR);
        bZOut[iPhi].emplace_back(bZ);
        bPhiOut[iPhi].emplace_back(bPhi);
        lPosOut[iPhi].emplace_back(lPos);
        sectorOut[iPhi].emplace_back(sector);
        const size_t idx = (iZ + nZPoints * (iR + iPhi * nRPoints));
        globalIdxOut[iPhi].emplace_back(idx);

        const float xDist = getXFromPolar(radiusDistorted, phiDistorted);
        const float yDist = getYFromPolar(radiusDistorted, phiDistorted);
        GlobalPosition3D posTmp(xDist, yDist, zPos);
        const DigitPos digiPadPos = o2::tpc::Mapper::instance().findDigitPosFromGlobalPosition(posTmp);
        isOnPadPlane[iPhi].emplace_back(digiPadPos.isValid());
      }
    }
  }

  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != sNThreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(sNThreads);
  ROOT::RDataFrame dFrame(nPhiPoints);

  TStopwatch timer;
  auto dfStore = dFrame.DefineSlotEntry("x", [&xOut = xOut](unsigned int, ULong64_t entry) { return xOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("y", [&yOut = yOut](unsigned int, ULong64_t entry) { return yOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("phi", [&phiPosOut = phiPosOut](unsigned int, ULong64_t entry) { return phiPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("r", [&rPosOut = rPosOut](unsigned int, ULong64_t entry) { return rPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("z", [&zPosOut = zPosOut](unsigned int, ULong64_t entry) { return zPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iPhi", [&iPhiOut = iPhiOut](unsigned int, ULong64_t entry) { return iPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iR", [&iROut = iROut](unsigned int, ULong64_t entry) { return iROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iZ", [&iZOut = iZOut](unsigned int, ULong64_t entry) { return iZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lPos", [&lPosOut = lPosOut](unsigned int, ULong64_t entry) { return lPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("sector", [&sectorOut = sectorOut](unsigned int, ULong64_t entry) { return sectorOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("scdensity", [&densityOut = densityOut](unsigned int, ULong64_t entry) { return densityOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("potential", [&potentialOut = potentialOut](unsigned int, ULong64_t entry) { return potentialOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("eZ", [&eZOut = eZOut](unsigned int, ULong64_t entry) { return eZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("eR", [&eROut = eROut](unsigned int, ULong64_t entry) { return eROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ePhi", [&ePhiOut = ePhiOut](unsigned int, ULong64_t entry) { return ePhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("distZ", [&distZOut = distZOut](unsigned int, ULong64_t entry) { return distZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("distR", [&distROut = distROut](unsigned int, ULong64_t entry) { return distROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("distRPhi", [&distRPhiOut = distRPhiOut](unsigned int, ULong64_t entry) { return distRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrZ", [&corrZOut = corrZOut](unsigned int, ULong64_t entry) { return corrZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrR", [&corrROut = corrROut](unsigned int, ULong64_t entry) { return corrROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrRPhi", [&corrRPhiOut = corrRPhiOut](unsigned int, ULong64_t entry) { return corrRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lcorrZ", [&lcorrZOut = lcorrZOut](unsigned int, ULong64_t entry) { return lcorrZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lcorrR", [&lcorrROut = lcorrROut](unsigned int, ULong64_t entry) { return lcorrROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lcorrRPhi", [&lcorrRPhiOut = lcorrRPhiOut](unsigned int, ULong64_t entry) { return lcorrRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ldistZ", [&ldistZOut = ldistZOut](unsigned int, ULong64_t entry) { return ldistZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ldistR", [&ldistROut = ldistROut](unsigned int, ULong64_t entry) { return ldistROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ldistRPhi", [&ldistRPhiOut = ldistRPhiOut](unsigned int, ULong64_t entry) { return ldistRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("bR", [&bROut = bROut](unsigned int, ULong64_t entry) { return bROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("bZ", [&bZOut = bZOut](unsigned int, ULong64_t entry) { return bZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("bPhi", [&bPhiOut = bPhiOut](unsigned int, ULong64_t entry) { return bPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("globalIndex", [&globalIdxOut = globalIdxOut](unsigned int, ULong64_t entry) { return globalIdxOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("isOnPadPlane", [&isOnPadPlane = isOnPadPlane](unsigned int, ULong64_t entry) { return isOnPadPlane[entry]; });
  dfStore.Snapshot("tree", outFileName);
  timer.Print("u");
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToTree(const char* outFileName, const Sector& sector, const int nZPoints) const
{
  const Side side = sector.side();
  const DataT zSpacing = (side == Side::A) ? GridProp::getGridSpacingZ(nZPoints) : -GridProp::getGridSpacingZ(nZPoints);
  const Mapper& mapper = Mapper::instance();

  const int nPads = Mapper::getPadsInSector();
  std::vector<std::vector<float>> phiPosOut(nZPoints);
  std::vector<std::vector<float>> rPosOut(nZPoints);
  std::vector<std::vector<float>> zPosOut(nZPoints);
  std::vector<std::vector<int>> rowOut(nZPoints);
  std::vector<std::vector<float>> lxOut(nZPoints);
  std::vector<std::vector<float>> lyOut(nZPoints);
  std::vector<std::vector<float>> xOut(nZPoints);
  std::vector<std::vector<float>> yOut(nZPoints);
  std::vector<std::vector<float>> corrZOut(nZPoints);
  std::vector<std::vector<float>> corrROut(nZPoints);
  std::vector<std::vector<float>> corrRPhiOut(nZPoints);
  std::vector<std::vector<float>> erOut(nZPoints);
  std::vector<std::vector<float>> ezOut(nZPoints);
  std::vector<std::vector<float>> ephiOut(nZPoints);
  std::vector<std::vector<float>> potentialOut(nZPoints);
  std::vector<std::vector<int>> izOut(nZPoints);
  std::vector<std::vector<size_t>> globalIdxOut(nZPoints);

#pragma omp parallel for num_threads(sNThreads)
  for (int iZ = 0; iZ < nZPoints; ++iZ) {
    phiPosOut[iZ].reserve(nPads);
    rPosOut[iZ].reserve(nPads);
    zPosOut[iZ].reserve(nPads);
    corrZOut[iZ].reserve(nPads);
    corrROut[iZ].reserve(nPads);
    corrRPhiOut[iZ].reserve(nPads);
    rowOut[iZ].reserve(nPads);
    lxOut[iZ].reserve(nPads);
    lyOut[iZ].reserve(nPads);
    xOut[iZ].reserve(nPads);
    yOut[iZ].reserve(nPads);
    erOut[iZ].reserve(nPads);
    ezOut[iZ].reserve(nPads);
    ephiOut[iZ].reserve(nPads);
    izOut[iZ].reserve(nPads);
    potentialOut[iZ].reserve(nPads);
    globalIdxOut[iZ].reserve(nPads);

    DataT zPos = getZMin(side) + iZ * zSpacing;
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          GlobalPadNumber globalpad = Mapper::getGlobalPadNumber(irow, ipad, region);
          const PadCentre& padcentre = mapper.padCentre(globalpad);
          auto lx = padcentre.X();
          auto ly = padcentre.Y();
          // local to global
          auto globalPos = Mapper::LocalToGlobal(padcentre, sector);
          auto x = globalPos.X();
          auto y = globalPos.Y();

          auto r = getRadiusFromCartesian(x, y);
          auto phi = getPhiFromCartesian(x, y);
          DataT corrZ{};
          DataT corrR{};
          DataT corrRPhi{};
          getCorrectionsCyl(zPos, r, phi, side, corrZ, corrR, corrRPhi);

          DataT eZ{};
          DataT eR{};
          DataT ePhi{};
          getElectricFieldsCyl(zPos, r, phi, side, eZ, eR, ePhi);

          potentialOut[iZ].emplace_back(getPotentialCyl(zPos, r, phi, side));
          erOut[iZ].emplace_back(eR);
          ezOut[iZ].emplace_back(eZ);
          ephiOut[iZ].emplace_back(ePhi);
          phiPosOut[iZ].emplace_back(phi);
          rPosOut[iZ].emplace_back(r);
          zPosOut[iZ].emplace_back(zPos);
          corrZOut[iZ].emplace_back(corrZ);
          corrROut[iZ].emplace_back(corrR);
          corrRPhiOut[iZ].emplace_back(corrRPhi);
          rowOut[iZ].emplace_back(irow + Mapper::ROWOFFSET[region]);
          lxOut[iZ].emplace_back(lx);
          lyOut[iZ].emplace_back(ly);
          xOut[iZ].emplace_back(x);
          yOut[iZ].emplace_back(y);
          izOut[iZ].emplace_back(iZ);
          const size_t idx = globalpad + Mapper::getPadsInSector() * iZ;
          globalIdxOut[iZ].emplace_back(idx);
        }
      }
    }
  }

  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != sNThreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(sNThreads);
  ROOT::RDataFrame dFrame(nZPoints);

  TStopwatch timer;
  auto dfStore = dFrame.DefineSlotEntry("phi", [&phiPosOut = phiPosOut](unsigned int, ULong64_t entry) { return phiPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("r", [&rPosOut = rPosOut](unsigned int, ULong64_t entry) { return rPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("z", [&zPosOut = zPosOut](unsigned int, ULong64_t entry) { return zPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iz", [&izOut = izOut](unsigned int, ULong64_t entry) { return izOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrZ", [&corrZOut = corrZOut](unsigned int, ULong64_t entry) { return corrZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrR", [&corrROut = corrROut](unsigned int, ULong64_t entry) { return corrROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrRPhi", [&corrRPhiOut = corrRPhiOut](unsigned int, ULong64_t entry) { return corrRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("row", [&rowOut = rowOut](unsigned int, ULong64_t entry) { return rowOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lx", [&lxOut = lxOut](unsigned int, ULong64_t entry) { return lxOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ly", [&lyOut = lyOut](unsigned int, ULong64_t entry) { return lyOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("x", [&xOut = xOut](unsigned int, ULong64_t entry) { return xOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("y", [&yOut = yOut](unsigned int, ULong64_t entry) { return yOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("er", [&erOut = erOut](unsigned int, ULong64_t entry) { return erOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ez", [&ezOut = ezOut](unsigned int, ULong64_t entry) { return ezOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ephi", [&ephiOut = ephiOut](unsigned int, ULong64_t entry) { return ephiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("potential", [&potentialOut = potentialOut](unsigned int, ULong64_t entry) { return potentialOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("globalIndex", [&globalIdxOut = globalIdxOut](unsigned int, ULong64_t entry) { return globalIdxOut[entry]; });
  dfStore.Snapshot("tree", outFileName);
  timer.Print("u");
}

template <typename DataT>
int SpaceCharge<DataT>::dumpElectricFields(std::string_view file, const Side side, std::string_view option) const
{
  if (!mElectricFieldEr[side].getNDataPoints()) {
    LOGP(info, "============== E-Fields are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mElectricFieldEr[side].writeToFile(file, option, fmt::format("fieldEr_side{}", sideName), sNThreads);
  const int ez = mElectricFieldEz[side].writeToFile(file, "UPDATE", fmt::format("fieldEz_side{}", sideName), sNThreads);
  const int ephi = mElectricFieldEphi[side].writeToFile(file, "UPDATE", fmt::format("fieldEphi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return er + ez + ephi;
}

template <typename DataT>
void SpaceCharge<DataT>::setElectricFieldsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  std::string_view treeEr{fmt::format("fieldEr_side{}", sideName)};
  if (!checkGridFromFile(file, treeEr)) {
    return;
  }
  initContainer(mElectricFieldEr[side], true);
  initContainer(mElectricFieldEz[side], true);
  initContainer(mElectricFieldEphi[side], true);
  mElectricFieldEr[side].initFromFile(file, treeEr, sNThreads);
  mElectricFieldEz[side].initFromFile(file, fmt::format("fieldEz_side{}", sideName), sNThreads);
  mElectricFieldEphi[side].initFromFile(file, fmt::format("fieldEphi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpGlobalDistortions(std::string_view file, const Side side, std::string_view option) const
{
  if (!mGlobalDistdR[side].getNDataPoints()) {
    LOGP(info, "============== global distortions are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mGlobalDistdR[side].writeToFile(file, option, fmt::format("distR_side{}", sideName), sNThreads);
  const int ez = mGlobalDistdZ[side].writeToFile(file, "UPDATE", fmt::format("distZ_side{}", sideName), sNThreads);
  const int ephi = mGlobalDistdRPhi[side].writeToFile(file, "UPDATE", fmt::format("distRphi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return er + ez + ephi;
}

template <typename DataT>
void SpaceCharge<DataT>::setGlobalDistortionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  std::string_view tree{fmt::format("distR_side{}", sideName)};
  if (!checkGridFromFile(file, tree)) {
    return;
  }
  initContainer(mGlobalDistdR[side], true);
  initContainer(mGlobalDistdZ[side], true);
  initContainer(mGlobalDistdRPhi[side], true);
  mGlobalDistdR[side].initFromFile(file, tree, sNThreads);
  mGlobalDistdZ[side].initFromFile(file, fmt::format("distZ_side{}", sideName), sNThreads);
  mGlobalDistdRPhi[side].initFromFile(file, fmt::format("distRphi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpGlobalCorrections(std::string_view file, const Side side, std::string_view option) const
{
  if (!mGlobalCorrdR[side].getNDataPoints()) {
    LOGP(info, "============== global corrections are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mGlobalCorrdR[side].writeToFile(file, option, fmt::format("corrR_side{}", sideName), sNThreads);
  const int ez = mGlobalCorrdZ[side].writeToFile(file, "UPDATE", fmt::format("corrZ_side{}", sideName), sNThreads);
  const int ephi = mGlobalCorrdRPhi[side].writeToFile(file, "UPDATE", fmt::format("corrRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return er + ez + ephi;
}

template <typename DataT>
void SpaceCharge<DataT>::setGlobalCorrectionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("corrR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }

  initContainer(mGlobalCorrdR[side], true);
  initContainer(mGlobalCorrdZ[side], true);
  initContainer(mGlobalCorrdRPhi[side], true);
  mGlobalCorrdR[side].initFromFile(file, treename, sNThreads);
  mGlobalCorrdZ[side].initFromFile(file, fmt::format("corrZ_side{}", sideName), sNThreads);
  mGlobalCorrdRPhi[side].initFromFile(file, fmt::format("corrRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpLocalCorrections(std::string_view file, const Side side, std::string_view option) const
{
  if (!mLocalCorrdR[side].getNDataPoints()) {
    LOGP(info, "============== local corrections are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lCorrdR = mLocalCorrdR[side].writeToFile(file, option, fmt::format("lcorrR_side{}", sideName), sNThreads);
  const int lCorrdZ = mLocalCorrdZ[side].writeToFile(file, "UPDATE", fmt::format("lcorrZ_side{}", sideName), sNThreads);
  const int lCorrdRPhi = mLocalCorrdRPhi[side].writeToFile(file, "UPDATE", fmt::format("lcorrRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return lCorrdR + lCorrdZ + lCorrdRPhi;
}

template <typename DataT>
void SpaceCharge<DataT>::setLocalCorrectionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("lcorrR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mLocalCorrdR[side], true);
  initContainer(mLocalCorrdZ[side], true);
  initContainer(mLocalCorrdRPhi[side], true);
  const bool lCorrdR = mLocalCorrdR[side].initFromFile(file, treename, sNThreads);
  const bool lCorrdZ = mLocalCorrdZ[side].initFromFile(file, fmt::format("lcorrZ_side{}", sideName), sNThreads);
  const bool lCorrdRPhi = mLocalCorrdRPhi[side].initFromFile(file, fmt::format("lcorrRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpLocalDistortions(std::string_view file, const Side side, std::string_view option) const
{
  if (!mLocalDistdR[side].getNDataPoints()) {
    LOGP(info, "============== local distortions are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lDistdR = mLocalDistdR[side].writeToFile(file, option, fmt::format("ldistR_side{}", sideName), sNThreads);
  const int lDistdZ = mLocalDistdZ[side].writeToFile(file, "UPDATE", fmt::format("ldistZ_side{}", sideName), sNThreads);
  const int lDistdRPhi = mLocalDistdRPhi[side].writeToFile(file, "UPDATE", fmt::format("ldistRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return lDistdR + lDistdZ + lDistdRPhi;
}

template <typename DataT>
int SpaceCharge<DataT>::dumpLocalDistCorrVectors(std::string_view file, const Side side, std::string_view option) const
{
  if (!mLocalVecDistdR[side].getNDataPoints()) {
    LOGP(info, "============== local distortion vectors are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lVecDistdR = mLocalVecDistdR[side].writeToFile(file, option, fmt::format("lvecdistR_side{}", sideName), sNThreads);
  const int lVecDistdZ = mLocalVecDistdZ[side].writeToFile(file, "UPDATE", fmt::format("lvecdistZ_side{}", sideName), sNThreads);
  const int lVecDistdRPhi = mLocalVecDistdRPhi[side].writeToFile(file, "UPDATE", fmt::format("lvecdistRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return lVecDistdR + lVecDistdZ + lVecDistdRPhi;
}

template <typename DataT>
void SpaceCharge<DataT>::setLocalDistortionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("ldistR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mLocalDistdR[side], true);
  initContainer(mLocalDistdZ[side], true);
  initContainer(mLocalDistdRPhi[side], true);
  const bool lDistdR = mLocalDistdR[side].initFromFile(file, treename, sNThreads);
  const bool lDistdZ = mLocalDistdZ[side].initFromFile(file, fmt::format("ldistZ_side{}", sideName), sNThreads);
  const bool lDistdRPhi = mLocalDistdRPhi[side].initFromFile(file, fmt::format("ldistRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
void SpaceCharge<DataT>::setLocalDistCorrVectorsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("lvecdistR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mLocalVecDistdR[side], true);
  initContainer(mLocalVecDistdZ[side], true);
  initContainer(mLocalVecDistdRPhi[side], true);
  const bool lVecDistdR = mLocalVecDistdR[side].initFromFile(file, treename, sNThreads);
  const bool lVecDistdZ = mLocalVecDistdZ[side].initFromFile(file, fmt::format("lvecdistZ_side{}", sideName), sNThreads);
  const bool lVecDistdRPhi = mLocalVecDistdRPhi[side].initFromFile(file, fmt::format("lvecdistRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpPotential(std::string_view file, const Side side, std::string_view option) const
{
  if (!mPotential[side].getNDataPoints()) {
    LOGP(info, "============== potential not set! returning ==============");
    return 0;
  }
  int status = mPotential[side].writeToFile(file, option, fmt::format("potential_side{}", getSideName(side)), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return status;
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialFromFile(std::string_view file, const Side side)
{
  const std::string_view treename{fmt::format("potential_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mPotential[side], true);
  mPotential[side].initFromFile(file, treename, sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpDensity(std::string_view file, const Side side, std::string_view option) const
{
  if (!mDensity[side].getNDataPoints()) {
    LOGP(info, "============== space charge density are not set! returning ==============");
    return 0;
  }
  int status = mDensity[side].writeToFile(file, option, fmt::format("density_side{}", getSideName(side)), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return status;
}

template <typename DataT>
void SpaceCharge<DataT>::setDensityFromFile(std::string_view file, const Side side)
{
  const std::string_view treename{fmt::format("density_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mDensity[side], true);
  mDensity[side].initFromFile(file, treename, sNThreads);
  readMetaData(file);
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToFile(std::string_view file, const Side side, std::string_view option) const
{
  if (option == "RECREATE") {
    // delete the file
    gSystem->Unlink(file.data());
  }
  dumpElectricFields(file, side, "UPDATE");
  dumpPotential(file, side, "UPDATE");
  dumpDensity(file, side, "UPDATE");
  dumpGlobalDistortions(file, side, "UPDATE");
  dumpGlobalCorrections(file, side, "UPDATE");
  dumpLocalCorrections(file, side, "UPDATE");
  dumpLocalDistortions(file, side, "UPDATE");
  dumpLocalDistCorrVectors(file, side, "UPDATE");
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToFile(std::string_view file) const
{
  dumpToFile(file, Side::A, "RECREATE");
  dumpToFile(file, Side::C, "UPDATE");
}

template <typename DataT>
void SpaceCharge<DataT>::dumpMetaData(std::string_view file, std::string_view option, const bool overwriteExisting) const
{
  TFile f(file.data(), option.data());
  if (!overwriteExisting && f.GetListOfKeys()->Contains("meta")) {
    return;
  }
  f.Close();

  // create meta objects
  std::vector<float> params{static_cast<float>(mC0), static_cast<float>(mC1), static_cast<float>(mC2)};
  auto helperA = mGrid3D[Side::A].getHelper();
  auto helperC = mGrid3D[Side::C].getHelper();

  // define dataframe
  ROOT::RDataFrame dFrame(1);
  auto dfStore = dFrame.DefineSlotEntry("paramsC", [&params = params](unsigned int, ULong64_t entry) { return params; });
  dfStore = dfStore.DefineSlotEntry("grid_A", [&helperA = helperA](unsigned int, ULong64_t entry) { return helperA; });
  dfStore = dfStore.DefineSlotEntry("grid_C", [&helperC = helperC](unsigned int, ULong64_t entry) { return helperC; });
  dfStore = dfStore.DefineSlotEntry("BField", [field = mBField.getBField()](unsigned int, ULong64_t entry) { return field; });
  dfStore = dfStore.DefineSlotEntry("metaInf", [meta = mMeta](unsigned int, ULong64_t entry) { return meta; });

  // write to TTree
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;
  opt.fOverwriteIfExists = true; // overwrite if already exists
  dfStore.Snapshot("meta", file, {"paramsC", "grid_A", "grid_C", "BField", "metaInf"}, opt);
}

template <typename DataT>
void SpaceCharge<DataT>::readMetaData(std::string_view file)
{
  if (mReadMetaData) {
    return;
  }

  // check if TTree exists
  TFile f(file.data(), "READ");
  if (!f.GetListOfKeys()->Contains("meta")) {
    return;
  }
  f.Close();

  auto readMeta = [&mC0 = mC0, &mC1 = mC1, &mC2 = mC2, &mGrid3D = mGrid3D, &mBField = mBField](const std::vector<float>& paramsC, const RegularGridHelper<double>& gridA, const RegularGridHelper<double>& gridC, int field) {
    mC0 = paramsC[0];
    mC1 = paramsC[1];
    mC2 = paramsC[2];
    mGrid3D[Side::A] = RegularGrid3D<DataT>(gridA.zmin, gridA.rmin, gridA.phimin, gridA.spacingZ, gridA.spacingR, gridA.spacingPhi, gridA.params);
    mGrid3D[Side::C] = RegularGrid3D<DataT>(gridC.zmin, gridC.rmin, gridC.phimin, gridC.spacingZ, gridC.spacingR, gridC.spacingPhi, gridC.params);
    mBField.setBField(field);
  };

  ROOT::RDataFrame dFrame("meta", file);
  dFrame.Foreach(readMeta, {"paramsC", "grid_A", "grid_C", "BField"});

  const auto& cols = dFrame.GetColumnNames();
  if (std::find(cols.begin(), cols.end(), "metaInf") != cols.end()) {
    auto readMetaInf = [&mMeta = mMeta](const SCMetaData& meta) {
      mMeta = meta;
    };
    dFrame.Foreach(readMetaInf, {"metaInf"});
  }

  LOGP(info, "Setting meta data: mC0={}  mC1={}  mC2={}", mC0, mC1, mC2);
  mReadMetaData = true;
}

template <typename DataT>
void SpaceCharge<DataT>::setFromFile(std::string_view file, const Side side)
{
  setDensityFromFile(file, side);
  setPotentialFromFile(file, side);
  setElectricFieldsFromFile(file, side);
  setLocalDistortionsFromFile(file, side);
  setLocalCorrectionsFromFile(file, side);
  setGlobalDistortionsFromFile(file, side);
  setGlobalCorrectionsFromFile(file, side);
  setLocalDistCorrVectorsFromFile(file, side);
}

template <typename DataT>
void SpaceCharge<DataT>::setFromFile(std::string_view file)
{
  setFromFile(file, Side::A);
  setFromFile(file, Side::C);
}


// explicit template instantiations of the moved members
template int o2::tpc::DataContainer3D<float>::writeToFile(std::string_view, std::string_view, std::string_view, const int) const;
template bool o2::tpc::DataContainer3D<float>::initFromFile(std::string_view, std::string_view, const int);
template void o2::tpc::DataContainer3D<float>::dumpSlice(std::string_view, std::string_view, std::string_view, std::string_view, std::pair<unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>, const int);
template void o2::tpc::DataContainer3D<float>::dumpInterpolation(std::string_view, std::string_view, std::string_view, std::string_view, std::pair<float, float>, std::pair<float, float>, std::pair<float, float>, const int, const int, const int, const int);
template void o2::tpc::SpaceCharge<float>::dumpToTree(const char*, const o2::tpc::Side, const int, const int, const int, const bool) const;
template void o2::tpc::SpaceCharge<float>::dumpToTree(const char*, const o2::tpc::Sector&, const int) const;
template int o2::tpc::SpaceCharge<float>::dumpElectricFields(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::setElectricFieldsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<float>::dumpGlobalDistortions(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::setGlobalDistortionsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<float>::dumpGlobalCorrections(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::setGlobalCorrectionsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<float>::dumpLocalCorrections(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::setLocalCorrectionsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<float>::dumpLocalDistortions(std::string_view, const o2::tpc::Side, std::string_view) const;
template int o2::tpc::SpaceCharge<float>::dumpLocalDistCorrVectors(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::setLocalDistortionsFromFile(std::string_view, const o2::tpc::Side);
template void o2::tpc::SpaceCharge<float>::setLocalDistCorrVectorsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<float>::dumpPotential(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::setPotentialFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<float>::dumpDensity(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::setDensityFromFile(std::string_view, const o2::tpc::Side);
template void o2::tpc::SpaceCharge<float>::dumpToFile(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<float>::dumpToFile(std::string_view) const;
template void o2::tpc::SpaceCharge<float>::dumpMetaData(std::string_view, std::string_view, const bool) const;
template void o2::tpc::SpaceCharge<float>::readMetaData(std::string_view);
template void o2::tpc::SpaceCharge<float>::setFromFile(std::string_view, const o2::tpc::Side);
template void o2::tpc::SpaceCharge<float>::setFromFile(std::string_view);
template int o2::tpc::DataContainer3D<double>::writeToFile(std::string_view, std::string_view, std::string_view, const int) const;
template bool o2::tpc::DataContainer3D<double>::initFromFile(std::string_view, std::string_view, const int);
template void o2::tpc::DataContainer3D<double>::dumpSlice(std::string_view, std::string_view, std::string_view, std::string_view, std::pair<unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>, const int);
template void o2::tpc::DataContainer3D<double>::dumpInterpolation(std::string_view, std::string_view, std::string_view, std::string_view, std::pair<float, float>, std::pair<float, float>, std::pair<float, float>, const int, const int, const int, const int);
template void o2::tpc::SpaceCharge<double>::dumpToTree(const char*, const o2::tpc::Side, const int, const int, const int, const bool) const;
template void o2::tpc::SpaceCharge<double>::dumpToTree(const char*, const o2::tpc::Sector&, const int) const;
template int o2::tpc::SpaceCharge<double>::dumpElectricFields(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::setElectricFieldsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<double>::dumpGlobalDistortions(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::setGlobalDistortionsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<double>::dumpGlobalCorrections(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::setGlobalCorrectionsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<double>::dumpLocalCorrections(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::setLocalCorrectionsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<double>::dumpLocalDistortions(std::string_view, const o2::tpc::Side, std::string_view) const;
template int o2::tpc::SpaceCharge<double>::dumpLocalDistCorrVectors(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::setLocalDistortionsFromFile(std::string_view, const o2::tpc::Side);
template void o2::tpc::SpaceCharge<double>::setLocalDistCorrVectorsFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<double>::dumpPotential(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::setPotentialFromFile(std::string_view, const o2::tpc::Side);
template int o2::tpc::SpaceCharge<double>::dumpDensity(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::setDensityFromFile(std::string_view, const o2::tpc::Side);
template void o2::tpc::SpaceCharge<double>::dumpToFile(std::string_view, const o2::tpc::Side, std::string_view) const;
template void o2::tpc::SpaceCharge<double>::dumpToFile(std::string_view) const;
template void o2::tpc::SpaceCharge<double>::dumpMetaData(std::string_view, std::string_view, const bool) const;
template void o2::tpc::SpaceCharge<double>::readMetaData(std::string_view);
template void o2::tpc::SpaceCharge<double>::setFromFile(std::string_view, const o2::tpc::Side);
template void o2::tpc::SpaceCharge<double>::setFromFile(std::string_view);
