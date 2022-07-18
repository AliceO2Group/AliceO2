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

/// \file TrackResiduals.cxx
/// \brief Implementation of the TrackResiduals class
///
/// \author Ole Schmidt, ole.schmidt@cern.ch
///
///

#include "SpacePoints/TrackResiduals.h"
#include "CommonConstants/MathConstants.h"
#include "ReconstructionDataFormats/Track.h"
#include "MathUtils/fit.h"

#include "TMatrixDSym.h"
#include "TDecompChol.h"
#include "TVectorD.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// for debugging
#include "TStopwatch.h"
#include "TSystem.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <iomanip>

#include <fairlogger/Logger.h>

using namespace o2::tpc;

///////////////////////////////////////////////////////////////////////////////
///
/// initialization + binning
///
///////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void TrackResiduals::init(bool doBinning, float bz)
{
  if (doBinning) {
    // initialize binning
    initBinning();
  }

  mSmoothPol2[VoxX] = true;
  mSmoothPol2[VoxF] = true;
  setKernelType();
  mBz = bz;
  mIsInitialized = true;
  LOGF(info, "Initialization complete for B=%.2f", mBz);
}

//______________________________________________________________________________
void TrackResiduals::setY2XBinning(const std::vector<float>& binning)
{
  if (mIsInitialized) {
    LOG(error) << "Binning already initialized, not changing y/x binning";
    return;
  }
  int nBins = binning.size() - 1;
  if (fabsf(binning[0] + 1.f) > param::sEps || fabsf(binning[nBins] - 1.f) > param::sEps) {
    LOG(error) << "Provided binning for y/x not in range -1 to 1: " << binning[0] << " - " << binning[nBins] << ". Not changing y/x binning";
    return;
  }
  setNY2XBins(nBins);
  mUniformBins[VoxF] = false;
  mY2XBinsDH.clear();
  mY2XBinsDI.clear();
  mY2XBinsCenter.clear();
  for (int iBin = 0; iBin < nBins; ++iBin) {
    mY2XBinsDH.push_back(.5f * (binning[iBin + 1] - binning[iBin]));
    mY2XBinsDI.push_back(.5f / mY2XBinsDH[iBin]);
    mY2XBinsCenter.push_back(binning[iBin] + mY2XBinsDH[iBin]);
  }
}

//______________________________________________________________________________
void TrackResiduals::setZ2XBinning(const std::vector<float>& binning)
{
  if (mIsInitialized) {
    LOG(error) << "Binning already initialized, not changing z/x binning";
    return;
  }
  int nBins = binning.size() - 1;
  if (fabsf(binning[0]) > param::sEps || fabsf(binning[nBins] - 1.f) > param::sEps) {
    LOG(error) << "Provided binning for z/x not in range 0 to 1: " << binning[0] << " - " << binning[nBins] << ". Not changing z/x binning";
    return;
  }
  setNZ2XBins(nBins);
  mUniformBins[VoxZ] = false;
  mZ2XBinsDH.clear();
  mZ2XBinsDI.clear();
  mZ2XBinsCenter.clear();
  for (int iBin = 0; iBin < nBins; ++iBin) {
    mZ2XBinsDH.push_back(.5f * (binning[iBin + 1] - binning[iBin]) * mMaxZ2X);
    mZ2XBinsDI.push_back(.5f / mZ2XBinsDH[iBin]);
    mZ2XBinsCenter.push_back(binning[iBin] * mMaxZ2X + mZ2XBinsDH[iBin]);
  }
}

//______________________________________________________________________________
void TrackResiduals::initBinning()
{
  // initialize binning structures
  //
  // X binning
  if (mNXBins > 0 && mNXBins < param::NPadRows) {
    // uniform binning in X
    LOGF(info, "X-binning is uniform with %i bins from %.2f to %.2f", mNXBins, param::MinX, param::MaxX);
    mDXI = mNXBins / (param::MaxX - param::MinX);
    mDX = 1.0f / mDXI;
    mUniformBins[VoxX] = true;
  } else {
    // binning per pad row
    LOGF(info, "X-binning is per pad-row");
    mNXBins = param::NPadRows;
    mUniformBins[VoxX] = false;
    mDX = param::RowDX[0];
    mDXI = 1.f / mDX; // should not be used
  }
  //
  // Y/X binning
  mMaxY2X.resize(mNXBins);
  mDY2XI.resize(mNXBins);
  mDY2X.resize(mNXBins);
  //
  for (int ix = 0; ix < mNXBins; ++ix) {
    float x = getX(ix);
    mMaxY2X[ix] = tan(.5f * SECPHIWIDTH) - sDeadZone / x;
    mDY2XI[ix] = mNY2XBins / (2.f * mMaxY2X[ix]);
    mDY2X[ix] = 1.f / mDY2XI[ix];
  }
  if (mUniformBins[VoxF]) {
    LOGF(info, "Y/X-binning is uniform with %i bins from -MaxY2X to +MaxY2X (values depend on X-bin)", mNY2XBins);
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      mY2XBinsDH.push_back(1.f / mNY2XBins);
      mY2XBinsDI.push_back(.5f / mY2XBinsDH[ip]);
      mY2XBinsCenter.push_back(-1.f + (ip + 0.5f) * 2.f * mY2XBinsDH[ip]);
      LOGF(info, "Bin %i: center (%.3f), half bin width (%.3f)", ip, mY2XBinsCenter.back(), mY2XBinsDH.back());
    }
  }
  //
  // Z/X binning
  mDZ2XI = mNZ2XBins / sMaxZ2X;
  mDZ2X = 1.0f / mDZ2XI; // for uniform case only
  if (mUniformBins[VoxZ]) {
    LOGF(info, "Z/X-binning is uniform with %i bins from 0 to %f", mNZ2XBins, sMaxZ2X);
    for (int iz = 0; iz < mNZ2XBins; ++iz) {
      mZ2XBinsDH.push_back(.5f * mDZ2X);
      mZ2XBinsDI.push_back(mDZ2XI);
      mZ2XBinsCenter.push_back((iz + 0.5f) * mDZ2X);
      LOGF(info, "Bin %i: center (%.3f), half bin width (%.3f)", iz, mZ2XBinsCenter.back(), mZ2XBinsDH.back());
    }
  }
  //
  mNVoxPerSector = mNY2XBins * mNZ2XBins * mNXBins;
  LOGF(info, "Each TPC sector is divided into %i voxels", mNVoxPerSector);
}

//______________________________________________________________________________
void TrackResiduals::initResultsContainer(int iSec)
{
  mVoxelResults[iSec].resize(mNVoxPerSector);
  for (int ix = 0; ix < mNXBins; ++ix) {
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZ2XBins; ++iz) {
        int binGlb = getGlbVoxBin(ix, ip, iz);
        VoxRes& resVox = mVoxelResults[iSec][binGlb];
        resVox.bvox[VoxX] = ix;
        resVox.bvox[VoxF] = ip;
        resVox.bvox[VoxZ] = iz;
        resVox.bsec = iSec;
      }
    }
  }
  LOG(debug) << "initialized the container for the main results";
}

//______________________________________________________________________________
void TrackResiduals::reset()
{
  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    mXBinsIgnore[iSec].reset();
    std::fill(mVoxelResults[iSec].begin(), mVoxelResults[iSec].end(), VoxRes());
    std::fill(mValidFracXBins[iSec].begin(), mValidFracXBins[iSec].end(), 0);
  }
}

//______________________________________________________________________________
void TrackResiduals::resetUnbinnedResiduals()
{
  mUnbinnedResiduals.clear();
}

//______________________________________________________________________________
int TrackResiduals::getRowID(float x) const
{
  int ix;
  if (x < param::RowX[param::NRowsAccumulated[0] - 1] + param::RowDX[0]) {
    // we are in the IROC
    ix = (x - (param::RowX[0] - .5f * param::RowDX[0])) / param::RowDX[0];
    if (ix < 0) {
      // x is smaller than the inner radius of the first pad row
      ix = -1;
    }
  } else if (x >= param::RowX[param::NRowsAccumulated[param::NROCTypes - 2]] - .5f * param::RowDX[param::NROCTypes - 1]) {
    // we are in the OROC3
    ix = (x - (param::RowX[param::NRowsAccumulated[param::NROCTypes - 2]] - .5f * param::RowDX[param::NROCTypes - 1])) / param::RowDX[param::NROCTypes - 1] + param::NRowsAccumulated[param::NROCTypes - 2];
    if (ix >= param::NPadRows) {
      // x is larger than the outer radius of the last OROC pad row
      ix = -1;
    }
  } else if (x > param::RowX[param::NRowsAccumulated[0]] - .5f * param::RowDX[1] && x < param::RowX[param::NRowsAccumulated[1] - 1] + .5f * param::RowDX[1]) {
    // we are in the OROC1
    ix = (x - (param::RowX[param::NRowsAccumulated[0]] - .5f * param::RowDX[1])) / param::RowDX[1] + param::NRowsAccumulated[0];
  } else if (x > param::RowX[param::NRowsAccumulated[1]] - .5f * param::RowDX[2] && x < param::RowX[param::NRowsAccumulated[2] - 1] + .5f * param::RowDX[2]) {
    // we are in the OROC2
    ix = (x - (param::RowX[param::NRowsAccumulated[1]] - .5f * param::RowDX[2])) / param::RowDX[2] + param::NRowsAccumulated[1];
  } else {
    // x is in one of the gaps between the ROCs
    ix = -1;
  }
  return ix;
}

bool TrackResiduals::findVoxelBin(int secID, float x, float y, float z, std::array<unsigned char, VoxDim>& bvox) const
{
  // Z/X bin
  if (fabs(z / x) > sMaxZ2X) {
    return false;
  }
  int bz = getZ2XBinExact(secID < SECTORSPERSIDE ? z / x : -z / x);
  if (bz < 0) {
    return false;
  }
  // X bin
  int bx = getXBinExact(x);
  if (bx < 0 || bx >= mNXBins) {
    return false;
  }
  // Y/X bin
  int bp = getY2XBinExact(y / x, bx);
  if (bp < 0 || bp >= mNY2XBins) {
    return false;
  }

  bvox[VoxZ] = bz;
  bvox[VoxX] = bx;
  bvox[VoxF] = bp;
  return true;
}

void TrackResiduals::setKernelType(KernelType kernel, float bwX, float bwP, float bwZ, float scX, float scP, float scZ)
{
  // set kernel type and widths in terms of binning in x, y/x, z/x and define aux variables
  mKernelType = kernel;

  mKernelScaleEdge[VoxX] = scX;
  mKernelScaleEdge[VoxF] = scP;
  mKernelScaleEdge[VoxZ] = scZ;

  mKernelWInv[VoxX] = (bwX > 0) ? 1. / bwX : 1.;
  mKernelWInv[VoxF] = (bwP > 0) ? 1. / bwP : 1.;
  mKernelWInv[VoxZ] = (bwZ > 0) ? 1. / bwZ : 1.;

  if (mKernelType == KernelType::Epanechnikov) {
    // bandwidth 1
    mStepKern[VoxX] = static_cast<int>(nearbyint(bwX + 0.5));
    mStepKern[VoxF] = static_cast<int>(nearbyint(bwP + 0.5));
    mStepKern[VoxZ] = static_cast<int>(nearbyint(bwZ + 0.5));
  } else if (mKernelType == KernelType::Gaussian) {
    // look in ~5 sigma
    mStepKern[VoxX] = static_cast<int>(nearbyint(bwX * 5. + 0.5));
    mStepKern[VoxF] = static_cast<int>(nearbyint(bwP * 5. + 0.5));
    mStepKern[VoxZ] = static_cast<int>(nearbyint(bwZ * 5. + 0.5));
  } else {
    LOG(error) << "given kernel type is not defined";
  }
  for (int i = VoxDim; i--;) {
    if (mStepKern[i] < 1) {
      mStepKern[i] = 1;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
///
/// processing functions
///
///////////////////////////////////////////////////////////////////////////////

void TrackResiduals::prepareLocalResidualTrees()
{
  // prepare tree structure
  mTmpFile = std::make_unique<TFile>(Form("%s.root", mLocalResFileName.c_str()), "recreate");
  mTmpTree = std::make_unique<TTree>(mLocalResTreeName.c_str(), "TPC unbinned residuals");
  mTmpTree->Branch(mLocalResBranchName.c_str(), &mUnbinnedResidualsPtr); // in the writer later on this should be one branch per sector?
}

void TrackResiduals::writeLocalResidualTreesToFile()
{
  // write trees with unbinned local residuals to file
  mTmpFile->cd();
  mTmpTree->Fill();
  mTmpTree->Write();
  mTmpTree.reset();
  mTmpFile->Close();
  mTmpFile.reset();
}

void TrackResiduals::loadInputFromFile()
{
  // open input file and access track data
  mFileIn = std::make_unique<TFile>(mInputFileNameResiduals.data(), "open");
  if (!mFileIn) {
    LOG(error) << "input file could not be opened";
    return;
  }
  mTreeInTracks = static_cast<TTree*>(mFileIn->Get("tracks"));
  if (!mTreeInTracks) {
    LOG(error) << "tree with track information not available in input file";
    return;
  }
  mTreeInTracks->SetBranchAddress("tracks", &mTrackDataPtr);
  mTreeInTracks->GetEntry(0);
  // and access also cluster residuals
  mTreeInClRes = static_cast<TTree*>(mFileIn->Get("residuals"));
  if (!mTreeInClRes) {
    LOG(error) << "tree with TPC cluster residuals not available in input file";
    return;
  }
  mTreeInClRes->SetBranchAddress("residuals", &mClResPtr);
  mTreeInClRes->GetEntry(0);
}

void TrackResiduals::setInputData(std::vector<TrackData>& trkData, std::vector<TPCClusterResiduals>& clResiduals)
{
  mTrackDataPtr = &trkData;
  mClResPtr = &clResiduals;
}

void TrackResiduals::doOutlierRejection(bool writeToFile)
{
  // Receives the collected residuals from TrackInterpolation, does outlier filtering
  // and writes the filtered residuals into a std::vector<UnbinnedResid> which will
  // be send to the aggregator afterwards

  if (!mIsInitialized) {
    LOG(error) << "TrackResiduals class not initialized. Aborting";
    return;
  }

  LOGF(info, "We got %lu tracks and %lu residuals", mTrackDataPtr->size(), mClResPtr->size());

  const std::vector<TrackData>& trackData = *mTrackDataPtr;
  const std::vector<TPCClusterResiduals>& clRes = *mClResPtr;

  if (writeToFile) {
    prepareLocalResidualTrees();
  }

  int nTracksValidated = 0;

  // loop over tracks
  for (const auto& trk : trackData) {
    TrackParams params;
    if (!validateTrack(trk, params) && mFilterOutliers) {
      continue;
    }
    TrackData trkDataOut = trk; // this track will be forwarded to the aggregator, but we still need to adapt the cluster range reference
    int nClValidated = 0;
    trkDataOut.clIdx.setFirstEntry(mUnbinnedResiduals.size());
    ++nTracksValidated;
    int iRow = 0;
    for (int iCl = 0; iCl < trk.clIdx.getEntries(); ++iCl) {
      int clIdx = trk.clIdx.getFirstEntry() + iCl;
      iRow += clRes[clIdx].dRow;
      if (params.flagRej[iCl]) {
        // skip masked cluster residual
        continue;
      }
      ++nClValidated;
      mUnbinnedResiduals.emplace_back(clRes[clIdx].dy, clRes[clIdx].dz, clRes[clIdx].tgl, clRes[clIdx].y, clRes[clIdx].z, iRow, clRes[clIdx].sec);
    }
    trkDataOut.clIdx.setEntries(nClValidated);
    mTrackDataOut.push_back(trkDataOut);
  }

  LOGF(info, "Could validate %i tracks out of %lu", nTracksValidated, mTrackDataPtr->size());

  if (writeToFile) {
    // write to file for debugging
    writeLocalResidualTreesToFile();
  }
}

bool TrackResiduals::validateTrack(const TrackData& trk, TrackParams& params) const
{
  if (trk.clIdx.getEntries() < param::MinNCl) {
    // no enough clusters for this track to be considered
    LOG(debug) << "Skipping track with too few clusters: " << trk.clIdx.getEntries();
    return false;
  }

  bool resHelix = compareToHelix(trk, params);
  if (!resHelix) {
    LOG(debug) << "Skipping track too far from helix approximation";
    return false;
  }
  if (fabsf(params.qpt) > param::MaxQ2Pt) {
    LOG(debug) << "Skipping track with too high q/pT: " << params.qpt;
    return false;
  }
  if (!outlierFiltering(trk, params)) {
    return false;
  }
  return true;
}

bool TrackResiduals::outlierFiltering(const TrackData& trk, TrackParams& params) const
{
  if (trk.clIdx.getEntries() < mNMALong) {
    LOG(debug) << "Skipping track with too few clusters for long moving average: " << trk.clIdx.getEntries();
    return false;
  }
  float rmsLong = checkResiduals(trk, params);
  if (static_cast<float>(params.flagRej.count()) / trk.clIdx.getEntries() > mMaxRejFrac) {
    LOG(debug) << "Skipping track with too many clusters rejected: " << static_cast<float>(params.flagRej.count()) / trk.clIdx.getEntries();
    return false;
  }
  if (rmsLong > mMaxRMSLong) {
    LOG(debug) << "Skipping track with too large RMS: " << rmsLong;
    return false;
  }
  return true;
}

float TrackResiduals::checkResiduals(const TrackData& trk, TrackParams& params) const
{
  float rmsLong = 0.f;

  int nCl = trk.clIdx.getEntries();
  int idxOffset = trk.clIdx.getFirstEntry();
  int iClFirst = 0;
  int iClLast = nCl - 1;
  int secStart = (*mClResPtr)[idxOffset].sec;

  // arrays with differences / abs(differences) of points to their neighbourhood, initialized to zero
  std::array<float, param::NPadRows> yDiffLL{};
  std::array<float, param::NPadRows> zDiffLL{};
  std::array<float, param::NPadRows> absDevY{};
  std::array<float, param::NPadRows> absDevZ{};

  for (int iCl = 0; iCl < nCl; ++iCl) {
    if (iCl < iClLast && (*mClResPtr)[idxOffset + iCl].sec == secStart) {
      continue;
    }
    // sector changed or last cluster reached
    // now run estimators for all points in the same sector
    int nClSec = iCl - iClFirst;
    if (iCl == iClLast) {
      ++nClSec;
    }
    diffToLocLine(nClSec, iClFirst, params.xTrk, params.dy, yDiffLL);
    diffToLocLine(nClSec, iClFirst, params.xTrk, params.dz, zDiffLL);
    iClFirst = iCl;
    secStart = (*mClResPtr)[idxOffset + iCl].sec;
  }
  // store abs deviations
  int nAccY = 0;
  int nAccZ = 0;
  for (int iCl = nCl; iCl--;) {
    if (fabsf(yDiffLL[iCl]) > param::sEps) {
      absDevY[nAccY++] = fabsf(yDiffLL[iCl]);
    }
    if (fabsf(zDiffLL[iCl]) > param::sEps) {
      absDevZ[nAccZ++] = fabsf(zDiffLL[iCl]);
    }
  }
  if (nAccY < param::MinNumberOfAcceptedResiduals || nAccZ < param::MinNumberOfAcceptedResiduals) {
    // mask all clusters
    params.flagRej.set();
    return 0.f;
  }
  // estimate rms on 90% of the smallest deviations
  int nKeepY = static_cast<int>(.9 * nAccY);
  int nKeepZ = static_cast<int>(.9 * nAccZ);
  std::nth_element(absDevY.begin(), absDevY.begin() + nKeepY, absDevY.begin() + nAccY);
  std::nth_element(absDevZ.begin(), absDevZ.begin() + nKeepZ, absDevZ.begin() + nAccZ);
  float rmsYkeep = 0.f;
  float rmsZkeep = 0.f;
  for (int i = nKeepY; i--;) {
    rmsYkeep += absDevY[i] * absDevY[i];
  }
  for (int i = nKeepZ; i--;) {
    rmsZkeep += absDevZ[i] * absDevZ[i];
  }
  rmsYkeep = std::sqrt(rmsYkeep / nKeepY);
  rmsZkeep = std::sqrt(rmsZkeep / nKeepZ);
  if (rmsYkeep < param::sEps || rmsZkeep < param::sEps) {
    LOG(warning) << "Too small RMS: " << rmsYkeep << "(y), " << rmsZkeep << "(z).";
    params.flagRej.set();
    return 0.f;
  }
  float rmsYkeepI = 1.f / rmsYkeep;
  float rmsZkeepI = 1.f / rmsZkeep;
  int nAcc = 0;
  std::array<float, param::NPadRows> yAcc;
  std::array<float, param::NPadRows> yDiffLong;
  for (int iCl = 0; iCl < nCl; ++iCl) {
    yDiffLL[iCl] *= rmsYkeepI;
    zDiffLL[iCl] *= rmsZkeepI;
    if (yDiffLL[iCl] * yDiffLL[iCl] + zDiffLL[iCl] * zDiffLL[iCl] > param::mMaxStdDevMA) {
      params.flagRej.set(iCl);
    } else {
      yAcc[nAcc++] = params.dy[iCl];
    }
  }
  if (nAcc > mNMALong) {
    diffToMA(nAcc, yAcc, yDiffLong);
    float average = 0.f;
    float rms = 0.f;
    for (int i = 0; i < nAcc; ++i) {
      average += yDiffLong[i];
      rms += yDiffLong[i] * yDiffLong[i];
    }
    average /= nAcc;
    rmsLong = rms / nAcc - average * average;
    rmsLong = (rmsLong > 0) ? std::sqrt(rmsLong) : 0.f;
  }
  return rmsLong;
}

bool TrackResiduals::compareToHelix(const TrackData& trk, TrackParams& params) const
{
  std::array<float, param::NPadRows> residHelixY;
  std::array<float, param::NPadRows> residHelixZ;

  std::array<float, param::NPadRows> xLab;
  std::array<float, param::NPadRows> yLab;
  std::array<float, param::NPadRows> sPath;

  float curvature = fabsf(trk.p[o2::track::ParLabels::kQ2Pt] * mBz * o2::constants::physics::LightSpeedCm2S * 1e-14f);
  int clIdxFirst = trk.clIdx.getFirstEntry();
  int nCl = trk.clIdx.getEntries();
  int secFirst = (*mClResPtr)[clIdxFirst].sec;
  float phiSect = (secFirst + .5f) * o2::constants::math::SectorSpanRad;
  float snPhi = sin(phiSect);
  float csPhi = cos(phiSect);
  sPath[0] = 0.f;

  int iRow = 0;
  for (int iP = 0; iP < nCl; ++iP) {
    iRow += (*mClResPtr)[clIdxFirst + iP].dRow;
    float yTrk = (*mClResPtr)[clIdxFirst + iP].y;
    // LOGF(info, "iRow(%i), yTrk(%f)", iRow, yTrk);
    xLab[iP] = param::RowX[iRow];
    if ((*mClResPtr)[clIdxFirst + iP].sec != secFirst) {
      float phiSectCurrent = ((*mClResPtr)[clIdxFirst + iP].sec + .5f) * o2::constants::math::SectorSpanRad;
      float cs = cos(phiSectCurrent - phiSect);
      float sn = sin(phiSectCurrent - phiSect);
      xLab[iP] = param::RowX[iRow] * cs - yTrk * sn;
      yLab[iP] = yTrk * cs + param::RowX[iRow] * sn;
    } else {
      xLab[iP] = param::RowX[iRow];
      yLab[iP] = yTrk;
    }
    // this is needed only later, but we retrieve it already now to save another loop
    params.zTrk[iP] = (*mClResPtr)[clIdxFirst + iP].z;
    params.xTrk[iP] = param::RowX[iRow];
    params.dy[iP] = (*mClResPtr)[clIdxFirst + iP].dy;
    params.dz[iP] = (*mClResPtr)[clIdxFirst + iP].dz;
    // done retrieving values for later
    if (iP > 0) {
      float dx = xLab[iP] - xLab[iP - 1];
      float dy = yLab[iP] - yLab[iP - 1];
      float ds2 = dx * dx + dy * dy;
      float ds = sqrt(ds2); // circular path (linear approximation)
      // if the curvature of the track or the (approximated) chord length is too large the more exact formula is used:
      // chord length = 2r * asin(ds/(2r))
      // using the first two terms of the tailer expansion for asin(x) ~ x + x^3 / 6
      if (ds * curvature > 0.05) {
        ds *= (1.f + ds2 * curvature * curvature / 24.f);
      }
      sPath[iP] = sPath[iP - 1] + ds;
    }
  }
  float xcSec = 0.f;
  float ycSec = 0.f;
  float r = 0.f;
  fitCircle(nCl, xLab, yLab, xcSec, ycSec, r, residHelixY);
  // LOGF(info, "Done with circle fit. nCl(%i), xcSec(%f), ycSec(%f), r(%f).", nCl, xcSec, ycSec, r);
  /*
  for (int i=0; i<nCl; ++i) {
    LOGF(info, "i(%i), xLab(%f), yLab(%f).", i, xLab[i], yLab[i]);
  }
  */
  // determine curvature
  float phiI = TMath::ATan2(yLab[0], xLab[0]);
  float phiF = TMath::ATan2(yLab[nCl - 1], xLab[nCl - 1]);
  if (phiI < 0) {
    phiI += o2::constants::math::TwoPI;
  }
  if (phiF < 0) {
    phiF += o2::constants::math::TwoPI;
  }
  float dPhi = phiF - phiI;
  float curvSign = -1.f;
  if (dPhi > 0) {
    if (dPhi < o2::constants::math::PI) {
      curvSign = 1.f;
    }
  } else if (dPhi < -o2::constants::math::PI) {
    curvSign = 1.f;
  }
  params.qpt = std::copysign(1.f / (r * mBz * o2::constants::physics::LightSpeedCm2S * 1e-14f), curvSign);

  // calculate circle coordinates in the lab frame
  float xc = xcSec * csPhi - ycSec * snPhi;
  float yc = xcSec * snPhi + ycSec * csPhi;

  std::array<float, 2> pol1Z;
  fitPoly1(nCl, sPath, params.zTrk, pol1Z);

  params.tgl = pol1Z[0];

  // max deviations in both directions from helix fit in y and z
  float hMinY = 1e9f;
  float hMaxY = -1e9f;
  float hMinZ = 1e9f;
  float hMaxZ = -1e9f;
  // extract residuals in Z and fill track slopes in sector frame
  int secCurr = secFirst;
  iRow = 0;
  for (int iCl = 0; iCl < nCl; ++iCl) {
    iRow += (*mClResPtr)[clIdxFirst + iCl].dRow;
    float resZ = params.zTrk[iCl] - (pol1Z[1] + sPath[iCl] * pol1Z[0]);
    residHelixZ[iCl] = resZ;
    if (resZ < hMinZ) {
      hMinZ = resZ;
    }
    if (resZ > hMaxZ) {
      hMaxZ = resZ;
    }
    if (residHelixY[iCl] < hMinY) {
      hMinY = residHelixY[iCl];
    }
    if (residHelixY[iCl] > hMaxY) {
      hMaxY = residHelixY[iCl];
    }
    int sec = (*mClResPtr)[clIdxFirst + iCl].sec;
    if (sec != secCurr) {
      secCurr = sec;
      phiSect = (.5f + sec) * o2::constants::math::SectorSpanRad;
      snPhi = sin(phiSect);
      csPhi = cos(phiSect);
      xcSec = xc * csPhi + yc * snPhi; // recalculate circle center in the sector frame
    }
    float cstalp = (param::RowX[iRow] - xcSec) / r;
    if (fabsf(cstalp) > 1.f - sFloatEps) {
      // track cannot reach this pad row
      cstalp = std::copysign(1.f - sFloatEps, cstalp);
    }
    params.tglArr[iCl] = cstalp / sqrt((1 - cstalp) * (1 + cstalp)); // 1 / tan(acos(cstalp)) = cstalp / sqrt(1 - cstalp^2)

    // In B+ the slope of q- should increase with x. Just look on q * B
    if (params.qpt * mBz > 0) {
      params.tglArr[iCl] *= -1.f;
    }
  }
  // LOGF(info, "CompareToHelix: hMaxY(%f), hMinY(%f), hMaxZ(%f), hMinZ(%f). Max deviation allowed: y(%.2f), z(%.2f)", hMaxY, hMinY, hMaxZ, hMinZ, param::MaxDevHelixY, param::MaxDevHelixZ);
  // LOGF(info, "New pt/Q (%f), old pt/Q (%f)", 1./params.qpt, 1./trk.qPt);
  return fabsf(hMaxY - hMinY) < param::MaxDevHelixY && fabsf(hMaxZ - hMinZ) < param::MaxDevHelixZ;
}

//______________________________________________________________________________
void TrackResiduals::processSectorResiduals(int iSec)
{
  initResultsContainer(iSec);
  std::vector<unsigned short> binData;
  for (const auto& res : mLocalResidualsIn) {
    binData.push_back(getGlbVoxBin(res.bvox));
  }
  // sort in voxel increasing order
  std::vector<size_t> binIndices(binData.size());
  o2::math_utils::SortData(binData, binIndices);
  // fill the voxel statistics into the results container
  std::vector<VoxRes>& secData = mVoxelResults[iSec];
  for (int iVox = 0; iVox < mNVoxPerSector; ++iVox) {
    const auto& voxStat = mVoxStatsIn[iVox];
    VoxRes& resVox = secData[iVox];
    for (int iDim = VoxDim; iDim--;) {
      resVox.stat[iDim] = voxStat.meanPos[iDim];
    }
    resVox.stat[VoxDim] = voxStat.nEntries;
  }
  // vectors holding the data for one voxel at a time
  std::vector<float> dyVec;
  std::vector<float> dzVec;
  std::vector<float> tgVec;
  // assuming we will always have around 1000 entries per voxel
  dyVec.reserve(1e3);
  dzVec.reserve(1e3);
  tgVec.reserve(1e3);
  int currVoxBin = -1;
  unsigned int nPointsInVox = 0;
  unsigned int nProcessed = 0;
  while (nProcessed < binData.size()) {
    // read all points, voxel by voxel
    int idx = binIndices[nProcessed];
    if (currVoxBin != binData[idx]) {
      if (nPointsInVox) {
        VoxRes& resVox = secData[currVoxBin];
        processVoxelResiduals(dyVec, dzVec, tgVec, resVox);
      }
      currVoxBin = binData[idx];
      nPointsInVox = 0;
      dyVec.clear();
      dzVec.clear();
      tgVec.clear();
    }
    dyVec.push_back(mLocalResidualsIn[idx].dy * param::MaxResid / 0x7fff);
    dzVec.push_back(mLocalResidualsIn[idx].dz * param::MaxResid / 0x7fff);
    tgVec.push_back(mLocalResidualsIn[idx].tgSlp * param::MaxTgSlp / 0x7fff);

    ++nPointsInVox;
    ++nProcessed;
  }
  if (nPointsInVox) {
    // process last voxel
    VoxRes& resVox = secData[currVoxBin];
    processVoxelResiduals(dyVec, dzVec, tgVec, resVox);
  }
  LOG(info) << "extracted residuals for sector " << iSec;

  int nRowsOK = validateVoxels(iSec);
  LOG(info) << "number of validated X rows: " << nRowsOK;
  if (!nRowsOK) {
    LOG(warning) << "sector " << iSec << ": all X-bins disabled, abandon smoothing";
    return;
  } else {
    smooth(iSec);
  }

  // process dispersions
  dyVec.clear();
  tgVec.clear();
  currVoxBin = -1;
  nProcessed = 0;
  nPointsInVox = 0;
  while (nProcessed < binData.size()) {
    int idx = binIndices[nProcessed];
    if (currVoxBin != binData[idx]) {
      if (nPointsInVox) {
        VoxRes& resVox = secData[currVoxBin];
        if (!getXBinIgnored(iSec, resVox.bvox[VoxX])) {
          processVoxelDispersions(tgVec, dyVec, resVox);
        }
      }
      currVoxBin = binData[idx];
      nPointsInVox = 0;
      dyVec.clear();
      tgVec.clear();
    }
    dyVec.push_back(mLocalResidualsIn[idx].dy * param::MaxResid / 0x7fff);
    tgVec.push_back(mLocalResidualsIn[idx].tgSlp * param::MaxTgSlp / 0x7fff);
    ++nPointsInVox;
    ++nProcessed;
  }
  if (nPointsInVox) {
    // process last voxel
    VoxRes& resVox = secData[currVoxBin];
    if (!getXBinIgnored(iSec, resVox.bvox[VoxX])) {
      processVoxelDispersions(tgVec, dyVec, resVox);
    }
  }
  // smooth dispersions
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (getXBinIgnored(iSec, ix)) {
      continue;
    }
    for (int iz = 0; iz < mNZ2XBins; ++iz) {
      for (int ip = 0; ip < mNY2XBins; ++ip) {
        int voxBin = getGlbVoxBin(ix, ip, iz);
        VoxRes& resVox = secData[voxBin];
        getSmoothEstimate(iSec, resVox.stat[VoxX], resVox.stat[VoxF], resVox.stat[VoxZ], resVox.DS, 0x1 << VoxV);
      }
    }
  }
  LOG(info) << "Done processing residuals for sector " << iSec;
  dumpResults(iSec);
}

//______________________________________________________________________________
void TrackResiduals::processVoxelResiduals(std::vector<float>& dy, std::vector<float>& dz, std::vector<float>& tg, VoxRes& resVox)
{
  int nPoints = dy.size();
  //LOG(debug) << "processing voxel residuals for vox " << getGlbVoxBin(resVox.bvox) << " with " << nPoints << " points";
  if (nPoints < mMinEntriesPerVoxel) {
    LOG(info) << "voxel " << getGlbVoxBin(resVox.bvox) << " is skipped due to too few entries (" << nPoints << " < " << mMinEntriesPerVoxel << ")";
    return;
  } else {
    LOGF(info, "Processing voxel %i with %i entries", getGlbVoxBin(resVox.bvox), nPoints);
  }
  std::array<float, 7> zResults;
  resVox.flags = 0;
  std::vector<size_t> indices(dz.size());
  if (!o2::math_utils::LTMUnbinned(dz, indices, zResults, mLTMCut)) {
    LOG(debug) << "failed trimming input array for voxel " << getGlbVoxBin(resVox.bvox);
    return;
  }
  std::array<float, 2> res{0.f};
  std::array<float, 3> err{0.f};
  float sigMAD = fitPoly1Robust(tg, dy, res, err, mLTMCut);
  if (sigMAD < 0) {
    LOG(debug) << "failed robust linear fit, sigMAD =  " << sigMAD;
    return;
  }
  float corrErr = err[0] * err[2];
  corrErr = corrErr > 0 ? err[1] / std::sqrt(corrErr) : -999;
  //
  resVox.D[ResX] = -res[1];
  resVox.D[ResY] = res[0];
  resVox.D[ResZ] = zResults[1];
  resVox.E[ResX] = std::sqrt(err[2]);
  resVox.E[ResY] = std::sqrt(err[0]);
  resVox.E[ResZ] = zResults[4];
  resVox.EXYCorr = corrErr;
  resVox.D[ResD] = resVox.dYSigMAD = sigMAD; // later will be overwritten by real dispersion
  resVox.dZSigLTM = zResults[2];

  resVox.flags |= DistDone;

  return;
}

void TrackResiduals::processVoxelDispersions(std::vector<float>& tg, std::vector<float>& dy, VoxRes& resVox)
{
  size_t nPoints = tg.size();
  LOG(debug) << "processing voxel dispersions for vox " << getGlbVoxBin(resVox.bvox) << " with " << nPoints << " points";
  if (nPoints < 2) {
    return;
  }
  for (size_t i = nPoints; i--;) {
    dy[i] -= resVox.DS[ResY] - resVox.DS[ResX] * tg[i];
  }
  resVox.D[ResD] = getMAD2Sigma(dy);
  resVox.E[ResD] = resVox.D[ResD] / sqrt(2.f * nPoints); // a la gaussioan RMS error (very crude)
  resVox.flags |= DispDone;
}

//______________________________________________________________________________
int TrackResiduals::validateVoxels(int iSec)
{
  // apply voxel validation cuts
  // return number of good voxels for given sector
  int cntMasked = 0;  // number of voxels masked due to fit error and / or distribution sigmas
  int cntInvalid = 0; // number of voxels which were invalid before + masked ones
  mXBinsIgnore[iSec].reset();
  std::vector<VoxRes>& secData = mVoxelResults[iSec];

  int cntMaskedFit = 0;
  int cntMaskedSigma = 0;

  // find bad voxels in sector
  for (int ix = 0; ix < mNXBins; ++ix) {
    int cntValid = 0;
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZ2XBins; ++iz) {
        int binGlb = getGlbVoxBin(ix, ip, iz);
        VoxRes& resVox = secData[binGlb];
        bool voxelOK = (resVox.flags & DistDone) && !(resVox.flags & Masked);
        if (voxelOK) {
          // check fit errors
          if (resVox.E[ResY] * resVox.E[ResY] > mMaxFitErrY2 ||
              resVox.E[ResX] * resVox.E[ResX] > mMaxFitErrX2 ||
              fabs(resVox.EXYCorr) > mMaxFitCorrXY) {
            voxelOK = false;
            ++cntMaskedFit;
          }
          // check raw distribution sigmas
          if (resVox.dYSigMAD > mMaxSigY ||
              resVox.dZSigLTM > mMaxSigZ) {
            voxelOK = false;
            ++cntMaskedSigma;
          }
          if (!voxelOK) {
            ++cntMasked;
          }
        }
        if (voxelOK) {
          ++cntValid;
        } else {
          ++cntInvalid;
          resVox.flags |= Masked;
        }
      } // loop over Z
    }   // loop over Y/X
    mValidFracXBins[iSec][ix] = static_cast<float>(cntValid) / (mNY2XBins * mNZ2XBins);
    LOG(debug) << "sector " << iSec << ": xBin " << ix << " has " << mValidFracXBins[iSec][ix] * 100 << "% of voxels valid";
  } // loop over X

  // mask X-bins which cannot be smoothed

  short nBadReg = 0;                           // count bad regions (one or more consecutive bad X-bins)
  std::array<short, param::NPadRows> badStart; // to store indices to the beginnings of the bad regions
  std::array<short, param::NPadRows> badEnd;   // to store indices to the end of the bad regions
  bool prevBad = false;
  float fracBadRows = 0.f;
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (mValidFracXBins[iSec][ix] < mMinValidVoxFracDrift) {
      LOG(debug) << "row " << ix << " is bad";
      ++fracBadRows;
      if (prevBad) {
        badEnd[nBadReg] = ix;
      } else {
        badStart[nBadReg] = ix;
        badEnd[nBadReg] = ix;
        prevBad = true;
      }
    } else {
      if (prevBad) {
        ++nBadReg;
        prevBad = false;
      }
    }
  }
  if (prevBad) {
    ++nBadReg;
  }
  fracBadRows /= mNXBins;
  if (fracBadRows > mMaxFracBadRowsPerSector) {
    LOG(warning) << "sector " << iSec << ": Fraction of bad X-bins: " << fracBadRows << " -> masking whole sector";
    mXBinsIgnore[iSec].set();
  } else {
    for (int iBad = 0; iBad < nBadReg; ++iBad) {
      LOG(debug) << "masking bad region " << iBad;
      short badInReg = badEnd[iBad] - badStart[iBad] + 1;
      short badInNextReg = iBad < (nBadReg - 1) ? badEnd[iBad] - badStart[iBad] + 1 : 0;
      if (badInReg > mMaxBadXBinsToCover) {
        // disable too large bad patches
        for (int i = 0; i < badInReg; ++i) {
          LOG(debug) << "disabling too large patch in bad region " << iBad << ", badStart(" << badStart[iBad] << "), i(" << i << ")";
          mXBinsIgnore[iSec].set(badStart[iBad] + i);
        }
      }
      if (badInNextReg > mMaxBadXBinsToCover && (badStart[iBad + 1] - badEnd[iBad] - 1) < mMinGoodXBinsToCover) {
        // disable too small isolated good patches`
        for (int i = badEnd[iBad] + 1; i < badStart[iBad + 1]; ++i) {
          LOG(debug) << "disabling too small good patch before bad region " << iBad + 1 << ", badStart(" << badEnd[iBad] << "), badEnd(" << badStart[iBad + 1] << ")";
          mXBinsIgnore[iSec].set(i);
        }
      }
    }
    if (nBadReg) {
      if (mXBinsIgnore[iSec].test(badStart[0]) && badStart[0] < mMinGoodXBinsToCover) {
        // 1st good patch is too small
        for (int i = 0; i < badStart[0]; ++i) {
          LOG(debug) << "disabling too small first good patch badStart(0), badEnd(" << badStart[0] << ")";
          mXBinsIgnore[iSec].set(i);
        }
      }
      if (mXBinsIgnore[iSec].test(badStart[nBadReg - 1]) && (mNXBins - badEnd[nBadReg - 1] - 1) < mMinGoodXBinsToCover) {
        // last good patch is too small
        for (int i = badEnd[nBadReg - 1] + 1; i < mNXBins; ++i) {
          LOG(debug) << "disabling too small last good patch badStart(" << badEnd[nBadReg - 1] << "), badEnd(" << mNXBins << ")";
          mXBinsIgnore[iSec].set(i);
        }
      }
    }
  }
  //
  int nMaskedRows = mXBinsIgnore[iSec].count();
  LOG(info) << "sector " << iSec << ": voxel stat: masked: " << cntMasked << " invalid: " << cntInvalid - cntMasked;
  //
  return mNXBins - nMaskedRows;
}

void TrackResiduals::smooth(int iSec)
{
  std::vector<VoxRes>& secData = mVoxelResults[iSec];
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (getXBinIgnored(iSec, ix)) {
      continue;
    }
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZ2XBins; ++iz) {
        int voxBin = getGlbVoxBin(ix, ip, iz);
        VoxRes& resVox = secData[voxBin];
        resVox.flags &= ~SmoothDone;
        bool res = getSmoothEstimate(resVox.bsec, resVox.stat[VoxX], resVox.stat[VoxF], resVox.stat[VoxZ], resVox.DS, (0x1 << VoxX | 0x1 << VoxF | 0x1 << VoxZ));
        if (!res) {
          mNSmoothingFailedBins[iSec]++;
        } else {
          resVox.flags |= SmoothDone;
        }
      }
    }
  }
  // substract dX contribution to dZ
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (getXBinIgnored(iSec, ix)) {
      continue;
    }
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZ2XBins; ++iz) {
        int voxBin = getGlbVoxBin(ix, ip, iz);
        VoxRes& resVox = secData[voxBin];
        if (!(resVox.flags & SmoothDone)) {
          continue;
        }
        resVox.DS[ResZ] += resVox.stat[VoxZ] * resVox.DS[ResX]; // remove slope*dX contribution from dZ
        resVox.D[ResZ] += resVox.stat[VoxZ] * resVox.DS[ResX];  // remove slope*dX contribution from dZ
      }
    }
  }
}

bool TrackResiduals::getSmoothEstimate(int iSec, float x, float p, float z, std::array<float, ResDim>& res, int whichDim)
{
  // get smooth estimate for distortions for point in sector coordinates
  /// \todo correct use of the symmetric matrix should speed up the code

  std::array<int, VoxDim> minPointsDir{0}; // min number of points per direction
  const float kTrialStep = 0.5;
  std::array<bool, ResDim> doDim{false};
  for (int i = 0; i < ResDim; ++i) {
    doDim[i] = (whichDim & (0x1 << i)) > 0;
    if (doDim[i]) {
      res[i] = 0.f;
    }
  }

  int matSize = sSmtLinDim;
  for (int i = 0; i < VoxDim; ++i) {
    minPointsDir[i] = 3; // for pol1 smoothing require at least 3 points
    if (mSmoothPol2[i]) {
      ++minPointsDir[i];
      ++matSize;
    }
  }

  int ix0, ip0, iz0;
  findVoxel(x, p, iSec < SECTORSPERSIDE ? z : -z, ix0, ip0, iz0); // find nearest voxel
  std::vector<VoxRes>& secData = mVoxelResults[iSec];
  int binCenter = getGlbVoxBin(ix0, ip0, iz0); // global bin of nearest voxel
  VoxRes& voxCenter = secData[binCenter];      // nearest voxel
  LOG(debug) << "getting smooth estimate around voxel " << binCenter;

  // cache
  // \todo maybe a 1-D cache would be more efficient?
  std::array<std::array<double, sMaxSmtDim*(sMaxSmtDim + 1) / 2>, ResDim> cmat;
  int maxNeighb = 10 * 10 * 10;
  std::vector<VoxRes*> currVox;
  currVox.reserve(maxNeighb);
  std::vector<float> currCache;
  currCache.reserve(maxNeighb * VoxHDim);

  std::array<int, VoxDim> maxTrials;
  maxTrials[VoxZ] = mNZ2XBins / 2;
  maxTrials[VoxF] = mNY2XBins / 2;
  maxTrials[VoxX] = mMaxBadXBinsToCover * 2;

  std::array<int, VoxDim> trial{0};

  while (true) {
    std::fill(mLastSmoothingRes.begin(), mLastSmoothingRes.end(), 0);
    memset(&cmat[0][0], 0, sizeof(cmat));

    int nbOK = 0; // accounted neighbours

    float stepX = mStepKern[VoxX] * (1. + kTrialStep * trial[VoxX]);
    float stepF = mStepKern[VoxF] * (1. + kTrialStep * trial[VoxF]);
    float stepZ = mStepKern[VoxZ] * (1. + kTrialStep * trial[VoxZ]);

    if (!(voxCenter.flags & DistDone) || (voxCenter.flags & Masked) || getXBinIgnored(iSec, ix0)) {
      // closest voxel has no data -> increase smoothing step
      stepX += kTrialStep * mStepKern[VoxX];
      stepF += kTrialStep * mStepKern[VoxF];
      stepZ += kTrialStep * mStepKern[VoxZ];
    }

    // effective kernel widths accounting for the increased bandwidth at the edges and missing data
    float kWXI = getDXI(ix0) * mKernelWInv[VoxX] * mStepKern[VoxX] / stepX;
    float kWFI = getDY2XI(ix0, ip0) * mKernelWInv[VoxF] * mStepKern[VoxF] / stepF;
    float kWZI = getDZ2XI(iz0) * mKernelWInv[VoxZ] * mStepKern[VoxZ] / stepZ;
    int iStepX = static_cast<int>(nearbyint(stepX + 0.5));
    int iStepF = static_cast<int>(nearbyint(stepF + 0.5));
    int iStepZ = static_cast<int>(nearbyint(stepZ + 0.5));
    int ixMin = ix0 - iStepX;
    int ixMax = ix0 + iStepX;
    if (ixMin < 0) {
      ixMin = 0;
      ixMax = std::min(static_cast<int>(nearbyint(ix0 + stepX * mKernelScaleEdge[VoxX])), mNXBins - 1);
      kWXI /= mKernelScaleEdge[VoxX];
    }
    if (ixMax >= mNXBins) {
      ixMax = mNXBins - 1;
      ixMin = std::max(static_cast<int>(nearbyint(ix0 - stepX * mKernelScaleEdge[VoxX])), 0);
      kWXI /= mKernelScaleEdge[VoxX];
    }

    int ipMin = ip0 - iStepF;
    int ipMax = ip0 + iStepF;
    if (ipMin < 0) {
      ipMin = 0;
      ipMax = std::min(static_cast<int>(nearbyint(ip0 + stepF * mKernelScaleEdge[VoxF])), mNY2XBins - 1);
      kWFI /= mKernelScaleEdge[VoxF];
    }
    if (ipMax >= mNY2XBins) {
      ipMax = mNY2XBins - 1;
      ipMin = std::max(static_cast<int>(nearbyint(ip0 - stepF * mKernelScaleEdge[VoxF])), 0);
      kWFI /= mKernelScaleEdge[VoxF];
    }

    int izMin = iz0 - iStepZ;
    int izMax = iz0 + iStepZ;
    if (izMin < 0) {
      izMin = 0;
      izMax = std::min(static_cast<int>(nearbyint(iz0 + stepZ * mKernelScaleEdge[VoxZ])), mNZ2XBins - 1);
      kWZI /= mKernelScaleEdge[VoxZ];
    }
    if (izMax >= mNZ2XBins) {
      izMax = mNZ2XBins - 1;
      izMin = std::max(static_cast<int>(nearbyint(iz0 - stepZ * mKernelScaleEdge[VoxZ])), 0);
      kWZI /= mKernelScaleEdge[VoxZ];
    }

    std::vector<unsigned short> nOccX(ixMax - ixMin + 1, 0);
    std::vector<unsigned short> nOccF(ipMax - ipMin + 1, 0);
    std::vector<unsigned short> nOccZ(izMax - izMin + 1, 0);

    int nbCheck = (ixMax - ixMin + 1) * (ipMax - ipMin + 1) * (izMax - izMin + 1);
    if (nbCheck >= maxNeighb) {
      maxNeighb = nbCheck + 100;
      currCache.reserve(maxNeighb * VoxHDim);
      currVox.reserve(maxNeighb);
    }
    std::array<double, 3> u2Vec;

    // first loop, check presence of enough points
    for (int ix = ixMin; ix <= ixMax; ++ix) {
      for (int ip = ipMin; ip <= ipMax; ++ip) {
        for (int iz = izMin; iz <= izMax; ++iz) {
          int binNb = getGlbVoxBin(ix, ip, iz);
          VoxRes& voxNb = secData[binNb];
          if (!(voxNb.flags & DistDone) ||
              (voxNb.flags & Masked) ||
              getXBinIgnored(iSec, ix)) {
            // skip voxels w/o data
            continue;
          }
          // estimate weighted distance
          float dx = voxNb.stat[VoxX] - x;
          float df = voxNb.stat[VoxF] - p;
          float dz = voxNb.stat[VoxZ] - z;
          float dxw = dx * kWXI;
          float dfw = df * kWFI;
          float dzw = dz * kWZI;
          u2Vec[0] = dxw * dxw;
          u2Vec[1] = dfw * dfw;
          u2Vec[2] = dzw * dzw;
          double kernelWeight = getKernelWeight(u2Vec);
          if (kernelWeight < 1e-6) {
            continue;
          }
          // new point is validated
          ++nOccX[ix - ixMin];
          ++nOccF[ip - ipMin];
          ++nOccZ[iz - izMin];
          currVox[nbOK] = &voxNb;
          currCache[nbOK * VoxHDim + VoxX] = dx;
          currCache[nbOK * VoxHDim + VoxF] = df;
          currCache[nbOK * VoxHDim + VoxZ] = dz;
          currCache[nbOK * VoxHDim + VoxV] = kernelWeight;
          ++nbOK;
        }
      }
    }

    // check if we have enough points in every dimension
    std::array<int, VoxDim> nPoints{0};
    for (int i = ixMax - ixMin + 1; i--;) {
      if (nOccX[i]) {
        ++nPoints[VoxX];
      }
    }
    for (int i = ipMax - ipMin + 1; i--;) {
      if (nOccF[i]) {
        ++nPoints[VoxF];
      }
    }
    for (int i = izMax - izMin + 1; i--;) {
      if (nOccZ[i]) {
        ++nPoints[VoxZ];
      }
    }
    bool enoughPoints = true;
    std::array<bool, VoxDim> incrDone{false};
    for (int i = 0; i < VoxDim; ++i) {
      if (nPoints[i] < minPointsDir[i]) {
        // need to extend smoothing neighbourhood
        enoughPoints = false;
        if (trial[i] < maxTrials[i] && !incrDone[i]) {
          // try to increment only missing direction
          ++trial[i];
          incrDone[i] = true;
        } else if (trial[i] == maxTrials[i]) {
          // cannot increment missing direction, try others
          for (int j = VoxDim; j--;) {
            if (i != j && trial[j] < maxTrials[j] && !incrDone[j]) {
              ++trial[j];
              incrDone[j] = true;
            }
          }
        }
      }
    }

    if (!enoughPoints) {
      if (!(incrDone[VoxX] || incrDone[VoxF] || incrDone[VoxZ])) {
        LOG(error) << "trial limit reached, skipping this voxel";
        return false;
      }
      LOG(debug) << "sector " << iSec << ": increasing filter bandwidth around voxel " << binCenter;
      //printf("Sector:%2d x=%.2f y/x=%.2f z/x=%.2f (iX: %d iY2X:%d iZ2X:%d)\n", iSec, x, p, z, ix0, ip0, iz0);
      //printf("not enough neighbours (need min %d) %d %d %d (tot: %d) | Steps: %.1f %.1f %.1f\n", 2, nPoints[VoxX], nPoints[VoxF], nPoints[VoxZ], nbOK, stepX, stepF, stepZ);
      //printf("trying to increase filter bandwidth (trialXFZ: %d %d %d)\n", trial[VoxX], trial[VoxF], trial[VoxZ]);
      continue;
    }

    // now fill matrices and solve
    for (int iNb = 0; iNb < nbOK; ++iNb) {
      double wiCache = currCache[iNb * VoxHDim + VoxV];
      double dxi = currCache[iNb * VoxHDim + VoxX];
      double dfi = currCache[iNb * VoxHDim + VoxF];
      double dzi = currCache[iNb * VoxHDim + VoxZ];
      double dxi2 = dxi * dxi;
      double dfi2 = dfi * dfi;
      double dzi2 = dzi * dzi;
      const VoxRes* voxNb = currVox[iNb];
      for (int iDim = 0; iDim < ResDim; ++iDim) {
        if (!doDim[iDim]) {
          continue;
        }
        double vi = voxNb->D[iDim];
        double wi = wiCache;
        if (mUseErrInSmoothing && fabs(voxNb->E[iDim]) > 1e-6) {
          // account for point error apart from kernel value
          wi /= (voxNb->E[iDim] * voxNb->E[iDim]);
        }
        std::array<double, sMaxSmtDim*(sMaxSmtDim + 1) / 2>& cmatD = cmat[iDim];
        double* rhsD = &mLastSmoothingRes[iDim * sMaxSmtDim];
        unsigned short iMat = 0;
        unsigned short iRhs = 0;
        // linear part
        cmatD[iMat++] += wi;
        rhsD[iRhs++] += wi * vi;
        //
        cmatD[iMat++] += wi * dxi;
        cmatD[iMat++] += wi * dxi2;
        rhsD[iRhs++] += wi * dxi * vi;
        //
        cmatD[iMat++] += wi * dfi;
        cmatD[iMat++] += wi * dxi * dfi;
        cmatD[iMat++] += wi * dfi2;
        rhsD[iRhs++] += wi * dfi * vi;
        //
        cmatD[iMat++] += wi * dzi;
        cmatD[iMat++] += wi * dxi * dzi;
        cmatD[iMat++] += wi * dfi * dzi;
        cmatD[iMat++] += wi * dzi2;
        rhsD[iRhs++] += wi * dzi * vi;
        //
        // check if quadratic part is needed
        if (mSmoothPol2[VoxX]) {
          cmatD[iMat++] += wi * dxi2;
          cmatD[iMat++] += wi * dxi * dxi2;
          cmatD[iMat++] += wi * dfi * dxi2;
          cmatD[iMat++] += wi * dzi * dxi2;
          cmatD[iMat++] += wi * dxi2 * dxi2;
          rhsD[iRhs++] += wi * dxi2 * vi;
        }
        if (mSmoothPol2[VoxF]) {
          cmatD[iMat++] += wi * dfi2;
          cmatD[iMat++] += wi * dxi * dfi2;
          cmatD[iMat++] += wi * dfi * dfi2;
          cmatD[iMat++] += wi * dzi * dfi2;
          cmatD[iMat++] += wi * dxi2 * dfi2;
          cmatD[iMat++] += wi * dfi2 * dfi2;
          rhsD[iRhs++] += wi * dfi2 * vi;
        }
        if (mSmoothPol2[VoxZ]) {
          cmatD[iMat++] += wi * dzi2;
          cmatD[iMat++] += wi * dxi * dzi2;
          cmatD[iMat++] += wi * dfi * dzi2;
          cmatD[iMat++] += wi * dzi * dzi2;
          cmatD[iMat++] += wi * dxi2 * dzi2;
          cmatD[iMat++] += wi * dfi2 * dzi2;
          cmatD[iMat++] += wi * dzi2 * dzi2;
          rhsD[iRhs++] += wi * dzi2 * vi;
        }
      }
    }

    bool fitRes = true;

    // solve system of linear equations

    TMatrixDSym matrix(matSize);
    TDecompChol chol(matSize);
    TVectorD rhsVec(matSize);
    for (int iDim = 0; iDim < ResDim; ++iDim) {
      if (!doDim[iDim]) {
        continue;
      }
      matrix.Zero(); // reset matrix
      std::array<double, sMaxSmtDim*(sMaxSmtDim + 1) / 2>& cmatD = cmat[iDim];
      double* rhsD = &mLastSmoothingRes[iDim * sMaxSmtDim];
      short iMat = -1;
      short row = -1;

      // with the studid implementation of TMatrixDSym we need to set all elements of the matrix explicitly (or maybe only upper triangle?)
      matrix(++row, 0) = cmatD[++iMat];
      matrix(++row, 0) = cmatD[++iMat];
      matrix(row, 1) = cmatD[++iMat];
      matrix(0, row) = matrix(row, 0);
      matrix(++row, 0) = cmatD[++iMat];
      matrix(row, 1) = cmatD[++iMat];
      matrix(row, 2) = cmatD[++iMat];
      matrix(0, row) = matrix(row, 0);
      matrix(1, row) = matrix(row, 1);
      matrix(++row, 0) = cmatD[++iMat];
      matrix(row, 1) = cmatD[++iMat];
      matrix(row, 2) = cmatD[++iMat];
      matrix(row, 3) = cmatD[++iMat];
      matrix(0, row) = matrix(row, 0);
      matrix(1, row) = matrix(row, 1);
      matrix(2, row) = matrix(row, 2);
      // add pol2 elements if needed
      if (mSmoothPol2[VoxX]) {
        const int colLim = (++row) + 1;
        for (int iCol = 0; iCol < colLim; ++iCol) {
          matrix(row, iCol) = cmatD[++iMat];
          matrix(iCol, row) = matrix(row, iCol);
        }
      }
      if (mSmoothPol2[VoxF]) {
        const int colLim = (++row) + 1;
        for (int iCol = 0; iCol < colLim; ++iCol) {
          matrix(row, iCol) = cmatD[++iMat];
          matrix(iCol, row) = matrix(row, iCol);
        }
      }
      if (mSmoothPol2[VoxZ]) {
        const int colLim = (++row) + 1;
        for (int iCol = 0; iCol < colLim; ++iCol) {
          matrix(row, iCol) = cmatD[++iMat];
          matrix(iCol, row) = matrix(row, iCol);
        }
      }
      rhsVec.SetElements(rhsD);
      chol.SetMatrix(matrix);
      chol.Decompose();
      fitRes = chol.Solve(rhsVec);
      if (!fitRes) {
        for (int i = VoxDim; i--;) {
          trial[i]++;
        }
        LOG(error) << "solution for smoothing failed, trying to increase filter bandwidth";
        continue;
      }
      res[iDim] = rhsVec[0];
    }

    break;
  }

  return true;
}

double TrackResiduals::getKernelWeight(std::array<double, 3> u2vec) const
{
  double w = 1.;
  if (mKernelType == KernelType::Epanechnikov) {
    for (size_t i = u2vec.size(); i--;) {
      if (u2vec[i] > 1) {
        return 0.;
      }
      w *= 3. / 4. * (1. - u2vec[i]);
    }
  } else if (mKernelType == KernelType::Gaussian) {
    double u2 = 0.;
    for (size_t i = u2vec.size(); i--;) {
      u2 += u2vec[i];
    }
    w = u2 < mMaxGaussStdDev * mMaxGaussStdDev * u2vec.size() ? std::exp(-u2) / std::sqrt(2. * M_PI) : 0;
  }
  return w;
}

///////////////////////////////////////////////////////////////////////////////
///
/// fitting + statistics helper functions
///
///////////////////////////////////////////////////////////////////////////////

void TrackResiduals::diffToMA(const int np, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffMA) const
{
  // Calculate
  std::vector<float> sumVec(np + 1);
  auto sum = &(sumVec[1]);
  for (int i = 0; i < np; ++i) {
    sum[i] = sum[i - 1] + y[i];
  }
  for (int i = 0; i < np; ++i) {
    diffMA[i] = 0;
    int iLeft = i - mNMALong;
    int iRight = i + mNMALong;
    if (iLeft < 0) {
      iLeft = 0;
    }
    if (iRight >= np) {
      iRight = np - 1;
    }
    int nPoints = iRight - iLeft;
    if (nPoints < mNMALong) {
      // this cannot happen, since at least mNMALong points are required as neighbours for this function to be called
      continue;
    }
    float movingAverage = (sum[iRight] - sum[iLeft - 1] - (sum[i] - sum[i - 1])) / nPoints;
    diffMA[i] = y[i] - movingAverage;
  }
}

void TrackResiduals::diffToLocLine(const int np, int idxOffset, const std::array<float, param::NPadRows>& x, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffY) const
{
  // Calculate the difference between the points and the linear extrapolations from the neighbourhood.
  // Nothing more than multiple 1-d fits at once. Instead of building 4 sums (x, x^2, y, xy), 4 * nPoints sums are calculated at once
  // compare to TrackResiduals::fitPoly1() method

  // adding one entry to the vectors saves an additional if statement when calculating the cumulants
  std::vector<float> sumX1vec(np + 1);
  std::vector<float> sumX2vec(np + 1);
  std::vector<float> sumY1vec(np + 1);
  std::vector<float> sumXYvec(np + 1);
  auto sumX1 = &(sumX1vec[1]);
  auto sumX2 = &(sumX2vec[1]);
  auto sumY1 = &(sumY1vec[1]);
  auto sumXY = &(sumXYvec[1]);

  // accumulate sums for all points
  for (int iCl = 0; iCl < np; ++iCl) {
    int idx = iCl + idxOffset;
    sumX1[iCl] = sumX1[iCl - 1] + x[idx];
    sumX2[iCl] = sumX2[iCl - 1] + x[idx] * x[idx];
    sumY1[iCl] = sumY1[iCl - 1] + y[idx];
    sumXY[iCl] = sumXY[iCl - 1] + x[idx] * y[idx];
  }

  for (int iCl = 0; iCl < np; ++iCl) {
    int iClLeft = iCl - mNMAShort;
    int iClRight = iCl + mNMAShort;
    if (iClLeft < 0) {
      iClLeft = 0;
    }
    if (iClRight >= np) {
      iClRight = np - 1;
    }
    int nPoints = iClRight - iClLeft;
    if (nPoints < mNMAShort) {
      continue;
    }
    float nPointsInv = 1.f / nPoints;
    int iClLeftP = iClLeft - 1;
    int iClCurrP = iCl - 1;
    // extract sum from iClLeft to iClRight from cumulants, excluding iCl from the fit
    float sX1 = sumX1[iClRight] - sumX1[iClLeftP] - (sumX1[iCl] - sumX1[iClCurrP]);
    float sX2 = sumX2[iClRight] - sumX2[iClLeftP] - (sumX2[iCl] - sumX2[iClCurrP]);
    float sY1 = sumY1[iClRight] - sumY1[iClLeftP] - (sumY1[iCl] - sumY1[iClCurrP]);
    float sXY = sumXY[iClRight] - sumXY[iClLeftP] - (sumXY[iCl] - sumXY[iClCurrP]);
    float det = sX2 - nPointsInv * sX1 * sX1;
    if (fabsf(det) < 1e-12f) {
      continue;
    }
    float slope = (sXY - nPointsInv * sX1 * sY1) / det;
    float offset = nPointsInv * sY1 - nPointsInv * slope * sX1;
    diffY[iCl + idxOffset] = y[iCl + idxOffset] - slope * x[iCl + idxOffset] - offset;
  }
}

float TrackResiduals::fitPoly1Robust(std::vector<float>& x, std::vector<float>& y, std::array<float, 2>& res, std::array<float, 3>& err, float cutLTM) const
{
  // robust pol1 fit, modifies input arrays order
  if (x.size() != y.size()) {
    LOG(error) << "x and y must not have different sizes for fitPoly1Robust (" << x.size() << " != " << y.size() << ")";
  }
  size_t nPoints = x.size();
  res[0] = res[1] = 0.f;
  if (nPoints < 2) {
    return -1;
  }
  std::array<float, 7> yResults;
  std::vector<size_t> indY(nPoints);
  if (!o2::math_utils::LTMUnbinned(y, indY, yResults, cutLTM)) {
    return -1;
  }
  // rearrange used events in increasing order
  o2::math_utils::Reorder(y, indY);
  o2::math_utils::Reorder(x, indY);
  //
  // 1st fit to get crude slope
  int nPointsUsed = std::lrint(yResults[0]);
  int vecOffset = std::lrint(yResults[5]);
  // use only entries selected by LTM for the fit
  float a, b;
  medFit(nPointsUsed, vecOffset, x, y, a, b, err);
  //
  std::vector<float> ycm(nPoints);
  for (size_t i = nPoints; i-- > 0;) {
    ycm[i] = y[i] - (a + b * x[i]);
  }
  std::vector<size_t> indices(nPoints);
  o2::math_utils::SortData(ycm, indices);
  o2::math_utils::Reorder(ycm, indices);
  o2::math_utils::Reorder(y, indices);
  o2::math_utils::Reorder(x, indices);
  //
  // robust estimate of sigma after crude slope correction
  float sigMAD = getMAD2Sigma({ycm.begin() + vecOffset, ycm.begin() + vecOffset + nPointsUsed});
  // find LTM estimate matching to sigMAD, keaping at least given fraction
  if (!o2::math_utils::LTMUnbinnedSig(ycm, indY, yResults, mMinFracLTM, sigMAD, true)) {
    return -1;
  }
  // final fit
  nPointsUsed = std::lrint(yResults[0]);
  vecOffset = std::lrint(yResults[5]);
  medFit(nPointsUsed, vecOffset, x, y, a, b, err);
  res[0] = a;
  res[1] = b;
  return sigMAD;
}

//___________________________________________________________________
void TrackResiduals::medFit(int nPoints, int offset, const std::vector<float>& x, const std::vector<float>& y, float& a, float& b, std::array<float, 3>& err) const
{
  // fitting a straight line y(x|a, b) = a + b * x
  // to given x and y data minimizing the absolute deviation
  float aa, bb, chi2 = 0.f;
  if (nPoints < 2) {
    a = b = 0.f;
    err[0] = err[1] = err[2] = 999.f;
    return;
  }
  // do least squares minimization as first guess
  float sx = 0.f, sxx = 0.f, sy = 0.f, sxy = 0.f;
  for (int j = nPoints + offset; j-- > offset;) { // same order as in AliRoot version such that resulting sums are identical
    sx += x[j];
    sxx += x[j] * x[j];
    sy += y[j];
    sxy += x[j] * y[j];
  }
  float del = nPoints * sxx - sx * sx;
  float delI = 1. / del;
  aa = (sxx * sy - sx * sxy) * delI;
  bb = (nPoints * sxy - sx * sy) * delI;
  err[0] = sxx * delI;
  err[1] = sx * delI;
  err[2] = nPoints * delI;

  for (int j = nPoints + offset; j-- > offset;) {
    float tmp = y[j] - (aa + bb * x[j]);
    chi2 += tmp * tmp;
  }
  float sigb = std::sqrt(chi2 * delI); // expected sigma for b
  float b1 = bb;
  float f1 = roFunc(nPoints, offset, x, y, b1, aa);
  if (sigb > 0) {
    float b2 = bb + std::copysign(3.f * sigb, f1);
    float f2 = roFunc(nPoints, offset, x, y, b2, aa);
    if (fabs(f1 - f2) < sFloatEps) {
      a = aa;
      b = bb;
      return;
    }
    while (f1 * f2 > 0.f) {
      bb = b2 + 1.6f * (b2 - b1);
      b1 = b2;
      f1 = f2;
      b2 = bb;
      f2 = roFunc(nPoints, offset, x, y, b2, aa);
    }
    sigb = .01f * sigb;
    while (fabs(b2 - b1) > sigb) {
      bb = b1 + .5f * (b2 - b1);
      if (bb == b1 || bb == b2) {
        break;
      }
      float f = roFunc(nPoints, offset, x, y, bb, aa);
      if (f * f1 >= .0f) {
        f1 = f;
        b1 = bb;
      } else {
        f2 = f;
        b2 = bb;
      }
    }
  }
  a = aa;
  b = bb;
}

float TrackResiduals::roFunc(int nPoints, int offset, const std::vector<float>& x, const std::vector<float>& y, float b, float& aa) const
{
  // calculate sum(x_i * sgn(y_i - a - b * x_i)) for given b
  // see numberical recipies paragraph 15.7.3
  std::vector<float> vecTmp(nPoints);
  float sum = 0.f;
  for (int j = nPoints; j-- > 0;) {
    vecTmp[j] = y[j + offset] - b * x[j + offset];
  }
  int nPointsHalf = nPoints / 2;
  if (nPoints < 20) { // it is faster to do insertion sort
    for (int i = 1; i < nPoints; i++) {
      float v = vecTmp[i];
      int j;
      for (j = i; j--;) {
        if (vecTmp[j] > v) {
          vecTmp[j + 1] = vecTmp[j];
        } else {
          break;
        }
      }
      vecTmp[j + 1] = v;
    }
    aa = (nPoints & 0x1) ? vecTmp[nPointsHalf] : .5f * (vecTmp[nPointsHalf - 1] + vecTmp[nPointsHalf]);
  } else {
    std::vector<float>::iterator nth = vecTmp.begin() + vecTmp.size() / 2;
    if (nPoints & 0x1) {
      std::nth_element(vecTmp.begin(), nth, vecTmp.end());
      aa = *nth;
    } else {
      std::nth_element(vecTmp.begin(), nth - 1, vecTmp.end());
      std::nth_element(nth, nth, vecTmp.end());
      aa = 0.5 * (*(nth - 1) + *(nth));
    }
    //aa = (nPoints & 0x1) ? selectKthMin(nPointsHalf, vecTmp) : .5f * (selectKthMin(nPointsHalf - 1, vecTmp) + selectKthMin(nPointsHalf, vecTmp));
  }
  for (int j = nPoints; j-- > 0;) {
    float d = y[j + offset] - (b * x[j + offset] + aa);
    if (y[j + offset] != 0.f) {
      d /= fabs(y[j + offset]);
    }
    if (fabs(d) > sFloatEps) {
      sum += (d >= 0.f ? x[j + offset] : -x[j + offset]);
    }
  }
  return sum;
}

//______________________________________________________________________________
float TrackResiduals::selectKthMin(const int k, std::vector<float>& data) const
{
  // Returns the k th smallest value in the vector. The input vector will be rearranged
  // to have this value in location data[k] , with all smaller elements moved before it
  // (in arbitrary order) and all larger elements after (also in arbitrary order).
  // From Numerical Recipes in C++ (paragraph 8.5)
  // Probably it is not needed anymore, since std::nth_element() can also be used

  int i, ir, j, l, mid, n = data.size();
  float a;    // partitioning element
  l = 0;      // left hand side of active partition
  ir = n - 1; // right hand side of active partition

  while (true) {
    if (ir <= l + 1) {                         // active partition with 1 or 2 elements
      if (ir == l + 1 && data[ir] < data[l]) { // case of 2 elements
        std::swap(data[l], data[ir]);
      }
      return data[k];
    } else {
      mid = (l + ir) >> 1;
      std::swap(data[mid], data[l + 1]);
      if (data[l] > data[ir]) {
        std::swap(data[l], data[ir]);
      }
      if (data[l + 1] > data[ir]) {
        std::swap(data[l + 1], data[ir]);
      }
      if (data[l] > data[l + 1]) {
        std::swap(data[l], data[l + 1]);
      }
      i = l + 1; // initialize pointers for partitioning
      j = ir;
      a = data[l + 1];
      while (true) {
        // innermost loop used for partitioning
        do {
          i++;
        } while (data[i] < a);
        do {
          j--;
        } while (data[j] > a);
        if (j < i) {
          // pointers crossed -> partitioning complete
          break;
        }
        std::swap(data[i], data[j]);
      }
      data[l + 1] = data[j];
      data[j] = a;
      // keep the partition which contains kth element active
      if (j >= k) {
        ir = j - 1;
      }
      if (j <= k) {
        l = i;
      }
    }
  }
}

//___________________________________________________________________
float TrackResiduals::getMAD2Sigma(std::vector<float> data) const
{
  // Sigma calculated from median absolute deviations
  // see: https://en.wikipedia.org/wiki/Median_absolute_deviation
  // the data is passed by value (copied!), such that the original vector
  // is not rearranged

  int nPoints = data.size();
  if (nPoints < 2) {
    return 0;
  }

  // calculate median of the input data
  float medianOfData;
  std::vector<float>::iterator nth = data.begin() + data.size() / 2;
  if (nPoints & 0x1) {
    std::nth_element(data.begin(), nth, data.end());
    medianOfData = *nth;
  } else {
    std::nth_element(data.begin(), nth - 1, data.end());
    std::nth_element(nth, nth, data.end());
    medianOfData = .5f * (*(nth - 1) + (*nth));
  }

  // fill vector with absolute deviations to median
  for (auto& entry : data) {
    entry = fabs(entry - medianOfData);
  }

  // calculate median of abs deviations
  float medianOfAbsDeviations;
  if (nPoints & 0x1) {
    std::nth_element(data.begin(), nth, data.end());
    medianOfAbsDeviations = *nth;
  } else {
    std::nth_element(data.begin(), nth - 1, data.end());
    std::nth_element(nth, nth, data.end());
    medianOfAbsDeviations = .5f * (*(nth - 1) + (*nth));
  }

  float k = 1.4826f; // scale factor for normally distributed data
  return k * medianOfAbsDeviations;
}

void TrackResiduals::fitCircle(int nCl, std::array<float, param::NPadRows>& x, std::array<float, param::NPadRows>& y, float& xc, float& yc, float& r, std::array<float, param::NPadRows>& residHelixY) const
{
  // this fast algebraic circle fit is described here:
  // https://dtcenter.org/met/users/docs/write_ups/circle_fit.pdf
  float xMean = 0.f;
  float yMean = 0.f;

  for (int i = nCl; i--;) {
    xMean += x[i];
    yMean += y[i];
  }
  xMean /= nCl;
  yMean /= nCl;
  // define sums needed for circular fit
  float su2 = 0.f, sv2 = 0.f, suv = 0.f, su3 = 0.f, sv3 = 0.f, su2v = 0.f, suv2 = 0.f;
  for (int i = nCl; i--;) {
    float ui = x[i] - xMean;
    float vi = y[i] - yMean;
    float ui2 = ui * ui;
    float vi2 = vi * vi;
    suv += ui * vi;
    su2 += ui2;
    sv2 += vi2;
    su3 += ui2 * ui;
    sv3 += vi2 * vi;
    su2v += ui2 * vi;
    suv2 += ui * vi2;
  }
  float rhsU = .5f * (su3 + suv2);
  float rhsV = .5f * (sv3 + su2v);
  float det = su2 * sv2 - suv * suv;
  float uc = (rhsU * sv2 - rhsV * suv) / det;
  float vc = (su2 * rhsV - suv * rhsU) / det;
  float r2 = uc * uc + vc * vc + (su2 + sv2) / nCl;
  xc = uc + xMean;
  yc = vc + yMean;
  r = sqrt(r2);
  // write residuals to residHelixY
  for (int i = nCl; i--;) {
    float dx = x[i] - xc;
    float dxr = r2 - dx * dx;
    float ys = dxr > 0 ? sqrt(dxr) : 0.f; // distance of point in y from the circle center (using fit results for r and xc)
    float dy = y[i] - yc;                 // distance of point in y from the circle center (using fit result for yc)
    float dysp = dy - ys;
    float dysm = dy + ys;
    residHelixY[i] = fabsf(dysp) < fabsf(dysm) ? dysp : dysm;
  }
  //printf("r = %.4f m, xc = %.4f, yc = %.4f\n", r/100.f, xc, yc);
}

bool TrackResiduals::fitPoly1(int nCl, std::array<float, param::NPadRows>& x, std::array<float, param::NPadRows>& y, std::array<float, 2>& res) const
{
  // fit a straight line y = ax + b to a given set of points (x,y)
  // no measurement errors assumed, no fit errors calculated
  // res[0] = a (slope)
  // res[1] = b (offset)
  if (nCl < 2) {
    // not enough points
    return false;
  }
  float sumX = 0.f, sumY = 0.f, sumXY = 0.f, sumX2 = 0.f, nInv = 1.f / nCl;
  for (int i = nCl; i--;) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
  }
  float det = sumX2 - nInv * sumX * sumX;
  if (fabsf(det) < 1e-12f) {
    return false;
  }
  res[0] = (sumXY - nInv * sumX * sumY) / det;
  res[1] = nInv * sumY - nInv * res[0] * sumX;
  return true;
}

///////////////////////////////////////////////////////////////////////////////
///
/// debugging
///
///////////////////////////////////////////////////////////////////////////////

void TrackResiduals::createOutputFile()
{
  mFileOut = std::make_unique<TFile>("debugVoxRes.root", "recreate");
  mTreeOut = std::make_unique<TTree>("voxResTree", "Voxel results and statistics");
  mTreeOut->Branch("voxRes", &mVoxelResultsOutPtr);
}

void TrackResiduals::closeOutputFile()
{
  mFileOut->cd();
  mTreeOut->Write();
  mTreeOut.reset();
  mFileOut->Close();
  mFileOut.reset();
}

void TrackResiduals::dumpResults(int iSec)
{
  if (mTreeOut) {
    printf("Dumping results for sector %i. Don't forget the call to closeOutputFile() in the end...\n", iSec);
    for (int i = 0; i < mNVoxPerSector; ++i) {
      mVoxelResultsOut = mVoxelResults[iSec][i];
      mTreeOut->Fill();
    }
  }
}

void TrackResiduals::printMem() const
{
  static float mres = 0, mvir = 0, mres0 = 0, mvir0 = 0;
  static ProcInfo_t procInfo;
  static TStopwatch sw;
  const Long_t kMB = 1024;
  gSystem->GetProcInfo(&procInfo);
  mres = float(procInfo.fMemResident) / kMB;
  mvir = float(procInfo.fMemVirtual) / kMB;
  sw.Stop();
  printf("RSS: %.3f(%.3f) VMEM: %.3f(%.3f) MB | CpuTime:%.3f RealTime:%.3f s\n",
         mres, mres - mres0, mvir, mvir - mvir0, sw.CpuTime(), sw.RealTime());
  mres0 = mres;
  mvir0 = mvir;
  sw.Start();
}
