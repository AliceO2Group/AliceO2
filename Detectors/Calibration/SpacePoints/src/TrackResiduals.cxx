// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackResiduals.cxx
/// \brief Implementation of the TrackResiduals class
///
/// \author Ole Schmidt, ole.schmidt@cern.ch
///
/// \todo The COG for every voxel is still assumed in the voxel center, because only the compact trees are used as input so far
///

#include "SpacePoints/TrackResiduals.h"
#include "SpacePoints/Statistics.h"

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

using namespace o2::calib;


///////////////////////////////////////////////////////////////////////////////
///
/// initialization + binning
///
///////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void TrackResiduals::init()
{
  // initialize binning
  mNZBins = param::ZBins;
  mNXBins = 0;
  mNY2XBins = param::Y2XBins;
  initBinning();

  // initialize results container
  mVoxelResults.resize(param::NSectors2);
  for (int i = 0; i < param::NSectors2; i++) {
    mVoxelResults[i].resize(mNVoxPerSector);
  }
  mSmoothPol2[param::VoxX] = true;
  mSmoothPol2[param::VoxF] = true;
  setKernelType();
  mIsInitialized = true;
  LOG(info) << "initialization complete";
}

//______________________________________________________________________________
void TrackResiduals::initBinning()
{
  // initialize binning structures
  //
  // X binning
  if (mNXBins > 0 && mNXBins < param::NPadRows) {
    // uniform binning in X
    mDXI = mNXBins / (param::MaxX - param::MinX);
    mDX = 1.0f / mDXI;
    mUniformBins[param::VoxX] = true;
  } else {
    // binning per pad row
    mNXBins = param::NPadRows;
    mUniformBins[param::VoxX] = false;
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
    mMaxY2X[ix] = param::MaxY2X - param::DeadZone / x;
    mDY2XI[ix] = mNY2XBins / (2.f * mMaxY2X[ix]);
    mDY2X[ix] = 1.f / mDY2XI[ix];
  }
  mUniformBins[param::VoxF] = true;
  //
  // Z binning
  mDZI = mNZBins / param::MaxZ;
  mDZ = 1.0f / mDZI;
  mUniformBins[param::VoxZ] = true;
  //
  mNVoxPerSector = mNY2XBins * mNZBins * mNXBins;
}

//______________________________________________________________________________
void TrackResiduals::initResultsContainer(int iSec)
{
  for (int ix = 0; ix < mNXBins; ++ix) {
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZBins; ++iz) {
        int binGlb = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = mVoxelResults[iSec][binGlb];
        resVox.bvox[param::VoxX] = ix;
        resVox.bvox[param::VoxF] = ip;
        resVox.bvox[param::VoxZ] = iz;
        resVox.bsec = iSec;
        // COG estimates are set to the bin center by default
        getVoxelCoordinates(resVox.bsec, resVox.bvox[param::VoxX], resVox.bvox[param::VoxF], resVox.bvox[param::VoxZ],
                            resVox.stat[param::VoxX], resVox.stat[param::VoxF], resVox.stat[param::VoxZ]);
      }
    }
  }
  LOG(info) << "initialized the container for the main results";
}

//______________________________________________________________________________
void TrackResiduals::reset()
{
  for (int iSec = 0; iSec < param::NSectors2; ++iSec) {
    mXBinsIgnore[iSec].reset();
    std::fill(mVoxelResults[iSec].begin(), mVoxelResults[iSec].end(), bres_t());
  }
}

//______________________________________________________________________________
int TrackResiduals::getXBin(float x) const
{
  // convert x to bin ID, following pad row widths
  if (mUniformBins[param::VoxX]) {
    int ix = (x - param::MinX) * mDXI;
    if (ix < 0) {
      ix = 0;
    }
    return (ix < mNXBins) ? ix : mNXBins - 1;
  } else {
    int ix;
    if (x < param::RowX[param::NRowIROC - 1] + 0.5 * param::RowDX[param::NRowIROC - 1]) {
      // pad size is uniform in IROC
      ix = (x - (param::RowX[0] - param::RowDX[0] * 0.5)) / param::RowDX[0];
      return (ix >= 0) ? ix : 0;
    } else if (x >= param::RowX[param::NRowIROC + param::NRowOROC1] - 0.5 * param::RowDX[param::NRowIROC + param::NRowOROC1]) {
      // pad size is uniform in OROC2
      ix = (x - (param::RowX[param::NRowIROC + param::NRowOROC1] - 0.5 * param::RowDX[param::NRowIROC + param::NRowOROC1])) / param::RowDX[param::NPadRows - 1] + param::NRowIROC + param::NRowOROC1;
      return (ix < param::NPadRows) ? ix : param::NPadRows - 1;
    } else {
      // pad size is uniform in OROC1
      ix = (x - (param::RowX[param::NRowIROC] - 0.5 * param::RowDX[param::NRowIROC])) / param::RowDX[param::NRowIROC] + param::NRowIROC;
      if (ix < param::NRowIROC) {
        // we are in between IROC and OROC1
        if (x > 0.5 * (param::RowX[param::NRowIROC - 1] + param::RowX[param::NRowIROC])) {
          ix = param::NRowIROC; // 1st OROC1 row
        } else {
          ix = param::NRowIROC - 1;
        }
      }
      return ix;
    }
  }
}

void TrackResiduals::setKernelType(int type, float bwX, float bwP, float bwZ, float scX, float scP, float scZ)
{
  // set kernel type and widths in terms of binning in x, y/x, z/x and define aux variables
  mKernelType = type;

  mKernelScaleEdge[param::VoxX] = scX;
  mKernelScaleEdge[param::VoxF] = scP;
  mKernelScaleEdge[param::VoxZ] = scZ;

  mKernelWInv[param::VoxX] = (bwX > 0) ? 1. / bwX : 1.;
  mKernelWInv[param::VoxF] = (bwP > 0) ? 1. / bwP : 1.;
  mKernelWInv[param::VoxZ] = (bwZ > 0) ? 1. / bwZ : 1.;

  if (mKernelType == param::EpanechnikovKernel) {
    // bandwidth 1
    mStepKern[param::VoxX] = static_cast<int>(nearbyint(bwX + 0.5));
    mStepKern[param::VoxF] = static_cast<int>(nearbyint(bwP + 0.5));
    mStepKern[param::VoxZ] = static_cast<int>(nearbyint(bwZ + 0.5));
  } else if (mKernelType == param::GaussianKernel) {
    // look in ~5 sigma
    mStepKern[param::VoxX] = static_cast<int>(nearbyint(bwX * 5. + 0.5));
    mStepKern[param::VoxF] = static_cast<int>(nearbyint(bwP * 5. + 0.5));
    mStepKern[param::VoxZ] = static_cast<int>(nearbyint(bwZ * 5. + 0.5));
  } else {
    LOG(error) << "kernel type " << type << " is not defined";
  }
  for (int i = param::VoxDim; i--;) {
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

//______________________________________________________________________________
void TrackResiduals::processResiduals()
{
  if (!mIsInitialized) {
    init();
  }
  for (int iSec = 0; iSec < param::NSectors2; ++iSec) {
    processSectorResiduals(iSec);
  }
}

//______________________________________________________________________________
void TrackResiduals::processSectorResiduals(int iSec)
{
  if (iSec < 0 || iSec > 35) {
    LOG(error) << "wrong sector: " << iSec;
    return;
  }
  LOG(info) << "processing sector residuals for sector " << iSec;
  if (!mIsInitialized) {
    init();
  }
  // open file and retrieve data tree (only local files are supported at the moment)
  std::string filename = "data/tmpDeltaSect";
  filename += std::to_string(iSec);
  filename += ".root";
  std::unique_ptr<TFile> flin = std::make_unique<TFile>(filename.c_str());
  if (!flin || flin->IsZombie()) {
    LOG(error) << "failed to open " << filename.c_str();
    return;
  }
  std::string treename = "ts";
  treename += std::to_string(iSec);
  ;
  std::unique_ptr<TTree> tree((TTree*)flin->Get(treename.c_str()));
  if (!tree) {
    LOG(error) << "did not find the data tree " << treename.c_str();
    return;
  }
  AliTPCDcalibRes::dts_t trkRes;
  auto* pTrkRes = &trkRes;
  tree->SetBranchAddress("dts", &pTrkRes);
  auto nPoints = tree->GetEntries();
  if (!nPoints) {
    LOG(warning) << "no entries found for sector " << iSec;
    flin->Close();
    return;
  }
  if (nPoints > param::MaxPointsPerSector) {
    nPoints = param::MaxPointsPerSector;
  }
  // initialize container holding results
  initResultsContainer(iSec);

  LOG(info) << "extracted " << nPoints << " of unbinned data";

  std::vector<bres_t>& secData = mVoxelResults[iSec];

  unsigned int nAccepted = 0;

  std::vector<float> dyData(nPoints);
  std::vector<float> dzData(nPoints);
  std::vector<float> tgSlpData(nPoints);
  std::vector<unsigned short> binData(nPoints);

  if (mPrintMem) {
    printMem();
  }

  // read input data into internal vectors
  for (int i = 0; i < nPoints; ++i) {
    tree->GetEntry(i);
    if (fabs(trkRes.tgSlp) >= param::MaxTgSlp) {
      continue;
    }
    dyData[nAccepted] = trkRes.dy;
    dzData[nAccepted] = trkRes.dz;
    tgSlpData[nAccepted] = trkRes.tgSlp;
    binData[nAccepted] = getGlbVoxBin(trkRes.bvox[param::VoxX], trkRes.bvox[param::VoxF], trkRes.bvox[param::VoxZ]);
    nAccepted++;
  }

  tree.release();
  flin->Close();

  if (mPrintMem) {
    printMem();
  }

  LOG(info) << "Done reading input data (accepted " << nAccepted << " points)";

  std::vector<size_t> binIndices(nAccepted);

  dyData.resize(nAccepted);
  dzData.resize(nAccepted);
  tgSlpData.resize(nAccepted);
  binData.resize(nAccepted);

  // convert to short and back to float to be compatible with AliRoot version
  std::vector<short> dyDataShort(nAccepted);
  std::vector<short> dzDataShort(nAccepted);
  std::vector<short> tgSlpDataShort(nAccepted);
  for (unsigned int i = 0; i < nAccepted; ++i) {
    dyDataShort[i] = short(dyData[i] * 0x7fff / param::MaxResid);
    dzDataShort[i] = short(dzData[i] * 0x7fff / param::MaxResid);
    tgSlpDataShort[i] = short(tgSlpData[i] * 0x7fff / param::MaxTgSlp);

    dyData[i] = dyDataShort[i] * param::MaxResid / 0x7fff;
    dzData[i] = dzDataShort[i] * param::MaxResid / 0x7fff;
    tgSlpData[i] = tgSlpDataShort[i] * param::MaxTgSlp / 0x7fff;
  }

  // sort in voxel increasing order
  stat::SortData(binData, binIndices);
  if (mPrintMem) {
    printMem();
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
  while (nProcessed < nAccepted) {
    // read all points, voxel by voxel
    int idx = binIndices[nProcessed];
    if (currVoxBin != binData[idx]) {
      if (nPointsInVox) {
        bres_t& resVox = secData[currVoxBin];
        processVoxelResiduals(dyVec, dzVec, tgVec, resVox);
      }
      currVoxBin = binData[idx];
      nPointsInVox = 0;
      dyVec.clear();
      dzVec.clear();
      tgVec.clear();
    }
    dyVec.push_back(dyData[idx]);
    dzVec.push_back(dzData[idx]);
    tgVec.push_back(tgSlpData[idx]);
    ++nPointsInVox;
    ++nProcessed;
  }
  if (nPointsInVox) {
    // process last voxel
    bres_t& resVox = secData[currVoxBin];
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
  while (nProcessed < nAccepted) {
    int idx = binIndices[nProcessed];
    if (currVoxBin != binData[idx]) {
      if (nPointsInVox) {
        bres_t& resVox = secData[currVoxBin];
        if (!getXBinIgnored(iSec, resVox.bvox[param::VoxX])) {
          processVoxelDispersions(tgVec, dyVec, resVox);
        }
      }
      currVoxBin = binData[idx];
      nPointsInVox = 0;
      dyVec.clear();
      tgVec.clear();
    }
    dyVec.push_back(dyData[idx]);
    tgVec.push_back(tgSlpData[idx]);
    ++nPointsInVox;
    ++nProcessed;
  }
  if (nPointsInVox) {
    // process last voxel
    bres_t& resVox = secData[currVoxBin];
    if (!getXBinIgnored(iSec, resVox.bvox[param::VoxX])) {
      processVoxelDispersions(tgVec, dyVec, resVox);
    }
  }
  // smooth dispersions
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (getXBinIgnored(iSec, ix)) {
      continue;
    }
    for (int iz = 0; iz < mNZBins; ++iz) {
      for (int ip = 0; ip < mNY2XBins; ++ip) {
        int voxBin = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = secData[voxBin];
        getSmoothEstimate(iSec, resVox.stat[param::VoxX], resVox.stat[param::VoxF], resVox.stat[param::VoxZ], resVox.DS, 0x1 << param::VoxV);
      }
    }
  }
  dumpResults(iSec);
}

//______________________________________________________________________________
void TrackResiduals::processVoxelResiduals(std::vector<float>& dy, std::vector<float>& dz, std::vector<float>& tg, bres_t& resVox)
{
  size_t nPoints = dy.size();
  //LOG(debug) << "processing voxel residuals for vox " << getGlbVoxBin(resVox.bvox) << " with " << nPoints << " points";
  if (nPoints < param::MinEntriesPerVoxel) {
    LOG(info) << "voxel " << getGlbVoxBin(resVox.bvox) << " is skipped due to too few entries (" << nPoints << " < " << param::MinEntriesPerVoxel << ")";
    return;
  }
  std::array<float, 7> zResults;
  resVox.flags = 0;
  std::vector<size_t> indices(dz.size());
  if (!stat::LTMUnbinned(dz, indices, zResults, param::LTMCut)) {
    return;
  }
  std::array<float, 2> res{ 0.f };
  std::array<float, 3> err{ 0.f };
  float sigMAD = fitPoly1Robust(tg, dy, res, err, param::LTMCut);
  if (sigMAD < 0) {
    return;
  }
  float corrErr = err[0] * err[2];
  corrErr = corrErr > 0 ? err[1] / std::sqrt(corrErr) : -999;
  //
  resVox.D[param::ResX] = -res[1];
  resVox.D[param::ResY] = res[0];
  resVox.D[param::ResZ] = zResults[1];
  resVox.E[param::ResX] = std::sqrt(err[2]);
  resVox.E[param::ResY] = std::sqrt(err[0]);
  resVox.E[param::ResZ] = zResults[4];
  resVox.EXYCorr = corrErr;
  resVox.D[param::ResD] = resVox.dYSigMAD = sigMAD; // later will be overwritten by real dispersion
  resVox.dZSigLTM = zResults[2];
  //
  //
  // at this point the actual COG for each voxel should be stored in resVox.stat

  resVox.flags |= param::DistDone;

  return;
}

void TrackResiduals::processVoxelDispersions(std::vector<float>& tg, std::vector<float>& dy, bres_t& resVox)
{
  size_t nPoints = tg.size();
  LOG(debug) << "processing voxel dispersions for vox " << getGlbVoxBin(resVox.bvox) << " with " << nPoints << " points";
  if (nPoints < 2) {
    return;
  }
  for (size_t i = nPoints; i--;) {
    dy[i] -= resVox.DS[param::ResY] - resVox.DS[param::ResX] * tg[i];
  }
  resVox.D[param::ResD] = getMAD2Sigma(dy);
  resVox.E[param::ResD] = resVox.D[param::ResD] / sqrt(2.f * nPoints); // a la gaussioan RMS error (very crude)
  resVox.flags |= param::DispDone;
}

//______________________________________________________________________________
int TrackResiduals::validateVoxels(int iSec)
{
  // apply voxel validation cuts
  // return number of good voxels for given sector
  int cntMasked = 0;  // number of voxels masked due to fit error and / or distribution sigmas
  int cntInvalid = 0; // number of voxels which were invalid before + masked ones
  mXBinsIgnore[iSec].reset();
  std::vector<bres_t>& secData = mVoxelResults[iSec];

  int cntMaskedFit = 0;
  int cntMaskedSigma = 0;

  // find bad voxels in sector
  for (int ix = 0; ix < mNXBins; ++ix) {
    int cntValid = 0;
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZBins; ++iz) {
        int binGlb = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = secData[binGlb];
        bool voxelOK = (resVox.flags & param::DistDone) && !(resVox.flags & param::Masked);
        if (voxelOK) {
          // check fit errors
          if (resVox.E[param::ResY] * resVox.E[param::ResY] > param::MaxFitErrY2 ||
              resVox.E[param::ResX] * resVox.E[param::ResX] > param::MaxFitErrX2 ||
              fabs(resVox.EXYCorr) > param::MaxFitCorrXY) {
            voxelOK = false;
            ++cntMaskedFit;
          }
          // check raw distribution sigmas
          if (resVox.dYSigMAD > param::MaxSigY ||
              resVox.dZSigLTM > param::MaxSigZ) {
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
          resVox.flags |= param::Masked;
        }
      } // loop over Z
    }   // loop over Y/X
    mValidFracXBins[iSec][ix] = static_cast<float>(cntValid) / (mNY2XBins * mNZBins);
    LOG(debug) << "sector " << iSec << ": xBin " << ix << " has " << mValidFracXBins[iSec][ix] * 100 << "\% of voxels valid";
  } // loop over X

  // mask X-bins which cannot be smoothed

  short nBadReg = 0;               // count bad regions (one or more consecutive bad X-bins)
  std::array<short, param::NPadRows> badStart; // to store indices to the beginnings of the bad regions
  std::array<short, param::NPadRows> badEnd;   // to store indices to the end of the bad regions
  bool prevBad = false;
  float fracBadRows = 0.f;
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (mValidFracXBins[iSec][ix] < param::MinValidVoxFracDrift) {
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
  if (fracBadRows > param::MaxFracBadRowsPerSector) {
    LOG(warning) << "sector " << iSec << ": Fraction of bad X-bins: " << fracBadRows << " -> masking whole sector";
    mXBinsIgnore[iSec].set();
  } else {
    for (int iBad = 0; iBad < nBadReg; ++iBad) {
      LOG(debug) << "masking bad region " << iBad;
      short badInReg = badEnd[iBad] - badStart[iBad] + 1;
      short badInNextReg = iBad < (nBadReg - 1) ? badEnd[iBad] - badStart[iBad] + 1 : 0;
      if (badInReg > param::MaxBadXBinsToCover) {
        // disable too large bad patches
        for (int i = 0; i < badInReg; ++i) {
          LOG(debug) << "disabling too large patch in bad region " << iBad << ", badStart(" << badStart[iBad] << "), i(" << i << ")";
          mXBinsIgnore[iSec].set(badStart[iBad] + i);
        }
      }
      if (badInNextReg > param::MaxBadXBinsToCover && (badStart[iBad + 1] - badEnd[iBad] - 1) < param::MinGoodXBinsToCover) {
        // disable too small isolated good patches`
        for (int i = badEnd[iBad] + 1; i < badStart[iBad + 1]; ++i) {
          LOG(debug) << "disabling too small good patch before bad region " << iBad + 1 << ", badStart(" << badEnd[iBad] << "), badEnd(" << badStart[iBad + 1] << ")";
          mXBinsIgnore[iSec].set(i);
        }
      }
    }
    if (nBadReg) {
      if (mXBinsIgnore[iSec].test(badStart[0]) && badStart[0] < param::MinGoodXBinsToCover) {
        // 1st good patch is too small
        for (int i = 0; i < badStart[0]; ++i) {
          LOG(debug) << "disabling too small first good patch badStart(0), badEnd(" << badStart[0] << ")";
          mXBinsIgnore[iSec].set(i);
        }
      }
      if (mXBinsIgnore[iSec].test(badStart[nBadReg - 1]) && (mNXBins - badEnd[nBadReg - 1] - 1) < param::MinGoodXBinsToCover) {
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
  std::vector<bres_t>& secData = mVoxelResults[iSec];
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (getXBinIgnored(iSec, ix)) {
      continue;
    }
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZBins; ++iz) {
        int voxBin = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = secData[voxBin];
        resVox.flags &= ~param::SmoothDone;
        bool res = getSmoothEstimate(resVox.bsec, resVox.stat[param::VoxX], resVox.stat[param::VoxF], resVox.stat[param::VoxZ], resVox.DS, (0x1 << param::VoxX | 0x1 << param::VoxF | 0x1 << param::VoxZ));
        if (!res) {
          mNSmoothingFailedBins[iSec]++;
        } else {
          resVox.flags |= param::SmoothDone;
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
      for (int iz = 0; iz < mNZBins; ++iz) {
        int voxBin = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = secData[voxBin];
        if (!(resVox.flags & param::SmoothDone)) {
          continue;
        }
        resVox.DS[param::ResZ] += resVox.stat[param::VoxZ] * resVox.DS[param::ResX]; // remove slope*dX contribution from dZ
        resVox.D[param::ResZ] += resVox.stat[param::VoxZ] * resVox.DS[param::ResX];  // remove slope*dX contribution from dZ
      }
    }
  }
}

bool TrackResiduals::getSmoothEstimate(int iSec, float x, float p, float z, std::array<float, param::ResDim>& res, int whichDim)
{
  // get smooth estimate for distortions for point in sector coordinates
  /// \todo correct use of the symmetric matrix should speed up the code

  std::array<int, param::VoxDim> minPointsDir{ 0 }; // min number of points per direction
  const float kTrialStep = 0.5;
  std::array<bool, param::ResDim> doDim{ false };
  for (int i = 0; i < param::ResDim; ++i) {
    doDim[i] = (whichDim & (0x1 << i)) > 0;
    if (doDim[i]) {
      res[i] = 0.f;
    }
  }

  int matSize = param::SmtLinDim;
  for (int i = 0; i < param::VoxDim; ++i) {
    minPointsDir[i] = 3; // for pol1 smoothing require at least 3 points
    if (mSmoothPol2[i]) {
      ++minPointsDir[i];
      ++matSize;
    }
  }

  int ix0, ip0, iz0;
  findVoxel(x, p, iSec < param::NSectors ? z : -z, ix0, ip0, iz0); // find nearest voxel
  std::vector<bres_t>& secData = mVoxelResults[iSec];
  int binCenter = getGlbVoxBin(ix0, ip0, iz0); // global bin of nearest voxel
  bres_t& voxCenter = secData[binCenter];      // nearest voxel
  LOG(debug) << "getting smooth estimate around voxel " << binCenter;

  // cache
  // \todo maybe a 1-D cache would be more efficient?
  std::array<std::array<double, param::MaxSmtDim*(param::MaxSmtDim + 1) / 2>, param::ResDim> cmat;
  int maxNeighb = 10 * 10 * 10;
  std::vector<bres_t*> currVox;
  currVox.reserve(maxNeighb);
  std::vector<float> currCache;
  currCache.reserve(maxNeighb * param::VoxHDim);

  std::array<int, param::VoxDim> maxTrials;
  maxTrials[param::VoxZ] = mNZBins / 2;
  maxTrials[param::VoxF] = mNY2XBins / 2;
  maxTrials[param::VoxX] = param::MaxBadXBinsToCover * 2;

  std::array<int, param::VoxDim> trial{ 0 };

  while (true) {
    std::fill(mLastSmoothingRes.begin(), mLastSmoothingRes.end(), 0);
    memset(&cmat[0][0], 0, sizeof(cmat));

    int nbOK = 0; // accounted neighbours

    float stepX = mStepKern[param::VoxX] * (1. + kTrialStep * trial[param::VoxX]);
    float stepF = mStepKern[param::VoxF] * (1. + kTrialStep * trial[param::VoxF]);
    float stepZ = mStepKern[param::VoxZ] * (1. + kTrialStep * trial[param::VoxZ]);

    if (!(voxCenter.flags & param::DistDone) || (voxCenter.flags & param::Masked) || getXBinIgnored(iSec, ix0)) {
      // closest voxel has no data -> increase smoothing step
      stepX += kTrialStep * mStepKern[param::VoxX];
      stepF += kTrialStep * mStepKern[param::VoxF];
      stepZ += kTrialStep * mStepKern[param::VoxZ];
    }

    // effective kernel widths accounting for the increased bandwidth at the edges and missing data
    float kWXI = getDXI(ix0) * mKernelWInv[param::VoxX] * mStepKern[param::VoxX] / stepX;
    float kWFI = getDY2XI(ix0) * mKernelWInv[param::VoxF] * mStepKern[param::VoxF] / stepF;
    float kWZI = getDZ2XI() * mKernelWInv[param::VoxZ] * mStepKern[param::VoxZ] / stepZ;
    int iStepX = static_cast<int>(nearbyint(stepX + 0.5));
    int iStepF = static_cast<int>(nearbyint(stepF + 0.5));
    int iStepZ = static_cast<int>(nearbyint(stepZ + 0.5));
    int ixMin = ix0 - iStepX;
    int ixMax = ix0 + iStepX;
    if (ixMin < 0) {
      ixMin = 0;
      ixMax = std::min(static_cast<int>(nearbyint(ix0 + stepX * mKernelScaleEdge[param::VoxX])), mNXBins - 1);
      kWXI /= mKernelScaleEdge[param::VoxX];
    }
    if (ixMax >= mNXBins) {
      ixMax = mNXBins - 1;
      ixMin = std::max(static_cast<int>(nearbyint(ix0 - stepX * mKernelScaleEdge[param::VoxX])), 0);
      kWXI /= mKernelScaleEdge[param::VoxX];
    }

    int ipMin = ip0 - iStepF;
    int ipMax = ip0 + iStepF;
    if (ipMin < 0) {
      ipMin = 0;
      ipMax = std::min(static_cast<int>(nearbyint(ip0 + stepF * mKernelScaleEdge[param::VoxF])), mNY2XBins - 1);
      kWFI /= mKernelScaleEdge[param::VoxF];
    }
    if (ipMax >= mNY2XBins) {
      ipMax = mNY2XBins - 1;
      ipMin = std::max(static_cast<int>(nearbyint(ip0 - stepF * mKernelScaleEdge[param::VoxF])), 0);
      kWFI /= mKernelScaleEdge[param::VoxF];
    }

    int izMin = iz0 - iStepZ;
    int izMax = iz0 + iStepZ;
    if (izMin < 0) {
      izMin = 0;
      izMax = std::min(static_cast<int>(nearbyint(iz0 + stepZ * mKernelScaleEdge[param::VoxZ])), mNZBins - 1);
      kWZI /= mKernelScaleEdge[param::VoxZ];
    }
    if (izMax >= mNZBins) {
      izMax = mNZBins - 1;
      izMin = std::max(static_cast<int>(nearbyint(iz0 - stepZ * mKernelScaleEdge[param::VoxZ])), 0);
      kWZI /= mKernelScaleEdge[param::VoxZ];
    }

    std::vector<unsigned short> nOccX(ixMax - ixMin + 1, 0);
    std::vector<unsigned short> nOccF(ipMax - ipMin + 1, 0);
    std::vector<unsigned short> nOccZ(izMax - izMin + 1, 0);

    int nbCheck = (ixMax - ixMin + 1) * (ipMax - ipMin + 1) * (izMax - izMin + 1);
    if (nbCheck >= maxNeighb) {
      maxNeighb = nbCheck + 100;
      currCache.reserve(maxNeighb * param::VoxHDim);
      currVox.reserve(maxNeighb);
    }
    std::array<double, 3> u2Vec;

    // first loop, check presence of enough points
    for (int ix = ixMin; ix <= ixMax; ++ix) {
      for (int ip = ipMin; ip <= ipMax; ++ip) {
        for (int iz = izMin; iz <= izMax; ++iz) {
          int binNb = getGlbVoxBin(ix, ip, iz);
          bres_t& voxNb = secData[binNb];
          if (!(voxNb.flags & param::DistDone) ||
              (voxNb.flags & param::Masked) ||
              getXBinIgnored(iSec, ix)) {
            // skip voxels w/o data
            continue;
          }
          // estimate weighted distance
          float dx = voxNb.stat[param::VoxX] - x;
          float df = voxNb.stat[param::VoxF] - p;
          float dz = voxNb.stat[param::VoxZ] - z;
          float dxw = dx * kWXI;
          float dfw = df * kWFI;
          float dzw = dz * kWZI;
          u2Vec[0] = dxw * dxw;
          u2Vec[1] = dfw * dfw;
          u2Vec[2] = dzw * dzw;
          double kernelWeight = getKernelWeight(u2Vec, mKernelType);
          if (kernelWeight < 1e-6) {
            continue;
          }
          // new point is validated
          ++nOccX[ix - ixMin];
          ++nOccF[ip - ipMin];
          ++nOccZ[iz - izMin];
          currVox[nbOK] = &voxNb;
          currCache[nbOK * param::VoxHDim + param::VoxX] = dx;
          currCache[nbOK * param::VoxHDim + param::VoxF] = df;
          currCache[nbOK * param::VoxHDim + param::VoxZ] = dz;
          currCache[nbOK * param::VoxHDim + param::VoxV] = kernelWeight;
          ++nbOK;
        }
      }
    }

    // check if we have enough points in every dimension
    std::array<int, param::VoxDim> nPoints{ 0 };
    for (int i = ixMax - ixMin + 1; i--;) {
      if (nOccX[i]) {
        ++nPoints[param::VoxX];
      }
    }
    for (int i = ipMax - ipMin + 1; i--;) {
      if (nOccF[i]) {
        ++nPoints[param::VoxF];
      }
    }
    for (int i = izMax - izMin + 1; i--;) {
      if (nOccZ[i]) {
        ++nPoints[param::VoxZ];
      }
    }
    bool enoughPoints = true;
    std::array<bool, param::VoxDim> incrDone{ false };
    for (int i = 0; i < param::VoxDim; ++i) {
      if (nPoints[i] < minPointsDir[i]) {
        // need to extend smoothing neighbourhood
        enoughPoints = false;
        if (trial[i] < maxTrials[i] && !incrDone[i]) {
          // try to increment only missing direction
          ++trial[i];
          incrDone[i] = true;
        } else if (trial[i] == maxTrials[i]) {
          // cannot increment missing direction, try others
          for (int j = param::VoxDim; j--;) {
            if (i != j && trial[j] < maxTrials[j] && !incrDone[j]) {
              ++trial[j];
              incrDone[j] = true;
            }
          }
        }
      }
    }

    if (!enoughPoints) {
      if (!(incrDone[param::VoxX] || incrDone[param::VoxF] || incrDone[param::VoxZ])) {
        LOG(error) << "trial limit reached, skipping this voxel";
        return false;
      }
      LOG(debug) << "sector " << iSec << ": increasing filter bandwidth around voxel " << binCenter;
      //printf("Sector:%2d x=%.2f y/x=%.2f z/x=%.2f (iX: %d iY2X:%d iZ2X:%d)\n", iSec, x, p, z, ix0, ip0, iz0);
      //printf("not enough neighbours (need min %d) %d %d %d (tot: %d) | Steps: %.1f %.1f %.1f\n", 2, nPoints[param::VoxX], nPoints[param::VoxF], nPoints[param::VoxZ], nbOK, stepX, stepF, stepZ);
      //printf("trying to increase filter bandwidth (trialXFZ: %d %d %d)\n", trial[param::VoxX], trial[param::VoxF], trial[param::VoxZ]);
      continue;
    }

    // now fill matrices and solve
    for (int iNb = 0; iNb < nbOK; ++iNb) {
      double wiCache = currCache[iNb * param::VoxHDim + param::VoxV];
      double dxi = currCache[iNb * param::VoxHDim + param::VoxX];
      double dfi = currCache[iNb * param::VoxHDim + param::VoxF];
      double dzi = currCache[iNb * param::VoxHDim + param::VoxZ];
      double dxi2 = dxi * dxi;
      double dfi2 = dfi * dfi;
      double dzi2 = dzi * dzi;
      const bres_t* voxNb = currVox[iNb];
      for (int iDim = 0; iDim < param::ResDim; ++iDim) {
        if (!doDim[iDim]) {
          continue;
        }
        double vi = voxNb->D[iDim];
        double wi = wiCache;
        if (mUseErrInSmoothing && fabs(voxNb->E[iDim]) > 1e-6) {
          // account for point error apart from kernel value
          wi /= (voxNb->E[iDim] * voxNb->E[iDim]);
        }
        std::array<double, param::MaxSmtDim*(param::MaxSmtDim + 1) / 2>& cmatD = cmat[iDim];
        double* rhsD = &mLastSmoothingRes[iDim * param::MaxSmtDim];
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
        if (mSmoothPol2[param::VoxX]) {
          cmatD[iMat++] += wi * dxi2;
          cmatD[iMat++] += wi * dxi * dxi2;
          cmatD[iMat++] += wi * dfi * dxi2;
          cmatD[iMat++] += wi * dzi * dxi2;
          cmatD[iMat++] += wi * dxi2 * dxi2;
          rhsD[iRhs++] += wi * dxi2 * vi;
        }
        if (mSmoothPol2[param::VoxF]) {
          cmatD[iMat++] += wi * dfi2;
          cmatD[iMat++] += wi * dxi * dfi2;
          cmatD[iMat++] += wi * dfi * dfi2;
          cmatD[iMat++] += wi * dzi * dfi2;
          cmatD[iMat++] += wi * dxi2 * dfi2;
          cmatD[iMat++] += wi * dfi2 * dfi2;
          rhsD[iRhs++] += wi * dfi2 * vi;
        }
        if (mSmoothPol2[param::VoxZ]) {
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
    for (int iDim = 0; iDim < param::ResDim; ++iDim) {
      if (!doDim[iDim]) {
        continue;
      }
      matrix.Zero(); // reset matrix
      std::array<double, param::MaxSmtDim*(param::MaxSmtDim + 1) / 2>& cmatD = cmat[iDim];
      double* rhsD = &mLastSmoothingRes[iDim * param::MaxSmtDim];
      short iMat = -1;
      short iRhs = -1;
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
      if (mSmoothPol2[param::VoxX]) {
        const unsigned int colLim = (++row) + 1;
        for (int iCol = 0; iCol < colLim; ++iCol) {
          matrix(row, iCol) = cmatD[++iMat];
          matrix(iCol, row) = matrix(row, iCol);
        }
      }
      if (mSmoothPol2[param::VoxF]) {
        const unsigned int colLim = (++row) + 1;
        for (int iCol = 0; iCol < colLim; ++iCol) {
          matrix(row, iCol) = cmatD[++iMat];
          matrix(iCol, row) = matrix(row, iCol);
        }
      }
      if (mSmoothPol2[param::VoxZ]) {
        const unsigned int colLim = (++row) + 1;
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
        for (int i = param::VoxDim; i--;) {
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

double TrackResiduals::getKernelWeight(std::array<double, 3> u2vec, int kernelType) const
{
  double w = 1.;
  if (kernelType == param::EpanechnikovKernel) {
    for (size_t i = u2vec.size(); i--;) {
      if (u2vec[i] > 1) {
        return 0.;
      }
      w *= 3. / 4. * (1. - u2vec[i]);
    }
  } else if (kernelType == param::GaussianKernel) {
    double u2 = 0.;
    for (size_t i = u2vec.size(); i--;) {
      u2 += u2vec[i];
    }
    w = u2 < param::MaxGaussStdDev * param::MaxGaussStdDev * u2vec.size() ? std::exp(-u2) / std::sqrt(2. * M_PI) : 0;
  } else {
    LOG(error) << "kernel type " << kernelType << " is not defined";
  }
  return w;
}

///////////////////////////////////////////////////////////////////////////////
///
/// fitting + statistics helper functions
///
///////////////////////////////////////////////////////////////////////////////

float TrackResiduals::fitPoly1Robust(std::vector<float>& x, std::vector<float>& y, std::array<float, 2>& res, std::array<float, 3>& err, float cutLTM)
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
  if (!stat::LTMUnbinned(y, indY, yResults, cutLTM)) {
    return -1;
  }
  // rearrange used events in increasing order
  stat::Reorder(y, indY);
  stat::Reorder(x, indY);
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
  stat::SortData(ycm, indices);
  stat::Reorder(ycm, indices);
  stat::Reorder(y, indices);
  stat::Reorder(x, indices);
  //
  // robust estimate of sigma after crude slope correction
  float sigMAD = getMAD2Sigma({ ycm.begin() + vecOffset, ycm.begin() + vecOffset + nPointsUsed });
  // find LTM estimate matching to sigMAD, keaping at least given fraction
  if (!stat::LTMUnbinnedSig(ycm, indY, yResults, param::MinFracLTM, sigMAD, true)) {
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
void TrackResiduals::medFit(int nPoints, int offset, const std::vector<float>& x, const std::vector<float>& y, float& a, float& b, std::array<float, 3>& err, float delI)
{
  // fitting a straight line y(x|a, b) = a + b * x
  // to given x and y data minimizing the absolute deviation
  float aa, bb, chi2 = 0.f;
  if (nPoints < 2) {
    a = b = 0.f;
    err[0] = err[1] = err[2] = 999.f;
    return;
  }
  if (!delI) {
    // no initial values provided, do least squares minimization as first guess
    float sx = 0.f, sxx = 0.f, sy = 0.f, sxy = 0.f, del;
    for (int j = nPoints + offset; j-- > offset;) { // same order as in AliRoot version such that resulting sums are identical
      sx += x[j];
      sxx += x[j] * x[j];
      sy += y[j];
      sxy += x[j] * y[j];
    }
    del = nPoints * sxx - sx * sx;
    delI = 1. / del;
    aa = (sxx * sy - sx * sxy) * delI;
    bb = (nPoints * sxy - sx * sy) * delI;
    err[0] = sxx * delI;
    err[1] = sx * delI;
    err[2] = nPoints * delI;
  } else {
    // initial values provided
    aa = a;
    bb = b;
  }
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
    if (fabs(f1 - f2) < param::FloatEps) {
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

float TrackResiduals::roFunc(int nPoints, int offset, const std::vector<float>& x, const std::vector<float>& y, float b, float& aa)
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
        vecTmp[j + 1] = v;
      }
    }
    aa = (nPoints & 0x1) ? vecTmp[nPointsHalf] : .5f * (vecTmp[nPointsHalf - 1] + vecTmp[nPointsHalf]);
  } else {
    /*
    if (nPoints & 0x1) {
      std::nth_element(vecTmp.begin(), vecTmp.begin() + nPointsHalf, vecTmp.end());
      aa = vecTmp[nPointsHalf];
    }
    else {
      std::nth_element(vecTmp.begin(), vecTmp.begin() + nPointsHalf - 1, vecTmp.end());
      aa = vecTmp[nPointsHalf -  1];
      std::nth_element(vecTmp.begin(), vecTmp.begin() + nPointsHalf, vecTmp.end());
      aa += vecTmp[nPointsHalf];
      aa *= 0.5;
    }
    */
    aa = (nPoints & 0x1) ? selectKthMin(nPointsHalf, vecTmp) : .5f * (selectKthMin(nPointsHalf - 1, vecTmp) + selectKthMin(nPointsHalf, vecTmp));
  }
  for (int j = nPoints; j-- > 0;) {
    float d = y[j + offset] - (b * x[j + offset] + aa);
    if (y[j + offset] != 0.f) {
      d /= fabs(y[j + offset]);
    }
    if (fabs(d) > param::FloatEps) {
      sum += (d >= 0.f ? x[j + offset] : -x[j + offset]);
    }
  }
  return sum;
}

//______________________________________________________________________________
float TrackResiduals::selectKthMin(const int k, std::vector<float>& data)
{
  // Returns the k th smallest value in the vector. The input vector will be rearranged
  // to have this value in location data[k] , with all smaller elements moved before it
  // (in arbitrary order) and all larger elements after (also in arbitrary order).
  // From Numerical Recipes in C++ (paragraph 8.5)

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
float TrackResiduals::getMAD2Sigma(std::vector<float> data)
{
  // Sigma calculated from median absolute deviations
  // see: https://en.wikipedia.org/wiki/Median_absolute_deviation
  // the data is passed by value (copied!), such that the original vector
  // is not rearranged

  int nPoints = data.size();

  if (nPoints < 2) {
    return 0;
  }
  // sort the input
  std::sort(data.begin(), data.end());

  // calculate median
  bool oddNumberOfEntries = nPoints & 0x1;
  int mid = nPoints / 2;
  float medianOfData;
  if (oddNumberOfEntries) {
    medianOfData = data[mid];
  } else {
    medianOfData = (data[mid - 1] + data[mid]) / 2.;
  }

  // calculate absolute deviations to median
  for (auto& entry : data) {
    entry = fabs(entry - medianOfData);
  }
  std::sort(data.begin(), data.end());

  // calculate median of abs deviations
  float medianAbsDeviation;
  if (oddNumberOfEntries) {
    medianAbsDeviation = data[mid];
  } else {
    medianAbsDeviation = (data[mid - 1] + data[mid]) / 2.;
  }

  float k = 1.4826f; // scale factor for normally distributed data
  return k * medianAbsDeviation;
}

///////////////////////////////////////////////////////////////////////////////
///
/// debugging
///
///////////////////////////////////////////////////////////////////////////////

void TrackResiduals::dumpToFile(const std::vector<float>& vec, const std::string fName = "output.txt") const
{
  std::ofstream fOut(fName.data());
  if (fOut.is_open()) {
    for (const auto& elem : vec) {
      fOut << std::fixed << std::setprecision(std::numeric_limits<float>::digits10 + 1) << elem << std::endl;
    }
    fOut.close();
  }
}

void TrackResiduals::createOutputFile()
{
  mFileOut = std::make_unique<TFile>("voxelResultsO2.root", "recreate");
  mTreeOut = std::make_unique<TTree>("debugTree", "voxel results");
  mTreeOut->Branch("voxRes", &(mVoxelResults[0][0]), "D[4]/F:E[4]/F:DS[4]/F:DC[4]/F:EXYCorr/F:dySigMAD/F:dZSigLTM/F:stat[4]/F:bvox[3]/b:bsec/b:flags/b");
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
    for (int iBin = 0; iBin < mNVoxPerSector; ++iBin) {
      (mTreeOut->GetBranch("voxRes"))->SetAddress(&(mVoxelResults[iSec][iBin]));
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
