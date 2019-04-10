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
#include "CommonConstants/MathConstants.h"
#include "MathUtils/MathBase.h"

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

#define TPC_RUN2

using namespace o2::TPC;

///////////////////////////////////////////////////////////////////////////////
///
/// initialization + binning
///
///////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void TrackResiduals::init()
{
  // initialize binning
  initBinning();

  // initialize results container
  for (int i = 0; i < SECTORSPERSIDE * SIDES; i++) {
    mVoxelResults[i].resize(mNVoxPerSector);
  }
  mSmoothPol2[VoxX] = true;
  mSmoothPol2[VoxF] = true;
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
    mDXI = mNXBins / (param::MaxX - param::MinX[0]);
    mDX = 1.0f / mDXI;
    mUniformBins[VoxX] = true;
  } else {
    // binning per pad row
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
  mUniformBins[VoxF] = true;
  //
  // Z binning
  mDZI = mNZ2XBins / sMaxZ2X;
  mDZ = 1.0f / mDZI;
  mUniformBins[VoxZ] = true;
  //
  mNVoxPerSector = mNY2XBins * mNZ2XBins * mNXBins;
}

//______________________________________________________________________________
void TrackResiduals::initResultsContainer(int iSec)
{
  for (int ix = 0; ix < mNXBins; ++ix) {
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZ2XBins; ++iz) {
        int binGlb = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = mVoxelResults[iSec][binGlb];
        resVox.bvox[VoxX] = ix;
        resVox.bvox[VoxF] = ip;
        resVox.bvox[VoxZ] = iz;
        resVox.bsec = iSec;
        // COG estimates are set to the bin center by default
        getVoxelCoordinates(resVox.bsec, resVox.bvox[VoxX], resVox.bvox[VoxF], resVox.bvox[VoxZ],
                            resVox.stat[VoxX], resVox.stat[VoxF], resVox.stat[VoxZ]);
      }
    }
  }
  LOG(info) << "initialized the container for the main results";
}

//______________________________________________________________________________
void TrackResiduals::reset()
{
  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    mXBinsIgnore[iSec].reset();
    std::fill(mVoxelResults[iSec].begin(), mVoxelResults[iSec].end(), bres_t());
    std::fill(mValidFracXBins[iSec].begin(), mValidFracXBins[iSec].end(), 0);
  }
}

//______________________________________________________________________________
int TrackResiduals::getXBin(float x) const
{
  // convert x to bin ID, following pad row widths
  if (mUniformBins[VoxX]) {
    int ix = (x - param::MinX[0]) * mDXI;
    if (ix < 0) {
      ix = 0;
    }
    return (ix < mNXBins) ? ix : mNXBins - 1;
  } else {
    int ix;
    if (x < param::MinX[1] - .5f * param::ROCDX[0]) {
      // we are in the IROC
      ix = (x - param::MinX[0] + .5f * param::RowDX[0]) / param::RowDX[0];
      return (ix < 0) ? 0 : std::min(ix, param::NRowsPerROC[0] - 1);
    } else if (x > param::MinX[param::NROCTypes - 1] - .5f * param::ROCDX[param::NROCTypes - 2]) {
      // we are in the last OROC
      ix = param::NRowsAccumulated[param::NROCTypes - 2] + (x - param::MinX[param::NROCTypes - 1] + .5f * param::RowDX[param::NROCTypes - 1]) / param::RowDX[param::NROCTypes - 1];
      return (ix < param::NRowsAccumulated[param::NROCTypes - 2]) ? param::NRowsAccumulated[param::NROCTypes - 2] : std::min(ix, param::NPadRows - 1);
    }
#ifdef TPC_RUN2
    else {
      // we are in OROC1
      ix = param::NRowsPerROC[0] + (x - param::MinX[1] + .5f * param::RowDX[1]) / param::RowDX[1];
      return (ix < param::NRowsPerROC[0]) ? param::NRowsPerROC[0] : std::min(ix, param::NRowsAccumulated[1] - 1);
    }
#else
    else if (x < param::MinX[2] - .5f * param::ROCDX[1]) {
      // we are in OROC1
      ix = param::NRowsPerROC[0] + (x - param::MinX[1] + .5f * param::RowDX[1]) / param::RowDX[1];
      return (ix < param::NRowsPerROC[0]) ? param::NRowsPerROC[0] : std::min(ix, param::NRowsAccumulated[1] - 1);
    } else {
      // we are in OROC2
      ix = param::NRowsAccumulated[1] + (x - param::MinX[2] + .5f * param::RowDX[2]) / param::RowDX[2];
      return (ix < param::NRowsAccumulated[1]) ? param::NRowsAccumulated[1] : std::min(ix, param::NRowsAccumulated[2] - 1);
    }
#endif
  }
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

//______________________________________________________________________________
void TrackResiduals::processResiduals()
{
  if (!mIsInitialized) {
    init();
  }
  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
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
  std::string filename = mLocalResFileName + std::to_string(iSec) + ".root";
  std::unique_ptr<TFile> flin = std::make_unique<TFile>(filename.c_str());
  if (!flin || flin->IsZombie()) {
    LOG(error) << "failed to open " << filename.c_str();
    return;
  }
  std::string treename = "ts" + std::to_string(iSec);
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
  if (nPoints > mMaxPointsPerSector) {
    nPoints = mMaxPointsPerSector;
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
    if (fabs(trkRes.tgSlp) >= mMaxTgSlp) {
      continue;
    }
    dyData[nAccepted] = trkRes.dy;
    dzData[nAccepted] = trkRes.dz;
    tgSlpData[nAccepted] = trkRes.tgSlp;
    binData[nAccepted] = getGlbVoxBin(trkRes.bvox[VoxX], trkRes.bvox[VoxF], trkRes.bvox[VoxZ]);
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
    dyDataShort[i] = short(dyData[i] * 0x7fff / sMaxResid);
    dzDataShort[i] = short(dzData[i] * 0x7fff / sMaxResid);
    tgSlpDataShort[i] = short(tgSlpData[i] * 0x7fff / mMaxTgSlp);

    dyData[i] = dyDataShort[i] * sMaxResid / 0x7fff;
    dzData[i] = dzDataShort[i] * sMaxResid / 0x7fff;
    tgSlpData[i] = tgSlpDataShort[i] * mMaxTgSlp / 0x7fff;
  }

  // sort in voxel increasing order
  o2::math_utils::math_base::SortData(binData, binIndices);
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
        if (!getXBinIgnored(iSec, resVox.bvox[VoxX])) {
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
        bres_t& resVox = secData[voxBin];
        getSmoothEstimate(iSec, resVox.stat[VoxX], resVox.stat[VoxF], resVox.stat[VoxZ], resVox.DS, 0x1 << VoxV);
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
  if (nPoints < mMinEntriesPerVoxel) {
    LOG(info) << "voxel " << getGlbVoxBin(resVox.bvox) << " is skipped due to too few entries (" << nPoints << " < " << mMinEntriesPerVoxel << ")";
    return;
  }
  std::array<float, 7> zResults;
  resVox.flags = 0;
  std::vector<size_t> indices(dz.size());
  if (!o2::math_utils::math_base::LTMUnbinned(dz, indices, zResults, mLTMCut)) {
    LOG(debug) << "failed trimming input array for voxel " << getGlbVoxBin(resVox.bvox);
    return;
  }
  std::array<float, 2> res{ 0.f };
  std::array<float, 3> err{ 0.f };
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
  //
  //
  // at this point the actual COG for each voxel should be stored in resVox.stat

  resVox.flags |= DistDone;

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
  std::vector<bres_t>& secData = mVoxelResults[iSec];

  int cntMaskedFit = 0;
  int cntMaskedSigma = 0;

  // find bad voxels in sector
  for (int ix = 0; ix < mNXBins; ++ix) {
    int cntValid = 0;
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZ2XBins; ++iz) {
        int binGlb = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = secData[binGlb];
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
    LOG(debug) << "sector " << iSec << ": xBin " << ix << " has " << mValidFracXBins[iSec][ix] * 100 << "\% of voxels valid";
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
  std::vector<bres_t>& secData = mVoxelResults[iSec];
  for (int ix = 0; ix < mNXBins; ++ix) {
    if (getXBinIgnored(iSec, ix)) {
      continue;
    }
    for (int ip = 0; ip < mNY2XBins; ++ip) {
      for (int iz = 0; iz < mNZ2XBins; ++iz) {
        int voxBin = getGlbVoxBin(ix, ip, iz);
        bres_t& resVox = secData[voxBin];
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
        bres_t& resVox = secData[voxBin];
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

  std::array<int, VoxDim> minPointsDir{ 0 }; // min number of points per direction
  const float kTrialStep = 0.5;
  std::array<bool, ResDim> doDim{ false };
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
  std::vector<bres_t>& secData = mVoxelResults[iSec];
  int binCenter = getGlbVoxBin(ix0, ip0, iz0); // global bin of nearest voxel
  bres_t& voxCenter = secData[binCenter];      // nearest voxel
  LOG(debug) << "getting smooth estimate around voxel " << binCenter;

  // cache
  // \todo maybe a 1-D cache would be more efficient?
  std::array<std::array<double, sMaxSmtDim*(sMaxSmtDim + 1) / 2>, ResDim> cmat;
  int maxNeighb = 10 * 10 * 10;
  std::vector<bres_t*> currVox;
  currVox.reserve(maxNeighb);
  std::vector<float> currCache;
  currCache.reserve(maxNeighb * VoxHDim);

  std::array<int, VoxDim> maxTrials;
  maxTrials[VoxZ] = mNZ2XBins / 2;
  maxTrials[VoxF] = mNY2XBins / 2;
  maxTrials[VoxX] = mMaxBadXBinsToCover * 2;

  std::array<int, VoxDim> trial{ 0 };

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
    float kWFI = getDY2XI(ix0) * mKernelWInv[VoxF] * mStepKern[VoxF] / stepF;
    float kWZI = getDZ2XI() * mKernelWInv[VoxZ] * mStepKern[VoxZ] / stepZ;
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
          bres_t& voxNb = secData[binNb];
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
    std::array<int, VoxDim> nPoints{ 0 };
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
    std::array<bool, VoxDim> incrDone{ false };
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
      const bres_t* voxNb = currVox[iNb];
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
      if (mSmoothPol2[VoxX]) {
        const unsigned int colLim = (++row) + 1;
        for (int iCol = 0; iCol < colLim; ++iCol) {
          matrix(row, iCol) = cmatD[++iMat];
          matrix(iCol, row) = matrix(row, iCol);
        }
      }
      if (mSmoothPol2[VoxF]) {
        const unsigned int colLim = (++row) + 1;
        for (int iCol = 0; iCol < colLim; ++iCol) {
          matrix(row, iCol) = cmatD[++iMat];
          matrix(iCol, row) = matrix(row, iCol);
        }
      }
      if (mSmoothPol2[VoxZ]) {
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
  if (!o2::math_utils::math_base::LTMUnbinned(y, indY, yResults, cutLTM)) {
    return -1;
  }
  // rearrange used events in increasing order
  o2::math_utils::math_base::Reorder(y, indY);
  o2::math_utils::math_base::Reorder(x, indY);
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
  o2::math_utils::math_base::SortData(ycm, indices);
  o2::math_utils::math_base::Reorder(ycm, indices);
  o2::math_utils::math_base::Reorder(y, indices);
  o2::math_utils::math_base::Reorder(x, indices);
  //
  // robust estimate of sigma after crude slope correction
  float sigMAD = getMAD2Sigma({ ycm.begin() + vecOffset, ycm.begin() + vecOffset + nPointsUsed });
  // find LTM estimate matching to sigMAD, keaping at least given fraction
  if (!o2::math_utils::math_base::LTMUnbinnedSig(ycm, indY, yResults, mMinFracLTM, sigMAD, true)) {
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
        vecTmp[j + 1] = v;
      }
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
    /*
    aa = (nPoints & 0x1) ? selectKthMin(nPointsHalf, vecTmp) : .5f * (selectKthMin(nPointsHalf - 1, vecTmp) + selectKthMin(nPointsHalf, vecTmp));
    */
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
