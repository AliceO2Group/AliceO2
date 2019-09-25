// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MatLayerCyl.cxx
/// \brief Implementation of single cylindrical material layer

#include "DetectorsBase/MatLayerCyl.h"
#include "MathUtils/Utils.h"
#include "CommonConstants/MathConstants.h"
#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
#include "DetectorsBase/GeometryManager.h"
#include "GPUCommonLogger.h"
#endif

using namespace o2::base;
using flatObject = o2::gpu::FlatObject;

//________________________________________________________________________________
MatLayerCyl::MatLayerCyl() : mNZBins(0), mNPhiBins(0), mNPhiSlices(0), mZHalf(0.f), mRMin2(0.f), mRMax2(0.f), mDZ(0.f), mDZInv(0.f), mDPhi(0.f), mDPhiInv(0.f), mPhiBin2Slice(nullptr), mSliceCos(nullptr), mSliceSin(nullptr), mCells(nullptr)
{
}

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
//________________________________________________________________________________
MatLayerCyl::MatLayerCyl(float rMin, float rMax, float zHalfSpan, float dzMin, float drphiMin)
{
  // main constructor
  initSegmentation(rMin, rMax, zHalfSpan, dzMin, drphiMin);
}

//________________________________________________________________________________
void MatLayerCyl::initSegmentation(float rMin, float rMax, float zHalfSpan, float dzMin, float drphiMin)
{
  // init and precalculate aux parameters. The initialization is done in the own memory.
  if (dzMin < 0.001f) {
    dzMin = 0.001f;
  }
  if (drphiMin < 0.001f) {
    drphiMin = 0.001f;
  }
  float peri = (rMax + rMin) * o2::constants::math::PI;
  int nz = 2 * zHalfSpan / dzMin, nphi = peri / drphiMin;
  initSegmentation(rMin, rMax, zHalfSpan, nz < 1 ? 1 : nz, nphi < 1 ? 1 : nphi);
}

//________________________________________________________________________________
void MatLayerCyl::initSegmentation(float rMin, float rMax, float zHalfSpan, int nz, int nphi)
{
  // Init and precalculate aux parameters. The initialization is done in the own memory.
  assert(mConstructionMask == NotConstructed);
  assert(rMin < rMax);
  assert(nz > 0);
  assert(nphi > 0);

  // book local storage
  auto sz = estimateFlatBufferSize(nphi, nphi, nz);

  //--------------????
  mFlatBufferPtr = mFlatBufferContainer = new char[sz];
  mFlatBufferSize = sz;
  //--------------????

  mRMin2 = rMin * rMin;
  mRMax2 = rMax * rMax;
  mZHalf = zHalfSpan;
  mNZBins = nz;

  mDZ = 2. * zHalfSpan / nz;
  mDZInv = 1.f / mDZ;

  mDPhi = o2::constants::math::TwoPI / nphi;
  mDPhiInv = 1.f / mDPhi;
  //
  int offs = 0;

  o2::gpu::resizeArray(mPhiBin2Slice, 0, nphi, reinterpret_cast<short*>(mFlatBufferPtr + offs));
  mNPhiSlices = mNPhiBins = nphi;

  for (int i = nphi; i--;) {
    mPhiBin2Slice[i] = i; // fill with trivial mapping
  }

  offs = alignSize(offs + nphi * sizeof(short), getBufferAlignmentBytes()); // account for alignment

  o2::gpu::resizeArray(mSliceCos, 0, nphi, reinterpret_cast<float*>(mFlatBufferPtr + offs)); // in the beginning nslice = nphi
  offs = alignSize(offs + nphi * sizeof(float), getBufferAlignmentBytes());                  // account for alignment

  o2::gpu::resizeArray(mSliceSin, 0, nphi, reinterpret_cast<float*>(mFlatBufferPtr + offs)); // in the beginning nslice = nphi
  offs = alignSize(offs + nphi * sizeof(float), getBufferAlignmentBytes());                  // account for alignment

  for (int i = nphi; i--;) {
    mSliceCos[i] = std::cos(getPhiBinMin(i));
    mSliceSin[i] = std::sin(getPhiBinMin(i));
  }

  o2::gpu::resizeArray(mCells, 0, getNCells(), reinterpret_cast<MatCell*>(mFlatBufferPtr + offs));

  mConstructionMask = InProgress;
}

//________________________________________________________________________________
void MatLayerCyl::populateFromTGeo(int ntrPerCell)
{
  /// populate layer with info extracted from TGeometry, using ntrPerCell test tracks per cell
  assert(mConstructionMask != Constructed);
  mConstructionMask = InProgress;
  ntrPerCell = ntrPerCell > 1 ? ntrPerCell : 1;
  for (int iz = getNZBins(); iz--;) {
    for (int ip = getNPhiBins(); ip--;) {
      populateFromTGeo(ip, iz, ntrPerCell);
    }
  }
}

//________________________________________________________________________________
void MatLayerCyl::populateFromTGeo(int ip, int iz, int ntrPerCell)
{
  /// populate cell with info extracted from TGeometry, using ntrPerCell test tracks per cell

  float zmn = getZBinMin(iz), phmn = getPhiBinMin(ip), sn, cs, rMin = getRMin(), rMax = getRMax();
  double meanRho = 0., meanX2X0 = 0., lgt = 0.;
  ;
  float dz = getDZ() / ntrPerCell;
  for (int isz = ntrPerCell; isz--;) {
    float zs = zmn + (isz + 0.5) * dz;
    float dzt = zs > 0.f ? 0.25 * dz : -0.25 * dz; // to avoid 90 degree polar angle
    for (int isp = ntrPerCell; isp--;) {
      o2::utils::sincosf(phmn + (isp + 0.5) * getDPhi() / ntrPerCell, sn, cs);
      auto bud = o2::base::GeometryManager::meanMaterialBudget(rMin * cs, rMin * sn, zs - dzt, rMax * cs, rMax * sn, zs + dzt);
      if (bud.length > 0.) {
        meanRho += bud.length * bud.meanRho;
        meanX2X0 += bud.meanX2X0; // we store actually not X2X0 but 1./X0
        lgt += bud.length;
      }
    }
  }
  if (lgt > 0.) {
    auto& cell = mCells[getCellIDPhiBin(ip, iz)];
    cell.meanRho = meanRho / lgt;   // mean rho
    cell.meanX2X0 = meanX2X0 / lgt; // mean 1./X0 seen in this cell
  }
}

//________________________________________________________________________________
bool MatLayerCyl::canMergePhiSlices(int i, int j, float maxRelDiff, int maxDifferent) const
{
  if (std::abs(i - j) > 1 || i == j || std::max(i, j) >= getNPhiSlices()) {
    LOG(ERROR) << "Only existing " << getNPhiSlices() << " slices with diff. of 1 can be merged, input is " << i << " and " << j;
    return false;
  }
  int ndiff = 0; // number of different cells
  for (int iz = getNZBins(); iz--;) {
    const auto& cellI = getCellPhiBin(i, iz);
    const auto& cellJ = getCellPhiBin(j, iz);
    if (cellsDiffer(cellI, cellJ, maxRelDiff)) {
      if (++ndiff > maxDifferent) {
        return false;
      }
    }
  }
  return true;
}

//________________________________________________________________________________
bool MatLayerCyl::cellsDiffer(const MatCell& cellA, const MatCell& cellB, float maxRelDiff) const
{
  /// check if the cells content is different within the tolerance
  float rav = 0.5 * (cellA.meanRho + cellB.meanRho), xav = 0.5 * (cellA.meanX2X0 + cellB.meanX2X0);
  float rdf = 0.5 * (cellA.meanRho - cellB.meanRho), xdf = 0.5 * (cellA.meanX2X0 - cellB.meanX2X0);
  if (rav > 0 && std::abs(rdf / rav) > maxRelDiff)
    return true;
  if (xav > 0 && std::abs(xdf / xav) > maxRelDiff)
    return true;
  return false;
}

//________________________________________________________________________________
void MatLayerCyl::optimizePhiSlices(float maxRelDiff)
{
  // merge compatible phi slices
  if (getNPhiSlices() < getNPhiBins()) {
    LOG(ERROR) << getNPhiBins() << " phi bins were already merged to " << getNPhiSlices() << " slices";
    return;
  }
  int newSl = 0;
  for (int is = 1; is < getNPhiSlices(); is++) {
    if (!canMergePhiSlices(is - 1, is, maxRelDiff)) {
      newSl++;
    }
    mPhiBin2Slice[is] = newSl;
  }
  LOG(INFO) << newSl + 1 << " slices out of " << getNPhiBins();
  if (newSl + 1 == getNPhiSlices()) {
    return;
  }
  newSl = 0;
  int slMin = 0, slMax = 0, is = 0;
  while (is++ < getNPhiSlices()) {
    while (is < getNPhiSlices() && mPhiBin2Slice[is] == newSl) { // select similar slices
      slMax++;
      is++;
    }
    if (slMax > slMin || newSl != slMin) { // merge or shift slices
      mSliceCos[newSl] = mSliceCos[slMin];
      mSliceSin[newSl] = mSliceSin[slMin];
      float norm = 1.f / (1.f + slMax - slMin);
      for (int iz = getNZBins(); iz--;) {
        int iDest = newSl * getNZBins() + iz, iSrc = slMin * getNZBins() + iz;
        mCells[iDest] = mCells[iSrc];
        for (int ism = slMin + 1; ism <= slMax; ism++) {
          iSrc = ism * getNZBins() + iz;
          mCells[iDest].meanX2X0 += mCells[iSrc].meanX2X0;
          mCells[iDest].meanRho += mCells[iSrc].meanRho;
        }
        mCells[iDest].scale(norm);
      }
      LOG(INFO) << "mapping " << slMin << ":" << slMax << " to new slice " << newSl;
    }
    newSl++;
    slMin = slMax = is;
  }
  mNPhiSlices = newSl;

  // relocate arrays to avoid spaces after optimization
  // mSliceCos pointer does not change, but sliceSin needs to be relocated
  auto offs = alignSize(newSl * sizeof(float), getBufferAlignmentBytes());
  char* dst = ((char*)mSliceCos) + offs; // account for alignment
  o2::gpu::resizeArray(mSliceSin, getNPhiBins(), newSl, reinterpret_cast<float*>(dst));
  // adjust mCells array
  dst = ((char*)mSliceSin) + offs; // account for alignment
  o2::gpu::resizeArray(mCells, getNPhiBins() * getNZBins(), newSl * getNZBins(), reinterpret_cast<MatCell*>(dst));
  mFlatBufferSize = estimateFlatBufferSize();
  LOG(INFO) << "Updated Nslices = " << getNPhiSlices();
}

//________________________________________________________________________________
void MatLayerCyl::getMeanRMS(MatCell& mean, MatCell& rms) const
{
  // mean and RMS over layer
  mean.meanRho = rms.meanRho = 0.f;
  mean.meanX2X0 = rms.meanX2X0 = 0.f;
  for (int ip = getNPhiBins(); ip--;) {
    for (int iz = getNZBins(); iz--;) {
      const auto& cell = getCellPhiBin(ip, iz);
      mean.meanRho += cell.meanRho;
      mean.meanX2X0 += cell.meanX2X0;
      rms.meanRho += cell.meanRho * cell.meanRho;
      rms.meanX2X0 += cell.meanX2X0 * cell.meanX2X0;
    }
  }
  int nc = getNPhiBins() * getNZBins();
  mean.meanRho /= nc;
  mean.meanX2X0 /= nc;
  rms.meanRho /= nc;
  rms.meanX2X0 /= nc;
  rms.meanRho -= mean.meanRho * mean.meanRho;
  rms.meanX2X0 -= mean.meanX2X0 * mean.meanX2X0;
  rms.meanRho = rms.meanRho > 0.f ? std::sqrt(rms.meanRho) : 0.f;
  rms.meanX2X0 = rms.meanX2X0 > 0.f ? std::sqrt(rms.meanX2X0) : 0.f;
}

//________________________________________________________________________________
void MatLayerCyl::print(bool data) const
{
  ///< print layer data
  float szkb = float(getFlatBufferSize()) / 1024;
  printf("Cyl.Layer %.3f<r<%.3f %+.3f<Z<%+.3f | Nphi: %5d (%d slices) Nz: %5d Size: %.3f KB\n",
         getRMin(), getRMax(), getZMin(), getZMax(), getNPhiBins(), getNPhiSlices(), getNZBins(), szkb);
  if (!data) {
    return;
  }
  for (int ip = 0; ip < getNPhiSlices(); ip++) {
    int ib0, ib1;
    int nb = getNPhiBinsInSlice(ip, ib0, ib1);
    printf("phi slice: %d (%d bins %d-%d %.4f:%.4f) sn:%+.4f/cs:%+.4f ... [iz/<rho>/<x/x0>] \n",
           ip, nb, ib0, ib1, getDPhi() * ib0, getDPhi() * (ib1 + 1), getSliceSin(ip), getSliceCos(ip));
    for (int iz = 0; iz < getNZBins(); iz++) {
      auto cell = getCellPhiBin(ib0, iz);
      printf("%3d/%.2e/%.2e ", iz, cell.meanRho, cell.meanX2X0);
      if (((iz + 1) % 5) == 0) {
        printf("\n");
      }
    }
    if (getNZBins() % 5) {
      printf("\n");
    }
  }
}

//________________________________________________________________________________
void MatLayerCyl::flatten(char* newPtr)
{
  // make object flat: move all content to single internally allocated buffer
  assert(mConstructionMask == InProgress);
  fixPointers(mFlatBufferPtr, newPtr);
  auto old = o2::gpu::resizeArray(mFlatBufferPtr, getFlatBufferSize(), getFlatBufferSize(), newPtr);
  delete[] old;
  mFlatBufferContainer = nullptr;
  mConstructionMask = Constructed;
}

#endif // ! GPUCA_ALIGPUCODE

#ifndef GPUCA_GPUCODE
//________________________________________________________________________________
void MatLayerCyl::fixPointers(char* oldPtr, char* newPtr)
{
  // fix pointers on the internal structure of the flat buffer after retrieving it from the file
  mPhiBin2Slice = flatObject::relocatePointer(oldPtr, newPtr, mPhiBin2Slice);
  mSliceCos = flatObject::relocatePointer(oldPtr, newPtr, mSliceCos);
  mSliceSin = flatObject::relocatePointer(oldPtr, newPtr, mSliceSin);
  mCells = flatObject::relocatePointer(oldPtr, newPtr, mCells);
}
#endif // ! GPUCA_GPUCODE

//________________________________________________________________________________
GPUd() int MatLayerCyl::getNPhiBinsInSlice(int iSlice, int& binMin, int& binMax) const
{
  // slow method to get number of phi bins for given phi slice
  int nb = 0;
  binMin = binMax = -1;
  for (int ib = getNPhiBins(); ib--;) {
    if (phiBin2Slice(ib) == iSlice) {
      binMax < 0 ? binMin = binMax = ib : binMin = ib;
      nb++;
      continue;
    }
    if (binMax >= 0) {
      break; // no more bins since they are consecutive
    }
  }
  return nb;
}
