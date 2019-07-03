// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MatLayerCylSet.cxx
/// \brief Implementation of the wrapper for the set of cylindrical material layers

#include "DetectorsBase/MatLayerCylSet.h"
#include "CommonConstants/MathConstants.h"

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

#include "GPUCommonLogger.h"
#include <TFile.h>
#include "CommonUtils/TreeStreamRedirector.h"
//#define _DBG_LOC_ // for local debugging only

#endif // !GPUCA_ALIGPUCODE

using namespace o2::base;

using flatObject = o2::gpu::FlatObject;

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

//________________________________________________________________________________
void MatLayerCylSet::addLayer(float rmin, float rmax, float zmax, float dz, float drphi)
{
  // add new layer checking for overlaps
  assert(mConstructionMask != Constructed);
  assert(rmin < rmax && zmax > 0 && dz > 0 && drphi > 0);
  mConstructionMask = InProgress;
  int nlr = getNLayers();
  if (!nlr) {
    // book local storage
    auto sz = sizeof(MatLayerCylSetLayout);
    o2::gpu::resizeArray(mFlatBufferContainer, 0, sz);
    mFlatBufferPtr = mFlatBufferContainer;
    mFlatBufferSize = sz;
    //--------------????
    get()->mRMin = 1.e99;
    get()->mRMax = 0.;
  }

  for (int il = 0; il < nlr; il++) {
    const auto& lr = getLayer(il);
    if (lr.getRMax() > rmin && rmax > lr.getRMin()) {
      LOG(FATAL) << "new layer overlaps with layer " << il;
    }
  }
  auto* oldLayers = o2::gpu::resizeArray(get()->mLayers, nlr, nlr + 1);
  // dynamyc buffers of old layers were used in new ones, detach them
  for (int i = nlr; i--;) {
    oldLayers[i].clearInternalBufferPtr();
  }
  delete[] oldLayers;
  get()->mLayers[nlr].initSegmentation(rmin, rmax, zmax, dz, drphi);
  get()->mNLayers++;
  get()->mRMin = get()->mRMin > rmin ? rmin : get()->mRMin;
  get()->mRMax = get()->mRMax < rmax ? rmax : get()->mRMax;
  get()->mZMax = get()->mZMax < zmax ? zmax : get()->mZMax;
  get()->mRMin2 = get()->mRMin * get()->mRMin;
  get()->mRMax2 = get()->mRMax * get()->mRMax;
}

//________________________________________________________________________________
void MatLayerCylSet::populateFromTGeo(int ntrPerCell)
{
  ///< populate layers, using ntrPerCell test tracks per cell
  assert(mConstructionMask == InProgress);

  int nlr = getNLayers();
  if (!nlr) {
    LOG(ERROR) << "The LUT is not yet initialized";
    return;
  }
  if (get()->mR2Intervals) {
    LOG(ERROR) << "The LUT is already populated";
    return;
  }
  for (int i = 0; i < nlr; i++) {
    printf("Populating with %d trials Lr  %3d ", ntrPerCell, i);
    get()->mLayers[i].print();
    get()->mLayers[i].populateFromTGeo(ntrPerCell);
  }
  // build layer search structures
  int nR2Int = 2 * (nlr + 1);
  o2::gpu::resizeArray(get()->mR2Intervals, 0, nR2Int);
  o2::gpu::resizeArray(get()->mInterval2LrID, 0, nR2Int);
  get()->mR2Intervals[0] = get()->mRMin2;
  get()->mR2Intervals[1] = get()->mRMax2;
  get()->mInterval2LrID[0] = 0;
  auto& nRIntervals = get()->mNRIntervals;
  nRIntervals = 1;

  for (int i = 1; i < nlr; i++) {
    const auto& lr = getLayer(i);
    if (std::sqrt(lr.getRMin2()) > std::sqrt(get()->mR2Intervals[nRIntervals] + Ray::Tiny)) {
      // register gap
      get()->mInterval2LrID[nRIntervals] = -1;
      get()->mR2Intervals[++nRIntervals] = lr.getRMin2();
    }
    get()->mInterval2LrID[nRIntervals] = i;
    get()->mR2Intervals[++nRIntervals] = lr.getRMax2();
  }
  delete[] o2::gpu::resizeArray(get()->mInterval2LrID, nR2Int, nRIntervals); // rebook with precise size
  delete[] o2::gpu::resizeArray(get()->mR2Intervals, nR2Int, ++nRIntervals); // rebook with precise size
  //
}

//________________________________________________________________________________
void MatLayerCylSet::dumpToTree(const std::string outName) const
{
  /// dump per cell info to the tree

  o2::utils::TreeStreamRedirector dump(outName.data(), "recreate");
  for (int i = 0; i < getNLayers(); i++) {
    const auto& lr = getLayer(i);
    float r = 0.5 * (lr.getRMin() + lr.getRMax());
    // per cell dump
    int nphib = lr.getNPhiBins();
    for (int ip = 0; ip < nphib; ip++) {
      float phi = 0.5 * (lr.getPhiBinMin(ip) + lr.getPhiBinMax(ip));
      float sn, cs;
      int ips = lr.phiBin2Slice(ip);
      char merge = 0; // not mergeable
      if (ip + 1 < nphib) {
        int ips1 = lr.phiBin2Slice(ip + 1);
        merge = ips == ips1 ? -1 : lr.canMergePhiSlices(ips, ips1); // -1 for already merged
      } else
        merge = -2; // last one
      o2::utils::sincosf(phi, sn, cs);
      float x = r * cs, y = r * sn;
      for (int iz = 0; iz < lr.getNZBins(); iz++) {
        float z = 0.5 * (lr.getZBinMin(iz) + lr.getZBinMax(iz));
        auto cell = lr.getCellPhiBin(ip, iz);
        dump << "cell"
             << "ilr=" << i << "r=" << r << "phi=" << phi << "x=" << x << "y=" << y << "z=" << z << "ip=" << ip << "ips=" << ips << "iz=" << iz
             << "mrgnxt=" << merge << "val=" << cell << "\n";
      }
    }
    //
    // statistics per layer
    MatCell mean, rms;
    lr.getMeanRMS(mean, rms);
    dump << "lay"
         << "ilr=" << i << "r=" << r << "mean=" << mean << "rms=" << rms << "\n";
  }
}

//________________________________________________________________________________
void MatLayerCylSet::writeToFile(std::string outFName, std::string name)
{
  /// store to file

  TFile outf(outFName.data(), "recreate");
  if (outf.IsZombie()) {
    return;
  }
  if (name.empty()) {
    name = "matBud";
  }
  outf.WriteObjectAny(this, Class(), name.data());
  outf.Close();
}

//________________________________________________________________________________
MatLayerCylSet* MatLayerCylSet::loadFromFile(std::string inpFName, std::string name)
{
  if (name.empty()) {
    name = "MatBud";
  }
  TFile inpf(inpFName.data());
  if (inpf.IsZombie()) {
    LOG(ERROR) << "Failed to open input file " << inpFName;
    return nullptr;
  }
  MatLayerCylSet* mb = reinterpret_cast<MatLayerCylSet*>(inpf.GetObjectChecked(name.data(), Class()));
  if (!mb) {
    LOG(ERROR) << "Failed to load " << name << " from " << inpFName;
  }
  mb->fixPointers();
  return mb;
}

//________________________________________________________________________________
void MatLayerCylSet::optimizePhiSlices(float maxRelDiff)
{
  // merge similar (whose relative budget does not differ within maxRelDiff) phi slices
  assert(mConstructionMask == InProgress);
  for (int i = getNLayers(); i--;) {
    get()->mLayers[i].optimizePhiSlices(maxRelDiff);
  }
  // flatten();  // RS: TODO
}

//________________________________________________________________________________
void MatLayerCylSet::print(bool data) const
{
  ///< print layer data
  if (!get()) {
    printf("Not initialized yet\n");
    return;
  }
  if (mConstructionMask != Constructed) {
    LOG(WARNING) << "Object is not yet flattened";
  }
  for (int i = 0; i < getNLayers(); i++) {
    printf("#%3d | ", i);
    getLayer(i).print(data);
  }
  printf("%.2f < R < %.2f  %d layers with total size %.2f MB\n", getRMin(), getRMax(), getNLayers(),
         float(getFlatBufferSize()) / 1024 / 1024);
}

#endif //!GPUCA_ALIGPUCODE

#ifndef GPUCA_GPUCODE
//________________________________________________________________________________
std::size_t MatLayerCylSet::estimateFlatBufferSize() const
{
  std::size_t sz = alignSize(sizeof(MatLayerCylSetLayout), getBufferAlignmentBytes()); // hold data members

  sz = alignSize(sz + get()->mNLayers * sizeof(MatLayerCyl), MatLayerCyl::getClassAlignmentBytes());
  sz = alignSize(sz + (get()->mNRIntervals + 1) * sizeof(float), getBufferAlignmentBytes());
  sz = alignSize(sz + get()->mNRIntervals * sizeof(int), getBufferAlignmentBytes());

  for (int i = 0; i < getNLayers(); i++) {
    sz = alignSize(sz + getLayer(i).estimateFlatBufferSize(), getBufferAlignmentBytes());
  }
  return sz;
}
#endif // ! GPUCA_GPUCODE

//_________________________________________________________________________________________________
GPUd() MatBudget MatLayerCylSet::getMatBudget(float x0, float y0, float z0, float x1, float y1, float z1) const
{
  // get material budget traversed on the line between point0 and point1
  MatBudget rval;
  Ray ray(x0, y0, z0, x1, y1, z1);
  short lmin, lmax; // get innermost and outermost relevant layer
  if (!getLayersRange(ray, lmin, lmax)) {
    return rval;
  }
  short lrID = lmax;
  while (lrID >= lmin) { // go from outside to inside
    const auto& lr = getLayer(lrID);
    int nc = ray.crossLayer(lr);
    for (int ic = nc; ic--;) {
      float cross1, cross2;
      ray.getCrossParams(ic, cross1, cross2); // tmax,tmin of crossing the layer
      auto phi0 = ray.getPhi(cross1), phi1 = ray.getPhi(cross2), dPhi = phi0 - phi1;
      auto phiID = lr.getPhiSliceID(phi0), phiIDLast = lr.getPhiSliceID(phi1);
      // account for eventual wrapping around 0
      if (dPhi > 0.f) {
        if (dPhi > o2::constants::math::PI) { // wraps around phi=0
          phiIDLast += lr.getNPhiSlices();
        }
      } else {
        if (dPhi < -o2::constants::math::PI) { // wraps around phi=0
          phiID += lr.getNPhiSlices();
        }
      }
      int stepPhiID = phiID > phiIDLast ? -1 : 1;
      bool checkMorePhi = true;
      auto tStartPhi = cross1, tEndPhi = 0.f;
      do {
        // get the path in the current phi slice
        if (phiID == phiIDLast) {
          tEndPhi = cross2;
          checkMorePhi = false;
        } else { // last phi slice still not reached
          tEndPhi = ray.crossRadial(lr, (stepPhiID > 0 ? phiID + 1 : phiID) % lr.getNPhiSlices());
        }
        auto zID = lr.getZBinID(ray.getZ(tStartPhi));
        auto zIDLast = lr.getZBinID(ray.getZ(tEndPhi));
        // check if Zbins are crossed

#ifdef _DBG_LOC_
        printf("-- Zdiff (%3d : %3d) mode: t: %+e %+e\n", zID, zIDLast, tStartPhi, tEndPhi);
#endif

        if (zID != zIDLast) {
          auto stepZID = zID < zIDLast ? 1 : -1;
          bool checkMoreZ = true;
          auto tStartZ = tStartPhi, tEndZ = 0.f;
          do {
            if (zID == zIDLast) {
              tEndZ = tEndPhi;
              checkMoreZ = false;
            } else {
              tEndZ = ray.crossZ(lr.getZBinMin(stepZID > 0 ? zID + 1 : zID));
            }
            // account materials of this step
            float step = tEndZ - tStartZ; // the real step is ray.getDist(tEnd-tStart), will rescale all later
            const auto& cell = lr.getCell(phiID, zID);
            rval.meanRho += cell.meanRho * step;
            rval.meanX2X0 += cell.meanX2X0 * step;
            rval.length += step;

#ifdef _DBG_LOC_
            float pos0[3] = { ray.getPos(tStartZ, 0), ray.getPos(tStartZ, 1), ray.getPos(tStartZ, 2) };
            float pos1[3] = { ray.getPos(tEndZ, 0), ray.getPos(tEndZ, 1), ray.getPos(tEndZ, 2) };
            printf(
              "Lr#%3d / cross#%d : account %f<t<%f at phiSlice %d | Zbin: %3d (%3d) |[%+e %+e +%e]:[%+e %+e %+e] "
              "Step: %.3e StrpCor: %.3e\n",
              lrID, ic, tEndZ, tStartZ, phiID % lr.getNPhiSlices(), zID, zIDLast,
              pos0[0], pos0[1], pos0[2], pos1[0], pos1[1], pos1[2], step, ray.getDist(step));
#endif

            tStartZ = tEndZ;
            zID += stepZID;
          } while (checkMoreZ);
        } else {
          float step = tEndPhi - tStartPhi; // the real step is |ray.getDist(tEnd-tStart)|, will rescale all later
          const auto& cell = lr.getCell(phiID, zID);
          rval.meanRho += cell.meanRho * step;
          rval.meanX2X0 += cell.meanX2X0 * step;
          rval.length += step;

#ifdef _DBG_LOC_
          float pos0[3] = { ray.getPos(tStartPhi, 0), ray.getPos(tStartPhi, 1), ray.getPos(tStartPhi, 2) };
          float pos1[3] = { ray.getPos(tEndPhi, 0), ray.getPos(tEndPhi, 1), ray.getPos(tEndPhi, 2) };
          printf(
            "Lr#%3d / cross#%d : account %f<t<%f at phiSlice %d | Zbin: %3d ----- |[%+e %+e +%e]:[%+e %+e %+e]"
            "Step: %.3e StrpCor: %.3e\n",
            lrID, ic, tEndPhi, tStartPhi, phiID % lr.getNPhiSlices(), zID,
            pos0[0], pos0[1], pos0[2], pos1[0], pos1[1], pos1[2], step, ray.getDist(step));
#endif
        }
        //
        tStartPhi = tEndPhi;
        phiID += stepPhiID;

      } while (checkMorePhi);
    }
    lrID--;
  } // loop over layers

  if (rval.length != 0.f) {
    rval.meanRho /= rval.length;                                       // average
    float norm = (rval.length < 0.f) ? -ray.getDist() : ray.getDist(); // normalize
    rval.meanX2X0 *= norm;
    rval.length *= norm;
  }
#ifdef _DBG_LOC_
  printf("<rho> = %e, x2X0 = %e  | step = %e\n", rval.meanRho, rval.meanX2X0, rval.length);
#endif
  return rval;
}

//_________________________________________________________________________________________________
GPUd() bool MatLayerCylSet::getLayersRange(const Ray& ray, short& lmin, short& lmax) const
{
  // get range of layers corresponding to rmin/rmax
  //
  lmin = lmax = -1;
  float rmin2, rmax2;
  ray.getMinMaxR2(rmin2, rmax2);

  if (rmin2 >= getRMax2() || rmax2 <= getRMin2()) {
    return false;
  }
  int lmxInt, lmnInt;
  lmxInt = rmax2 < getRMax2() ? searchSegment(rmax2, 0) : get()->mNRIntervals - 2;
  lmnInt = rmin2 >= getRMin2() ? searchSegment(rmin2, 0, lmxInt + 1) : 0;
  const auto* interval2LrID = get()->mInterval2LrID;
  lmax = interval2LrID[lmxInt];
  lmin = interval2LrID[lmnInt];
  // make sure lmnInt and/or lmxInt are not in the gap
  if (lmax < 0) {
    lmax = interval2LrID[--lmxInt]; // rmax2 is in the gap, take highest layer below rmax2
  }
  if (lmin < 0) {
    lmin = interval2LrID[++lmnInt]; // rmin2 is in the gap, take lowest layer above rmin2
  }
  return lmin <= lmax; // valid if both are not in the same gap
}

GPUd() int MatLayerCylSet::searchSegment(float val, int low, int high) const
{
  ///< search segment val belongs to. The val MUST be within the boundaries
  if (low < 0) {
    low = 0;
  }
  if (high < 0) {
    high = get()->mNRIntervals;
  }
  int mid = (low + high) >> 1;
  const auto* r2Intervals = get()->mR2Intervals;
  while (mid != low) {
    if (val < r2Intervals[mid]) {
      high = mid;
    } else {
      low = mid;
    }
    mid = (low + high) >> 1;
  }
  return mid;
}

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

void MatLayerCylSet::flatten()
{
  // make object flat: move all content to single internally allocated buffer
  assert(mConstructionMask == InProgress);

  int sz = estimateFlatBufferSize();
  // create new internal buffer with total size and copy data
  delete[] o2::gpu::resizeArray(mFlatBufferContainer, mFlatBufferSize, sz);
  mFlatBufferPtr = mFlatBufferContainer;
  mFlatBufferSize = sz;
  int nLr = getNLayers();

  auto offs = alignSize(sizeof(MatLayerCylSetLayout), getBufferAlignmentBytes()); // account for the alignment
  // move array of layer pointers to the flat array
  auto* oldLayers = o2::gpu::resizeArray(get()->mLayers, nLr, nLr, (MatLayerCyl*)(mFlatBufferPtr + offs));
  // dynamyc buffers of old layers were used in new ones, detach them
  for (int i = nLr; i--;) {
    oldLayers[i].clearInternalBufferPtr();
  }
  delete[] oldLayers;
  offs = alignSize(offs + nLr * sizeof(MatLayerCyl), MatLayerCyl::getClassAlignmentBytes()); // account for the alignment

  // move array of R2 boundaries to the flat array
  delete[] o2::gpu::resizeArray(get()->mR2Intervals, nLr + 1, nLr + 1, (float*)(mFlatBufferPtr + offs));
  offs = alignSize(offs + (nLr + 1) * sizeof(float), getBufferAlignmentBytes()); // account for the alignment

  // move array of R2 boundaries to the flat array
  delete[] o2::gpu::resizeArray(get()->mInterval2LrID, nLr, nLr, (int*)(mFlatBufferPtr + offs));
  offs = alignSize(offs + nLr * sizeof(int), getBufferAlignmentBytes()); // account for the alignment

  for (int il = 0; il < nLr; il++) {
    MatLayerCyl& lr = get()->mLayers[il];
    lr.flatten(mFlatBufferPtr + offs);
    offs = alignSize(offs + lr.getFlatBufferSize(), getBufferAlignmentBytes()); // account for the alignment
  }
  mConstructionMask = Constructed;
}

//______________________________________________
void MatLayerCylSet::moveBufferTo(char* newFlatBufferPtr)
{
  /// sets buffer pointer to the new address, move the buffer content there.
  flatObject::moveBufferTo(newFlatBufferPtr);
  setActualBufferAddress(mFlatBufferPtr);
}
#endif // !GPUCA_ALIGPUCODE

#ifndef GPUCA_GPUCODE
//______________________________________________
void MatLayerCylSet::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// Sets the actual location of the external flat buffer before it was created
  ///
  fixPointers(mFlatBufferPtr, futureFlatBufferPtr, false); // flag that futureFlatBufferPtr is not valid yet
  flatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

//______________________________________________
void MatLayerCylSet::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// Sets the actual location of the external flat buffer after it has been moved (i.e. to another machine)
  ///
  fixPointers(actualFlatBufferPtr);
}
//______________________________________________
void MatLayerCylSet::cloneFromObject(const MatLayerCylSet& obj, char* newFlatBufferPtr)
{
  /// Initializes from another object, copies data to newBufferPtr
  flatObject::cloneFromObject(obj, newFlatBufferPtr);
  fixPointers(mFlatBufferPtr);
}

//______________________________________________
void MatLayerCylSet::fixPointers(char* newBasePtr)
{
  // fix pointers on the internal structure of the flat buffer after retrieving it from the file
  if (newBasePtr) {
    mFlatBufferPtr = newBasePtr; // used to impose external pointer
  } else {
    mFlatBufferPtr = mFlatBufferContainer; // impose pointer after reading from file
  }
  auto offs = alignSize(sizeof(MatLayerCylSetLayout), getBufferAlignmentBytes()); // account for the alignment
  char* newPtr = mFlatBufferPtr + offs;                                           // correct pointer on MatLayerCyl*
  char* oldPtr = reinterpret_cast<char*>(get()->mLayers);                         // old pointer read from the file
  fixPointers(oldPtr, newPtr);
}

//______________________________________________
void MatLayerCylSet::fixPointers(char* oldPtr, char* newPtr, bool newPtrValid)
{
  // fix pointers on the internal structure of the flat buffer after retrieving it from the file
  auto* layPtr = get()->mLayers;
  get()->mLayers = flatObject::relocatePointer(oldPtr, newPtr, get()->mLayers);
  get()->mR2Intervals = flatObject::relocatePointer(oldPtr, newPtr, get()->mR2Intervals);
  get()->mInterval2LrID = flatObject::relocatePointer(oldPtr, newPtr, get()->mInterval2LrID);
  if (newPtrValid) {
    layPtr = get()->mLayers;
  }
  for (int i = 0; i < getNLayers(); i++) {
    layPtr[i].setFlatPointer(flatObject::relocatePointer(oldPtr, newPtr, layPtr[i].getFlatBufferPtr()));
    layPtr[i].fixPointers(oldPtr, newPtr);
  }
}
#endif // !GPUCA_GPUCODE
