// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterFinderDump.cxx
/// \author David Rohr

#include "GPUTPCClusterFinder.h"
#include "GPUReconstruction.h"
#include "ClusterNative.h"
#include "Digit.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCClusterFinder::DumpDigits(std::ostream& out)
{
  out << "Clusterer - Digits - Slice " << mISlice << ": " << mPmemory->nDigits << "\n";
  for (size_t i = 0; i < mPmemory->nDigits; i++) {
    out << i << ": " << mPdigits[i].charge << ", " << mPdigits[i].time << ", " << (int)mPdigits[i].pad << ", " << (int)mPdigits[i].row << "\n";
  }
}

void GPUTPCClusterFinder::DumpChargeMap(std::ostream& out)
{
  out << "Clusterer - Charges - Slice " << mISlice << "\n";
  for (unsigned int i = 0; i < TPC_MAX_TIME_PADDED; i++) {
    out << "Line " << i;
    for (unsigned int j = 0; j < TPC_NUM_OF_PADS; j++) {
      if (mPchargeMap[i * TPC_NUM_OF_PADS + j]) {
        out << " " << mPchargeMap[i * TPC_NUM_OF_PADS + j];
      }
    }
    out << "\n";
  }
}

void GPUTPCClusterFinder::DumpPeaks(std::ostream& out)
{
  out << "Clusterer - Peaks - Slice " << mISlice << "\n";
  for (unsigned int i = 0; i < mPmemory->nDigits; i++) {
    out << (int)mPisPeak[i] << " ";
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
  out << "\n";
}

void GPUTPCClusterFinder::DumpPeaksCompacted(std::ostream& out)
{
  out << "Clusterer - Compacted Peaks - Slice " << mISlice << ": " << mPmemory->nPeaks << "\n";
  for (size_t i = 0; i < mPmemory->nPeaks; i++) {
    out << i << ": " << mPpeaks[i].charge << ", " << mPpeaks[i].time << ", " << (int)mPpeaks[i].pad << ", " << (int)mPpeaks[i].row << "\n";
  }
}

void GPUTPCClusterFinder::DumpSuppressedPeaks(std::ostream& out)
{
  out << "Clusterer - NoiseSuppression - Slice " << mISlice << "\n";
  for (unsigned int i = 0; i < mPmemory->nPeaks; i++) {
    out << (int)mPisPeak[i] << " ";
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
  out << "\n";
}

void GPUTPCClusterFinder::DumpSuppressedPeaksCompacted(std::ostream& out)
{
  out << "Clusterer - Noise Suppression Peaks Compacted - Slice " << mISlice << "\n";
  for (size_t i = 0; i < mPmemory->nClusters; i++) {
    out << i << ": " << mPfilteredPeaks[i].charge << ", " << mPfilteredPeaks[i].time << ", " << (int)mPfilteredPeaks[i].pad << ", " << (int)mPfilteredPeaks[i].row << "\n";
  }
}

void GPUTPCClusterFinder::DumpCountedPeaks(std::ostream& out)
{
  out << "Clusterer - Peak Counts - Slice " << mISlice << "\n";
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    out << i << ": " << mPclusterInRow[i] << "\n";
  }
}

void GPUTPCClusterFinder::DumpClusters(std::ostream& out)
{
  out << "Clusterer - Clusters - Slice " << mISlice << "\n";
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    out << "Row: " << i << ": " << mPclusterInRow[i] << "\n";
    for (int j = 0; j < mPclusterInRow[i]; j++) {
      const auto& cl = mPclusterByRow[i * mNMaxClusterPerRow + j];
      out << cl.timeFlagsPacked << ", " << cl.padPacked << ", " << (int)cl.sigmaTimePacked << ", " << (int)cl.sigmaPadPacked << ", " << cl.qmax << ", " << cl.qtot << "\n";
    }
  }
}
