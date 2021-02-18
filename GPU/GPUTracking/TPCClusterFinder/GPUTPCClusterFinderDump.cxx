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
#include "Array2D.h"
#include "Digit.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCClusterFinder::DumpDigits(std::ostream& out)
{
  out << "Clusterer - Digits - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << mPmemory->counters.nPositions << "\n";
  for (size_t i = 0; i < mPmemory->counters.nPositions; i++) {
    out << i << ": " << mPpositions[i].time() << ", " << (int)mPpositions[i].pad() << ", " << (int)mPpositions[i].row() << "\n";
  }
}

void GPUTPCClusterFinder::DumpChargeMap(std::ostream& out, std::string_view title)
{
  out << "Clusterer - " << title << " - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";
  Array2D<ushort> map(mPchargeMap);

  for (TPCFragmentTime i = 0; i < TPC_MAX_TIME_PADDED; i++) {
    out << "Line " << i;
    int zeros = 0;
    for (GlobalPad j = 0; j < TPC_NUM_OF_PADS; j++) {
      ushort q = map[{j, i}];
      zeros += (q == 0);
      if (q != 0) {
        if (zeros > 0) {
          out << " z" << zeros;
          zeros = 0;
        }

        out << " q" << q;
      }
    }
    if (zeros > 0) {
      out << " z" << zeros;
    }
    out << "\n";
  }
}

void GPUTPCClusterFinder::DumpPeaks(std::ostream& out)
{
  out << "Clusterer - Peaks - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";
  for (unsigned int i = 0; i < mPmemory->counters.nPositions; i++) {
    out << (int)mPisPeak[i] << " ";
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
  out << "\n";
}

void GPUTPCClusterFinder::DumpPeaksCompacted(std::ostream& out)
{
  out << "Clusterer - Compacted Peaks - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << mPmemory->counters.nPeaks << "\n";
  for (size_t i = 0; i < mPmemory->counters.nPeaks; i++) {
    out << i << ": " << mPpeakPositions[i].time() << ", " << (int)mPpeakPositions[i].pad() << ", " << (int)mPpeakPositions[i].row() << "\n";
  }
}

void GPUTPCClusterFinder::DumpSuppressedPeaks(std::ostream& out)
{
  out << "Clusterer - NoiseSuppression - Slice "
      << " - Fragment " << mPmemory->fragment.index << mISlice << "\n";
  for (unsigned int i = 0; i < mPmemory->counters.nPeaks; i++) {
    out << (int)mPisPeak[i] << " ";
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
  out << "\n";
}

void GPUTPCClusterFinder::DumpSuppressedPeaksCompacted(std::ostream& out)
{
  out << "Clusterer - Noise Suppression Peaks Compacted - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << mPmemory->counters.nClusters << "\n";
  for (size_t i = 0; i < mPmemory->counters.nClusters; i++) {
    out << i << ": " << mPfilteredPeakPositions[i].time() << ", " << (int)mPfilteredPeakPositions[i].pad() << ", " << (int)mPfilteredPeakPositions[i].row() << "\n";
  }
}

void GPUTPCClusterFinder::DumpCountedPeaks(std::ostream& out)
{
  out << "Clusterer - Peak Counts - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    out << i << ": " << mPclusterInRow[i] << "\n";
  }
}

void GPUTPCClusterFinder::DumpClusters(std::ostream& out)
{
  out << "Clusterer - Clusters - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";

  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    size_t N = mPclusterInRow[i];
    out << "Row: " << i << ": " << N << "\n";
    std::vector<tpc::ClusterNative> sortedCluster(N);
    tpc::ClusterNative* row = &mPclusterByRow[i * mNMaxClusterPerRow];
    std::copy(row, &row[N], sortedCluster.begin());

    std::sort(sortedCluster.begin(), sortedCluster.end());

    for (const auto& cl : sortedCluster) {
      out << std::hex << cl.timeFlagsPacked << std::dec << ", " << cl.padPacked << ", " << (int)cl.sigmaTimePacked << ", " << (int)cl.sigmaPadPacked << ", " << cl.qMax << ", " << cl.qTot << "\n";
    }
  }
}
