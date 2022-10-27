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

/// \file GPUTPCClusterFinderDump.cxx
/// \author David Rohr

#include "GPUTPCClusterFinder.h"
#include "GPUReconstruction.h"
#include "Array2D.h"
#include "DataFormatsTPC/Digit.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

void GPUTPCClusterFinder::DumpDigits(std::ostream& out)
{
  out << "\nClusterer - Digits - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << mPmemory->counters.nPositions << "\n";

  out << std::hex;
  for (size_t i = 0; i < mPmemory->counters.nPositions; i++) {
    out << mPpositions[i].time() << " " << mPpositions[i].gpad << '\n';
  }
  out << std::dec;
}

void GPUTPCClusterFinder::DumpChargeMap(std::ostream& out, std::string_view title, bool doGPU)
{
  out << "\nClusterer - " << title << " - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";
  Array2D<ushort> map(mPchargeMap);

  out << std::hex;

  for (TPCFragmentTime i = 0; i < TPC_MAX_FRAGMENT_LEN_PADDED(mRec->GetProcessingSettings().overrideClusterizerFragmentLen); i++) {
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
    out << '\n';
  }

  out << std::dec;
}

void GPUTPCClusterFinder::DumpPeaks(std::ostream& out)
{
  out << "\nClusterer - Peaks - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";
  for (unsigned int i = 0; i < mPmemory->counters.nPositions; i++) {
    out << (int)mPisPeak[i] << " ";
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
}

void GPUTPCClusterFinder::DumpPeaksCompacted(std::ostream& out)
{
  out << "\nClusterer - Compacted Peaks - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << mPmemory->counters.nPeaks << "\n";
  for (size_t i = 0; i < mPmemory->counters.nPeaks; i++) {
    out << mPpeakPositions[i].time() << ", " << (int)mPpeakPositions[i].pad() << ", " << (int)mPpeakPositions[i].row() << "\n";
  }
}

void GPUTPCClusterFinder::DumpSuppressedPeaks(std::ostream& out)
{
  out << "\nClusterer - NoiseSuppression - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << mISlice << "\n";
  for (unsigned int i = 0; i < mPmemory->counters.nPeaks; i++) {
    out << (int)mPisPeak[i] << " ";
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
}

void GPUTPCClusterFinder::DumpSuppressedPeaksCompacted(std::ostream& out)
{
  out << "\nClusterer - Noise Suppression Peaks Compacted - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << mPmemory->counters.nClusters << "\n";
  for (size_t i = 0; i < mPmemory->counters.nClusters; i++) {
    out << i << ": " << mPfilteredPeakPositions[i].time() << ", " << (int)mPfilteredPeakPositions[i].pad() << ", " << (int)mPfilteredPeakPositions[i].row() << "\n";
  }
}

void GPUTPCClusterFinder::DumpCountedPeaks(std::ostream& out)
{
  out << "\nClusterer - Peak Counts - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    out << i << ": " << mPclusterInRow[i] << "\n";
  }
}

void GPUTPCClusterFinder::DumpClusters(std::ostream& out)
{
  out << "\nClusterer - Clusters - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";

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
