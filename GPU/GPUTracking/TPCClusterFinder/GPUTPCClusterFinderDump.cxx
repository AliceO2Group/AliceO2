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
  const auto nPositions = mPmemory->counters.nPositions;

  out << "\nClusterer - Digits - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << nPositions << "\n";

  out << std::hex;
  for (size_t i = 0; i < mPmemory->counters.nPositions; i++) {
    const auto& pos = mPpositions[i];
    out << pos.time() << " " << pos.gpad << '\n';
  }
  out << std::dec;
}

void GPUTPCClusterFinder::DumpChargeMap(std::ostream& out, std::string_view title)
{
  out << "\nClusterer - " << title << " - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";
  Array2D<uint16_t> map(mPchargeMap);

  out << std::hex;

  TPCFragmentTime start = 0;
  TPCFragmentTime end = TPC_MAX_FRAGMENT_LEN_PADDED(mRec->GetProcessingSettings().overrideClusterizerFragmentLen);

  for (TPCFragmentTime i = start; i < end; i++) {
    int32_t zeros = 0;
    for (GlobalPad j = 0; j < TPC_NUM_OF_PADS; j++) {
      uint16_t q = map[{j, i}];
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

void GPUTPCClusterFinder::DumpPeakMap(std::ostream& out, std::string_view title)
{
  out << "\nClusterer - " << title << " - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";

  Array2D<uint8_t> map(mPpeakMap);

  out << std::hex;

  TPCFragmentTime start = 0;
  TPCFragmentTime end = TPC_MAX_FRAGMENT_LEN_PADDED(mRec->GetProcessingSettings().overrideClusterizerFragmentLen);

  for (TPCFragmentTime i = start; i < end; i++) {
    int32_t zeros = 0;

    out << i << ":";
    for (GlobalPad j = 0; j < TPC_NUM_OF_PADS; j++) {
      uint8_t q = map[{j, i}];
      zeros += (q == 0);
      if (q != 0) {
        if (zeros > 0) {
          out << " z" << zeros;
          zeros = 0;
        }

        out << " p" << int32_t{q};
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
  for (uint32_t i = 0; i < mPmemory->counters.nPositions; i++) {
    out << int32_t{mPisPeak[i]};
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
}

void GPUTPCClusterFinder::DumpPeaksCompacted(std::ostream& out)
{
  const auto nPeaks = mPmemory->counters.nPeaks;

  out << "\nClusterer - Compacted Peaks - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << ": " << nPeaks << "\n";
  for (size_t i = 0; i < nPeaks; i++) {
    const auto& pos = mPpeakPositions[i];
    out << pos.time() << " " << int32_t{pos.pad()} << " " << int32_t{pos.row()} << "\n";
  }
}

void GPUTPCClusterFinder::DumpSuppressedPeaks(std::ostream& out)
{
  const auto& fragment = mPmemory->fragment;
  const auto nPeaks = mPmemory->counters.nPeaks;

  out << "\nClusterer - NoiseSuppression - Slice " << mISlice << " - Fragment " << fragment.index << mISlice << "\n";
  for (uint32_t i = 0; i < nPeaks; i++) {
    out << int32_t{mPisPeak[i]};
    if ((i + 1) % 100 == 0) {
      out << "\n";
    }
  }
}

void GPUTPCClusterFinder::DumpSuppressedPeaksCompacted(std::ostream& out)
{
  const auto& fragment = mPmemory->fragment;
  const auto nPeaks = mPmemory->counters.nClusters;

  out << "\nClusterer - Noise Suppression Peaks Compacted - Slice " << mISlice << " - Fragment " << fragment.index << ": " << nPeaks << "\n";
  for (size_t i = 0; i < nPeaks; i++) {
    const auto& peak = mPfilteredPeakPositions[i];
    out << peak.time() << " " << int32_t{peak.pad()} << " " << int32_t{peak.row()} << "\n";
  }
}

void GPUTPCClusterFinder::DumpClusters(std::ostream& out)
{
  out << "\nClusterer - Clusters - Slice " << mISlice << " - Fragment " << mPmemory->fragment.index << "\n";

  for (int32_t i = 0; i < GPUCA_ROW_COUNT; i++) {
    size_t N = mPclusterInRow[i];
    const tpc::ClusterNative* row = &mPclusterByRow[i * mNMaxClusterPerRow];

    std::vector<tpc::ClusterNative> sortedCluster;
    sortedCluster.insert(sortedCluster.end(), row, row + N);
    std::sort(sortedCluster.begin(), sortedCluster.end());

    out << "Row: " << i << ": " << N << "\n";
    for (const auto& cl : sortedCluster) {
      out << std::hex << cl.timeFlagsPacked << std::dec << " " << cl.padPacked << " " << int32_t{cl.sigmaTimePacked} << " " << int32_t{cl.sigmaPadPacked} << " " << cl.qMax << " " << cl.qTot << "\n";
    }
  }
}
