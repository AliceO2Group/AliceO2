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

/// \file GPUExtractPbPbCollision.h
/// \author David Rohr

static void GPUExtractPbPbCollision()
{
  std::vector<unsigned int> counts(param().par.continuousMaxTimeBin + 1);
  std::vector<unsigned int> sums(param().par.continuousMaxTimeBin + 1);
  std::vector<unsigned int> countsTracks(param().par.continuousMaxTimeBin + 1);
  std::vector<unsigned int> sumsTracks(param().par.continuousMaxTimeBin + 1);
  const int driftlength = 520;
  for (unsigned int i = 0; i < mIOPtrs.clustersNative->nClustersTotal; i++) {
    int time = mIOPtrs.clustersNative->clustersLinear[i].getTime();
    if (time < 0 || time > param().par.continuousMaxTimeBin) {
      fprintf(stderr, "Invalid time %d > %d\n", time, param().par.continuousMaxTimeBin);
      throw std::runtime_error("Invalid Time");
    }
    counts[time]++;
  }
  for (unsigned int i = 0; i < mIOPtrs.nMergedTracks; i++) {
    if (mIOPtrs.mergedTracks[i].NClusters() < 40) {
      continue;
    }
    int time = mIOPtrs.mergedTracks[i].GetParam().GetTZOffset();
    if (time < 0 || time > param().par.continuousMaxTimeBin) {
      continue;
    }
    countsTracks[time]++;
  }
  int first = 0, last = 0;
  for (int i = driftlength; i < param().par.continuousMaxTimeBin; i++) {
    if (counts[i]) {
      first = i;
      break;
    }
  }
  for (int i = param().par.continuousMaxTimeBin + 1 - driftlength; i > 0; i--) {
    if (counts[i - 1]) {
      last = i;
      break;
    }
  }
  unsigned int count = 0;
  unsigned int countTracks = 0;
  unsigned int min = 1e9;
  unsigned long avg = 0;
  for (int i = first; i < last; i++) {
    count += counts[i];
    countTracks += countsTracks[i];
    if (i - first >= driftlength) {
      sums[i - driftlength] = count;
      sumsTracks[i - driftlength] = countTracks;
      if (count < min) {
        min = count;
      }
      avg += count;
      count -= counts[i - driftlength];
      countTracks -= countsTracks[i - driftlength];
    }
  }
  avg /= (last - first - driftlength);
  printf("BASELINE Min %d Avg %d\n", min, (int)avg);
  bool found = false;
  do {
    found = false;
    unsigned int max = 0, maxpos = 0;
    for (int i = first; i < last - driftlength; i++) {
      if (sums[i] > 10 * min && sums[i] > avg && sumsTracks[i] > 3 && sums[i] > max) {
        max = sums[i];
        maxpos = i;
        found = true;
      }
    }
    if (found) {
      printf("MAX %d: %u (Tracks %u)\n", maxpos, max, sumsTracks[maxpos]);
      for (int i = std::max<int>(first, maxpos - driftlength); i < std::min<int>(last, maxpos + driftlength); i++) {
        sums[i] = 0;
      }
    }
  } while (found);
  /*for (int i = first; i < last - driftlength; i++) {
    printf("STAT %d: %u (trks %u)\n", i, sums[i], sumsTracks[i]);
  }*/
}
