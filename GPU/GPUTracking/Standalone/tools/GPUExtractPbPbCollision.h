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

static void GPUExtractPbPbCollision(GPUParam& param, GPUTrackingInOutPointers& ioPtrs)
{
  std::vector<unsigned int> counts(param.par.continuousMaxTimeBin + 1);
  std::vector<unsigned int> sums(param.par.continuousMaxTimeBin + 1);
  std::vector<unsigned int> countsTracks(param.par.continuousMaxTimeBin + 1);
  std::vector<unsigned int> sumsTracks(param.par.continuousMaxTimeBin + 1);
  std::vector<bool> mask(param.par.continuousMaxTimeBin + 1);
  const int driftlength = 520;
  const bool checkAfterGlow = true;
  const int afterGlowLength = checkAfterGlow ? 8000 : 0;
  for (unsigned int i = 0; i < ioPtrs.clustersNative->nClustersTotal; i++) {
    int time = ioPtrs.clustersNative->clustersLinear[i].getTime();
    if (time < 0 || time > param.par.continuousMaxTimeBin) {
      fprintf(stderr, "Invalid time %d > %d\n", time, param.par.continuousMaxTimeBin);
      throw std::runtime_error("Invalid Time");
    }
    counts[time]++;
  }
  for (unsigned int i = 0; i < ioPtrs.nMergedTracks; i++) {
    if (ioPtrs.mergedTracks[i].NClusters() < 40) {
      continue;
    }
    int time = ioPtrs.mergedTracks[i].GetParam().GetTZOffset();
    if (time < 0 || time > param.par.continuousMaxTimeBin) {
      continue;
    }
    countsTracks[time]++;
  }
  int first = 0, last = 0;
  for (int i = driftlength; i < param.par.continuousMaxTimeBin; i++) {
    if (counts[i]) {
      first = i;
      break;
    }
  }
  for (int i = param.par.continuousMaxTimeBin + 1 - driftlength; i > 0; i--) {
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
  /*for (int i = first; i < last - driftlength; i++) {
    printf("STAT %d: %u %u (trks %u)\n", i, sums[i], counts[i], sumsTracks[i]);
  }*/
  bool found = false;
  do {
    found = false;
    unsigned int max = 0, maxpos = 0;
    for (int i = first; i < last - driftlength - afterGlowLength; i++) {
      if (sums[i] > 10 * min && sums[i] > avg && sumsTracks[i] > 3) {
        bool noColInAfterGlow = true;
        if (checkAfterGlow) {
          for (int ii = i + driftlength; ii < i + driftlength + afterGlowLength; ii++) {
            if (sums[ii] > 10 * min && sums[ii] > avg && sumsTracks[ii] > 3) {
              noColInAfterGlow = false;
            }
          }
        }
        if (noColInAfterGlow && sums[i] > max) {
          max = sums[i];
          maxpos = i;
          found = true;
        }
      }
    }
    if (found) {
      unsigned int glow = 0;
      unsigned int glowcount = 0;
      if (checkAfterGlow) {
        int glowstart = maxpos + driftlength;
        int glowend = std::min<int>(last, maxpos + driftlength + afterGlowLength);
        for (int i = glowstart; i < glowend; i++) {
          glowcount++;
          glow += counts[i];
        }
        // printf("AFTERGLOW RANGE %d %d\n", glowstart, glowend);
      }
      printf("MAX %d: %u (Tracks %u) Glow %d (%d)\n", maxpos, max, sumsTracks[maxpos], glow, glowcount);
      for (int i = std::max<int>(first, maxpos - driftlength); i < std::min<int>(last, maxpos + driftlength + afterGlowLength); i++) {
        sums[i] = 0;
        mask[i] = true;
      }
    }
  } while (found && !checkAfterGlow);
  unsigned int noise = 0;
  unsigned int noisecount = 0;
  for (int i = first; i < last; i++) {
    if (!mask[i]) {
      noise += counts[i];
      noisecount++;
    }
  }
  printf("AVERAGE NOISE: %d\n", noise / noisecount);
}
