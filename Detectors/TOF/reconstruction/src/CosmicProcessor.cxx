// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.cxx
/// \brief Implementation of the TOF cluster finder
#include <algorithm>
#include "FairLogger.h" // for LOG
#include "TOFReconstruction/CosmicProcessor.h"
#include <TStopwatch.h>

using namespace o2::tof;

//__________________________________________________
void CosmicProcessor::clear()
{
  mCosmicInfo.clear();
  for (int i = 0; i < Geo::NCHANNELS; i++) {
    mCounters[i] = 0;
  }
}
//__________________________________________________
void CosmicProcessor::process(DigitDataReader& reader, bool fill)
{
  TStopwatch timerProcess;
  timerProcess.Start();

  reader.init();

  auto array = reader.getDigitArray();
  int ndig = array->size();
  int ndig2 = ndig * fill;

  int bcdist = 200;
  float thr = 5000000; // in ps

  int volID1[5], volID2[5];
  float pos1[3], pos2[3];

  for (int i = 0; i < ndig; i++) {
    auto& dig1 = (*array)[i];
    int ch1 = dig1.getChannel();
    mCounters[ch1]++;
    if (mCounters[ch1] > 3) {
      continue;
    }
    int64_t bc1 = int64_t(dig1.getBC());
    int tdc1 = int(dig1.getTDC());
    float tot1 = dig1.getTOT() * 48.8E-3;
    Geo::getVolumeIndices(ch1, volID1);
    Geo::getPos(volID1, pos1);

    for (int j = i + 1; j < ndig2; j++) {
      auto& dig2 = (*array)[j];
      int64_t bc2 = int64_t(dig2.getBC()) - bc1;
      if (abs(bc2) > bcdist) {
        continue;
      }

      int ch2 = dig2.getChannel();
      if (mCounters[ch2] > 3) {
        continue;
      }
      int tdc2 = int(dig2.getTDC()) - tdc1;
      float tot2 = dig2.getTOT() * 48.8E-3;
      Geo::getVolumeIndices(ch2, volID2);
      Geo::getPos(volID2, pos2);
      pos2[0] -= pos1[0];
      pos2[1] -= pos1[1];
      pos2[2] -= pos1[2];

      float dtime = (bc2 * 1024 + tdc2) * Geo::TDCBIN; // in ps

      float l = sqrt(pos2[0] * pos2[0] + pos2[1] * pos2[1] + pos2[2] * pos2[2]);

      if (l < 500) {
        continue;
      }
      if (pos2[1] > 0)
        l = -l;

      dtime -= l * 33.356409; // corrected for pad distance assuiming muonn downward

      if (abs(dtime) > thr) {
        continue;
      }

      mCosmicInfo.emplace_back(ch1, ch2, dtime, tot1, tot2, l);
    }
  }

  LOG(DEBUG) << "We had " << ndig << " digits in this event";
  timerProcess.Stop();
}
