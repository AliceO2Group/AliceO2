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
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::tof;

//__________________________________________________
void CosmicProcessor::clear()
{
  mCosmicInfo.clear();
  mCosmicInfo.reserve(5200);
  mCosmicTrack.clear();
  mCosmicTrack.reserve(1200);
  mSizeTrack.clear();
  mSizeTrack.reserve(1200);

  mCosmicTrackTemp.reserve(200);

  memset(mCounters, 0, sizeof(int) * Geo::NCHANNELS);
}
//__________________________________________________
void CosmicProcessor::processTrack()
{
  if (mCosmicTrack.size() > 1000) {
    return;
  }

  // fit the track and check if is good
  mSizeTrack.push_back(int(mCosmicTrackTemp.size()));
  mCosmicTrack.insert(mCosmicTrack.end(), mCosmicTrackTemp.begin(), mCosmicTrackTemp.end());
}
//__________________________________________________
void CosmicProcessor::process(DigitDataReader& reader, bool fill)
{
  if (mCosmicInfo.size() > 5000) {
    return;
  }

  TStopwatch timerProcess;
  timerProcess.Start();

  reader.init();
  const o2::raw::HBFUtils& hbfutils = o2::raw::HBFUtils::Instance();

  auto array = reader.getDigitArray();
  int ndig = array->size();
  int ndig2 = ndig * fill;

  int npair = 0;

  int bcdist = 200;
  float thr = 5000000; // in ps

  int volID1[5], volID2[5];
  float pos1[3], pos2[3], pos2or[3];

  std::vector<int> mIcandClus;
  mIcandClus.reserve(200);

  bool trackFound = false;

  for (int i = 0; i < ndig; i++) {
    if (npair >= 150) {
      break;
    }
    auto& dig1 = (*array)[i];
    int ch1 = dig1.getChannel();
    mCounters[ch1]++;
    if (mCounters[ch1] > 3) {
      continue;
    }
    auto ir0 = hbfutils.getFirstIRofTF(dig1.getIR());

    int64_t bc1 = int64_t(dig1.getBC());
    int tdc1 = int(dig1.getTDC());
    float tot1 = dig1.getTOT() * 48.8E-3;
    Geo::getVolumeIndices(ch1, volID1);
    Geo::getPos(volID1, pos1);

    float tm1 = (dig1.getIR().differenceInBC(ir0) * 1024 + tdc1) * Geo::TDCBIN; // in ps

    mIcandClus.clear();
    if (!trackFound) {
      mCosmicTrackTemp.clear();
      mCosmicTrackTemp.emplace_back(ch1, pos1[0], pos1[1], pos1[2], 0.0, tot1);
    }

    for (int j = i + 1; j < ndig2; j++) {
      auto& dig2 = (*array)[j];
      int64_t bc2 = int64_t(dig2.getBC()) - bc1;
      if (labs(bc2) > bcdist) {
        continue;
      }

      int ch2 = dig2.getChannel();
      if (mCounters[ch2] > 3) {
        continue;
      }
      float tm2 = (dig2.getIR().differenceInBC(ir0) * 1024 + int(dig2.getTDC())) * Geo::TDCBIN; // in ps

      int tdc2 = int(dig2.getTDC()) - tdc1;
      float tot2 = dig2.getTOT() * 48.8E-3;
      Geo::getVolumeIndices(ch2, volID2);
      Geo::getPos(volID2, pos2or);
      pos2[0] = pos2or[0] - pos1[0];
      pos2[1] = pos2or[1] - pos1[1];
      pos2[2] = pos2or[2] - pos1[2];

      float dtime = (bc2 * 1024 + tdc2) * Geo::TDCBIN; // in ps

      float lt2 = pos2[0] * pos2[0] + pos2[1] * pos2[1];
      float l = sqrt(lt2 + pos2[2] * pos2[2]);

      if (!trackFound && lt2 < 40000 && fabs(dtime) < 10000) {
        mCosmicTrackTemp.emplace_back(ch2, pos2or[0], pos2or[1], pos2or[2], dtime, tot2);
      }

      if (l < 500) {
        continue;
      }
      if (pos2[1] > 0) {
        l = -l;
      }

      dtime -= l * 33.356409; // corrected for pad distance assuiming muonn downward

      if (fabs(dtime) > thr) {
        continue;
      }

      npair++;
      mCosmicInfo.emplace_back(ch1, ch2, dtime, tot1, tot2, l, tm1, tm2);
    }
    if (!trackFound && ndig < 1000 && mCosmicTrackTemp.size() > 4) {
      trackFound = true;
    }
  }

  if (trackFound) {
    processTrack();
  }

  LOG(DEBUG) << "We had " << ndig << " digits in this event";
  timerProcess.Stop();
}
