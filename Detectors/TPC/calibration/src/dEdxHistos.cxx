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

#include "TPCCalibration/dEdxHistos.h"

#include <array>
#include <cstddef>
#include <string_view>
#include <utility>

//o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "Framework/Logger.h"
#include "TPCCalibration/FastHisto.h"

//root includes
#include "TFile.h"

using namespace o2::tpc;

dEdxHistos::dEdxHistos(unsigned int nBins, const TrackCuts& cuts)
  : mCuts{cuts}, mHist{{{nBins, 0, static_cast<float>(nBins)}, {nBins, 0, static_cast<float>(nBins)}}}
{
}

void dEdxHistos::fill(const gsl::span<const TrackTPC> tracks)
{
  for (const auto& track : tracks) {

    // applying cut
    if (!mApplyCuts || mCuts.goodTrack(track)) {
      // filling histogram
      if (track.hasASideClustersOnly()) {
        mEntries[0]++;
        mHist[0].fill(track.getdEdx().dEdxTotTPC);
      } else if (track.hasCSideClustersOnly()) {
        mEntries[1]++;
        mHist[1].fill(track.getdEdx().dEdxTotTPC);
      }
    }
  }
}

void dEdxHistos::merge(const dEdxHistos* other)
{
  for (size_t i = 0; i < mHist.size(); i++) {
    const auto binCount = mHist[i].getNBins();
    for (size_t bin = 0; bin < binCount; bin++) {
      float bin_content = other->getHists()[i].getBinContent(bin);
      mHist[i].fillBin(bin, bin_content);
    }
  }
}

void dEdxHistos::print() const
{
  LOG(INFO) << "Total number of entries: " << mEntries[0] << " in A side, " << mEntries[1] << " in C side";
}

void dEdxHistos::dumpToFile(std::string_view fileName) const
{
  TFile file(fileName.data(), "recreate");
  file.WriteObject(&mHist[0], "dEdxTotTPC A side");
  file.WriteObject(&mHist[1], "dEdxTotTPC C side");

  file.Close();
}
