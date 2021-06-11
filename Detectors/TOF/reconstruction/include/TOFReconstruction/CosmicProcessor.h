// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CosmicProcessor.h
/// \brief Managing digitsi in a RO window to provide cosmics candidates
#ifndef ALICEO2_TOF_COSMICPROCESSOR_H
#define ALICEO2_TOF_COSMICPROCESSOR_H

#include <utility>
#include <vector>
#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "TOFReconstruction/DataReader.h"
#include "DataFormatsTOF/CosmicInfo.h"

namespace o2
{

namespace tof
{
class CosmicProcessor
{
  using Digit = o2::tof::Digit;
  using StripData = o2::tof::DataReader::StripData;

 public:
  CosmicProcessor() = default;
  ~CosmicProcessor() = default;

  CosmicProcessor(const CosmicProcessor&) = delete;
  CosmicProcessor& operator=(const CosmicProcessor&) = delete;

  void process(DigitDataReader& r, bool fill = true);
  void processTrack();
  void clear();
  std::vector<CosmicInfo>* getCosmicInfo() { return (&mCosmicInfo); }
  std::vector<CalibInfoTrackCl>* getCosmicTrack() { return (&mCosmicTrack); }
  std::vector<int>* getCosmicTrackSize() { return (&mSizeTrack); }

 private:
  std::vector<CosmicInfo> mCosmicInfo;
  std::vector<CalibInfoTrackCl> mCosmicTrack;
  std::vector<CalibInfoTrackCl> mCosmicTrackTemp;
  std::vector<int> mSizeTrack;
  int mCounters[Geo::NCHANNELS];
};

} // namespace tof
} // namespace o2
#endif /* ALICEO2_TOF_COSMICPROCESSOR_H */
