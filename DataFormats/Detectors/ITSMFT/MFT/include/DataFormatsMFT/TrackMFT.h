// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.h
/// \brief Definition of the MFT track
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 8, 2018

#ifndef ALICEO2_MFT_TRACKMFT_H
#define ALICEO2_MFT_TRACKMFT_H

#include <vector>

#include "ReconstructionDataFormats/Track.h"

namespace o2
{

namespace itsmft
{
class Cluster;
}

namespace mft
{
class TrackMFT : public o2::track::TrackParCov
{
  using Cluster = o2::itsmft::Cluster;

 public:
  using o2::track::TrackParCov::TrackParCov;

  TrackMFT() = default;
  TrackMFT(const TrackMFT& t) = default;
  TrackMFT& operator=(const TrackMFT& tr) = default;
  ~TrackMFT() = default;

 private:
  ClassDefNV(TrackMFT, 1);
};
} // namespace mft
} // namespace o2

#endif
