// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CASCADE_H
#define ALICEO2_CASCADE_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include <array>
#include <Math/SVector.h>

namespace o2
{
namespace dataformats
{

class Cascade : public V0
{
 public:
  Cascade() = default;
  Cascade(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz, const std::array<float, 6>& covxyz,
          const o2::track::TrackParCov& v0, const o2::track::TrackParCov& bachelor,
          int v0ID, GIndex bachelorID, o2::track::PID pid = o2::track::PID::XiMinus);

  GIndex getBachelorID() const { return mProngIDs[1]; }
  void setBachelorID(GIndex gid) { mProngIDs[1] = gid; }

  int getV0ID() const { return int(mProngIDs[0].getRaw()); }
  void setV0ID(int vid) { mProngIDs[0].setRaw(GIndex::Base_t(vid)); }

  const Track& getV0Track() const { return mProngs[0]; }
  Track& getV0Track() { return mProngs[0]; }

  const Track& getBachelorTrack() const { return mProngs[1]; }
  Track& getBachelorTrack() { return mProngs[1]; }

  void setV0Track(const Track& t) { mProngs[0] = t; }
  void setBachelorTrack(const Track& t) { mProngs[1] = t; }

 protected:
  GIndex getProngID(int i) const = delete;

  ClassDefNV(Cascade, 1);
};

} // namespace dataformats
} // namespace o2
#endif
