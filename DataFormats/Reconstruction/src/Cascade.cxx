// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/Cascade.h"

using namespace o2::dataformats;

Cascade::Cascade(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz, const std::array<float, 6>& covxyz,
                 const o2::track::TrackParCov& v0, const o2::track::TrackParCov& bachelor,
                 int v0ID, GIndex bachelorID, o2::track::PID pid)
{
  std::array<float, 21> covV{}, covB{};
  v0.getCovXYZPxPyPzGlo(covV);
  bachelor.getCovXYZPxPyPzGlo(covB);
  for (int i = 0; i < 21; i++) {
    covV[i] += covB[i];
  }
  for (int i = 0; i < 6; i++) {
    covV[i] = covxyz[i];
  }
  this->set(xyz, pxyz, covV, v0.getCharge() + bachelor.getCharge(), true, pid);
  setV0ID(v0ID);
  setBachelorID(bachelorID);
  setV0Track(v0);
  setBachelorTrack(bachelor);
}
