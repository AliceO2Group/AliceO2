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

/// \file EventTimeMaker.h
/// \brief Definition of the TOF event time maker

#ifndef ALICEO2_TOF_EVENTTIMEMAKER_H
#define ALICEO2_TOF_EVENTTIMEMAKER_H

namespace o2
{

namespace tof
{

struct eventTimeContainer {
  eventTimeContainer(const float& e) : eventTime{e} {};
  float eventTime = 0.f;
};

template <typename tTracks>
eventTimeContainer evTimeMaker(const tTracks& tracks)
{
  for (auto track : tracks) {
    track.tofSignal();
  }
  return eventTimeContainer{0};
}

} // namespace tof
} // namespace o2

#endif /* ALICEO2_TOF_EVENTTIMEMAKER_H */