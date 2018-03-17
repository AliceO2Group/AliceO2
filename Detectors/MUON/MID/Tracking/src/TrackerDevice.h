// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   TrackerDevice.h
/// \brief  Track reconstruction device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 May 2017

#ifndef O2_MID_TRACKERDEVICE_H
#define O2_MID_TRACKERDEVICE_H

#include "FairMQDevice.h"
#include "Tracker.h"

namespace o2
{
namespace mid
{
/// Tracking device for MID
class TrackerDevice : public FairMQDevice
{
 public:
  TrackerDevice();
  ~TrackerDevice() override = default;

  TrackerDevice(const TrackerDevice&) = delete;
  TrackerDevice& operator=(const TrackerDevice&) = delete;
  TrackerDevice(TrackerDevice&&) = delete;
  TrackerDevice& operator=(TrackerDevice&&) = delete;

 private:
  bool handleData(FairMQMessagePtr&, int);
  Tracker mTracker; ///< Tracking class
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACKERDEVICE_H */
