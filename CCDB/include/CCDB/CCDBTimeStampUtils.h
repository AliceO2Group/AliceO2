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

//
// Created by Sandro Wenzel on 2019-08-20.
//

#ifndef O2_CCDBTIMESTAMPUTILS_H
#define O2_CCDBTIMESTAMPUTILS_H

/// a couple of static helper functions to create timestamp values for CCDB queries or override obsolete objects

namespace o2
{
namespace ccdb
{
class CcdbApi;
class CcdbObjectInfo;

/// returns the timestamp in long corresponding to "now + secondsInFuture"
long getFutureTimestamp(int secondsInFuture);

/// returns the timestamp in long corresponding to "now"
long getCurrentTimestamp();

/// \brief Converting time into numerical time stamp representation
long createTimestamp(int year, int month, int day, int hour, int minutes, int seconds);

/// \brief set EOV of overriden objects to SOV-1 of overriding one if it is allowed
int adjustOverriddenEOV(CcdbApi& api, const CcdbObjectInfo& infoNew);

} // namespace ccdb
} // namespace o2

#endif //O2_CCDBTIMESTAMPUTILS_H
