// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackletLabels.h
/// \brief Used for storing the MC labels for the TRD tracklets

/// \author Ole Schmidt

#ifndef GPUTRDTRACKLETLABELS_H
#define GPUTRDTRACKLETLABELS_H

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct GPUTRDTrackletLabels {
  int mLabel[3];
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDTRACKLETLABELS_H
