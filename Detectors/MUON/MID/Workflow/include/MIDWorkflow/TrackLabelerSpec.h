// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDWorkflow/TrackLabelerSpec.h
/// \brief  Data processor spec for MID TrackLabeler device
/// \author Diego Stocco <diego.stocco at cern.ch>
/// \date   11 June 2019

#ifndef O2_MID_TRACKLABELERSPEC_H
#define O2_MID_TRACKLABELERSPEC_H

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mid
{
framework::DataProcessorSpec getTrackLabelerSpec();
}
} // namespace o2

#endif //O2_MID_TRACKLABELERSPEC_H
