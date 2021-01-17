// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Configuration.h"

namespace o2
{
namespace its
{
static auto& sVertexerParamITS = o2::its::VertexerParamConfig::Instance();
static auto& sCATrackerParamITS = o2::its::TrackerParamConfig::Instance();

O2ParamImpl(o2::its::VertexerParamConfig);
O2ParamImpl(o2::its::TrackerParamConfig);
} // namespace its
} // namespace o2
