// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EndCapsSimulation/DPLDigitizerParam.h"

namespace o2
{
namespace endcaps
{
// this makes sure that the constructor of the parameters is statically called
// so that these params are part of the parameter database
static auto& sDigitizerParamEC0 = o2::endcaps::DPLDigitizerParam<o2::detectors::DetID::EC0>::Instance();
} // namespace endcaps
} // namespace o2
