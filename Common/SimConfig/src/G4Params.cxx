// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimConfig/G4Params.h"
O2ParamImpl(o2::conf::G4Params);

namespace o2
{
namespace conf
{

namespace
{
static const std::string confstrings[3] = {"FTFP_BERT_EMV+optical", "FTFP_BERT_EMV+optical+biasing", "FTFP_INCLXX_EMV+optical"};
}

std::string const& G4Params::getPhysicsConfigString() const
{
  return confstrings[(int)physicsmode];
}

} // namespace conf
} // namespace o2
