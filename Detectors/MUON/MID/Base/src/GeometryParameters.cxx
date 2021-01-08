// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/src/GeometryParameters.cxx
/// \brief  Implementation of the geometrical parameters for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 March 2018
#include "MIDBase/GeometryParameters.h"

#include <stdexcept>
#include <sstream>
#include <string>

namespace o2
{
namespace mid
{
namespace geoparams
{
std::string getRPCVolumeName(RPCtype type, int chamber)
{
  /// Gets the RPC volume name
  std::string name = "";
  switch (type) {
    case RPCtype::Long:
      name += "long";
      break;
    case RPCtype::BottomCut:
      name += "bottomCut";
      break;
    case RPCtype::TopCut:
      name += "topCut";
      break;
    case RPCtype::Short:
      name += "short";
      break;
  }
  name += "RPC_" + std::to_string(11 + chamber);
  return name;
}

RPCtype getRPCType(int deId)
{
  /// Gets the RPC type
  int irpc = deId % 9;
  if (irpc == 4) {
    return RPCtype::Short;
  }
  if (irpc == 3) {
    return RPCtype::TopCut;
  }
  if (irpc == 5) {
    return RPCtype::BottomCut;
  }
  return RPCtype::Long;
}

std::string getChamberVolumeName(int chamber)
{
  /// Returns the chamber name in the geometry
  return "SC" + std::to_string(11 + chamber);
}

} // namespace geoparams
} // namespace mid
} // namespace o2
