//
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche
#include "SegmentationCreator.h"
#include <iostream>
#include <map>

namespace o2 {
namespace mch {
namespace mapping {
namespace impl3 {

std::map<int, SegmentationCreator>& Creators()
{
  static std::map<int, SegmentationCreator> creators;
  return creators;
}

void registerSegmentationCreator(int segType, SegmentationCreator func)
{
  if (Creators().find(segType)!=Creators().end())
  {
    std::cerr << "WARNING: there is already a creator registered for segType=" << segType
                                                                               << ". Will override it\n";
  }
  Creators()[segType] = func;
}


SegmentationCreator getSegmentationCreator(int segType)
{
  return Creators()[segType];
}

}
}
}
}
