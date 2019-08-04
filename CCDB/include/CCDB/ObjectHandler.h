// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICE_O2_OBJECTHANDLER_H_
#define ALICE_O2_OBJECTHANDLER_H_

#include <string>
#include <vector>

namespace o2
{
namespace ccdb
{

class ObjectHandler
{
 public:
  ObjectHandler();
  virtual ~ObjectHandler();

  /// Returns the binary payload of a ROOT file as an std::string
  static void GetObject(const std::string& path, std::string& object);
};
} // namespace ccdb
} // namespace o2
#endif
