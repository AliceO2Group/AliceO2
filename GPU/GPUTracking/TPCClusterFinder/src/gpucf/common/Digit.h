// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/Object.h>
#include <gpucf/common/RawDigit.h>
#include <gpucf/common/SectorMap.h>
#include <gpucf/common/serialization.h>

#include <shared/Digit.h>

#include <iosfwd>
#include <string>
#include <vector>

namespace gpucf
{

class Digit : public PackedDigit
{

 public:
  static SectorMap<std::vector<Digit>> bySector(
    const SectorMap<std::vector<RawDigit>>&);

  Digit();
  Digit(const RawDigit&);
  Digit(float, int, int, int);

  Object serialize() const;
  void deserialize(const Object&);

  float getCharge() const;

  int localRow() const;
  int cru() const;

  bool operator==(const Digit&) const;
};

std::ostream& operator<<(std::ostream&, const Digit&);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
