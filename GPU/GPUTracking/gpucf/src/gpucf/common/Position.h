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

#include <gpucf/common/Digit.h>

#include <shared/tpc.h>

namespace gpucf
{

class Position
{

 public:
  row_t row;
  pad_t pad;
  timestamp time;

  Position(const Digit&);
  Position(const Digit&, int, int);
  Position(row_t, pad_t, timestamp);

  bool operator==(const Position&) const;

  size_t idx() const;
};

} // namespace gpucf

namespace std
{

template <>
struct hash<gpucf::Position> {

  size_t operator()(const gpucf::Position& p) const
  {
    return std::hash<size_t>()(p.idx());
  }
};

} // namespace std

// vim: set ts=4 sw=4 sts=4 expandtab:
