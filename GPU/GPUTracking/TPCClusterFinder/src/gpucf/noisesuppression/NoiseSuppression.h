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
#include <gpucf/common/Map.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/View.h>

#include <vector>

namespace gpucf
{

class NoiseSuppression
{

 public:
  RowMap<std::vector<Digit>> run(
    const RowMap<std::vector<Digit>>&,
    const RowMap<Map<bool>>&,
    const Map<float>&);

  std::vector<Digit> runOnAllRows(
    View<Digit>,
    const Map<bool>&,
    const Map<float>&);

  std::string getName() const
  {
    return name;
  }

 protected:
  NoiseSuppression(const std::string& name)
    : name(name)
  {
  }

  virtual std::vector<Digit> runImpl(
    View<Digit>,
    const Map<bool>&,
    const Map<float>&) = 0;

 private:
  std::string name;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
