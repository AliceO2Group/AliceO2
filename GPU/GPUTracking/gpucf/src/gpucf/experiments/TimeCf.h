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

#include <gpucf/algorithms/ClusterFinderConfig.h>
#include <gpucf/common/Digit.h>
#include <gpucf/experiments/Experiment.h>

#include <nonstd/span.h>

namespace gpucf
{

class TimeCf : public Experiment
{
 public:
  TimeCf(const std::string&,
         filesystem::path,
         ClusterFinderConfig,
         nonstd::span<const Digit>,
         size_t);

  void run(ClEnv&) override;

 private:
  std::string name;
  filesystem::path tgtFile;

  size_t repeats;
  nonstd::span<const Digit> digits;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
