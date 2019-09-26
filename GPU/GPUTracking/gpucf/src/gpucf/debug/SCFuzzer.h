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

#include <gpucf/algorithms/StreamCompaction.h>

#include <CL/cl2.h>

namespace gpucf
{

class ClEnv;

class SCFuzzer
{
 public:
  SCFuzzer(ClEnv&);

  bool run(size_t);

 private:
  StreamCompaction streamCompaction;

  cl::Context context;
  cl::Device device;

  cl::CommandQueue queue;

  cl::Buffer digitsInBuf;
  cl::Buffer digitsOutBuf;
  cl::Buffer predicateBuf;

  void setup(ClEnv&);

  void dumpResult(const std::vector<std::vector<int>>&);

  bool repeatTest(size_t, size_t);
  bool runTest(size_t);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
