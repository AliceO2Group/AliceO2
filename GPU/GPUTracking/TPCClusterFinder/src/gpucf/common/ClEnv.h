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

#include <CL/cl2.h>

#include <args/args.h>
#include <filesystem/path.h>

#include <memory>
#include <string>
#include <vector>

namespace gpucf
{

class ClEnv
{

 public:
  class Flags
  {

   public:
    args::ValueFlag<std::string> clSrcDir;
    args::ValueFlag<size_t> gpuId;
    args::Flag useCpu;

    Flags(args::Group& required, args::Group& optional)
      : clSrcDir(required, "clsrc", "Base directory of cl source files.",
                 {'s', "src"}),
        gpuId(optional, "gpuid", "Id of the gpu device.",
              {'g', "gpu"}, 0),
        useCpu(optional, "", "Use cpu as openCl device.", {"clcpu"})
    {
    }
  };

  ClEnv(
    const filesystem::path& srcDir,
    ClusterFinderConfig cfg,
    size_t gpuid = 0,
    bool useCpu = false);

  ClEnv(Flags& flags, ClusterFinderConfig cfg)
    : ClEnv(
        args::get(flags.clSrcDir),
        cfg,
        args::get(flags.gpuId),
        args::get(flags.useCpu))
  {
  }

  cl::Context getContext() const
  {
    return context;
  }

  cl::Program getProgram() const
  {
    return program;
  }

  cl::Device getDevice() const
  {
    return devices[gpuId];
  }

 private:
  static const std::vector<filesystem::path> srcs;

  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;

  std::vector<std::string> defines;

  size_t gpuId;

  cl::Context context;
  cl::Program program;

  filesystem::path sourceDir;

  cl::Program buildFromSrc(bool);

  cl::Program::Sources loadSrc(const std::vector<filesystem::path>& srcFiles);

  void addDefine(const std::string&);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
