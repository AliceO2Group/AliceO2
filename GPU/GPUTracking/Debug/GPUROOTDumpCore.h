// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUROOTDumpCore.h
/// \author David Rohr

#ifndef GPUROOTDUMPCORE_H
#define GPUROOTDUMPCORE_H

#include "GPUCommonDef.h"
#include <memory>
#include <vector>

class TFile;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUROOTDumpCore;

class GPUROOTDumpBase
{
 public:
  virtual void write() = 0;

 protected:
  GPUROOTDumpBase();
  std::weak_ptr<GPUROOTDumpCore> mCore;
};

class GPUROOTDumpCore
{
#if !defined(GPUCA_NO_ROOT) && !defined(GPUCA_GPUCODE)
  friend class GPUReconstruction;
  friend class GPUROOTDumpBase;

 private:
  struct GPUROOTDumpCorePrivate {
  };

 public:
  GPUROOTDumpCore(const GPUROOTDumpCore&) = delete;
  GPUROOTDumpCore operator=(const GPUROOTDumpCore&) = delete;
  GPUROOTDumpCore(GPUROOTDumpCorePrivate); // Cannot be declared private directly since used with new
  ~GPUROOTDumpCore();

 private:
  static std::shared_ptr<GPUROOTDumpCore> getAndCreate();
  static std::weak_ptr<GPUROOTDumpCore> get() { return sInstance; }
  static std::weak_ptr<GPUROOTDumpCore> sInstance;
  std::unique_ptr<TFile> mFile;
  std::vector<GPUROOTDumpBase*> mBranches;
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
