// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  //template <class T> friend class GPUROOTDump;

 public:
  GPUROOTDumpCore(const GPUROOTDumpCore&) = delete;
  GPUROOTDumpCore operator=(const GPUROOTDumpCore&) = delete;
  GPUROOTDumpCore(); // Public since used with new, but should not be created manually.
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
