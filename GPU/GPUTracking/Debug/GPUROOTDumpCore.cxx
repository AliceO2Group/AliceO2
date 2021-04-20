// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUROOTDumpCore.cxx
/// \author David Rohr

#include "GPUROOTDumpCore.h"

#if (!defined(GPUCA_STANDALONE) || defined(GPUCA_BUILD_QA)) && !defined(GPUCA_GPUCODE)
#include <atomic>
#include <memory>
#include <TFile.h>

using namespace GPUCA_NAMESPACE::gpu;

std::weak_ptr<GPUROOTDumpCore> GPUROOTDumpCore::sInstance;

GPUROOTDumpCore::GPUROOTDumpCore()
{
  mFile.reset(new TFile("gpudebug.root", "recreate"));
}

GPUROOTDumpCore::~GPUROOTDumpCore()
{
  for (unsigned int i = 0; i < mBranches.size(); i++) {
    mBranches[i]->write();
  }
  mFile->Close();
}

std::shared_ptr<GPUROOTDumpCore> GPUROOTDumpCore::getAndCreate()
{
  static std::atomic_flag lock = ATOMIC_FLAG_INIT;
  while (lock.test_and_set(std::memory_order_acquire)) {
  }
  std::shared_ptr<GPUROOTDumpCore> retVal = sInstance.lock();
  if (!retVal) {
    retVal = std::make_shared<GPUROOTDumpCore>();
    sInstance = retVal;
  }
  lock.clear(std::memory_order_release);
  return retVal;
}

GPUROOTDumpBase::GPUROOTDumpBase()
{
  std::shared_ptr<GPUROOTDumpCore> p = GPUROOTDumpCore::get().lock();
  if (!p) {
    throw std::runtime_error("No instance of GPUROOTDumpCore exists");
  }
  p->mBranches.emplace_back(this);
  p->mFile->cd();
}

#endif
