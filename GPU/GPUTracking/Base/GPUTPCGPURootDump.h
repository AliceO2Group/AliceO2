// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGPURootDump.h
/// \author David Rohr

#ifndef GPUTPCGPUROOTDUMP_H
#define GPUTPCGPUROOTDUMP_H

#if (!defined(GPUCA_STANDALONE) || defined(GPUCA_BUILD_QA)) && !defined(GPUCA_GPUCODE)
#include <TTree.h>
#include <TFile.h>
#include <TNtuple.h>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
namespace
{
template <class S>
struct internal_Branch {
  template <typename... Args>
  static void Branch(S* p, Args... args)
  {
  }
};
template <>
struct internal_Branch<TTree> {
  template <typename... Args>
  static void Branch(TTree* p, Args... args)
  {
    p->Branch(args...);
  }
};
} // namespace

template <class T>
class GPUTPCGPURootDump
{
 public:
  GPUTPCGPURootDump() = delete;
  GPUTPCGPURootDump(const GPUTPCGPURootDump<T>&) = delete;
  GPUTPCGPURootDump<T> operator=(const GPUTPCGPURootDump<T>&) = delete;
  template <typename... Args>
  GPUTPCGPURootDump(const char* filename, Args... args)
  {
    fFile = new TFile(filename, "recreate");
    fTree = new T(args...);
  }

  ~GPUTPCGPURootDump()
  {
    fTree->Write();
    fFile->Write();
    fFile->Close();
    delete fFile;
  }

  template <typename... Args>
  void Fill(Args... args)
  {
    fTree->Fill(args...);
  }
  template <typename... Args>
  void Branch(Args... args)
  {
    internal_Branch<T>::Branch(fTree, args...);
  }

 private:
  TFile* fFile = nullptr;
  T* fTree = nullptr;
};
#else
template <class T>
class GPUTPCGPURootDump
{
 public:
  GPUTPCGPURootDump() = delete;
  template <typename... Args>
  GPUTPCGPURootDump(const char* filename, Args... args)
  {
  }
  template <typename... Args>
  void Fill(Args... args)
  {
  }
  template <typename... Args>
  void Branch(Args... args)
  {
  }

 private:
  void *a, *b;
};
}
}
#endif

#endif
