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

/// \file GPUROOTDump.h
/// \author David Rohr

#ifndef GPUROOTDUMP_H
#define GPUROOTDUMP_H

#include "GPUCommonDef.h"
#if !defined(GPUCA_NO_ROOT) && !defined(GPUCA_GPUCODE)
#include "GPUROOTDumpCore.h"
#include <TTree.h>
#include <TNtuple.h>
#else
class TNtuple;
#endif
#ifndef GPUCA_GPUCODE
#include <memory>
#include <stdexcept>
#endif

namespace o2::gpu
{
#if !defined(GPUCA_NO_ROOT) && !defined(GPUCA_GPUCODE)
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

template <typename... Args>
class GPUROOTDump;

template <class T, typename... Args>
class GPUROOTDump<T, Args...> : public GPUROOTDump<Args...>
{
 public:
  template <typename... Names>
  static GPUROOTDump<T, Args...>& get(const char* name1, Names... names) // return always the same instance, identified by template
  {
    static GPUROOTDump<T, Args...> instance(name1, names...);
    return instance;
  }
  template <typename... Names>
  static GPUROOTDump<T, Args...> getNew(const char* name1, Names... names) // return new individual instance
  {
    return GPUROOTDump<T, Args...>(name1, names...);
  }

  void Fill(const T& o, const Args&... args)
  {
    stdspinlock spinlock(GPUROOTDumpBase::mMutex);
    FillInternal(o, args...);
  }

 protected:
  void FillInternal(const T& o, const Args&... args)
  {
    mObj = o;
    GPUROOTDump<Args...>::Fill(args...);
  }

  using GPUROOTDump<Args...>::mTree;
  template <typename... Names>
  GPUROOTDump(const char* name1, Names... names) : GPUROOTDump<Args...>(names...)
  {
    stdspinlock spinlock(GPUROOTDumpBase::mMutex);
    mTree->Branch(name1, &mObj);
  }

 private:
  T mObj;
};

template <>
class GPUROOTDump<> : public GPUROOTDumpBase
{
 public:
  void write() override { mTree->Write(); }

 protected:
  void Fill()
  {
    mTree->Fill();
  }

  GPUROOTDump(const char* name1, const char* nameTree = nullptr)
  {
    if (nameTree == nullptr) {
      nameTree = name1;
    }
    stdspinlock spinlock(GPUROOTDumpBase::mMutex);
    mTree = new TTree(nameTree, nameTree);
  }
  TTree* mTree = nullptr;
};

template <>
class GPUROOTDump<TNtuple> : public GPUROOTDumpBase
{
 public:
  static GPUROOTDump<TNtuple>& get(const char* name, const char* options)
  {
    static GPUROOTDump<TNtuple> instance(name, options);
    return instance;
  }
  static GPUROOTDump<TNtuple> getNew(const char* name, const char* options)
  {
    return GPUROOTDump<TNtuple>(name, options);
  }

  void write() override { mNTuple->Write(); }

  template <typename... Args>
  void Fill(const Args&... args)
  {
    stdspinlock spinlock(GPUROOTDumpBase::mMutex);
    mNTuple->Fill(args...);
  }

 private:
  GPUROOTDump(const char* name, const char* options)
  {
    stdspinlock spinlock(GPUROOTDumpBase::mMutex);
    mNTuple = new TNtuple(name, name, options);
  }
  TNtuple* mNTuple;
};
#else
template <typename... Args>
class GPUROOTDump
{
 public:
  template <typename... Names>
  GPUd() static void Fill(Args... args)
  {
  }
  template <typename... Names>
  GPUd() static GPUROOTDump<Args...>& get(Args... args)
  {
    return GPUROOTDump<Args...>();
  }
  template <typename... Names>
  GPUd() static GPUROOTDump<Args...>& getNew(Args... args)
  {
    return GPUROOTDump<Args...>();
  }
};
#endif
} // namespace o2::gpu

#endif
