// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <benchmark/benchmark.h>

#include "Framework/ContextRegistry.h"
#include "Framework/ArrowContext.h"
#include "Framework/StringContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/MessageContext.h"
#include <TObject.h>

using namespace o2::framework;

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
static void BM_ContextRegistrySingleGet(benchmark::State& state)
{
  FairMQDeviceProxy proxy(nullptr);
  ArrowContext c0(proxy);
  StringContext c1(proxy);
  RawBufferContext c2(proxy);
  RootObjectContext c3(proxy);
  MessageContext c4(proxy);
  ContextRegistry registry({&c0, &c1, &c2, &c3, &c4});

  for (auto _ : state) {
    registry.get<MessageContext>();
  }
}

BENCHMARK(BM_ContextRegistrySingleGet);

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
static void BM_ContextRegistryMultiGet(benchmark::State& state)
{
  FairMQDeviceProxy proxy(nullptr);
  ArrowContext c0(proxy);
  StringContext c1(proxy);
  RawBufferContext c2(proxy);
  RootObjectContext c3(proxy);
  MessageContext c4(proxy);
  ContextRegistry registry({&c0, &c1, &c2, &c3, &c4});

  for (auto _ : state) {
    registry.get<MessageContext>();
    registry.get<RawBufferContext>();
    registry.get<RootObjectContext>();
  }
}

BENCHMARK(BM_ContextRegistryMultiGet);

BENCHMARK_MAIN();
