// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef o2_framework_readers_AODReaderHelpers_INCLUDED_H
#define o2_framework_readers_AODReaderHelpers_INCLUDED_H

#include "Framework/TableBuilder.h"
#include "Framework/AlgorithmSpec.h"

namespace o2
{
namespace framework
{
namespace readers
{

struct RuntimeWatchdog {
  int numberTimeFrames;
  clock_t startTime;
  clock_t lastTime;
  double runTime;
  Long64_t runTimeLimit;

  RuntimeWatchdog(Long64_t limit)
  {
    numberTimeFrames = -1;
    startTime = clock();
    lastTime = startTime;
    runTime = 0.;
    runTimeLimit = limit;
  }

  bool update()
  {
    numberTimeFrames++;
    if (runTimeLimit <= 0) {
      return true;
    }

    auto nowTime = clock();

    // time spent to process the time frame
    double time_spent = numberTimeFrames < 1 ? (double)(nowTime - lastTime) / CLOCKS_PER_SEC : 0.;
    runTime += time_spent;
    lastTime = nowTime;

    return ((double)(lastTime - startTime) / CLOCKS_PER_SEC + runTime / (numberTimeFrames + 1)) < runTimeLimit;
  }

  void printOut()
  {
    LOGP(INFO, "RuntimeWatchdog");
    LOGP(INFO, "  run time limit: {}", runTimeLimit);
    LOGP(INFO, "  number of time frames: {}", numberTimeFrames);
    LOGP(INFO, "  estimated run time per time frame: {}", (numberTimeFrames >= 0) ? runTime / (numberTimeFrames + 1) : 0.);
    LOGP(INFO, "  estimated total run time: {}", (double)(lastTime - startTime) / CLOCKS_PER_SEC + ((numberTimeFrames >= 0) ? runTime / (numberTimeFrames + 1) : 0.));
  }
};

struct AODReaderHelpers {
  static AlgorithmSpec rootFileReaderCallback();
  static AlgorithmSpec aodSpawnerCallback(std::vector<InputSpec> requested);
};

} // namespace readers
} // namespace framework
} // namespace o2

#endif
