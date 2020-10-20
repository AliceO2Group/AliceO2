// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_AODREADERHELPERS_H_
#define O2_FRAMEWORK_AODREADERHELPERS_H_

#include "Framework/TableBuilder.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/Logger.h"
#include <uv.h>

namespace o2::framework::readers
{

struct RuntimeWatchdog {
  int numberTimeFrames;
  uint64_t startTime;
  uint64_t lastTime;
  double runTime;
  uint64_t runTimeLimit;

  RuntimeWatchdog(Long64_t limit)
  {
    numberTimeFrames = -1;
    startTime = uv_hrtime();
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

    auto nowTime = uv_hrtime();

    // time spent to process the time frame
    double time_spent = numberTimeFrames < 1 ? (double)(nowTime - lastTime) / 1.E9 : 0.;
    runTime += time_spent;
    lastTime = nowTime;

    return ((double)(lastTime - startTime) / 1.E9 + runTime / (numberTimeFrames + 1)) < runTimeLimit;
  }

  void printOut()
  {
    LOGP(INFO, "RuntimeWatchdog");
    LOGP(INFO, "  run time limit: {}", runTimeLimit);
    LOGP(INFO, "  number of time frames: {}", numberTimeFrames);
    LOGP(INFO, "  estimated run time per time frame: {}", (numberTimeFrames >= 0) ? runTime / (numberTimeFrames + 1) : 0.);
    LOGP(INFO, "  estimated total run time: {}", (double)(lastTime - startTime) / 1.E9 + ((numberTimeFrames >= 0) ? runTime / (numberTimeFrames + 1) : 0.));
  }
};

struct AODReaderHelpers {
  static AlgorithmSpec rootFileReaderCallback();
  static AlgorithmSpec aodSpawnerCallback(std::vector<InputSpec> requested);
};

} // namespace o2::framework::readers

#endif // O2_FRAMEWORK_AODREADERHELPERS_H_
