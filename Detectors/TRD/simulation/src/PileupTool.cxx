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

#include "TRDSimulation/PileupTool.h"
#include "TRDSimulation/TRDSimParams.h"

using namespace o2::trd;
using namespace o2::trd::constants;

SignalContainer PileupTool::addSignals(std::deque<std::array<SignalContainer, constants::MAXCHAMBER>>& pileupSignals, const double& triggerTime)
{
  SignalContainer addedSignalsMap;
  int nSignalsToRemove = 0;
  for (const auto& collection : pileupSignals) {
    bool pileupSignalBecomesObsolete = false;
    for (int det = 0; det < MAXCHAMBER; ++det) {
      const auto& signalMap = collection[det]; //--> a map with active pads only for this chamber
      for (const auto& signal : signalMap) {   // loop over active pads only, if there is any
        const int& key = signal.first;
        const SignalArray& signalArray = signal.second;
        // check if the signal is from a previous event
        if (signalArray.firstTBtime < triggerTime) {
          pileupSignalBecomesObsolete = true;
          if ((triggerTime - signalArray.firstTBtime) > TRDSimParams::Instance().readoutTimeNS) { // OS: READOUT_TIME should actually be drift time (we want to ignore signals which don't contribute signal anymore at triggerTime)
            continue;                                                              // ignore the signal if it  is too old.
          }
          // add only what's leftover from this signal
          // 0.01 = samplingRate/1000, 1/1000 to go from ns to micro-s, the sampling rate is in 1/micro-s
          int idx = (int)((triggerTime - signalArray.firstTBtime) * 0.01); // number of bins to skip
          auto it0 = signalArray.signals.begin() + idx;
          auto it1 = addedSignalsMap[key].signals.begin();
          while (it0 < signalArray.signals.end()) {
            *it1 += *it0;
            it0++;
            it1++;
          }
        } else {
          // the signal is from a subsequent event
          int idx = (int)((signalArray.firstTBtime - triggerTime) * 0.01); // time bin offset of the pileup signal wrt trigger time. Number of time bins to be added to the signal is constants::TIMEBINS - idx
          auto it0 = signalArray.signals.begin();
          auto it1 = addedSignalsMap[key].signals.begin() + idx;
          while (it1 < addedSignalsMap[key].signals.end()) {
            *it1 += *it0;
            it0++;
            it1++;
          }
        }
        // keep the labels
        for (const auto& label : signalArray.labels) {
          // do we want to keep all labels? what about e.g. a TR signal which does not contribute to the pileup of a previous event since the signal arrives too late, but then we will have its label?
          (addedSignalsMap[key].labels).push_back(label); // maybe check if the label already exists? is that even possible?
        }
      } // loop over active pads in detector
    }   // loop over detectors
    if (pileupSignalBecomesObsolete) {
      ++nSignalsToRemove;
    }
  } // loop over pileup container
  // remove all used added signals, keep those that can pileup to newer events.
  for (int i = 0; i < nSignalsToRemove; ++i) {
    pileupSignals.pop_front();
  }
  return addedSignalsMap;
}
