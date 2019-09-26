// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "PeakCountTest.h"

using namespace gpucf;

PeakCountTest::PeakCountTest(ClusterFinderConfig config, ClEnv env)
  : ClusterFinder(config, 100, env)
{
}

bool PeakCountTest::run(
  const Array2D<float>& charges,
  const Array2D<unsigned char>& /*isPeakGT*/,
  const Array2D<char>& /*peakCountGT*/)
{
  std::vector<Digit> digits = digitize(charges);

  for (const Digit& d : digits) {
    log::Debug() << d;
  }

  digitsToGPU.call(state, digits, queue);

  fillChargeMap.call(state, queue);

  findPeaks.call(state, queue);

  countPeaks.call(state, queue);

  size_t timebins = getWidthTime(charges);
  size_t pads = getWidthPad(charges);
  size_t elems = TPC_NUM_OF_PADS * (timebins + PADDING_TIME);

  std::vector<unsigned short> chargeMapBuf(elems);
  gpucpy<unsigned short>(
    state.chargeMap,
    chargeMapBuf,
    chargeMapBuf.size(),
    queue,
    true);
  Map<unsigned short> chargeMap = mapify<unsigned short>(chargeMapBuf, 0, pads, timebins);

  std::vector<unsigned char> isPeakBuf(elems);
  gpucpy<unsigned char>(state.peakMap, isPeakBuf, isPeakBuf.size(), queue, true);
  Map<unsigned char> isPeak = mapify<unsigned char>(isPeakBuf, 0, pads, timebins);

  log::Debug() << "chargeMap\n"
               << print(chargeMap, pads, timebins);
  log::Debug() << "isPeakMap\n"
               << print(isPeak, pads, timebins);

  log::Debug() << "isPeak:";
  std::vector<unsigned char> isPeakPred(digits.size());
  gpucpy<unsigned char>(state.isPeak, isPeakPred, isPeakPred.size(), queue, true);
  for (size_t i = 0; i < digits.size(); i++) {
    log::Debug() << digits[i] << " " << int(isPeakPred[i]);
  }

  resetMaps.call(state, queue);

  queue.finish();

  return true;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
