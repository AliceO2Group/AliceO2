// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file Digitizer.h
 * C++  MCH Digitizer.
 * @author Michael Winn, Laurent Aphecetche
 */

#ifndef O2_MCH_SIMULATION_MCHDIGITIZER_H_
#define O2_MCH_SIMULATION_MCHDIGITIZER_H_

#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Hit.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace mch
{

class Digitizer
{
 public:
  Digitizer(int mode = 0);

  ~Digitizer() = default;

  void init();

  //process hits: fill digit vector with digits
  void process(const std::vector<Hit> hits, std::vector<Digit>& digits);
  void provideMC(o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer);
  void mergeDigits(const std::vector<Digit> digits, const std::vector<o2::MCCompLabel> trackLabels);
  //external pile-up adding up
  void mergeDigits(std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer);

  void fillOutputContainer(std::vector<Digit>& digits, std::vector<o2::MCCompLabel>& trackLabels);

  void setEventTime(double timeNS) { mEventTime = timeNS; }

  void setContinuous(bool val) { mContinuous = val; }
  bool isContinuous() const { return mContinuous; }

  void setSrcID(int v);
  int getSrcID() const { return mSrcID; }

  void setEventID(int v);
  int getEventID() const { return mEventID; }

  //for debugging
  std::vector<Digit> getDigits() { return mDigits; }
  std::vector<o2::MCCompLabel> getTrackLabels() { return mTrackLabels; }

 private:
  double mEventTime;
  int mEventID = 0;
  int mSrcID = 0;

  bool mContinuous = false;

  //number of detector elements
  const static int mNdE = 156;
  // digit per pad
  std::vector<Digit> mDigits;

  //MCLabel container (transient)
  std::vector<o2::MCCompLabel> mTrackLabels;
  //MCLabel container (output)
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mMCTruthOutputContainer;

  int processHit(const Hit& hit, int detID, double event_time);
};

} // namespace mch
} // namespace o2
#endif
