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

#ifndef ALICEO2_EMCAL_DIGITSWRITEOUTBUFFER_H_
#define ALICEO2_EMCAL_DIGITSWRITEOUTBUFFER_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <deque>
#include <list>
#include <gsl/span>
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALSimulation/LabeledDigit.h"

namespace o2
{
namespace emcal
{

/// \class DigitsWriteoutBuffer
/// \brief Container class for time sampled digits
/// \ingroup EMCALsimulation
/// \author Hadi Hassan, ORNL
/// \author Markus Fasel, ORNL
/// \date 08/03/2021

class DigitsWriteoutBuffer
{
 public:
  /// Default constructor
  DigitsWriteoutBuffer(unsigned int nTimeBins = 30, unsigned int binWidth = 100);

  /// Destructor
  ~DigitsWriteoutBuffer() = default;

  /// clear the container
  void clear();
  void reserve();

  /// Add digit to the container
  /// \param towerID Cell ID
  /// \param dig Labaled digit to add
  /// \param eventTime The time of the event (w.r.t Tigger time)
  void addDigit(unsigned int towerID, LabeledDigit dig, double eventTime);

  /// Getter for the last N entries (time samples) in the ring buffer
  /// \param nsample number of entries to be written
  /// \return List of map of cell IDs and labeled digits in that cell
  std::list<std::unordered_map<int, std::list<LabeledDigit>>> getLastNSamples(int nsample = 15);

  void setNumberReadoutSamples(unsigned int nsamples) { mNumberReadoutSamples = nsamples; }

  std::deque<std::unordered_map<int, std::list<LabeledDigit>>> getTheWholeSample() { return mTimedDigits; }

 private:
  unsigned int mBufferSize = 30;                                             ///< The size of the buffer, it has to be at least 2 times the size of the readout cycle
  unsigned int mTimeBinWidth = 100;                                          ///< The size of the time bin (ns)
  unsigned int mNumberReadoutSamples = 15;                                   ///< The number of smaples in a readout window
  double mLastEventTime = 1500;                                              ///< The event time of last collisions in the readout window
  std::deque<std::unordered_map<int, std::list<LabeledDigit>>> mTimedDigits; ///< Container for time sampled digits per tower ID

  ClassDefNV(DigitsWriteoutBuffer, 1);
};

} // namespace emcal

} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITSWRITEOUTBUFFER_H_ */
