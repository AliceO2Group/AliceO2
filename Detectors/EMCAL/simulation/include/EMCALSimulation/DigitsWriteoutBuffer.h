// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  /// Add digit to the container
  /// \param towerID Cell ID
  /// \param dig Labaled digit to add
  /// \param eventTime The time of the event (w.r.t Tigger time)
  void addDigit(unsigned int towerID, LabeledDigit dig, double eventTime);

  /// Getter for the last N entries (time samples) in the ring buffer
  /// \param nsample number of entries to be written
  /// \return Vector of map of cell IDs and labeled digits in that cell
  gsl::span<std::unordered_map<int, std::list<LabeledDigit>>> getLastNSamples(int nsample = 15);

  /// Forwards the marker to the next time sample every mTimeBinWidth
  void forwardMarker(double eventTime);

  void setNumberReadoutSamples(unsigned int nsamples) { mNumberReadoutSamples = nsamples; }

  /// Getter for current position in the ring buffer
  /// \return current position
  std::tuple<double, int> getCurrentTimeAndPosition() const { return std::make_tuple(mMarker.mReferenceTime, mMarker.mPositionInBuffer - mTimedDigits.begin()); }

 private:
  struct Marker {
    double mReferenceTime;
    std::deque<std::unordered_map<int, std::list<LabeledDigit>>>::iterator mPositionInBuffer;
  };

  unsigned int mBufferSize = 30;                                             ///< The size of the buffer, it has to be at least 2 times the size of the readout cycle
  unsigned int mTimeBinWidth = 100;                                          ///< The size of the time bin (ns)
  unsigned int mNumberReadoutSamples = 15;                                   ///< The number of smaples in a readout window
  Marker mMarker;                                                            ///< Marker for the current time sample
  std::deque<std::unordered_map<int, std::list<LabeledDigit>>> mTimedDigits; ///< Container for time sampled digits per tower ID

  ClassDefNV(DigitsWriteoutBuffer, 1);
};

} // namespace emcal

} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITSWRITEOUTBUFFER_H_ */
