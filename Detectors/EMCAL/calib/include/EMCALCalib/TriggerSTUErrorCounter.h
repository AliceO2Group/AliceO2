// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iosfwd>
#include <array>
#include <Rtypes.h>

namespace o2
{

namespace emcal
{

/// \class TriggerSTUErrorCounter
/// \brief CCDB container for STU error counts
/// \ingroup EMCALcalib
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since December 4th, 2019
///
/// based on AliEMCALTriggerSTUConfig class authored by R. GUERNANE

class TriggerSTUErrorCounter
{
 public:
  /// \brief default constructor
  TriggerSTUErrorCounter() = default;

  /// \brief copy constructor
  TriggerSTUErrorCounter(const TriggerSTUErrorCounter& error) = default;

  /// \brief Assignment operator
  TriggerSTUErrorCounter& operator=(const TriggerSTUErrorCounter& source) = default;

  /// \brief nomral constructor
  explicit TriggerSTUErrorCounter(std::pair<int, unsigned long> TimeAndError);

  /// \brief nomral constructor
  explicit TriggerSTUErrorCounter(int Time, unsigned long Error);

  /// \brief Destructor
  ~TriggerSTUErrorCounter() = default;

  /// \brief Comparison of two TRU data
  /// \return true if the TRU data are identical, false otherwise
  ///
  /// Testing two TRUs for equalness. TRUs are considered identical
  /// if the contents are identical.
  bool operator==(const TriggerSTUErrorCounter& other) const;

  /// \brief Checks for equalness according to the time stamp.
  bool isEqual(TriggerSTUErrorCounter& counter) const;

  /// \brief Compare time-dependent error counts based on the time information.
  int compare(TriggerSTUErrorCounter& counter) const;

  void setValue(int time, unsigned long errorcount)
  {
    mTimeErrorCount.first = time;
    mTimeErrorCount.second = errorcount;
  }
  void setValue(std::pair<int, unsigned long> TimeAndError) { mTimeErrorCount = TimeAndError; }

  int getTime() const { return mTimeErrorCount.first; }
  unsigned long getErrorCount() const { return mTimeErrorCount.second; }
  std::pair<int, unsigned long> getTimeAndErrorCount() const { return mTimeErrorCount; }

 private:
  std::pair<int, unsigned long> mTimeErrorCount; ///< Pair for the time and the number of errors at this time

  ClassDefNV(TriggerSTUErrorCounter, 1);
};

} // namespace emcal

} // namespace o2
