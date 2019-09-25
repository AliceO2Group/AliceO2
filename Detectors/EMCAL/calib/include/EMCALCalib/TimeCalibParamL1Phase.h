// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class TimeCalibParamL1Phase
/// \brief CCDB container for the L1 phase shifts
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since Aug 12th, 2019
///
/// The L1Phase shift can be added for each SM by
/// ~~~.{cxx}
/// o2::emcal::TimeCalibParamL1Phase TCP;
/// TCP.addTimeCalibParamL1Phase(13, 2);
/// ~~~
///
/// One can read the L1 Phase shift for each SM by calling
/// ~~~.{cxx}
/// auto param = TCP.getTimeCalibParamL1Phase(13);
/// This will return the L1 Phase shift for a SM.
/// ~~~
///

#ifndef TIMECALIBPARAML1PHASE_H_
#define TIMECALIBPARAML1PHASE_H_

#include <iosfwd>
#include <array>
#include <Rtypes.h>

class TH1;

namespace o2
{

namespace emcal
{

class TimeCalibParamL1Phase
{
 public:
  /// \brief Constructor
  TimeCalibParamL1Phase() = default;

  /// \brief Destructor
  ~TimeCalibParamL1Phase() = default;

  /// \brief Comparison of two L1 phase shifts
  /// \return true if the two lists of L1 phase shifts are the same, false otherwise
  bool operator==(const TimeCalibParamL1Phase& other) const;

  /// \brief Add L1 phase shifts to the container
  /// \param iSM is the Super Module
  /// \param L1Phase is the L1 phase shift
  void addTimeCalibParamL1Phase(unsigned short iSM, unsigned char L1Phase);

  /// \brief Get the L1 phase for a certain SM
  /// \param iSM is the Super Module
  /// \return L1 phase shifts of the SM
  unsigned char getTimeCalibParamL1Phase(unsigned short iSM) const;

  /// \brief Convert the L1 phase shift per SM array to a histogram
  TH1* getHistogramRepresentation() const;

 private:
  std::array<unsigned char, 20> mTimeCalibParamsL1Phase; ///< Container for the L1 phase shift

  ClassDefNV(TimeCalibParamL1Phase, 1);
};

} // namespace emcal

} // namespace o2
#endif
