// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class TempCalibParamSM
/// \brief CCDB container for the temperature calibration coefficients per SM
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since August 13th, 2019
///
/// The calibration parameters for each SM can be added by
/// ~~~.{cxx}
/// o2::emcal::TempCalibParamSM TCP;
/// TCP.addTempCalibParamPerSM(13, 2);
/// ~~~
///
/// One can read the temperature calibration coefficients for each SM by calling
/// ~~~.{cxx}
/// auto param = TCP.getTempCalibParamPerSM(13);
/// This will return the temperature calibration coefficients for a SM.
/// ~~~
///

#ifndef TEMPCALIBPARAMSM_H_
#define TEMPCALIBPARAMSM_H_

#include <iosfwd>
#include <array>
#include <Rtypes.h>

class TH1;

namespace o2
{

namespace emcal
{

class TempCalibParamSM
{
 public:
  /// \brief Constructor
  TempCalibParamSM() = default;

  /// \brief Destructor
  ~TempCalibParamSM() = default;

  /// \brief Comparison of two temperature calibration coefficients per SM
  /// \return true if the two list of Temp calibration coefficients per SM are the same, false otherwise
  bool operator==(const TempCalibParamSM& other) const;

  /// \brief Add temperature coefficients to the container
  /// \param iSM is the Super Module
  /// \param TempSM is the temperature coefficients for the SM
  void addTempCalibParamPerSM(unsigned short iSM, float TempSM);

  /// \brief Get the temperature calibration coefficient for a certain SM
  /// \param iSM is the Super Module
  /// \return temperature calibration coefficient of the SM
  float getTempCalibParamPerSM(unsigned short iSM) const;

  /// \brief Convert the temperature calibration coefficient per SM arrays to histograms
  /// \return Histogram representation for temperature calibration coefficient per SM
  TH1* getHistogramRepresentation() const;

 private:
  std::array<float, 20> mTempCalibParamsPerSM; ///< Temp calibration coefficient per SM.

  ClassDefNV(TempCalibParamSM, 1);
};

} // namespace emcal

} // namespace o2
#endif
