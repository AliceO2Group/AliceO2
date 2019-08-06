// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class GainCalibrationFactors
/// \brief CCDB container for the gain calibration factors
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since August 5th, 2019
///
/// The gain calibration factors can be added for each channel
/// ~~~.{cxx}
/// o2::emcal::GainCalibrationFactors GCF;
/// TCP.addGainCalibFactor(iCell, gainValue);
/// ~~~
///
/// One can read the gain calibration factors by calling
/// ~~~.{cxx}
/// auto param = GCF.getGainCalibFactors(iCell);
/// This will return the gain calibration factor for a certain channel.
/// ~~~
///

#ifndef GAINCALIBRATIONFACTORS_H_
#define GAINCALIBRATIONFACTORS_H_

#include <iosfwd>
#include <array>
#include <Rtypes.h>

class TH1;

namespace o2
{

namespace emcal
{

class GainCalibrationFactors
{
 public:
  /// \brief Constructor
  GainCalibrationFactors() = default;

  /// \brief Destructor
  ~GainCalibrationFactors() = default;

  /// \brief Comparison of two gain calibration factors containers
  /// \return true if the two list of gain calibration factors are the same, false otherwise
  bool operator==(const GainCalibrationFactors& other) const;

  /// \brief Add gain calibration factors to the container
  /// \param iCell is the cell index
  /// \param gainFactor is the gain calibration factor
  void addGainCalibFactor(unsigned short iCell, float gainFactor);

  /// \brief Get the gain calibration factor for a certain cell
  /// \param iCell is the cell index
  /// \return gain calibration factor of the cell
  float getGainCalibFactors(unsigned short iCell) const;

  /// \brief Convert the gain calibration factors to a histogram
  /// \return Histogram representation for gain calibration factors for each channel
  TH1* getHistogramRepresentation() const;

 private:
  std::array<float, 17664> mGainCalibFactors; ///< Container for the gain calibration factors

  ClassDefNV(GainCalibrationFactors, 1);
};

} // namespace emcal

} // namespace o2
#endif
