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
#ifndef ALICEO2_EMCAL_PEDESTAL_H
#define ALICEO2_EMCAL_PEDESTAL_H

#include <iosfwd>
#include <array>
#include <Rtypes.h>

class TH1;
class TH2;

namespace o2
{

namespace emcal
{

/// \class Pedestal
/// \brief CCDB container for pedestal values
/// \ingroup EMCALcalib
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since July 16th, 2019
///
/// Pedestal values can be added for each channel by
/// ~~~.{cxx}
/// o2::emcal::Pedestal ped;
/// ped.addPedestalValue(23, 3, false);
/// ~~~
/// For the High Gain cells the last parameter should be set to false, for low
/// gain it should be set to true.
///
/// One can read the pedestal values by calling
/// ~~~.{cxx}
/// auto param = ped.getPedestalValue(23, false);
/// ~~~
/// This will return the pedestal value for a certain HG cell.
/// For low gain cells you have to set the last parameter false
class Pedestal
{
 public:
  /// \brief Constructor
  Pedestal() = default;

  /// \brief Destructor
  ~Pedestal() = default;

  /// \brief Comparison of two pedestal containers
  /// \return true if the two list of pedestal values are the same, false otherwise
  bool operator==(const Pedestal& other) const;

  /// \brief Add pedestal to the container
  /// \param cellID Absolute ID of cell
  /// \param isLowGain Cell type is low gain cell
  /// \param pedestal Pedestal value
  /// \throw CalibContainerIndexException in case the cell ID exceeds the range of cells in EMCAL
  void addPedestalValue(unsigned short cellID, short pedestal, bool isLowGain, bool isLEDMON);

  /// \brief Get the time calibration coefficient for a certain cell
  /// \param cellID Absolute ID of cell
  /// \param isLowGain Cell type is low gain cell
  /// \return Pedestal value of the cell
  /// \throw CalibContainerIndexException in case the cell ID exceeds the range of cells in EMCAL
  short getPedestalValue(unsigned short cellID, bool isLowGain, bool isLEDMON) const;

  /// \brief Convert the pedestal container to a histogram
  /// \param isLowGain Monitor low gain cells
  /// \return Histogram representation of the pedestal container
  TH1* getHistogramRepresentation(bool isLowGain, bool isLEDMON) const;

  /// \brief Convert the pedestal container to a 2D histogram
  /// \param isLowGain Monitor low gain cells
  /// \return 2D Histogram representation (heatmap with respect to column and row) of the pedestal container
  TH2* getHistogramRepresentation2D(bool isLowGain, bool isLEDMON) const;

 private:
  std::array<short, 17664> mPedestalValuesHG;     ///< Container for the pedestal values (high gain)
  std::array<short, 17664> mPedestalValuesLG;     ///< Container for the pedestal values (low gain)
  std::array<short, 480> mPedestalValuesLEDMONHG; ///< Container for the LEDMON pedestal values (high gain)
  std::array<short, 480> mPedestalValuesLEDMONLG; ///< Container for the LEDMON pedestal values (low gain)

  ClassDefNV(Pedestal, 1);
};

} // namespace emcal

} // namespace o2
#endif
