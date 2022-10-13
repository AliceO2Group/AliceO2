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

#ifndef ALICEO2_EMCAL_CELL_H_
#define ALICEO2_EMCAL_CELL_H_

#include <bitset>
#include "DataFormatsEMCAL/Constants.h"

namespace o2
{
namespace emcal
{

/// \class Cell
/// \brief EMCAL uncompressed cell information
/// \author Florian Jonas <florian.jonas@cern.ch>, Oak Ridge National Laboratory
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since October 12, 2022
/// \ingroup EMCALDataFormat
///
/// # Base format for EMCAL cell information in the Compressed Timeframe
///
/// The cell class contains the relevant information for each tower per event
/// - Tower ID
/// - Energy of the raw fit
/// - Time of the raw fit
/// - Type of the cell
class Cell
{
 public:
  // constants
  Cell();
  Cell(short tower, float energy, float time, ChannelType_t type = ChannelType_t::LOW_GAIN, float chi2 = 10.);
  ~Cell() = default; // override

  void setTower(short tower) { mtower = tower; }
  short getTower() const { return mtower; }

  /// \brief Set the time stamp
  /// \param time Time in ns
  ///
  /// The time stamp is expressed in ns
  void setTimeStamp(float time) { mtime = time; }

  /// \brief Get the time stamp
  /// \return Time in ns
  ///
  /// Time stamp is expressed in ns
  float getTimeStamp() const { return mtime; }

  /// \brief Set the energy of the cell
  /// \brief Energy of the cell in GeV
  ///
  /// The energy range covered by the cell
  /// is 0 - 250 GeV
  void setEnergy(float energy) { menergy = energy; }

  /// \brief Get the energy of the cell
  /// \return Energy of the cell GeV
  ///
  /// Return the energy in GeV
  float getEnergy() const { return menergy; }

  /// \brief Set the amplitude of the cell
  /// \param amplitude Cell amplitude
  ///
  /// See setEnergy for more information
  void setAmplitude(float amplitude) { setEnergy(amplitude); }

  /// \brief Get cell amplitude
  /// \return cell Amplitude
  ///
  /// Set getEnergy for more information
  float getAmplitude() const { return getEnergy(); }

  /// \brief Set the type of the cell
  /// \param type Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  void setType(ChannelType_t type) { mtype = static_cast<ChannelType_t>(type); }

  /// \brief Get the type of the cell
  /// \return Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  ChannelType_t getType() const { return mtype; }

  /// \brief Check whether the cell is of a given type
  /// \param type Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  /// \return True if the type of the cell matches the requested type, false otherwise
  bool isChannelType(ChannelType_t type) const { return getType() == type; }

  /// \brief Mark cell as low gain cell
  void setLowGain() { setType(ChannelType_t::LOW_GAIN); }

  /// \brief Check whether the cell is a low gain cell
  /// \return True if the cell type is low gain, false otherwise
  Bool_t getLowGain() const { return isChannelType(ChannelType_t::LOW_GAIN); }

  /// \brief Mark cell as high gain cell
  void setHighGain() { setType(ChannelType_t::HIGH_GAIN); }

  /// \brief Check whether the cell is a high gain cell
  /// \return True if the cell type is high gain, false otherwise
  Bool_t getHighGain() const { return isChannelType(ChannelType_t::HIGH_GAIN); };

  /// \brief Mark cell as LED monitor cell
  void setLEDMon() { setType(ChannelType_t::LEDMON); }

  /// \brief Check whether the cell is a LED monitor cell
  /// \return True if the cell type is LED monitor, false otherwise
  Bool_t getLEDMon() const { return isChannelType(ChannelType_t::LEDMON); }

  /// \brief Mark cell as TRU cell
  void setTRU() { setType(ChannelType_t::TRU); }

  /// \brief Check whether the cell is a TRU cell
  /// \return True if the cell type is TRU, false otherwise
  Bool_t getTRU() const { return isChannelType(ChannelType_t::TRU); }

  /// \brief Set the chi2 of the raw fitter
  /// \param chi2 Chi2 of the raw fitter
  void setChi2(float chi2) { mchi2 = chi2; }

  /// \brief Get the chi2 of the raw fitter
  /// \return Chi2 of the raw fitter
  float getChi2() const { return mchi2; }

  void setAll(short tower, float energy, float time, ChannelType_t type = ChannelType_t::LOW_GAIN, float chi2 = 10.);

  void PrintStream(std::ostream& stream) const;

 private:
  short mtower;
  float menergy;
  float mtime;
  ChannelType_t mtype;
  float mchi2;

  ClassDefNV(Cell, 1);
};

/// \brief Stream operator for EMCAL cell
/// \param stream Stream where to print the EMCAL cell
/// \param cell Cell to be printed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const Cell& cell);
} // namespace emcal
} // namespace o2

#endif
