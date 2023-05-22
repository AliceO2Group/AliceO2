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
/// \brief EMCAL compressed cell information
/// \author Anders Knospe, University of Houston
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since March 6, 2019
/// \ingroup EMCALDataFormat
///
/// # Base format for EMCAL cell information in the Compressed Timeframe
///
/// The cell class contains the relevant information for each tower per event
/// - Tower ID
/// - Energy of the raw fit
/// - Time of the raw fit
/// - Type of the cell
/// While cell type and tower ID have a predefined range based on the hardware
/// design, energy and time have a finite resolution influenced by the resolution
/// of the digitizer. This is used in order to compress the information stored
/// in the compressed timeframe by not storing the full double values but instead
/// assigning a certain amount of bits to each information. Therefore for certain
/// information (energy, time) precision loss has to be taken into account.
///
/// # Internal structure and resolution
///
/// The internal structure is a bit field compressing the information to
/// 48 bits. The definition of the bit field as well as the value range and the resolution
/// is listed in the table below:
///
/// | Bits  | Content       | Resolution    | Range                       |
/// |-------|---------------|---------------|-----------------------------|
/// | 0-14  | Tower ID      | -             | 0 to 17644                  |
/// | 15-26 | Time (ns)     | 0.73 ns       | -600 to 900 ns              |
/// | 27-40 | Energy (GeV)  | 0.0153 GeV    | 0 to 250 GeV                |
/// | 41-42 | Cell type     | -             | 0=LG, 1=HG, 2=LEMon, 4=TRU  |
///
/// The remaining bits are 0
class Cell
{
 public:
  Cell();
  Cell(short tower, float energy, float time, ChannelType_t ctype = ChannelType_t::LOW_GAIN);
  ~Cell() = default; // override

  void setTower(short tower) { getDataRepresentation()->mTowerID = tower; }
  short getTower() const { return getDataRepresentation()->mTowerID; }

  /// \brief Set the time stamp
  /// \param time Time in ns
  ///
  /// The time stamp is expressed in ns and has
  /// a resolution of 1 ns. The time range which can
  /// be stored is from -1023 to 1023 ns. In case the
  /// range is exceeded the time is set to the limit
  /// of the range.
  void setTimeStamp(float time);

  /// \brief Get the time stamp
  /// \return Time in ns
  ///
  /// Time has a resolution of 1 ns and can cover
  /// a range from -1023 to 1023 ns
  float getTimeStamp() const;

  /// \brief Set the energy of the cell
  /// \brief Energy of the cell in GeV
  ///
  /// The energy range covered by the cell
  /// is 0 - 250 GeV, with a resolution of
  /// 0.0153 GeV. In case an energy exceeding
  /// the limits is provided the energy is
  /// set to the limits (0 in case of negative
  /// energy, 250. in case of energies > 250 GeV)
  void setEnergy(float energy);

  /// \brief Get the energy of the cell
  /// \return Energy of the cell
  ///
  /// The energy is truncated to a range
  /// covering 0 to 250 GeV with a resolution
  /// of 0.0153 GeV
  float getEnergy() const;

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
  /// \param ctype Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  void setType(ChannelType_t ctype) { getDataRepresentation()->mCellStatus = static_cast<uint16_t>(ctype); }

  /// \brief Get the type of the cell
  /// \return Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  ChannelType_t getType() const { return static_cast<ChannelType_t>(getDataRepresentation()->mCellStatus); }

  /// \brief Check whether the cell is of a given type
  /// \param ctype Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  /// \return True if the type of the cell matches the requested type, false otherwise
  bool isChannelType(ChannelType_t ctype) const { return getType() == ctype; }

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

  void PrintStream(std::ostream& stream) const;

  /// used for CTF encoding/decoding: access to packed data
  void setPacked(uint16_t tower, uint16_t t, uint16_t en, uint16_t status)
  {
    auto dt = getDataRepresentation();
    dt->mTowerID = tower;
    dt->mTime = t;
    dt->mEnergy = en;
    dt->mCellStatus = status;
  }

  auto getPackedTowerID() const { return getDataRepresentation()->mTowerID; }
  auto getPackedTime() const { return getDataRepresentation()->mTime; }
  auto getPackedEnergy() const { return getDataRepresentation()->mEnergy; }
  auto getPackedCellStatus() const { return getDataRepresentation()->mCellStatus; }

 private:
  struct __attribute__((packed)) CellData {
    uint16_t mTowerID : 15;   ///< bits 0-14   Tower ID
    uint16_t mTime : 11;      ///< bits 15-25: Time (signed, can become negative after calibration)
    uint16_t mEnergy : 14;    ///< bits 26-39: Energy
    uint16_t mCellStatus : 2; ///< bits 40-41: Cell status
    uint16_t mZerod : 6;      ///< bits 42-47: Zerod
  };

  CellData* getDataRepresentation() { return reinterpret_cast<CellData*>(mCellWords); }
  const CellData* getDataRepresentation() const { return reinterpret_cast<const CellData*>(mCellWords); }

  char mCellWords[6]; ///< data word

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
