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
#include <cfloat>
#include <climits>
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
/// # Cell content
///
/// The cell class contains the relevant information for each tower per event
/// - Tower ID
/// - Energy of the raw fit
/// - Time of the raw fit
/// - Type of the cell
///
/// # Compression for CTF
///
/// While cell type and tower ID have a predefined range based on the hardware
/// design, energy and time have a finite resolution influenced by the resolution
/// of the digitizer. This is used in order to compress the information stored
/// in the compressed timeframe by not storing the full double values but instead
/// assigning a certain amount of bits to each information. Therefore for certain
/// information (energy, time) precision loss has to be taken into account. The number
/// of bits assigned to each data member in the encoding are as follows:
///
/// | Content       | Number of bits |Resolution    | Range                       |
/// |---------------|----------------|--------------|-----------------------------|
/// | Tower ID      | 15             | -            | 0 to 17644                  |
/// | Time (ns)     | 11             | 0.73 ns      | -600 to 900 ns              |
/// | Energy (GeV)  | 14             | 0.0153 GeV   | 0 to 250 GeV                |
/// | Cell type     | 2              | -            | 0=LG, 1=HG, 2=LEMon, 4=TRU  |
///
/// The remaining bits are 0
class Cell
{
 public:
  enum class EncoderVersion {
    EncodingV0,
    EncodingV1,
    EncodingV2
  };
  /// \brief Default constructor
  Cell() = default;

  /// \brief Constructor
  /// \param tower Tower ID
  /// \param energy Energy
  /// \param timestamp Cell time
  /// \param ctype Channel type
  Cell(short tower, float energy, float timestamp, ChannelType_t ctype = ChannelType_t::LOW_GAIN);

  /// \brief Constructor, from encoded bit representation
  /// \param tower Tower bitsets
  /// \param energy Energy bits
  /// \param timestamp Cell time bits
  /// \param ctype Channel type bits
  /// \param version Encoding version
  Cell(uint16_t towerBits, uint16_t energyBits, uint16_t timestampBits, uint16_t channelBits, EncoderVersion version = EncoderVersion::EncodingV1);

  /// \brief Destructor
  ~Cell() = default; // override

  /// \brief Set the tower ID
  /// \param tower Tower ID
  void setTower(short tower) { mTowerID = tower; }

  /// \brief Get the tower ID
  /// \return Tower ID
  short getTower() const { return mTowerID; }

  /// \brief Set the time stamp
  /// \param timestamp Time in ns
  void setTimeStamp(float timestamp) { mTimestamp = timestamp; }

  /// \brief Get the time stamp
  /// \return Time in ns
  float getTimeStamp() const { return mTimestamp; }

  /// \brief Set the energy of the cell
  /// \brief Energy of the cell in GeV
  void setEnergy(float energy) { mEnergy = energy; }

  /// \brief Get the energy of the cell
  /// \return Energy of the cell
  float getEnergy() const { return mEnergy; }

  /// \brief Set the amplitude of the cell
  /// \param amplitude Cell amplitude
  void setAmplitude(float amplitude) { setEnergy(amplitude); }

  /// \brief Get cell amplitude
  /// \return Cell amplitude in GeV
  float getAmplitude() const { return getEnergy(); }

  /// \brief Set the type of the cell
  /// \param ctype Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  void setType(ChannelType_t ctype) { mChannelType = ctype; }

  /// \brief Get the type of the cell
  /// \return Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  ChannelType_t getType() const { return mChannelType; }

  /// \brief Check whether the cell is of a given type
  /// \param ctype Type of the cell (HIGH_GAIN, LOW_GAIN, LEDMON, TRU)
  /// \return True if the type of the cell matches the requested type, false otherwise
  bool isChannelType(ChannelType_t ctype) const { return mChannelType == ctype; }

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

  /// \brief Apply compression as done during writing to / reading from CTF
  /// \param version Encoder version
  void truncate(EncoderVersion version = EncoderVersion::EncodingV1);

  void PrintStream(std::ostream& stream) const;

  /// \brief Initialize cell class from bit representation (for CTF decoding)
  /// \param towerIDBits Encoded tower ID
  /// \param timestampBits Encoded timestamp
  /// \param energyBits Encoded energy
  /// \param celltypeBits Encoded cell type
  /// \param version Encoder version
  void initialiseFromEncoded(uint16_t towerIDBits, uint16_t timestampBits, uint16_t energyBits, uint16_t celltypeBits, EncoderVersion version = EncoderVersion::EncodingV1)
  {
    setEnergyEncoded(energyBits, static_cast<ChannelType_t>(celltypeBits), version);
    setTimestampEncoded(timestampBits);
    setTowerIDEncoded(towerIDBits);
    setChannelTypeEncoded(celltypeBits);
  }

  /// \brief Get encoded bit representation of tower ID (for CTF)
  /// \return Encoded bit representation
  ///
  /// Same as getTower - no compression applied for tower ID
  uint16_t getTowerIDEncoded() const;

  /// \brief Get encoded bit representation of timestamp (for CTF)
  /// \return Encoded bit representation
  ///
  /// The time stamp is expressed in ns and has
  /// a resolution of 1 ns. The time range which can
  /// be stored is from -1023 to 1023 ns. In case the
  /// range is exceeded the time is set to the limit
  /// of the range.
  uint16_t getTimeStampEncoded() const;

  /// \brief Get encoded bit representation of energy (for CTF)
  /// \param version Encoding verions
  /// \return Encoded bit representation
  ///
  /// The energy range covered by the cell
  /// is 0 - 250 GeV, with a resolution of
  /// 0.0153 GeV. In case an energy exceeding
  /// the limits is provided the energy is
  /// set to the limits (0 in case of negative
  /// energy, 250. in case of energies > 250 GeV)
  uint16_t getEnergyEncoded(EncoderVersion version = EncoderVersion::EncodingV2) const;

  /// \brief Get encoded bit representation of cell type (for CTF)
  /// \return Encoded bit representation
  uint16_t getCellTypeEncoded() const;

  void initializeFromPackedBitfieldV0(const char* bitfield);

  static float getEnergyFromPackedBitfieldV0(const char* bitfield);
  static float getTimeFromPackedBitfieldV0(const char* bitfield);
  static ChannelType_t getCellTypeFromPackedBitfieldV0(const char* bitfield);
  static short getTowerFromPackedBitfieldV0(const char* bitfield);

  static uint16_t encodeTime(float timestamp);
  static uint16_t encodeEnergyV0(float energy);
  static uint16_t encodeEnergyV1(float energy, ChannelType_t celltype);
  static uint16_t encodeEnergyV2(float energy, ChannelType_t celltype);
  static uint16_t V0toV1(uint16_t energybits, ChannelType_t celltype);
  static uint16_t V0toV2(uint16_t energybits, ChannelType_t celltype);
  static uint16_t V1toV2(uint16_t energybits, ChannelType_t celltype);
  static float decodeTime(uint16_t timestampBits);
  static float decodeEnergyV0(uint16_t energybits);
  static float decodeEnergyV1(uint16_t energybits, ChannelType_t celltype);
  static float decodeEnergyV2(uint16_t energybits, ChannelType_t celltype);

 private:
  /// \brief Set cell energy from encoded bit representation (from CTF)
  /// \param energyBits Bit representation of energy
  /// \param cellTypeBits Bit representation of cell type
  void setEnergyEncoded(uint16_t energyBits, uint16_t cellTypeBits, EncoderVersion version = EncoderVersion::EncodingV1);

  /// \brief Set cell time from encoded bit representation (from CTF)
  /// \param timestampBits Bit representation of timestamp
  void setTimestampEncoded(uint16_t timestampBits);

  /// \brief Set tower ID from encoded bit representation (from CTF)
  /// \param towerIDBits Bit representation of towerID
  void setTowerIDEncoded(uint16_t towerIDBits);

  /// \brief Set cell type from encoded bit representation (from CTF)
  /// \param channelTypeBits Bit representation of cell type
  void setChannelTypeEncoded(uint16_t channelTypeBits);

  float mEnergy = FLT_MIN;                               ///< Energy
  float mTimestamp = FLT_MIN;                            ///< Timestamp
  short mTowerID = SHRT_MAX;                             ///< Tower ID
  ChannelType_t mChannelType = ChannelType_t::HIGH_GAIN; ///< Cell type

  ClassDefNV(Cell, 3);
};

/// \brief Stream operator for EMCAL cell
/// \param stream Stream where to print the EMCAL cell
/// \param cell Cell to be printed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const Cell& cell);
} // namespace emcal
} // namespace o2

#endif
