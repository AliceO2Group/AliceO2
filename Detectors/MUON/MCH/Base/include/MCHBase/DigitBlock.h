// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @since 2016-10-19
/// @author P. Pillot
/// @brief Structures defining the digits and the data blocks to hold them

#ifndef ALICEO2_MCH_DIGITBLOCK_H_
#define ALICEO2_MCH_DIGITBLOCK_H_

#include <cstdint>
#include <ostream>

namespace o2
{
namespace mch
{

struct DataBlockHeader {

  uint16_t fType;        // The type of the data block. Must contain a value defined by DataBlockType.
  uint16_t fRecordWidth; // The number of bytes each record uses.
  uint32_t fNrecords;    // Number of records in this data block.

  bool operator==(const DataBlockHeader& that) const
  {
    return (fType == that.fType && fRecordWidth == that.fRecordWidth && fNrecords == that.fNrecords);
  }

  bool operator!=(const DataBlockHeader& that) const { return not this->operator==(that); }
};

/**
 * Gives the fired digit/pad information.
 */
struct DigitStruct {

  
  uint32_t uid;   // Digit ID in the current mapping (from OCDB)
  uint16_t index; // Digit index in the new mapping (produced internally)
  uint16_t adc;   // ADC value of signal
  
  DigitStruct(uid,index,adc);

  
  bool operator==(const DigitStruct& that) const { return (uid == that.uid && index == that.index && adc == that.adc); }

  bool operator!=(const DigitStruct& that) const { return not this->operator==(that); }
};

/**
 * DigitBlock defines the format of the internal digit data block.
 */
struct DigitBlock {

  DataBlockHeader header; // Common data block header

  // Array of digits/pads.
  // DigitStruct fDigit[/*header.fNrecords*/];

  bool operator==(const DigitBlock& that) const;

  bool operator!=(const DigitBlock& that) const { return not this->operator==(that); }
};

/**
 * Stream operator for usage with std::ostream classes which prints the common
 * data block header in the following format:
 *  {fType = xx, fRecordWidth = yy, fNrecords = zz}
 */
inline std::ostream& operator<<(std::ostream& stream, const DataBlockHeader& header)
{
  stream << "{fType = " << header.fType << ", fRecordWidth = " << header.fRecordWidth
         << ", fNrecords = " << header.fNrecords << "}";
  return stream;
}

/**
 * Stream operator for usage with std::ostream classes which prints the
 * DigitStruct in the following format:
 *   {uid = xx, index = yy, adc = zz}
 */
std::ostream& operator<<(std::ostream& stream, const DigitStruct& digit);

/**
 * Stream operator for usage with std::ostream classes which prints the
 * DigitBlock in the following format:
 *   {header = xx, fDigit[] = [{..}, {..}, ...]}
 */
std::ostream& operator<<(std::ostream& stream, const DigitBlock& block);

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_DIGITBLOCK_H_
