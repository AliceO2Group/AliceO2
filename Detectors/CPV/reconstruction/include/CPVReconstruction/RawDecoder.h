// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_CPV_RAWDECODER_H
#define ALICEO2_CPV_RAWDECODER_H

#include <iosfwd>
#include <gsl/span>
#include <string>
#include <utility>
#include "DataFormatsCPV/Digit.h"
#include "CPVReconstruction/RawReaderMemory.h"
class Digits;

namespace o2
{
namespace cpv
{

class RawDecoderError
{
 public:
  RawDecoderError() = default; //Constructors for vector::emplace_back methods
  RawDecoderError(short c, short d, short g, short p, RawErrorType_t e) : ccId(c), dil(d), gas(g), pad(p), errortype(e) {}
  RawDecoderError(const RawDecoderError& e) = default;
  ~RawDecoderError() = default;

  short ccId;
  short dil;
  short gas;
  short pad;
  RawErrorType_t errortype;
  ClassDefNV(RawDecoderError, 1);
};

union AddressCharge {
  uint32_t mDataWord;
  struct {
    uint32_t Address : 18; ///< Bits  0 - 17 : Address
    uint32_t Charge : 14;  ///< Bits 18 - 32 : charge
  };
};

/// BC reference to digits
struct BCRecord {
  BCRecord() = default;
  BCRecord(uint16_t bunchCrossing, unsigned int first, unsigned int last) : bc(bunchCrossing), firstDigit(first), lastDigit(last) {}
  uint16_t bc;
  unsigned int firstDigit;
  unsigned int lastDigit;
};

/// \class RawDecoder
/// \brief Decoder of the ALTRO data in the raw page
/// \ingroup CPVreconstruction
/// \author Dmitri Peresunko
/// \since Dec, 2020
///
/// This is a base class for reading raw data digits.
/// It takes raw cpv payload from RawReaderMemory and produces
/// std::vector<uint32_t> mDigits and std::vector<BCRecord> mBCRecords

class RawDecoder
{
 public:
  /// \brief Constructor
  /// \param reader Raw reader instance to be decoded
  RawDecoder(RawReaderMemory& reader);

  /// \brief Destructor
  ~RawDecoder() = default;

  /// \brief Decode the raw cpv payload stream
  /// \throw RawDecoderError if the RCUTrailer or raw cpv payload cannot be decoded
  ///
  /// Decoding and checking the cpvheader,
  /// cpvwords and cpvtrailer.
  /// After successfull decoding the Decoder can provide
  /// a reference to a vector
  /// with the decoded chanenels and their bc reference
  RawErrorType_t decode();

  /// \brief Get the reference to the digits container
  /// \return Reference to the digits container
  const std::vector<uint32_t>& getDigits() const { return mDigits; };

  /// \brief Get the reference to the BC records
  /// \return reference to the BC records
  const std::vector<o2::cpv::BCRecord>& getBCRecords() const { return mBCRecords; };

  /// \brief Get the reference to the list of decoding errors
  /// \return Reference to the list of decoding errors
  const std::vector<o2::cpv::RawDecoderError>& getErrors() const { return mErrors; }

 protected:
  /// \brief Read channels for the current event in the raw buffer
  RawErrorType_t readChannels();

 private:
  void addDigit(uint32_t padWord, short ddl, uint16_t bc);
  void removeLastNDigits(int n);

  RawReaderMemory& mRawReader;               ///< underlying raw reader
  std::vector<uint32_t> mDigits;             ///< vector of channels and BCs in the raw stream
  std::vector<o2::cpv::BCRecord> mBCRecords; ///< vector of bc references to digits
  std::vector<RawDecoderError> mErrors;      ///< vector of decoding errors
  bool mChannelsInitialized = false;         ///< check whether the channels are initialized

  ClassDefNV(RawDecoder, 2);
};

} // namespace cpv

} // namespace o2

#endif
