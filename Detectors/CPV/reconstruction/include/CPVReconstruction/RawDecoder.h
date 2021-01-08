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
#include "CPVBase/RCUTrailer.h"
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
  RawDecoderError(short l, short r, short d, short p, RawErrorType_t e) : Ddl(l), Row(r), Dilogic(d), Pad(p), errortype(e) {}
  RawDecoderError(const RawDecoderError& e) = default;
  ~RawDecoderError() = default;

  short Ddl;
  short Row;
  short Dilogic;
  short Pad;
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

/// \class RawDecoder
/// \brief Decoder of the ALTRO data in the raw page
/// \ingroup CPVreconstruction
/// \author Dmitri Peresunko
/// \since Dec, 2020
///
/// This is a base class for reading raw data digits.

class RawDecoder
{
 public:
  /// \brief Constructor
  /// \param reader Raw reader instance to be decoded
  RawDecoder(RawReaderMemory& reader);

  /// \brief Destructor
  ~RawDecoder() = default;

  /// \brief Decode the ALTRO stream
  /// \throw RawDecoderError if the RCUTrailer or ALTRO payload cannot be decoded
  ///
  /// Decoding and checking the RCUTtrailer and
  /// all channels and bunches in the ALTRO stream.
  /// After successfull decoding the Decoder can provide
  /// a reference to the RCU trailer and a vector
  /// with the decoded chanenels, each containing
  /// its bunches.
  RawErrorType_t decode();

  /// \brief Get reference to the RCU trailer object
  /// \return const reference to the RCU trailer
  const RCUTrailer& getRCUTrailer() const;

  /// \brief Get the reference to the digits container
  /// \return Reference to the digits container
  const std::vector<uint32_t>& getDigits() const;

  /// \brief Get the reference to the list of decoding errors
  /// \return Reference to the list of decoding errors
  const std::vector<o2::cpv::RawDecoderError>& getErrors() const { return mErrors; }

 protected:
  /// \brief Read RCU trailer for the current event in the raw buffer
  RawErrorType_t readRCUTrailer();

  /// \brief Read channels for the current event in the raw buffer
  RawErrorType_t readChannels();

 private:
  /// \brief run checks on the RCU trailer
  /// \throw Error if the RCU trailer has inconsistencies
  ///
  /// Performing various consistency checks on the RCU trailer
  /// In case of failure an exception is thrown.
  void checkRCUTrailer();

  void addDigit(uint32_t padWord, short ddl);

  RawReaderMemory& mRawReader;          ///< underlying raw reader
  RCUTrailer mRCUTrailer;               ///< RCU trailer
  std::vector<uint32_t> mDigits;        ///< vector of channels in the raw stream
  std::vector<RawDecoderError> mErrors; ///< vector of decoding errors
  bool mChannelsInitialized = false;    ///< check whether the channels are initialized

  ClassDefNV(RawDecoder, 1);
};

} // namespace cpv

} // namespace o2

#endif
