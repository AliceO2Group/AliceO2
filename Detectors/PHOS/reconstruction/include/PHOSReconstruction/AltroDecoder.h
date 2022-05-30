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
#ifndef ALICEO2_PHOS_ALTRODECODER_H
#define ALICEO2_PHOS_ALTRODECODER_H

#include <iosfwd>
#include <gsl/span>
#include <string>
#include <bitset>
#include "PHOSBase/RCUTrailer.h"
#include "DataFormatsPHOS/Cell.h"
#include "PHOSBase/Mapping.h"
#include "PHOSReconstruction/RawReaderError.h"
#include "PHOSReconstruction/CaloRawFitter.h"
#include "PHOSReconstruction/RawReaderMemory.h"

namespace o2
{
namespace phos
{

/// \class AltroDecoderError
/// \brief Error handling of the ALTRO Decoder
/// \ingroup EMCALreconstruction
class AltroDecoderError : public std::exception
{
 public:
  enum ErrorType_t {
    kOK,                      ///< NoError
    RCU_TRAILER_ERROR,        ///< RCU trailer cannot be decoded or invalid
    RCU_VERSION_ERROR,        ///< RCU trailer version not matching with the version in the raw header
    RCU_TRAILER_SIZE_ERROR,   ///< RCU trailer size length
    ALTRO_BUNCH_HEADER_ERROR, ///< ALTRO bunch header cannot be decoded or is invalid
    ALTRO_BUNCH_LENGTH_ERROR, ///< ALTRO bunch has incorrect length
    ALTRO_PAYLOAD_ERROR,      ///< ALTRO payload cannot be decoded
    ALTRO_MAPPING_ERROR,      ///< Incorrect ALTRO channel mapping
    CHANNEL_ERROR             ///< Channels not initialized
  };

  /// \brief Constructor
  ///
  /// Defining error code and error message. To be called when the
  /// exception is thrown
  AltroDecoderError(ErrorType_t errtype, const char* message) : mErrorType(errtype), mErrorMessage(message) {}

  /// \brief Destructor
  ~AltroDecoderError() noexcept override = default;

  /// \brief Access to error message cnnected to the error
  /// \return Error message
  const char* what() const noexcept override { return mErrorMessage.data(); }

  /// \brief Access to the error type connected to the error
  /// \return Error type
  ErrorType_t getErrorType() const noexcept { return mErrorType; }

 private:
  ErrorType_t mErrorType;    ///< Code of the decoding error type
  std::string mErrorMessage; ///< Message connected to the error type
};

/// \class AltroDecoder
/// \brief Decoder of the ALTRO data in the raw page
/// \ingroup PHOSreconstruction
/// \author Dmitri Peresunko aftesr Markus Fasel
/// \since Sept, 2020
///
/// This is a base class for reading raw data digits in Altro format.
/// The class is able to read the RCU v3 and above formats.
/// The main difference between the format V3 and older ones is in
/// the coding of the 10-bit Altro payload words. In V3 3 10-bit words
/// are coded in one 32-bit word. The bits 30 and 31 are used to identify
/// the payload, altro header and RCU trailer contents.
///
/// Based on AliAltroRawStreamV3 and AliCaloRawStreamV3 by C. Cheshkov

class AltroDecoder
{
 public:
  /// \brief Constructor
  AltroDecoder() = default;

  /// \brief Destructor
  ~AltroDecoder() = default;

  /// \brief Decode the ALTRO stream
  /// \throw AltroDecoderError if the RCUTrailer or ALTRO payload cannot be decoded
  ///
  /// Decoding and checking the RCUTtrailer and
  /// all channels and bunches in the ALTRO stream.
  /// After successfull decoding the Decoder can provide
  /// a reference to the RCU trailer and a vector
  /// with the decoded chanenels, each containing
  /// its bunches.
  AltroDecoderError::ErrorType_t decode(RawReaderMemory& rawreader, CaloRawFitter* rawFitter,
                                        std::vector<o2::phos::Cell>& cellContainer,
                                        std::vector<o2::phos::Cell>& truContainer);

  /// \brief Get list of hw errors found in decoding
  const std::vector<o2::phos::RawReaderError>& hwerrors() { return mOutputHWErrors; }

  const std::vector<short>& chi2list() { return mOutputFitChi; }

  /// \brief Get reference to the RCU trailer object
  /// \return reference to the RCU trailers vector
  const RCUTrailer& getRCUTrailer() const { return mRCUTrailer; }

  /// \brief Read channels for the current event in the raw buffer
  void readChannels(const std::vector<uint32_t>& payloadwords, CaloRawFitter* rawFitter,
                    std::vector<o2::phos::Cell>& cellContainer,
                    std::vector<o2::phos::Cell>& truContainer);
  void setPedestalRun()
  {
    mPedestalRun = true;
    mCombineGHLG = false;
  }
  void setCombineHGLG(bool a) { mCombineGHLG = a; }

  void setKeepTruNoise(bool a) { mKeepTruNoise = a; }

  void setPresamples(int ps) { mPreSamples = ps; }

 private:
  union truDigitPack {
    int32_t mDataWord;
    struct {
      int32_t mHeader : 1; ///< Bit  0 : digit exist
      int32_t mAmp : 15;   ///< Bits  1 - 15: amplitude in channel
      int32_t mTime : 16;  ///< Bits 16 - 25: Time in TRU ticks
    };
  };

  static constexpr int kGeneralSRUErr = 15; ///< Non-existing FEE card to store general SRU errors
  static constexpr int kGeneralTRUErr = 16; ///< Non-existing FEE card to store general TRU errors
  // check and convert HW address to absId and caloFlag
  bool hwToAbsAddress(short hwaddress, short& absId, Mapping::CaloFlag& caloFlag);
  // read trigger digits
  void readTRUDigits(short absId, int payloadSize);
  // read trigger summary tables
  void readTRUFlags(short hwAddress, int payloadSize);
  // Check if TRU digit belongs/matches TRU flag
  bool matchTruDigits(const Cell& cTruFlag, float& sumAmp);

  bool mCombineGHLG = true;                              ///< Combine or not HG and LG channels (def: combine, LED runs: not combine)
  bool mPedestalRun = false;                             ///< Analyze pedestal run (calculate pedestal mean and RMS)
  bool mKeepTruNoise = false;                            ///< Keep all TRU channels for noise scan
  short mddl;                                            ///< Current DDL
  short mPreSamples = 0;                                 ///< Number of pre-samples in time calculation
  std::vector<uint16_t> mBunchwords;                     ///< (transient) bunch of samples for current channel
  std::vector<o2::phos::RawReaderError> mOutputHWErrors; ///< Errors occured in reading data
  std::vector<short> mOutputFitChi;                      ///< Raw sample fit quality
  std::vector<Cell> mTRUFlags;                           ///< trigger summary table
  std::array<int32_t, 224> mTRUDigits;                   ///< list of active TRU digits in DDL
  std::bitset<128> mFlag4x4Bitset;
  std::bitset<128> mFlag2x2Bitset;
  RCUTrailer mRCUTrailer; ///< RCU trailer

  ClassDefNV(AltroDecoder, 2);
};

} // namespace phos

} // namespace o2

#endif
