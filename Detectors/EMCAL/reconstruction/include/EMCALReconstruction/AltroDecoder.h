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
#ifndef ALICEO2_EMCAL_ALTRODECODER_H
#define ALICEO2_EMCAL_ALTRODECODER_H

#include <exception>
#include <iosfwd>
#include <gsl/span>
#include <string>
#include "EMCALBase/RCUTrailer.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/Channel.h"
#include "EMCALReconstruction/RawReaderMemory.h"

namespace o2
{
namespace emcal
{

/// \class AltroDecoderError
/// \brief Error handling of the ALTRO Decoder
/// \ingroup EMCALreconstruction
class AltroDecoderError : public std::exception
{
 public:
  /// \enum ErrorType_t
  /// \brief Error codes connected with the ALTRO decoding
  enum class ErrorType_t {
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

  /// \brief convert the error type from symoblic constant into int
  /// \return the error number
  static int errorTypeToInt(ErrorType_t errortype);

  /// \brief convert the error from number into a type (symbolic constant)
  /// \return the error type
  static ErrorType_t intToErrorType(int errornumber);

  /// \brief Get the number of error types handled by the AltroDecoderError
  /// \return Number of error types
  static constexpr int getNumberOfErrorTypes() noexcept { return 8; }

  /// \brief Access to the error type connected to the erro
  /// \return Error type
  const ErrorType_t getErrorType() const noexcept { return mErrorType; }

 private:
  ErrorType_t mErrorType;    ///< Code of the decoding error type
  std::string mErrorMessage; ///< Message connected to the error type
};

/// \class MinorAltroDecodingError
/// \brief Error handling for the ALTRO decoder for non-crashing errors
/// \ingroup EMCALreconstruction
class MinorAltroDecodingError
{
 public:
  /// \enum ErrorType_t
  /// \brief Error codes connected with the ALTRO decoding
  enum class ErrorType_t {
    BUNCH_HEADER_NULL,            ///< Bunch header is 0
    CHANNEL_END_PAYLOAD_UNEXPECT, ///< Unexpected end of payload (channel or trailer word in bunch words)
    CHANNEL_PAYLOAD_EXCEED,       ///< Exceeding channel payload block
    BUNCH_LENGTH_EXCEED           ///< Bunch length exceeding channel payload size
  };

  /// \brief Dummy constructor
  MinorAltroDecodingError() = default;

  /// \brief Constructor, initializing the object
  /// \param errtype Type of the error
  /// \param channelHeader Header of the channel raising the error
  /// \param payloadword Payload word raising the error
  MinorAltroDecodingError(ErrorType_t errtype, uint32_t channelHeader, uint32_t payloadword) : mErrorType(errtype),
                                                                                               mChannelHeader(channelHeader),
                                                                                               mPayloadWord(payloadword)
  {
  }

  /// \brief Destructor
  ~MinorAltroDecodingError() noexcept = default;

  /// \brief Get the header of the channel raising the error
  /// \return Hardware address
  uint32_t getChannelHeader() const noexcept { return mChannelHeader; };

  /// \brief Get the payload word raising the error
  /// \return Payload word
  uint32_t getPayloadWord() const noexcept { return mPayloadWord; }

  /// \brief Get the type of the error
  /// \return Error type
  ErrorType_t getErrorType() const noexcept { return mErrorType; }

  /// \brief Create and return error message for different error types
  /// \return Error message
  ///
  /// Object returning a std::string which can be owned by the caller.
  /// This is in contrast to exceptions inheriting from std::exception
  /// which must return const char * in order to comply with the interface.
  std::string what() const noexcept;

  /// \brief convert the error type from symoblic constant into int
  /// \return the error number
  static int errorTypeToInt(ErrorType_t errortype);

  /// \brief convert the error from number into a type (symbolic constant)
  /// \return the error type
  static ErrorType_t intToErrorType(int errornumber);

  /// \brief Get the number of error types handled by the AltroDecoderError
  /// \return Number of error types
  static constexpr int getNumberOfErrorTypes() noexcept { return 2; }

 private:
  ErrorType_t mErrorType;  ///< Type of the error
  uint32_t mChannelHeader; ///< Hardware address raising the error
  uint32_t mPayloadWord;   ///< Payload word raising the error
};

/// \class AltroDecoder
/// \brief Decoder of the ALTRO data in the raw page
/// \ingroup EMCALreconstruction
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Aug. 12, 2019
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
  /// \param reader Raw reader instance to be decoded
  AltroDecoder(RawReaderMemory& reader);

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
  void decode();

  /// \brief Get reference to the RCU trailer object
  /// \return const reference to the RCU trailer
  /// \throw AltroDecoderError with type RCU_TRAILER_ERROR if the RCU trailer was not initialized
  const RCUTrailer& getRCUTrailer() const;

  /// \brief Get the reference to the channel container
  /// \return Reference to the channel container
  /// \throw AltroDecoderError with CHANNEL_ERROR if the channel container was not initialized for the current event
  const std::vector<Channel>& getChannels() const;

  /// \brief Read RCU trailer for the current event in the raw buffer
  void readRCUTrailer();

  /// \brief Read channels for the current event in the raw buffer
  void readChannels();

  /// \brief Get list of minor decoding errors
  /// \return List of minor decoding errors
  const std::vector<MinorAltroDecodingError>& getMinorDecodingErrors() const { return mMinorDecodingErrors; }

 private:
  /// \brief run checks on the RCU trailer
  /// \throw Error if the RCU trailer has inconsistencies
  ///
  /// Performing various consistency checks on the RCU trailer
  /// In case of failure an exception is thrown.
  void checkRCUTrailer();

  RawReaderMemory& mRawReader;                               ///< underlying raw reader
  RCUTrailer mRCUTrailer;                                    ///< RCU trailer
  std::vector<Channel> mChannels;                            ///< vector of channels in the raw stream
  std::vector<MinorAltroDecodingError> mMinorDecodingErrors; ///< Container for minor (non-crashing) errors
  bool mChannelsInitialized = false;                         ///< check whether the channels are initialized

  ClassDefNV(AltroDecoder, 1);
};

} // namespace emcal

} // namespace o2

#endif