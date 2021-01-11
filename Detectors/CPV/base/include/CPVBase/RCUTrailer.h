// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_CPV_RCUTRAILER_H
#define ALICEO2_CPV_RCUTRAILER_H

#include <exception>
#include <iosfwd>
#include <string>
#include <cstdint>
#include <gsl/span>
#include "Rtypes.h"

namespace o2
{

namespace cpv
{

/// \class RCUTrailer
/// \brief Information stored in the RCU trailer
/// \ingroup CPVbase
///
/// The RCU trailer can be found at the end of
/// the payload and contains general information
/// sent by the SRU.
class RCUTrailer
{
 public:
  /// \class Error
  /// \brief Error handling of the
  class Error : public std::exception
  {
   public:
    /// \enum ErrorType_t
    /// \brief Error codes for different error types
    enum class ErrorType_t {
      DECODING_INVALID,     ///< Invalid words during decoding
      SIZE_INVALID,         ///< Invalid trailer size
      SAMPLINGFREQ_INVALID, ///< Invalid sampling frequency
      L1PHASE_INVALID       ///< Invalid L1 phase
    };

    /// \brief Constructor
    /// \param errtype Code of the error type
    /// \param message corresponding error message
    ///
    /// Initializing the error with error code and message.
    /// To be called when the exception is raised.
    Error(ErrorType_t errtype, const char* message) : mErrorType(errtype), mErrorMessage(message) {}

    /// \brief Destructor
    ~Error() noexcept override = default;

    /// \brief Access to the error message
    /// \return Error message related to the exception type
    const char* what() const noexcept override { return mErrorMessage.data(); }

    /// \brief Access to error code
    /// \return Error code of the exception type
    ErrorType_t getErrorType() const noexcept { return mErrorType; }

   private:
    ErrorType_t mErrorType;    ///< Type of the error
    std::string mErrorMessage; ///< Error Message
  };

  /// \brief Constructor
  RCUTrailer() = default;

  /// \brief destructor
  ~RCUTrailer() = default;

  /// \brief Reset the RCU trailer
  ///
  /// Setting all values to 0
  void reset();

  /// \brief Prints the contents of the RCU trailer data
  /// \param stream stream the trailer has to be put on
  void printStream(std::ostream& stream) const;

  /// \brief Decode RCU trailer from the 32-bit words in the raw buffer
  /// \param buffer Raw buffer from which to read the trailer
  ///
  /// Read the RCU trailer according to the RCU formware version
  /// specified in CDH.
  void constructFromRawPayload(const gsl::span<const char> payload);

  unsigned int getFECErrorsA() const { return mFECERRA; }
  unsigned int getFECErrorsB() const { return mFECERRB; }
  unsigned short getErrorsG2() const { return mERRREG2; }
  unsigned int getErrorsG3() const { return mERRREG3; }
  unsigned short getActiveFECsA() const { return mActiveFECsA; }
  unsigned short getActiveFECsB() const { return mActiveFECsB; }
  unsigned int getAltroCFGReg1() const { return mAltroCFG1; }
  unsigned int getAltroCFGReg2() const { return mAltroCFG2; }
  int getRCUID() const { return mRCUId; }
  unsigned int getTrailerSize() const { return mTrailerSize; }
  unsigned int getPayloadSize() const { return mPayloadSize; }
  unsigned char getFirmwareVersion() const { return mFirmwareVersion; }

  unsigned short getNumberOfChannelAddressMismatch() const { return (mERRREG3 & 0xFFF); }
  unsigned short getNumberOfChannelLengthMismatch() const { return ((mERRREG3 >> 12) & 0x1FFF); }
  unsigned char getBaselineCorrection() const { return mAltroCFG1 & 0xF; }
  bool getPolarity() const { return (mAltroCFG1 >> 4) & 0x1; }
  unsigned char getNumberOfPresamples() const { return (mAltroCFG1 >> 5) & 0x3; }
  unsigned char getNumberOfPostsamples() const { return (mAltroCFG1 >> 7) & 0xF; }
  bool hasSecondBaselineCorr() const { return (mAltroCFG1 >> 11) & 0x1; }
  unsigned char getGlitchFilter() const { return (mAltroCFG1 >> 12) & 0x3; }
  unsigned char getNumberOfNonZeroSuppressedPostsamples() const { return (mAltroCFG1 >> 14) & 0x7; }
  unsigned char getNumberOfNonZeroSuppressedPresamples() const { return (mAltroCFG1 >> 17) & 0x3; }
  bool hasZeroSuppression() const { return (mAltroCFG1 >> 19) & 0x1; }
  bool getNumberOfAltroBuffers() const { return (mAltroCFG2 >> 24) & 0x1; }
  unsigned char getNumberOfPretriggerSamples() const { return (mAltroCFG2 >> 20) & 0xF; }
  unsigned short getNumberOfSamplesPerChannel() const { return (mAltroCFG2 >> 10) & 0x3FF; }
  bool isSparseReadout() const { return (mAltroCFG2 >> 9) & 0x1; }

  /// \brief Access to the sampling time
  /// \return Sampling time in seconds.
  /// \throw Error if the RCU trailer was not properly initializied
  double getTimeSample() const;

  /// \brief set time sample
  /// \param timesample Time sample (in ns)
  void setTimeSample(double timesample);

  /// \brief Access to the L1 phase
  /// \return L1 phase w.r.t to the LHC clock
  double getL1Phase() const;

  /// \brief Set the L1 phase
  /// \param l1phase L1 phase (in ns)
  void setL1Phase(double l1phase);

  void setFECErrorsA(unsigned int value) { mFECERRA = value; }
  void setFECErrorsB(unsigned int value) { mFECERRB = value; }
  void setErrorsG2(unsigned short value) { mERRREG2 = value; }
  void setErrorsG3(unsigned int value) { mERRREG3 = value; }
  void setActiveFECsA(unsigned short value) { mActiveFECsA = value; }
  void setActiveFECsB(unsigned short value) { mActiveFECsB = value; }
  void setAltroCFGReg1(unsigned int value) { mAltroCFG1 = value; }
  void setAltroCFGReg2(unsigned int value) { mAltroCFG2 = value; }
  void setFirmwareVersion(unsigned char version) { mFirmwareVersion = version; }
  void setPayloadSize(unsigned int size) { mPayloadSize = size; }

  /// \brief checlks whether the RCU trailer is initialzied
  /// \return True if the trailer is initialized, false otherwise
  bool isInitialized() const { return mIsInitialized; }

  std::vector<uint32_t> encode() const;

  static RCUTrailer constructFromPayloadWords(const gsl::span<const uint32_t> payloadwords);
  static RCUTrailer constructFromPayload(const gsl::span<const char> payload);

 private:
  int mRCUId = -1;                    ///< current RCU identifier
  unsigned char mFirmwareVersion = 0; ///< RCU firmware version
  unsigned int mTrailerSize = 0;      ///< Size of the trailer (in number of 32 bit words)
  unsigned int mPayloadSize = 0;      ///< Size of the payload (in nunber of 32 bit words)
  unsigned int mFECERRA = 0;          ///< contains errors related to ALTROBUS transactions
  unsigned int mFECERRB = 0;          ///< contains errors related to ALTROBUS transactions
  unsigned short mERRREG2 = 0;        ///< contains errors related to ALTROBUS transactions or trailer of ALTRO channel block
  unsigned int mERRREG3 = 0;          ///< contains number of altro channels skipped due to an address mismatch
  unsigned short mActiveFECsA = 0;    ///< bit pattern of active FECs in branch A
  unsigned short mActiveFECsB = 0;    ///< bit pattern of active FECs in branch B
  unsigned int mAltroCFG1 = 0;        ///< ALTROCFG1 register
  unsigned int mAltroCFG2 = 0;        ///< ALTROCFG2 and ALTROIF register
  bool mIsInitialized = false;        ///< Flag whether RCU trailer is initialized for the given raw event

  ClassDefNV(RCUTrailer, 1);
};

std::ostream& operator<<(std::ostream& stream, const RCUTrailer& trailer);

} // namespace cpv

} // namespace o2

#endif