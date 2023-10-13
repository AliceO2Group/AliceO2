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
#ifndef ALICEO2_EMCAL_RCUTRAILER_H
#define ALICEO2_EMCAL_RCUTRAILER_H

#include <exception>
#include <iosfwd>
#include <string>
#include <cstdint>
#include <gsl/span>
#include <Rtypes.h>

namespace o2
{

namespace emcal
{

/// \class RCUTrailer
/// \brief Information stored in the RCU trailer
/// \ingroup EMCALbase
///
/// The RCU trailer can be found at the end of
/// the payload and contains general information
/// sent by the SRU.
///
/// Definition of the trailer words:
/// - mFirmwareVersion: Firmware version
/// - mTrailerSize: Size of the trailer in DDL words (32-bit)
/// - mPayloadSize: Size of the payload in DDL words (32-bit)
/// - mFECERRA
/// - mFECERRB
/// - mERRREG2
/// - mERRREG3
///   |    Bits   |                  Error type                   |
///   |-----------|-----------------------------------------------|
///   | 0 - 11    | Number of channels with address mismatch      |
///   | 12 - 24   | Number of channels with length mismatch       |
///   | 25 - 31   | Zeroed (used for trailer word markers)        |
/// - mActiveFECsA
/// - mActiveFECsB
/// - mAltroCFG1 (32 bit)
///   |    Bits   |                   Setting                     |
///   |-----------|-----------------------------------------------|
///   |  0-3      | Baseline correction                           |
///   |  4        | Polarity                                      |
///   |  5-6      | Number of presamples                          |
///   |  7-10     | Number of postsamples                         |
///   |  11       | Second baseline correction                    |
///   |  12-13    | Glitch filter                                 |
///   |  14-16    | Number of postsamples before zero suppression |
///   |  17-18    | Number of presamples before zero suppression  |
///   |  19       | Zero suppression on / off                     |
///   |  20 - 31  | Zeroed (used for trailer word markers)        |
/// - mAltroCFG2 (32 bit)
///   |    Bits   |                   Setting                     |
///   |-----------|-----------------------------------------------|
///   | 0 - 4     | L1 phase                                      |
///   | 5 - 9     | Length of the time sample                     |
///   | 9         | Sparse readout on / off                       |
///   | 10 - 19   | Number of samples per channel                 |
///   | 20 - 23   | Number of pretrigger samples                  |
///   | 24        | ALTRO buffers (0 - 4 buffers, 1 - 8 buffers)  |
///   | 25 - 31   | Zeroed (used for trailer word markers)        |
///
class RCUTrailer
{
 public:
  /// \class Error
  /// \brief Error handling of the RCU trailer
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

  /// \enum BufferMode_t
  /// \brief Handler for encoding of the number of ALTRO buffers in the configuration
  enum BufferMode_t {
    NBUFFERS4 = 0, ///< 4 ALTRO buffers
    NBUFFERS8 = 1  ///< 8 ALTRO buffers
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
  void constructFromRawPayload(const gsl::span<const uint32_t> payloadwords);

  /// \brief Get index of the RCU the trailer belongs to
  /// \return RCU index
  int getRCUID() const { return mRCUId; }

  /// \brief Get the trailer size in number of DDL (32 bit) words
  /// \return Size of the RCU trailer
  uint32_t getTrailerSize() const { return mTrailerSize; }

  /// \brief Get size of the payload as number of DDL (32-bit) words
  /// \return Size of the payload as number of 32 bit workds
  uint32_t getPayloadSize() const { return mPayloadSize; }

  /// \brief Get number of corrupted trailer words (undefined trailer word code)
  /// \return Number of trailer word corruptions
  uint32_t getTrailerWordCorruptions() const { return mWordCorruptions; }

  /// \brief Get the firmware version
  /// \return Firmware version
  uint8_t getFirmwareVersion() const { return mFirmwareVersion; }

  /// \brief Set the firmware version
  /// \param version Firmware version
  void setFirmwareVersion(uint8_t version) { mFirmwareVersion = version; }

  /// \brief Set the ID of the RCU
  /// \param rcuid ID of the RCU
  void setRCUID(int rcuid) { mRCUId = rcuid; }

  /// \brief set the payload size in number of DDL (32-bit) words
  /// \param size Payload size
  void setPayloadSize(uint32_t size) { mPayloadSize = size; }

  /// \brief Access to the sampling time
  /// \return Sampling time in seconds.
  /// \throw Error if the RCU trailer was not properly initializied
  double getTimeSampleNS() const;

  /// \brief Access to the L1 phase
  /// \return L1 phase w.r.t to the LHC clock
  double getL1PhaseNS() const;

  /// \brief Set the time sample length and L1 phase based on the trigger time
  /// \param time Trigger time (in ns)
  /// \param timesample Time sample (in ns)
  ///
  /// L1 phase: Collision time with respect to the sample length. Number
  /// of phases: Sample length / bunch spacing (25 ns)
  void setTimeSamplePhaseNS(uint64_t triggertime, uint64_t timesample);

  //
  // Error counters
  //

  /// \brief Get the number of channels with address mismatch
  /// \return Number of channels
  uint16_t getNumberOfChannelAddressMismatch() const { return mErrorCounter.mNumChannelAddressMismatch; }

  /// \brief Get the number of channels with length mismatch
  /// \return Number of channels
  uint16_t getNumberOfChannelLengthMismatch() const { return mErrorCounter.mNumChannelLengthMismatch; }

  /// \brief Set the number of channels with address mismatch
  /// \param nchannel Number of channels
  void setNumberOfChannelAddressMismatch(uint16_t nchannel) { mErrorCounter.mNumChannelAddressMismatch = nchannel; }

  /// \brief Set the number of channels with length mismatch
  /// \param nchannel Number of channels
  void setNumberOfChannelLengthMismatch(uint8_t nchannel) { mErrorCounter.mNumChannelLengthMismatch = nchannel; }

  uint32_t getFECErrorsA() const { return mFECERRA; }
  uint32_t getFECErrorsB() const { return mFECERRB; }
  uint16_t getActiveFECsA() const { return mActiveFECsA; }
  uint16_t getActiveFECsB() const { return mActiveFECsB; }
  void setFECErrorsA(uint32_t value) { mFECERRA = value; }
  void setFECErrorsB(uint32_t value) { mFECERRB = value; }
  void setActiveFECsA(uint16_t value) { mActiveFECsA = value; }
  void setActiveFECsB(uint16_t value) { mActiveFECsB = value; }

  //
  // ALTRO configuration
  //

  /// \brief Get baseline correction method
  /// \return baseline correction method
  uint16_t getBaselineCorrection() const { return mAltroConfig.mBaselineCorrection; }

  /// \brief Check polarity setting
  /// \return Polarity setting
  bool getPolarity() const { return mAltroConfig.mPolarity; }

  /// \brief Get the number of presamples (after zero suppression)
  /// \return Number of presamples
  uint16_t getNumberOfPresamples() const { return mAltroConfig.mNumPresamples; }

  /// \brief Get the number of postsamples (after zero suppression)
  /// \return Number of postsamples
  uint16_t getNumberOfPostsamples() const { return mAltroConfig.mNumPostsamples; }

  /// \brief Check if second baseline correction is applied
  /// \return True if second baseline correction has been applied, false otherwise
  bool hasSecondBaselineCorr() const { return mAltroConfig.mSecondBaselineCorrection; }

  /// \brief Get the glitch filter
  /// \return Glitch filter setting
  uint16_t getGlitchFilter() const { return mAltroConfig.mGlitchFilter; }

  /// \brief Get the number of postsamples before zero suppression
  /// \return Number of postsamples
  uint16_t getNumberOfNonZeroSuppressedPostsamples() const { return mAltroConfig.mNumPostsamplesNoZS; }

  /// \brief Get the number of presamples before zero suppression
  /// \return Number of presamples
  uint16_t getNumberOfNonZeroSuppressedPresamples() const { return mAltroConfig.mNumPresamplesNoZS; }

  /// \brief Check whether zero suppression has been applied
  /// \return True if zero suppression has been applied, false otherwise
  bool hasZeroSuppression() const { return mAltroConfig.mZeroSuppression; }

  /// \brief Get the number of pretrigger samples
  /// \return Number of samples
  uint16_t getNumberOfPretriggerSamples() const { return mAltroConfig.mNumSamplesPretrigger; }

  /// \brief Get the number of samples per channel
  /// \return Number of samples per channel
  uint16_t getNumberOfSamplesPerChannel() const { return mAltroConfig.mNumSamplesChannel; }

  /// \brief Get the number of ALTRO buffers
  /// \return Number of Altro Buffers
  uint16_t getNumberOfAltroBuffers() const { return BufferMode_t(mAltroConfig.mAltroBuffers) == BufferMode_t::NBUFFERS4 ? 4 : 8; }

  /// \brief Check whether readout is in sparse mode
  /// \return True if the readout is in sparse mode, false otherwise
  bool isSparseReadout() const { return mAltroConfig.mSparseReadout; }

  /// \brief Set baseline correction method
  /// \param baselineCorrection Baseline correction method
  void setBaselineCorrection(uint16_t baselineCorrection) { mAltroConfig.mBaselineCorrection = baselineCorrection; }

  /// \brief Set the polarity
  /// \param doSet If true polarity is set
  void setPolarity(bool doSet) { mAltroConfig.mPolarity = doSet; }

  /// \brief Set the number of presamples (after zero suppression)
  /// \param npresamples Number of presamples
  void setNumberOfPresamples(uint16_t npresamples) { mAltroConfig.mNumPresamples = npresamples; }

  /// \brief Set the number of postsamples (after zero suppression)
  /// \param
  void setNumberOfPostsamples(uint16_t npostsamples) { mAltroConfig.mNumPostsamples = npostsamples; }

  /// \brief Specify whether second basedline correction has been applied
  /// \param doHave If true a second baseline correction has bben applied
  void setSecondBaselineCorrection(bool doHave) { mAltroConfig.mSecondBaselineCorrection = doHave; }

  /// \brief Set the glitch filter
  /// \param glitchfilter Glitch filter
  void setGlitchFilter(uint16_t glitchfilter) { mAltroConfig.mGlitchFilter = glitchfilter; }

  /// \brief Set the number of postsamples before zero suppression
  /// \param npostsamples Number of postsamples
  void setNumberOfNonZeroSuppressedPostsamples(uint16_t npostsamples) { mAltroConfig.mNumPostsamplesNoZS = npostsamples; }

  /// \brief Set the number of presamples after zero suppression
  /// \param npresamples Number of presamples
  void setNumberOfNonZeroSuppressedPresamples(uint16_t npresamples) { mAltroConfig.mNumPresamplesNoZS = npresamples; }

  /// \brief Set the number of pretrigger samples
  /// \param nsamples Number of samples
  void setNumberOfPretriggerSamples(uint16_t nsamples) { mAltroConfig.mNumSamplesPretrigger = nsamples; }

  /// \brief Set the number of samples per channel
  /// \param nsamples Number of samples
  void setNumberOfSamplesPerChannel(uint16_t nsamples) { mAltroConfig.mNumSamplesChannel = nsamples; }

  /// \brief Specify whether zero suppression has been applied
  /// \param doHave If true zero suppression has been applied
  void setZeroSuppression(bool doHave) { mAltroConfig.mZeroSuppression = doHave ? 1 : 0; }

  /// \brief Set sparse readout mode
  /// \param isSparse True if readout is in sparse mode, false otherwise
  void setSparseReadout(bool isSparse) { mAltroConfig.mSparseReadout = isSparse; }

  /// \brief Set the number of ALTRO buffers
  /// \param bufmode Number of ALTRO buffers (4 or 8 buffers)
  void setNumberOfAltroBuffers(BufferMode_t bufmode) { mAltroConfig.mAltroBuffers = uint8_t(bufmode); }

  ///
  /// Direct access to RCU trailer registers (not recommended)
  ///

  /// \brief Get value stored in error counter register 2
  /// \return Value of the register
  uint16_t getErrorsG2() const { return mErrorCounter.mErrorRegister2; }

  /// \brief Get value stored in error counter register 3
  /// \return Value of the register
  /// \deprecated Use dedicated getters for error counters
  uint32_t getErrorsG3() const { return mErrorCounter.mErrorRegister3; }

  /// \brief Get value stored in ALTRO config register 1
  /// \return Value of the register
  /// \deprecated Use dedicated getters for ALTRO configuration
  uint32_t getAltroCFGReg1() const { return mAltroConfig.mWord1; }

  /// \brief Get value stored in ALTRO config register 1
  /// \return Value of the register
  /// \deprecated Use dedicated getters for ALTRO configuration
  uint32_t getAltroCFGReg2() const { return mAltroConfig.mWord2; }

  /// \brief Set error counter register 2
  /// \param value Value for register
  void setErrorsG2(uint16_t value) { mErrorCounter.mErrorRegister2 = value; }

  /// \brief Set error counter register 3
  /// \param value Value for register
  /// \deprecated Use dedicated setters for error counters
  void setErrorsG3(uint32_t value) { mErrorCounter.mErrorRegister3 = value; }

  /// \brief Set ALTRO config register 1
  /// \param value Value for register
  /// \deprecated Use dedicated setters for configuration
  void setAltroCFGReg1(uint32_t value) { mAltroConfig.mWord1 = value; }

  /// \brief Set ALTRO config register 2
  /// \param value Value for register
  /// \deprecated Use dedicated setters for configuration
  void setAltroCFGReg2(uint32_t value) { mAltroConfig.mWord2 = value; }

  /// \brief checlks whether the RCU trailer is initialzied
  /// \return True if the trailer is initialized, false otherwise
  bool isInitialized() const { return mIsInitialized; }

  /// \brief Encode RCU trailer as array of DDL (32-bit) words
  /// \return array of trailer words after encoding
  ///
  /// Encoded trailer words always contain the trailer pattern
  /// (bit 30 and bit 31 set)
  std::vector<uint32_t> encode() const;

  /// \brief Decode RCU trailer from payload
  /// \return RCU trailer found at the end of the given payload
  ///
  /// The trailer is expected at the end of the paylaod. Trailer words
  /// are identified via their trailer marker (bits 30 and 31), and
  /// are assigned based on the trailer word marker.
  static RCUTrailer constructFromPayloadWords(const gsl::span<const uint32_t> payloadwords);

  /// \brief Check whether the word is a valid last trailer word
  /// \param trailerword Word to be checked
  /// \return True if the word is a valid last trailer word, false if there are inconsistencies
  static bool checkLastTrailerWord(uint32_t trailerword);

 private:
  /// \struct AltroConfig
  /// \brief Bit field configuration of the ALTRO config registers
  struct AltroConfig {
    union {
      uint32_t mWord1 = 0; ///< ALTROCFG1 register
      struct {
        uint32_t mBaselineCorrection : 4;       ///< Baseline correction setting
        uint32_t mPolarity : 1;                 ///< Polarity
        uint32_t mNumPresamples : 2;            ///< Number of presamples
        uint32_t mNumPostsamples : 4;           ///< Number of postsamples
        uint32_t mSecondBaselineCorrection : 1; ///< Second baseline correction
        uint32_t mGlitchFilter : 2;             ///< Glitch filter
        uint32_t mNumPostsamplesNoZS : 3;       ///< Number of postsamples without zero suppression
        uint32_t mNumPresamplesNoZS : 2;        ///< Number of presamples without zero suppression
        uint32_t mZeroSuppression : 1;          ///< Zero suppression
        uint32_t mZero1_1 : 12;                 ///< Zeroed bits
      };
    };

    union {
      uint32_t mWord2 = 0; ///< ALTROCFG2 and ALTROIF register (see class description for bit configuration)
      struct {
        uint32_t mL1Phase : 5;              ///< L1 phase
        uint32_t mSampleTime : 4;           ///< Length of the time sample
        uint32_t mSparseReadout : 1;        ///< Sparse readout
        uint32_t mNumSamplesChannel : 10;   ///< Number of samples per channel
        uint32_t mNumSamplesPretrigger : 4; ///< Number of samples per pretrigger
        uint32_t mAltroBuffers : 1;         ///< Number of ALTRO buffers
        uint32_t mZero2_2 : 7;              ///< Zeroed bits
      };
    };
  };

  /// \struct ErrorCounters
  /// \brief Bit definition for error counter registers
  struct ErrorCounters {
    union {
      uint32_t mErrorRegister2 = 0; ///< contains errors related to ALTROBUS transactions or trailer of ALTRO channel block
    };
    union {
      uint32_t mErrorRegister3 = 0; ///< contains number of altro channels skipped due to an address mismatch
      struct {
        uint32_t mNumChannelAddressMismatch : 12; ///< Number of channels with address mismatch
        uint32_t mNumChannelLengthMismatch : 13;  ///< Number of channels with link mismatch
        uint32_t mZero3_1 : 7;                    ///< Zeroed bits
      };
    };
  };

  int mRCUId = -1;                      ///< current RCU identifier
  uint8_t mFirmwareVersion = 0;         ///< RCU firmware version
  uint32_t mTrailerSize = 0;            ///< Size of the trailer (in number of 32 bit words)
  uint32_t mPayloadSize = 0;            ///< Size of the payload (in nunber of 32 bit words)
  uint32_t mWordCorruptions = 0;        ///< Number of trailer word corruptions (decoding only)
  uint32_t mFECERRA = 0;                ///< contains errors related to ALTROBUS transactions
  uint32_t mFECERRB = 0;                ///< contains errors related to ALTROBUS transactions
  ErrorCounters mErrorCounter = {0, 0}; ///< Error counter registers
  uint16_t mActiveFECsA = 0;            ///< bit pattern of active FECs in branch A
  uint16_t mActiveFECsB = 0;            ///< bit pattern of active FECs in branch B
  AltroConfig mAltroConfig = {0, 0};    ///< ALTRO configuration registers
  bool mIsInitialized = false;          ///< Flag whether RCU trailer is initialized for the given raw event

  ClassDefNV(RCUTrailer, 1);
};

std::ostream& operator<<(std::ostream& stream, const RCUTrailer& trailer);

} // namespace emcal

} // namespace o2

#endif