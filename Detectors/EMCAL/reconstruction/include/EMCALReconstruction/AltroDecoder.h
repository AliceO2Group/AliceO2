// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef __O2_EMCAL_ALTRODECODER_H__
#define __O2_EMCAL_ALTRODECODER_H__

#include <exception>
#include <iosfwd>
#include <gsl/span>
#include <string>
#include "EMCALReconstruction/RawReaderFile.h"

namespace o2
{
namespace emcal
{

/// \class AltroDecoder
/// \brief Decoder of the ALTRO data in the raw page
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
  /// \class RCUTrailer
  /// \brief Information stored in the RCU trailer
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
      /// \param Message corresponding error message
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
    void constructFromRawBuffer(const RawBuffer& buffer);

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

    /// \brief Access to the L1 phase
    /// \return L1 phase w.r.t to the LHC clock
    double getL1Phase() const;

    void setFECErrorsA(unsigned int value) { mFECERRA = value; }
    void setFECErrorsB(unsigned int value) { mFECERRB = value; }
    void setErrorsG2(unsigned short value) { mERRREG2 = value; }
    void setErrorsG3(unsigned int value) { mERRREG3 = value; }
    void setActiveFECsA(unsigned short value) { mActiveFECsA = value; }
    void setActiveFECsB(unsigned short value) { mActiveFECsB = value; }
    void setAltroCFGReg1(unsigned int value) { mAltroCFG1 = value; }
    void setAltroCFGReg2(unsigned int value) { mAltroCFG2 = value; }

    /// \brief checlks whether the RCU trailer is initialzied
    /// \return True if the trailer is initialized, false otherwise
    bool isInitialized() const { return mIsInitialized; }

   private:
    int mRCUId = -1;                    ///< current RCU identifier
    unsigned char mFirmwareVersion = 0; ///< RCU firmware version
    unsigned int mTrailerSize = 0;      ///< Size of the trailer
    unsigned int mPayloadSize = 0;      ///< Size of the payload
    unsigned int mFECERRA = 0;          ///< contains errors related to ALTROBUS transactions
    unsigned int mFECERRB = 0;          ///< contains errors related to ALTROBUS transactions
    unsigned short mERRREG2 = 0;        ///< contains errors related to ALTROBUS transactions or trailer of ALTRO channel block
    unsigned int mERRREG3 = 0;          ///< contains number of altro channels skipped due to an address mismatch
    unsigned short mActiveFECsA = 0;    ///< bit pattern of active FECs in branch A
    unsigned short mActiveFECsB = 0;    ///< bit pattern of active FECs in branch B
    unsigned int mAltroCFG1 = 0;        ///< ALTROCFG1 register
    unsigned int mAltroCFG2 = 0;        ///< ALTROCFG2 and ALTROIF register
    bool mIsInitialized = false;        ///< Flag whether RCU trailer is initialized for the given raw event
  };

  /// \class Bunch
  /// \brief ALTRO bunch information
  ///
  /// The bunch contains the ADC values of a given
  /// data bunch for a channel in the ALTRO stream.
  /// The ADC values are stored in reversed order in
  /// time both in the ALTRO stream and in the bunch
  /// object.
  ///
  /// For iteration one should assume that the end time
  /// is 0, however it can also be larger than 0. In this
  /// case the first value has to be mapped to the end timebin
  /// and the last value to the start timebin, still iterating
  /// only over the number of samples.
  class Bunch
  {
   public:
    /// \brief Constructor
    Bunch() = default;

    /// \brief Initialize the bunch with start time and bunch length
    /// \param length Length of the bunch
    /// \param start Start time of the bunch
    Bunch(uint8_t length, uint8_t start) : mBunchLength(length), mStartTime(start), mADC() {}

    /// \brief
    ~Bunch() = default;

    /// \add ADC value to the bunch
    /// \param adc Next ADC value
    ///
    /// ADC values are stored in reversed order. The next ADC value
    /// has to be earlier in time compared to the previous one.
    void addADC(uint16_t adc) { mADC.emplace_back(adc); }

    /// \brief Initialize the ADC values in the bunch from a range
    /// \param ranage Range of ADC values
    ///
    /// The ADC values are stored in reversed order in time. Therefore
    /// the last entry is the one earliest in time.
    void initFromRange(gsl::span<uint16_t> range);

    /// \brief Get range of ADC values in the bunch
    /// \return ADC values in the bunch
    ///
    /// The ADC values are stored in reversed order in time. Therefore
    /// the last entry is the one earliest in time.
    const std::vector<uint16_t>& getADC() const { return mADC; }

    /// \brief Get the length of the bunch (number of time bins)
    /// \return Length of the bunch
    uint8_t getBunchLength() const { return mBunchLength; }

    /// \brief Get the start time bin
    /// \return Start timebin
    ///
    /// The start timebin is the higher of the two,
    /// the samples are in reversed order.
    uint8_t getStartTime() const { return mStartTime; }

    /// \brief Get the end time bin
    /// \return End timebin
    ///
    /// The end timebin is the lower of the two,
    /// the samples are in reversed order.
    uint8_t getEndTime() const { return mStartTime - mBunchLength + 1; }

   private:
    uint8_t mBunchLength = 0;   ///< Number of ADC samples in buffer
    uint8_t mStartTime = 0;     ///< Start timebin (larger time bin, samples are in reversed order)
    std::vector<uint16_t> mADC; ///< ADC samples in bunch
  };

  /// \class Channel
  /// \brief ALTRO channel representation
  ///
  /// The channel contains information about
  /// a hardware channel in the raw stream. Those
  /// information are:
  /// - Hardware address
  /// - Size of the payload of all bunches in the channel
  ///   as total number of 10-bit words
  /// - Channel status (good or bad)
  /// In addition it contains the data of all bunches in the
  /// raw stream.
  ///
  /// The hardware address itself encods
  /// - Branch ID (bit 12)
  /// - FEC ID (bits 7-10)
  /// - ALTRO ID (bits 4-6)
  /// - Channel ID (bits 0-3)
  class Channel
  {
   public:
    /// \class HardwareAddressError
    /// \brief Handling of uninitialized hardware addresses
    class HardwareAddressError : public std::exception
    {
     public:
      /// \brief Constructor
      HardwareAddressError() = default;

      /// \brief Destructor
      ~HardwareAddressError() noexcept override = default;

      /// \brief Access to error message
      /// \return error message
      const char* what() const noexcept override
      {
        return "Hardware address not initialized";
      }
    };

    /// \brief Dummy constructor
    Channel() = default;

    /// \brief Constructor initializing hardware address and payload size
    /// \param hardwareAddress Harware address
    /// \param payloadSize Size of the payload
    Channel(int32_t hardwareAddress, uint8_t payloadSize) : mHardwareAddress(hardwareAddress),
                                                            mPayloadSize(payloadSize),
                                                            mBunches()
    {
    }

    /// \brief Destructor
    ~Channel() = default;

    /// \brief Check whether the channel is bad
    /// \return true if the channel is bad, false otherwise
    bool isBadChannel() const { return mBadChannel; }

    /// \brief Get the full hardware address
    /// \return Hardware address
    ///
    /// The hardware address contains:
    /// - Branch ID (bit 12)
    /// - FEC ID (bits 7-10)
    /// - ALTRO ID (bits 4-6)
    /// - Channel ID (bits 0-3)
    uint16_t getHardwareAddress() const { return mHardwareAddress; }

    /// \brief Get the size of the payload
    /// \return Size of the payload as number of 10-bit samples (1/3rd words)
    uint8_t getPayloadSize() const { return mPayloadSize; }

    /// \brief Get list of bunches in the channel
    /// \return List of bunches
    const std::vector<Bunch>& getBunches() const { return mBunches; }

    /// \brief Provide the branch index for the current hardware address
    /// \return RCU branch index (0 or 1)
    /// \throw HadrwareAddressError in case the hardware address is not initialized
    int getBranchIndex() const;

    /// \brief Provide the front-end card index for the current hardware address
    /// \return Front-end card index for the current hardware address
    /// \throw HadrwareAddressError in case the hardware address is not initialized
    int getFECIndex() const;

    /// \brief Provide the altro chip index for the current hardware address
    /// \return Altro chip index for the current hardware address
    /// \throw HadrwareAddressError in case the hardware address is not initialized
    int getAltroIndex() const;

    /// \brief Provide the channel index for the current hardware address
    /// \return Channel index for the current hardware address
    /// \throw HadrwareAddressError in case the hardware address is not initialized
    int getChannelIndex() const;

    /// \brief Add bunch to the channel
    /// \param bunch Bunch to be added
    ///
    /// This function will copy the bunch information to the
    /// object, which might be expensive. Better use the
    /// function createBunch.
    void addBunch(const Bunch& bunch) { mBunches.emplace_back(bunch); }

    /// \brief Set the hardware address
    /// \param hardwareAddress Hardware address
    void setHardwareAddress(uint16_t hardwareAddress) { mHardwareAddress = hardwareAddress; }

    /// \brief Set the size of the payload in number of 10-bit words
    /// \param payloadSize Size of the payload
    void setPayloadSize(uint8_t payloadSize) { mPayloadSize = payloadSize; }

    /// \brief Mark the channel status
    /// \param badchannel Bad channel status (true if bad)
    void setBadChannel(bool badchannel) { mBadChannel = badchannel; }

    /// \brief Create and initialize a new bunch and return reference to it
    /// \param bunchlength Length of the bunch
    /// \param starttime Start time of the bunch
    Bunch& createBunch(uint8_t bunchlength, uint8_t starttime);

   private:
    int32_t mHardwareAddress = -1; ///< Hardware address
    uint8_t mPayloadSize = 0;      ///< Payload size
    bool mBadChannel;              ///< Bad channel status
    std::vector<Bunch> mBunches;   ///< Bunches in channel;
  };

  /// \class Error
  /// \brief Error handling of the ALTRO Decoder
  class Error : public std::exception
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
    Error(ErrorType_t errtype, const char* message) : mErrorType(errtype), mErrorMessage(message) {}

    /// \brief Destructor
    ~Error() noexcept override = default;

    /// \brief Access to error message cnnected to the error
    /// \return Error message
    const char* what() const noexcept override { return mErrorMessage.data(); }

    /// \brief Access to the error type connected to the erro
    /// \return Error type
    const ErrorType_t getErrorType() const noexcept { return mErrorType; }

   private:
    ErrorType_t mErrorType;    ///< Code of the decoding error type
    std::string mErrorMessage; ///< Message connected to the error type
  };

  /// \brief Constructor
  /// \param reader Raw reader instance to be decoded
  AltroDecoder(RawReaderFile& reader);

  /// \brief Destructor
  ~AltroDecoder() = default;

  /// \brief Decode the ALTRO stream
  /// \throw Error if the RCUTrailer or ALTRO payload cannot be decoded
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
  /// \throw Error with type RCU_TRAILER_ERROR if the RCU trailer was not initialized
  const RCUTrailer& getRCUTrailer() const;

  /// \Get the reference to the channel container
  /// \return Reference to the channel container
  /// \throw Error with CHANNEL_ERROR if the channel container was not initialized for the current event
  const std::vector<Channel>& getChannels() const;

  /// \read RCU trailer for the current event in the raw buffer
  void readRCUTrailer();

  /// \brief Read channels for the current event in the raw buffer
  void readChannels();

 private:
  /// \brief run checks on the RCU trailer
  /// \throw Error if the RCU trailer has inconsistencies
  ///
  /// Performing various consistency checks on the RCU trailer
  /// In case of failure an exception is thrown.
  void checkRCUTrailer();

  RawReaderFile& mRawReader;         ///< underlying raw reader
  RCUTrailer mRCUTrailer;            ///< RCU trailer
  std::vector<Channel> mChannels;    ///< vector of channels in the raw stream
  bool mChannelsInitialized = false; ///< check whether the channels are initialized

  ClassDefNV(AltroDecoder, 1);
};

std::ostream& operator<<(std::ostream& stream, const AltroDecoder::RCUTrailer& trailer);

} // namespace emcal

} // namespace o2

#endif