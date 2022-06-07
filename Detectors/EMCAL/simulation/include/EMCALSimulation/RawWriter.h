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

#ifndef ALICEO2_EMCAL_RAWWRITER_H
#define ALICEO2_EMCAL_RAWWRITER_H

#include <gsl/span>

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include <vector>

#include "Rtypes.h"

#include "DetectorsRaw/RawFileWriter.h"
#include "EMCALBase/Mapper.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALReconstruction/AltroHelper.h"

namespace o2
{

namespace emcal
{

class Geometry;

/// \union ChannelHeader
/// \brief Bitfield encoding channel headers
union ChannelHeader {
  uint32_t mDataWord; ///< Full data word representation
  struct {
    uint32_t mHardwareAddress : 16; ///< Bits  0 - 15: Hardware address
    uint32_t mPayloadSize : 10;     ///< Bits 16 - 25: Payload size
    uint32_t mZero1 : 3;            ///< Bits 26 - 28: zeroed
    uint32_t mBadChannel : 1;       ///< Bit  29: Bad channel status
    uint32_t mHeaderBits : 2;       ///< Bits 30 - 31: channel header bits (1)
  };
};

/// \union CaloBunchWord
/// \brief Encoding of ALTRO words (32 bit consisting of 3 10-bit words)
union CaloBunchWord {
  uint32_t mDataWord; ///< Full data word representation
  struct {
    uint32_t mWord2 : 10; ///< Bits  0 - 9  : Word 2
    uint32_t mWord1 : 10; ///< Bits 10 - 19 : Word 1
    uint32_t mWord0 : 10; ///< Bits 20 - 29 : Word 0
    uint32_t mZero : 2;   ///< Bits 30 - 31 : zeroed
  };
};

/// \class RawWriter
/// \brief Raw data creator for EMCAL raw data based on EMCAL digits
/// \ingroup EMCALsimulation
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \author Hadi Hassan, Oak Ridge National Laboratory
/// \since Jan 24, 2020
class RawWriter
{
 public:
  /// \enum FileFor_t
  /// \brief Definition of the granularity of the raw files
  enum class FileFor_t {
    kFullDet, ///< Full detector (EMCAL + DCAL)
    kSubDet,  ///< Subdetector (EMCAL/DCAL separate)
    kCRORC,   ///< C-RORC card
    kLink     ///< Per link
  };

  /// \brief Dummy constructor
  RawWriter() = default;

  /// \brief Constructor, defining output location
  /// \param outputdir Output directiory
  ///
  /// Initializing the output directory. The output files
  /// can be found in the output directory with the granularity
  /// defined via setFileFor
  RawWriter(const char* outputdir) { setOutputLocation(outputdir); }

  /// \brief Destructor
  ~RawWriter() = default;

  /// \brief Get access to underlying RawFileWriter
  /// \return RawFileWriter
  o2::raw::RawFileWriter& getWriter() const { return *mRawWriter; }

  void setOutputLocation(const char* outputdir) { mOutputLocation = outputdir; }
  void setDigits(gsl::span<o2::emcal::Digit> digits) { mDigits = digits; }

  /// \brief Set the granularity of the output file
  /// \param filefor Output granularity
  ///
  /// Output files can be created for
  /// - Whole EMCAL
  /// - Subdetector (EMCAL or DCAL)
  /// - Link
  void setFileFor(FileFor_t filefor) { mFileFor = filefor; }

  /// \brief Set the number of ADC samples in the readout window
  /// \param nsapmles Number of time samples
  void setNumberOfADCSamples(int nsamples) { mNADCSamples = nsamples; }

  /// \brief Set min. ADC samples expected in a calo bunch
  /// \param nsamples Minimum number of ADC samples
  void setMinADCSamplesBunch(int nsamples) { mMinADCBunch = nsamples; }

  /// \brief Set pedestal threshold used to accept ADC values when creating the bunches
  /// \param pedestal Pedestal value
  void setPedestal(int pedestal) { mPedestal = pedestal; }

  /// \brief Set the geometry parameters
  /// \param geo EMCAL geometry
  void setGeometry(o2::emcal::Geometry* geo) { mGeometry = geo; }

  void init();

  /// \brief Converting digits from a full timeframe to raw pages
  /// \param digits Vector of digits belonging to the same timeframe
  /// \param triggers trigger records with ranges in digits container of data for the various events in the timeframe
  ///
  /// Processing all events from within a timeframe. See processTrigger for more information
  /// about the digit to raw converion of a single event.
  void digitsToRaw(gsl::span<o2::emcal::Digit> digits, gsl::span<o2::emcal::TriggerRecord> triggers);

  /// \brief Processing digits to raw conversion for the digits from the current event
  /// \param trg Trigger record providing collision information and data range for the current event
  ///
  /// Digits are first sorted according to SRUs and within their channels and time samples.
  /// For each SRU and channel within the SRU calo bunches are created. In case at least one
  /// valid calo bunch is found channels are created, and they data words are organized in a
  /// raw stream, which is closed by the RCU trailer of the given stream. The content of the
  /// stream is then passed to the RawFileWriter for page splitting and output streaming.
  bool processTrigger(const o2::emcal::TriggerRecord& trg);

  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const;

 protected:
  void createPayload(o2::emcal::ChannelData channel, o2::emcal::ChannelType_t chanType, int ddlID, std::vector<char>& payload, bool& saturatedBunch);

  /// \brief Parse digits vector in channel and create ALTRO bunches
  /// \param channelDigits Vector with digits in the channel for the current event
  ///
  /// Channels are parsed in a time-reversed order. Bunches are selected for ranges of
  /// digits where the ADC value is consecutively above the pedestal. Only bunches having
  /// a min. amount of ADC samples are selected.
  std::vector<AltroBunch> findBunches(const std::vector<o2::emcal::Digit*>& channelDigits, ChannelType_t channelType);

  /// \brief Create channel header
  /// \param hardwareAddress Hardware address
  /// \param payloadSize Size of the payload of the channel in 10-bit ALTRO words
  /// \param isBadChannel If true the channel is a bad channel at hardware level
  ChannelHeader createChannelHeader(int hardwareAddress, int payloadSize, bool isBadChannel);

  /// \brief Creating RCU trailer
  /// \param payloadsize Size of the payload as 32bit word
  /// \param timesampe Length of the time sample (for L1 phase calculation)
  /// \param triggertime Time of the trigger (for L1 phase calculation)
  /// \param feeID Link ID
  ///
  /// Creating RCU trailer. Also setting the values of the ALTRO Config registers based
  /// on the settings in the raw writer. The RCU trailer is then encoded and converted to
  /// 8-bit words.
  std::vector<char> createRCUTrailer(int payloadsize, double timesample, uint64_t triggertime, int feeID);

  /// \brief Encoding words of the ALTRO bunch into 32-bit words
  /// \param data ALTRO bunch information
  /// \return Encoded ALTRO words
  ///
  /// Converting ALTRO bunch into ALTRO words. For the ALTRO bunch the following structure is
  /// expected:
  /// - bunch size including bunch header size (2)
  /// - start time
  /// - ADC samples
  /// The input data is converted to 10 but ALTRO words and put on the stream.
  std::vector<int> encodeBunchData(const std::vector<int>& data);

 private:
  int mNADCSamples = 15;                                      ///< Number of time samples
  int mPedestal = 1;                                          ///< Pedestal
  int mMinADCBunch = 3;                                       ///< Min. number of ADC samples in ALTRO bunch
  FileFor_t mFileFor = FileFor_t::kFullDet;                   ///< Granularity of the output files
  o2::emcal::Geometry* mGeometry = nullptr;                   ///< EMCAL geometry
  std::string mOutputLocation;                                ///< Rawfile name
  std::unique_ptr<o2::emcal::MappingHandler> mMappingHandler; ///< Mapping handler
  gsl::span<o2::emcal::Digit> mDigits;                        ///< Digits input vector - must be in digitized format including the time response
  std::vector<SRUDigitContainer> mSRUdata;                    ///< Internal helper of digits assigned to SRUs
  std::unique_ptr<o2::raw::RawFileWriter> mRawWriter;         ///< Raw writer

  ClassDefNV(RawWriter, 1);
};

} // namespace emcal

} // namespace o2

#endif