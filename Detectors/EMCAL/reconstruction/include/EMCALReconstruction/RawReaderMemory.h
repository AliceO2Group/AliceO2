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
#ifndef ALICEO2_EMCAL_RAWREADERMEMORY_H
#define ALICEO2_EMCAL_RAWREADERMEMORY_H

#include <vector>
#include <gsl/span>
#include <Rtypes.h>

#include "EMCALBase/RCUTrailer.h"
#include "EMCALReconstruction/RawBuffer.h"
#include "EMCALReconstruction/RawDecodingError.h"
#include "EMCALReconstruction/RawPayload.h"
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"

namespace o2
{

namespace emcal
{

/// \class RawReaderMemory
/// \brief Reader for raw data produced by the Readout application in in-memory format
/// \ingroup EMCALreconstruction
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Nov. 14, 2019
///
///
class RawReaderMemory
{
 public:
  /// \class MinorError
  /// \brief Minor (non-crashing) raw decoding errors
  ///
  /// Minor errors share the same codes as major raw decoding errors,
  /// however are not crashing.
  class MinorError
  {
   public:
    /// \brief Dummy constructor
    MinorError() = default;

    /// \brief Main constructor
    /// \param errortype Type of the error
    /// \param feeID ID of the FEE equipment
    MinorError(RawDecodingError::ErrorType_t errortype, int feeID) : mErrorType(errortype), mFEEID(feeID) {}

    /// \brief Destructor
    ~MinorError() = default;

    /// \brief Set the type of the error
    /// \param errortype Type of the error
    void setErrorType(RawDecodingError::ErrorType_t errortype) { mErrorType = errortype; }

    /// \brief Set the ID of the FEE equipment
    /// \param feeID ID of the FEE
    void setFEEID(int feeID) { mFEEID = feeID; }

    /// \brief Get type of the error
    /// \return Type of the error
    RawDecodingError::ErrorType_t getErrorType() const { return mErrorType; }

    /// \brief Get ID of the FEE
    /// \return ID of the FEE
    int getFEEID() const { return mFEEID; }

   private:
    RawDecodingError::ErrorType_t mErrorType; ///< Type of the error
    int mFEEID;                               ///< ID of the FEC responsible for the ERROR
  };

  /// \brief Constructor
  RawReaderMemory(const gsl::span<const char> rawmemory);

  /// \brief Destructor
  ~RawReaderMemory() = default;

  /// \brief set new raw memory chunk
  /// \param rawmemory New raw memory chunk
  void setRawMemory(const gsl::span<const char> rawmemory);

  /// \brief Set range for DDLs from SRU (for RCU trailer merging)
  /// \param minDDL Min DDL of the SRU DDL range
  /// \param maxDDL Max DDL of the SRU DDL range
  void setRangeSRUDDLs(uint16_t minDDL, uint16_t maxDDL)
  {
    mMinSRUDDL = minDDL;
    mMaxSRUDDL = maxDDL;
  }

  /// \brief Read next payload from the stream
  ///
  /// Read the next pages until the stop bit is found.
  void next();

  /// \brief Read the next page from the stream (single DMA page)
  /// \param resetPayload If true the raw payload is reset
  /// \throw Error if the page cannot be read or header or payload cannot be deocded
  ///
  /// Function reading a single DMA page from the stream. It is called
  /// inside the next() function for reading payload from multiple DMA
  /// pages. As the function cannot handle payload from multiple pages
  /// it should not be called directly by the user.
  void nextPage(bool resetPayload = true);

  /// \brief access to the raw header of the current page
  /// \return Raw header of the current page
  /// \throw RawDecodingError with HEADER_INVALID if the header was not decoded
  const o2::header::RDHAny& getRawHeader() const;

  /// \brief access to the raw buffer (single DMA page)
  /// \return Raw buffer of the current page
  /// \throw Error with PAYLOAD_INCALID if payload was not decoded
  const RawBuffer& getRawBuffer() const;

  /// \brief access to the full raw payload (single or multiple DMA pages)
  /// \return Raw Payload of the data until the stop bit is received.
  const RawPayload& getPayload() const { return mRawPayload; }

  /// \brief Get minor (non-crashing) raw decoding errors
  /// \return Minor raw decoding errors
  gsl::span<const MinorError> getMinorErrors() const { return mMinorErrors; }

  /// \brief Return size of the payload
  /// \return size of the payload
  int getPayloadSize() const { return mRawPayload.getPayloadSize(); }

  /// \brief get the size of the file in bytes
  /// \return size of the file in byte
  int getFileSize() const noexcept { return mRawMemoryBuffer.size(); }

  /// \brief check if more pages are available in the raw file
  /// \return true if there is a next page
  bool hasNext() const { return mCurrentPosition < mRawMemoryBuffer.size(); }

 protected:
  /// \brief Initialize the raw stream
  ///
  /// Rewind stream to the first entry
  void init();

  /// \brief Decode raw header words
  /// \param headerwords Headerwords
  /// \return Decoded RDH
  /// \throw RawDecodingError with code HEADER_DECODING if the payload does not correspond to an expected header
  o2::header::RDHAny decodeRawHeader(const void* headerwords);

 private:
  gsl::span<const char> mRawMemoryBuffer; ///< Memory block with multiple DMA pages
  RawBuffer mRawBuffer;                   ///< Raw buffer
  o2::header::RDHAny mRawHeader;          ///< Raw header
  RawPayload mRawPayload;                 ///< Raw payload (can consist of multiple pages)
  RCUTrailer mCurrentTrailer;             ///< RCU trailer
  uint64_t mTrailerPayloadWords = 0;      ///< Payload words in common trailer
  uint16_t mMinSRUDDL = 0;                ///< Min. range of SRU DDLs (for RCU trailer merging)
  uint16_t mMaxSRUDDL = 39;               ///< Max. range of SRU DDls (for RCU trailer merging)
  int mCurrentPosition = 0;               ///< Current page in file
  int mCurrentFEE = -1;                   ///< Current FEE in the data stream
  bool mRawHeaderInitialized = false;     ///< RDH for current page initialized
  bool mPayloadInitialized = false;       ///< Payload for current page initialized
  std::vector<MinorError> mMinorErrors;   ///< Minor raw decoding errors

  ClassDefNV(RawReaderMemory, 1);
};

} // namespace emcal

} // namespace o2

#endif