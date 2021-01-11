// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_CPV_RAWREADERMEMORY_H
#define ALICEO2_CPV_RAWREADERMEMORY_H

#include <gsl/span>
#include <Rtypes.h>

#include "CPVBase/RCUTrailer.h"
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"

namespace o2
{

namespace cpv
{

enum RawErrorType_t {
  kOK,         ///< NoError
  kNO_PAYLOAD, ///< No payload per ddl
  kHEADER_DECODING,
  kPAGE_NOTFOUND,
  kPAYLOAD_DECODING,
  kHEADER_INVALID,
  kRCU_TRAILER_ERROR,      ///< RCU trailer cannot be decoded or invalid
  kRCU_VERSION_ERROR,      ///< RCU trailer version not matching with the version in the raw header
  kRCU_TRAILER_SIZE_ERROR, ///< RCU trailer size length
  kSEGMENT_HEADER_ERROR,
  kROW_HEADER_ERROR,
  kEOE_HEADER_ERROR,
  kPADERROR,
  kPadAddress
};

/// \class RawReaderMemory
/// \brief Reader for raw data produced by the Readout application in in-memory format
/// \ingroup CPVreconstruction
/// \author Dmitri Peresunko after Markus Fasel
/// \since Sept. 25, 2020
///
///
class RawReaderMemory
{
 public:
  /// \brief Constructor
  RawReaderMemory(const gsl::span<const char> rawmemory);

  /// \brief Destructor
  ~RawReaderMemory() = default;

  /// \brief set new raw memory chunk
  /// \param rawmemory New raw memory chunk
  void setRawMemory(const gsl::span<const char> rawmemory);

  /// \brief Read next payload from the stream
  ///
  /// Read the next pages until the stop bit is found.
  RawErrorType_t next();

  /// \brief Read the next page from the stream (single DMA page)
  /// \param resetPayload If true the raw payload is reset
  /// \throw Error if the page cannot be read or header or payload cannot be deocded
  ///
  /// Function reading a single DMA page from the stream. It is called
  /// inside the next() function for reading payload from multiple DMA
  /// pages. As the function cannot handle payload from multiple pages
  /// it should not be called directly by the user.
  RawErrorType_t nextPage();

  /// \brief access to the raw header of the current page
  /// \return Raw header of the current page
  const o2::header::RDHAny& getRawHeader() const { return mRawHeader; }

  /// \brief access to the full raw payload (single or multiple DMA pages)
  /// \return Raw Payload of the data until the stop bit is received.
  const std::vector<uint32_t>& getPayload() const { return mRawPayload; }

  /// \brief Return size of the payload
  /// \return size of the payload
  int getPayloadSize() const { return mRawPayload.size(); }

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

  o2::header::RDHAny decodeRawHeader(const void* headerwords);

 private:
  gsl::span<const char> mRawMemoryBuffer; ///< Memory block with multiple DMA pages
  o2::header::RDHAny mRawHeader;          ///< Raw header
  std::vector<uint32_t> mRawPayload;      ///< Raw payload (can consist of multiple pages)
  RCUTrailer mCurrentTrailer;             ///< RCU trailer
  uint64_t mTrailerPayloadWords = 0;      ///< Payload words in common trailer
  int mCurrentPosition = 0;               ///< Current page in file
  bool mRawHeaderInitialized = false;     ///< RDH for current page initialized
  bool mPayloadInitialized = false;       ///< Payload for current page initialized

  ClassDefNV(RawReaderMemory, 1);
};

} // namespace cpv

} // namespace o2

#endif
