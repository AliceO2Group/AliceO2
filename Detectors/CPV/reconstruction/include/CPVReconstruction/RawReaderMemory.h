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
#ifndef ALICEO2_CPV_RAWREADERMEMORY_H
#define ALICEO2_CPV_RAWREADERMEMORY_H

#include <gsl/span>
#include <Rtypes.h>

#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"

namespace o2
{

namespace cpv
{

enum RawErrorType_t {
  kOK,            ///< NoError
  kOK_NO_PAYLOAD, ///< No payload per ddl (not error)
  kRDH_DECODING,
  kRDH_INVALID,
  kNOT_CPV_RDH,
  kSTOPBIT_NOTFOUND,
  kPAGE_NOTFOUND,
  kOFFSET_TO_NEXT_IS_0,
  kPAYLOAD_INCOMPLETE,
  kNO_CPVHEADER,
  kNO_CPVTRAILER,
  kCPVHEADER_INVALID,
  kCPVTRAILER_INVALID,
  kSEGMENT_HEADER_ERROR,
  kROW_HEADER_ERROR,
  kEOE_HEADER_ERROR,
  kPADERROR,
  kUNKNOWN_WORD,
  kPadAddress
};

/// \class RawReaderMemory
/// \brief Reader for raw data produced by the Readout application in in-memory format
/// \ingroup CPVreconstruction
/// \author Dmitri Peresunko after Markus Fasel
/// \since Sept. 25, 2020
///
/// It reads one HBF, stores HBF orbit number in getCurrentHBFOrbit() and produces digits in AddressChargeBC format
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
  const std::vector<char>& getPayload() const { return mRawPayload; }

  /// \brief Return size of the payload
  /// \return size of the payload
  int getPayloadSize() const { return mRawPayload.size(); }

  /// \brief get the size of the file in bytes
  /// \return size of the file in byte
  int getFileSize() const noexcept { return mRawMemoryBuffer.size(); }

  /// \brief check if more pages are available in the raw file
  /// \return true if there is a next page
  bool hasNext() const { return mCurrentPosition < mRawMemoryBuffer.size(); }

  /// \return HeartBeatFrame orbit number
  uint32_t getCurrentHBFOrbit() const { return mCurrentHBFOrbit; }

 protected:
  /// \brief Initialize the raw stream
  ///
  /// Rewind stream to the first entry
  void init();

  o2::header::RDHAny decodeRawHeader(const void* headerwords);

 private:
  gsl::span<const char> mRawMemoryBuffer; ///< Memory block with multiple DMA pages
  o2::header::RDHAny mRawHeader;          ///< Raw header
  std::vector<char> mRawPayload;          ///< Raw payload (can consist of multiple pages)
  unsigned int mCurrentPosition = 0;      ///< Current page in file
  bool mRawHeaderInitialized = false;     ///< RDH for current page initialized
  bool mPayloadInitialized = false;       ///< Payload for current page initialized
  uint32_t mCurrentHBFOrbit = 0;          ///< Current orbit of HBF
  bool mStopBitWasNotFound;               ///< True if StopBit was not found but HBF orbit changed
  bool mIsJustInited = false;             ///< True if init() was just called

  ClassDefNV(RawReaderMemory, 2);
};

} // namespace cpv

} // namespace o2

#endif
