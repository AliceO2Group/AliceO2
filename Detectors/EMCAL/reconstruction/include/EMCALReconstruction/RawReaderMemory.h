// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_RAWREADERMEMORY_H
#define ALICEO2_EMCAL_RAWREADERMEMORY_H

#include <gsl/span>
#include <Rtypes.h>

#include "EMCALReconstruction/RawBuffer.h"
#include "EMCALReconstruction/RAWDataHeader.h"
#include "EMCALReconstruction/RawPayload.h"
#include "Headers/RAWDataHeader.h"

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
template <class RawHeader>
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

  /// \brief Read page with a given index
  /// \param page Index of the page to be decoded
  /// \throw RawDecodingError if the page cannot be read or header or payload cannot be deocded
  ///
  /// The reader will try to read the page with a certain index. In
  /// case the page cannot be decoded (page index outside range,
  /// decoding of header or payload failed), and excpetion is raised.
  void readPage(int page);

  /// \brief access to the raw header of the current page
  /// \return Raw header of the current page
  /// \throw RawDecodingError with HEADER_INVALID if the header was not decoded
  const RawHeader& getRawHeader() const;

  /// \brief access to the raw buffer (single DMA page)
  /// \return Raw buffer of the current page
  /// \throw Error with PAYLOAD_INCALID if payload was not decoded
  const RawBuffer& getRawBuffer() const;

  /// \brief access to the full raw payload (single or multiple DMA pages)
  /// \return Raw Payload of the data until the stop bit is received.
  const RawPayload& getPayload() const { return mRawPayload; }

  /// \brief Return size of the payload
  /// \return size of the payload
  int getPayloadSize() const { return mRawPayload.getPayloadSize(); }

  /// \brief get the size of the file in bytes
  /// \return size of the file in byte
  int getFileSize() const noexcept { return mRawMemoryBuffer.size(); }

  /// \brief get the number of pages in the file
  /// \return number of pages in the file
  int getNumberOfPages() const noexcept { return mNumData; }

  /// \brief check if more pages are available in the raw file
  /// \return true if there is a next page
  bool hasNext() const { return mCurrentPosition < mRawMemoryBuffer.size(); }

 protected:
  /// \brief Initialize the raw stream
  ///
  /// Rewind stream to the first entry
  void init();

  bool isStop(const o2::emcal::RAWDataHeader& hdr) { return true; }
  bool isStop(const o2::header::RAWDataHeaderV4& hdr) { return hdr.stop; }

 private:
  gsl::span<const char> mRawMemoryBuffer; ///< Memory block with multiple DMA pages
  RawBuffer mRawBuffer;                   ///< Raw buffer
  RawHeader mRawHeader;                   ///< Raw header
  RawPayload mRawPayload;                 ///< Raw payload (can consist of multiple pages)
  int mCurrentPosition = 0;               ///< Current page in file
  int mNumData = 0;                       ///< Number of pages
  bool mRawHeaderInitialized = false;     ///< RDH for current page initialized
  bool mPayloadInitialized = false;       ///< Payload for current page initialized

  ClassDefNV(RawReaderMemory, 1);
};

// For template specifications
using RawReaderMemoryRDHvE = RawReaderMemory<o2::emcal::RAWDataHeader>;
using RawReaderMemoryRDHv4 = RawReaderMemory<o2::header::RAWDataHeaderV4>;

} // namespace emcal

} // namespace o2

#endif