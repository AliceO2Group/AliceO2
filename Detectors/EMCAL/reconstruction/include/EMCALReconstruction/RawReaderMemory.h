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
#include "Headers/RAWDataHeader.h"

namespace o2
{

namespace emcal
{

template <class RawHeader>
class RawReaderMemory
{
 public:
  RawReaderMemory(const gsl::span<const char> rawmemory);

  /// \brief Destructor
  ~RawReaderMemory() = default;

  /// \brief set new raw memory chunk
  /// \param rawmemory New raw memory chunk
  void setRawMemory(const gsl::span<const char> rawmemory);

  /// \brief Read the next page from the stream
  /// \throw Error if the page cannot be read or header or payload cannot be deocded
  void nextPage();

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

  /// \brief access to the
  const RawBuffer& getRawBuffer() const;

  /// \brief get the size of the file in bytes
  /// \return size of the file in byte
  int getFileSize() const noexcept { return mRawMemoryBuffer.size(); }

  /// \brief get the number of pages in the file
  /// \return number of pages in the file
  int getNumberOfPages() const noexcept { return mNumData; }

  /// \brief check if more pages are available in the raw file
  /// \return true if there is a next page
  bool hasNext() const { return mCurrentPosition < mNumData; }

 protected:
  void init();

 private:
  gsl::span<const char> mRawMemoryBuffer;
  RawBuffer mRawBuffer;
  RawHeader mRawHeader;
  int mCurrentPosition = 0;           ///< Current page in file
  int mNumData = 0;                   ///< Number of pages
  bool mRawHeaderInitialized = false; ///< RDH for current page initialized
  bool mPayloadInitialized = false;   ///< Payload for current page initialized

  ClassDefNV(RawReaderMemory, 1);
};

// For template specifications
using RawReaderMemoryRDHvE = RawReaderMemory<o2::emcal::RAWDataHeader>;
using RawReaderMemoryRDHv4 = RawReaderMemory<o2::header::RAWDataHeaderV4>;

} // namespace emcal

} // namespace o2

#endif