// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef __O2_EMCAL_RAWREADERFILE_H__
#define __O2_EMCAL_RAWREADERFILE_H__

#include <array>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <string>

#include "Rtypes.h"
#include "RStringView.h"

#include "Headers/RAWDataHeader.h"
#include "EMCALReconstruction/RawBuffer.h"
#include "EMCALReconstruction/RAWDataHeader.h"

namespace o2
{

namespace emcal
{

/// \class RawReaderFile
/// \brief Reader for raw data produced by the ReadoutCard from a binary file
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Aug. 12, 2019
///
///
template <class RawHeader>
class RawReaderFile
{
 public:
  /// \brief Constructor
  ///
  /// Opening the raw file and determining its size and the number
  /// of pages.
  RawReaderFile(const std::string_view filename);

  /// \brief Destructor
  ///
  /// Closing the raw file
  ~RawReaderFile();

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
  /// \throw Error with HEADER_INVALID if the header was not decoded
  const RawHeader& getRawHeader() const;

  /// \brief access to the
  const RawBuffer& getRawBuffer() const;

  /// \brief get the size of the file in bytes
  /// \return size of the file in byte
  int getFileSize() const noexcept { return mFileSize; }

  /// \brief get the number of pages in the file
  /// \return number of pages in the file
  int getNumberOfPages() const noexcept { return mNumData; }

  /// \brief check if more pages are available in the raw file
  /// \return true if there is a next page
  bool hasNext() const { return mCurrentPosition < mNumData; }

  static void readFile(const std::string_view filename);

 protected:
  /// \bried Init the raw reader
  ///
  /// Opening the raw file and determining the number of superpages
  void init();

  /// \brief Decode the Raw Data Header
  /// \throw RawDecodingError with HEADER_DECODING in case the header decoding failed
  ///
  /// Decoding the raw header. Function assumes that the pointer
  /// is at the beginning of the raw header
  void readHeader();

  /// \brief Decode the payload
  /// \throw RawDecodingError with PAYLOAD_DECODING in case the payload decoding failed
  ///
  /// Decoding the payload. The function assumes that the pointer is at
  /// the beginning of the payload of the page. Needs the raw header of the
  /// page to be decoded before in order to determine size of the payload
  /// and offset.
  void readPayload();

 private:
  std::string mInputFileName;         ///< Name of the input file
  std::ifstream mDataFile;            ///< Stream of the inputfile
  RawHeader mRawHeader;               ///< Raw header
  RawBuffer mRawBuffer;               ///< Raw bufffer
  int mCurrentPosition = 0;           ///< Current page in file
  int mFileSize = 0;                  ///< Size of the file in bytes
  int mNumData = 0;                   ///< Number of pages
  bool mRawHeaderInitialized = false; ///< RDH for current page initialized
  bool mPayloadInitialized = false;   ///< Payload for current page initialized

  ClassDefNV(RawReaderFile, 1);
};

// template specifications
using RawReaderFileRDHvE = RawReaderFile<o2::emcal::RAWDataHeader>;
using RawReaderFileRDHv4 = RawReaderFile<o2::header::RAWDataHeaderV4>;

} // namespace emcal

} // namespace o2
#endif
