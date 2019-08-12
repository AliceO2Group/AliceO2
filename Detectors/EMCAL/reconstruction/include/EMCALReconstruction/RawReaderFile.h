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
#include <exception>
#include <fstream>
#include <string>

#include "Rtypes.h"
#include "RStringView.h"

#include "EMCALReconstruction/RAWDataHeader.h"
#include "EMCALReconstruction/RawBuffer.h"

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
class RawReaderFile
{
 public:
  /// \class Error
  /// \brief Error handling of the raw reader
  ///
  /// The following error types are defined:
  /// - Page not found
  /// - Raw header decoding error
  /// - Payload decoding error
  class Error : public std::exception
  {
   public:
    /// \enum ErrorType_t
    /// \brief Codes for different error types
    enum class ErrorType_t {
      PAGE_NOTFOUND,    ///< Page was not found (page index outside range)
      HEADER_DECODING,  ///< Header cannot be decoded (format incorrect)
      PAYLOAD_DECODING, ///< Payload cannot be decoded (format incorrect)
      HEADER_INVALID,   ///< Header in memory not belonging to requested superpage
      PAYLOAD_INVALID,  ///< Payload in memory not belonging to requested superpage
    };

    /// \brief Constructor
    /// \param errtype Identifier code of the error type
    ///
    /// Constructing the error with error code. To be called when the
    /// exception is thrown.
    Error(ErrorType_t errtype) : mErrorType(errtype)
    {
    }

    /// \brief destructor
    ~Error() noexcept override = default;

    /// \brief Providing error message of the exception
    /// \return Error message of the exception
    const char* what() const noexcept override
    {
      switch (mErrorType) {
        case ErrorType_t::PAGE_NOTFOUND:
          return "Page with requested index not found";
        case ErrorType_t::HEADER_DECODING:
          return "RDH of page cannot be decoded";
        case ErrorType_t::PAYLOAD_DECODING:
          return "Payload of page cannot be decoded";
        case ErrorType_t::HEADER_INVALID:
          return "Access to header not belonging to requested superpage";
        case ErrorType_t::PAYLOAD_INVALID:
          return "Access to payload not belonging to requested superpage";
      };
      return "Undefined error";
    }

    /// \brief Get the type identifier of the error handled with this exception
    /// \return Error code of the exception
    ErrorType_t getErrorType() const { return mErrorType; }

   private:
    ErrorType_t mErrorType; ///< Type of the error
  };

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
  /// \throw Error if the page cannot be read or header or payload cannot be deocded
  ///
  /// The reader will try to read the page with a certain index. In
  /// case the page cannot be decoded (page index outside range,
  /// decoding of header or payload failed), and excpetion is raised.
  void readPage(int page);

  /// \brief access to the raw header of the current page
  /// \return Raw header of the current page
  /// \throw Error with HEADER_INVALID if the header was not decoded
  const RAWDataHeader& getRawHeader() const;

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
  /// \throw Error with HEADER_DECODING in case the header decoding failed
  ///
  /// Decoding the raw header. Function assumes that the pointer
  /// is at the beginning of the raw header
  void readHeader();

  /// \brief Decode the payload
  /// \throw Error with PAYLOAD_DECODING in case the payload decoding failed
  ///
  /// Decoding the payload. The function assumes that the pointer is at
  /// the beginning of the payload of the page. Needs the raw header of the
  /// page to be decoded before in order to determine size of the payload
  /// and offset.
  void readPayload();

 private:
  std::string mInputFileName;         ///< Name of the input file
  std::ifstream mDataFile;            ///< Stream of the inputfile
  RAWDataHeader mRawHeader;           ///< Raw header
  RawBuffer mRawBuffer;               ///< Raw bufffer
  int mCurrentPosition = 0;           ///< Current page in file
  int mFileSize = 0;                  ///< Size of the file in bytes
  int mNumData = 0;                   ///< Number of pages
  bool mRawHeaderInitialized = false; ///< RDH for current page initialized
  bool mPayloadInitialized = false;   ///< Payload for current page initialized

  ClassDefNV(RawReaderFile, 1);
};

} // namespace emcal

} // namespace o2
#endif
