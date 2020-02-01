// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_DMAOUTPUTSTREAM_H
#define ALICEO2_EMCAL_DMAOUTPUTSTREAM_H

#include <exception>
#include <fstream>
#include <string>

#include <gsl/span>

#include "Rtypes.h"
#include "RStringView.h"

#include "Headers/RAWDataHeader.h"

namespace o2
{

namespace emcal
{

/// \class DMAOutputStream
/// \brief Output stream of a payload to DMA raw files
/// \ingroup EMCALsimulation
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Nov 6, 2019
///
/// Stream of output payload with variable size to DMA pages in
/// a binary raw file. The output payload can be larger than the
/// size of a DMA page (default: 8 kB) the output is split into
/// multiple pages. Page counter, memory size, page size and
/// stop bit are handled internally and are overwritten in the
/// raw data header provided. All other header information must be
/// handled externally.
class DMAOutputStream
{
 public:
  using RawHeader = o2::header::RAWDataHeaderV4;

  class OutputFileException : public std::exception
  {
   public:
    OutputFileException() = default;
    OutputFileException(const std::string_view filename) : std::exception(), mFilePath(filename), mMessage("Path \"" + mFilePath + "\" invalid") {}
    ~OutputFileException() noexcept final = default;

    const char* what() const noexcept final
    {
      return mMessage.data();
    }

   private:
    std::string mFilePath = "";
    std::string mMessage = "";
  };

  /// \brief Constructor
  DMAOutputStream() = default;

  /// \brief Constructor
  /// \param filename Name of the output file
  DMAOutputStream(const char* filename);

  /// \brief Destructor
  ///
  /// Closing file I/O
  ~DMAOutputStream();

  /// \brief Open the output stream
  /// \throw OutputFileException
  ///
  /// Opening output file I/O
  void open();

  /// \brief Set the name of the output file
  /// \param filename Name of the output file
  void setOutputFilename(const char* filename) { mFilename = filename; }

  /// \brief Write output payload to the output stream
  /// \param header Raw data header
  /// \param buffer Raw data payload
  ///
  /// Converting output payload to DMA papges. If the payload is larger than
  /// the pagesize - header size the payload is automatically split to multiple
  /// pages. Page counter, memory size, page size and stop bit of the header are
  /// handled internally and are overwritten from the header provided. All other
  /// header information must be provided externally.
  void writeData(RawHeader header, gsl::span<char> buffer);

 protected:
  /// \brief Write DMA page
  /// \param header Raw data header for page
  /// \param payload Page payload (includes size of teh payload)
  /// \param pagesize Size of the DMA page (not size of the payload)
  ///
  /// Expects that the size of the payload is smaller than the size ot the DMA
  /// page. Parameter pagesize supporting variable page size.
  void writeDMAPage(const RawHeader& header, gsl::span<char> payload, int pagesize);

 private:
  std::string mFilename = ""; ///< Name of the output file
  std::ofstream mOutputFile;  ///< Handler for output raw file
  bool mInitialized = false;  ///< Switch for whether the output stream is initialized

  ClassDefNV(DMAOutputStream, 1);
};
} // namespace emcal
} // namespace o2

#endif