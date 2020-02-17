// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_RAWBUFFER_H
#define ALICEO2_EMCAL_RAWBUFFER_H

#include <array>
#include <cstdint>
#include <iosfwd>
#include <gsl/span>

namespace o2
{
namespace emcal
{

/// \class RawBuffer
/// \brief Buffer for EMCAL raw pages
/// \ingroup EMCALreconstruction
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Aug. 9, 2019
class RawBuffer
{
 public:
  RawBuffer() = default;
  ~RawBuffer() = default;

  void reset() { mCurrentDataWord = 0; }

  /// \brief Flush the buffer
  /// Does not overwrite the word buffer but just resets the counter and iterator
  void flush();

  /// \brief Read page from stream
  /// \param in Input file stream
  /// \param payloadsize Number of char words in payload
  /// Read a whole superpage from the raw stream
  /// and convert the bitwise representation directly
  /// into 32 bit words
  void readFromStream(std::istream& in, uint32_t payloadsize);

  /// \brief Read page from raw memory buffer
  /// \param rawmemory Raw memory buffer (as char words) with size of the payload from the raw data header
  /// Converts the char word raw memory buffer of a pages into
  /// into the 32 bit word buffer
  void readFromMemoryBuffer(const gsl::span<const char> rawmemory);

  /// \brief Get the number of data words read for the superpage
  /// \return Number of data words in the superpage
  int getNDataWords() const { return mNDataWords; }

  /// \brief Get the next data word in the superpage
  /// \return next data word in the superpage
  /// \throw std::runtime_error if there exists no next data word
  uint32_t getNextDataWord();

  /// \brief Get the data word at a given index
  /// \param index index of the word in the buffer
  /// \return word at requested index
  /// \throw std::runtime_error if the index is out-of-range
  uint32_t getWord(int index) const;

  /// \brief Get all data words from the raw buffer
  /// \return Span with data words in the buffer (removing trailing null entries)
  const gsl::span<const uint32_t> getDataWords() const { return gsl::span<const uint32_t>(mDataWords.data(), mNDataWords); }

  /// \brief Check whether the next data word exists
  /// \return True if more data words exist, false otherwise
  /// Check is done starting from the current position
  /// of the iterator
  bool hasNext() const { return mCurrentDataWord < mNDataWords; }

 private:
  std::array<uint32_t, 2048> mDataWords; ///< Data words in one superpage
  int mNDataWords = 0;                   ///< Number of data words read from superpage
  int mCurrentDataWord = 0;              ///< Iterator over words in superpage
};

} // namespace emcal

} // namespace o2

#endif