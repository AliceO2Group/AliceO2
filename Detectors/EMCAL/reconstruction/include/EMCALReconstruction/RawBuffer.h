// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef __O2_EMCAL_RAWBUFFER_H__
#define __O2_EMCAL_RAWBUFFER_H__

#include <array>
#include <cstdint>
#include <iosfwd>

namespace o2
{
namespace emcal
{

/// \class RawBuffer
/// \brief Buffer for
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

  /// \brief Read superpage from stream
  /// Read a whole superpage from the raw stream
  /// and convert the bitwise representation directly
  /// into 32 bit words
  void readFromStream(std::istream& in, uint32_t payloadsize);

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