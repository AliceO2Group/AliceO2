// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_RAWPAYLOAD_H
#define ALICEO2_PHOS_RAWPAYLOAD_H

#include <cstdint>
#include <vector>
#include <gsl/span>
#include "Rtypes.h"

namespace o2
{

namespace phos
{

/// \class RawPayload
/// \brief Class for raw payload excluding raw data headers from one or multiple DMA pages
/// \ingroup PHOSreconstruction
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Nov 14, 2019
///
/// Container of 32-bit words in the current payload which can come from a single DMA page
/// or, in case the payload had to be split into multiple pages due to exceeding of the
/// page size, from multiple DMA pages. A page counter provides the amount of DMA pages
/// contributing to the current payload.
class RawPayload
{
 public:
  /// \brief Constructor
  RawPayload() = default;

  /// \brief Constructor
  /// \param payloadwords Payload words of one or multiple pages
  /// \param numpages Number of DMA pages contributing to the payload
  RawPayload(const gsl::span<const uint32_t> payloadwords, int numpages);

  /// \brief Destructor
  ~RawPayload() = default;

  /// \brief Set the number of pages contributing to the current payload
  /// \param numpages Number of DMA pages contributing to the payload
  void setNumberOfPages(int numpages) { mNumberOfPages = numpages; }

  /// \brief Append many words to the current payload (usually of a given DMA page)
  /// \param payloadwords Payload words to be appened to the current payload
  void appendPayloadWords(const gsl::span<const uint32_t> payloadwords);

  /// \brief Append single payload word to the current payload
  /// \param payloadword Payload word to be appended to the current payload
  void appendPayloadWord(uint32_t payloadword) { mPayloadWords.emplace_back(payloadword); };

  /// \brief Increase the page counter of the current payload
  void increasePageCount() { mNumberOfPages++; }

  /// \brief Get the payload words (as 32 bit words) contributing to the current payload
  /// \return Words of the current payload
  const std::vector<uint32_t>& getPayloadWords() const { return mPayloadWords; }

  /// \brief Get the number of pages contributing to the payload
  /// \return Number of pages
  int getNumberOfPages() const { return mNumberOfPages; }

  /// \brief Resetting payload words and page counter
  void reset();

  /// \brief Get the size of the payload
  /// \return Size of the payload
  int getPayloadSize() const { return mPayloadWords.size(); }

 private:
  std::vector<uint32_t> mPayloadWords; ///< Payload words (excluding raw header)
  int mNumberOfPages;                  ///< Number of DMA pages

  ClassDefNV(RawPayload, 1);
};

} // namespace phos

} // namespace o2
#endif
