// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file DecoderError.h
 * C++ definition of a decoder error
 * @author  Andrea Ferrero, CEE-Saclay
 */

#ifndef ALICEO2_MCH_BASE_DECODERERROR_H_
#define ALICEO2_MCH_BASE_DECODERERROR_H_

#include "Rtypes.h"

namespace o2
{
namespace mch
{

// \class DecoderError
/// \brief MCH decoder error implementation
class DecoderError
{
 public:
  DecoderError() = default;

  DecoderError(int solarid, int dsid, int chip, uint32_t error) : mSolarID(solarid), mChipID(dsid * 2 + chip), mError(error) {}
  ~DecoderError() = default;

  uint16_t getSolarID() const { return mSolarID; }
  uint8_t getDsID() const { return mChipID / 2; }
  uint8_t getChip() const { return mChipID % 2; }

  uint32_t getError() const { return mError; }

 private:
  uint16_t mSolarID;
  uint8_t mChipID;
  uint32_t mError;

  ClassDefNV(DecoderError, 1);
}; //class DecoderError

} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_BASE_DECODERERROR_H_
