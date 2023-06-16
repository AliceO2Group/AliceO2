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
#ifndef ALICEO2_EMCAL_COMPRESSEDTRIGGERDATA_H
#define ALICEO2_EMCAL_COMPRESSEDTRIGGERDATA_H

#include <cstdint>
#include <iosfwd>

namespace o2::emcal
{

/// \struct CompressedTRU
/// \brief Compressed reconstructed TRU information
/// \ingroup EMCALDataFormat
struct CompressedTRU {
  uint8_t mTRUIndex;        ///< TRU index
  uint8_t mTriggerTime;     ///< Trigger time of the TRU
  bool mFired;              ///< Fired status of the TRU
  uint8_t mNumberOfPatches; ///< Number of patches found for the TRU
};

/// \struct CompressedTriggerPatch
/// \brief Compressed reconstructed L0 trigger patch information
/// \ingroup EMCALDataFormat
struct CompressedTriggerPatch {
  uint8_t mTRUIndex;        ///< Index of the TRU where the trigger patch has been found
  uint8_t mPatchIndexInTRU; ///< Index of the trigger patch in the TRU
  uint8_t mTime;            ///< Reconstructed time of the trigger patch
  uint16_t mADC;            ///< ADC sum of the trigger patch
};

/// \struct CompressedL0TimeSum
/// \brief Compressed L0 timesum information
/// \ingroup EMCALDataFormat
struct CompressedL0TimeSum {
  uint16_t mIndex;   ///< Absolute ID of the FastOR
  uint16_t mTimesum; ///< ADC value of the time-sum (4-integral)
};

/// \brief Output stream operator of the CompressedTRU
/// \param stream Stream to write to
/// \param tru TRU data to be streamed
/// \return Stream after writing
std::ostream& operator<<(std::ostream& stream, const CompressedTRU& tru);

/// \brief Output stream operator of the CompressedTriggerPatch
/// \param stream Stream to write to
/// \param patch Trigger patch to be streamed
/// \return Stream after writing
std::ostream& operator<<(std::ostream& stream, const CompressedTriggerPatch& patch);

/// \brief Output stream operator of the CompressedL0TimeSum
/// \param stream Stream to write to
/// \param timesum FastOR L0 timesum to be streamed
/// \return Stream after writing
std::ostream& operator<<(std::ostream& stream, const CompressedL0TimeSum& timesum);

} // namespace o2::emcal

#endif // ALICEO2_EMCAL_COMPRESSEDTRIGGERDATA_H