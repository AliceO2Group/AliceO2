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
#ifndef ALICEO2_EMCAL_TRUDataHandler_H
#define ALICEO2_EMCAL_TRUDataHandler_H

#include <array>
#include <bitset>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <string>

#include "Rtypes.h"

#include "EMCALBase/TriggerMappingV2.h"

namespace o2::emcal
{

/// \class TRUDataHandler
/// \brief Helper class to handle decoded TRU data during the reconstruction
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \ingroup EMCALreconstruction
/// \since April 19, 2024
///
/// The decoded TRU data contains the following information
/// - Index of the TRU
/// - Trigger time of the TRU
/// - Fired or not
/// - Index of fired patches with time the patch has fired
/// The information is decoded in columns 96 to 105 of the FakeALTRO data. The
/// class does not handle FastOR timesums (colums 0-96), they are handled by a
/// separate class FastORTimeSeries.
class TRUDataHandler
{
 public:
  /// \class PatchIndexException
  /// \brief Handler of errors related to invalid trigger patch IDs
  /// \ingroup EMCALreconstruction
  class PatchIndexException final : public std::exception
  {
   public:
    /// \brief Constructor
    /// \param index Patch index raising the exception
    PatchIndexException(int8_t index);

    /// \brief  Destructor
    ~PatchIndexException() noexcept final = default;

    /// \brief Get patch index raising the exception
    /// \return Patch index
    int8_t getIndex() const { return mIndex; }

    /// \brief Access Error message
    /// \return Error message
    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    /// \brief Print error on output stream
    /// \param stream Stream to be printed to
    void printStream(std::ostream& stream) const;

   private:
    int8_t mIndex = -1;   ///< Patch index rainsing the exception
    std::string mMessage; ///< Buffer for error message
  };

  /// \brief Constructor
  TRUDataHandler();

  /// \brief Destructor
  ~TRUDataHandler() = default;

  /// \brief Reset handler
  void reset();

  /// \brief Set reconstructed trigger patch
  /// \param index Index of the trigger patch in the TRU
  /// \param time Decoded time of the patch
  /// \throw PatchIndexException in case the patch index is invalid (>= 77)
  void setPatch(unsigned int index, unsigned int time)
  {
    checkPatchIndex(index);
    mPatchTimes[index] = time;
  }

  /// \brief Mark TRU as fired (containing at least one patch above threshold)
  /// \param fired
  void setFired(bool fired) { mL0Fired = fired; }

  /// \brief Set the L0 time of the TRU
  /// \param l0time L0 time of the TRU
  void setL0time(int l0time) { mL0Time = l0time; }

  /// \brief Set the index of the TRU (in global STU indexing scheme)
  /// \param index Index of the TRU
  void setTRUIndex(int index) { mTRUIndex = index; }

  /// \brief Check whether the TRU was fired (at least one patch above threshold)
  /// \return True if the TRU was fired, false otherwise
  bool isFired() const { return mL0Fired; }

  int8_t getL0time() const { return mL0Time; }

  /// \brief Check whehther the patch at the given index has fired
  /// \param index Index of the patch
  /// \return True if the patch has fired, false otherwise
  /// \throw PatchIndexException in case the patch index is invalid (>= 77)
  bool hasPatch(unsigned int index) const
  {
    checkPatchIndex(index);
    return mPatchTimes[index] < UCHAR_MAX;
  }

  /// \brief Get the trigger time of the trigger patch at a given index
  /// \param index Index of the trigger patch
  /// \return Reconstructed patch time (UCHAR_MAX in case the patch has not fired)
  /// \throw PatchIndexException in case the patch index is invalid (>= 77)
  uint8_t getPatchTime(unsigned int index) const
  {
    checkPatchIndex(index);
    return mPatchTimes[index];
  }

  /// \brief Check whether the TRU has any patch fired
  /// \return True if at least one fired patch was found, false otherwise
  bool hasAnyPatch() const
  {
    for (int ipatch = 0; ipatch < mPatchTimes.size(); ipatch++) {
      if (hasPatch(ipatch)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Get the index of the TRU in global (STU) index schemes
  /// \return Index of the TRU
  int getTRUIndex() const { return mTRUIndex; }

  /// \brief Print TRU information to an output stream
  /// \param stream Stream to print on
  void printStream(std::ostream& stream) const;

 private:
  /// \brief Check whether the patch index is valid
  /// \throw PatchIndexException in case the patch index is invalid (>= 77)
  void checkPatchIndex(unsigned int index) const
  {
    if (index >= mPatchTimes.size()) {
      throw PatchIndexException(index);
    }
  }

  std::array<uint8_t, o2::emcal::TriggerMappingV2::PATCHESINTRU> mPatchTimes; ///< Patch times: In case the patch time is smaller than UCHAR_MAX then the patch has fired
  bool mL0Fired = false;                                                      ///< TRU has fired
  int8_t mL0Time = -1;                                                        ///< L0 time of the TRU
  int8_t mTRUIndex = -1;                                                      ///< Index of the TRU
  ClassDefNV(TRUDataHandler, 1);
};

/// \brief Output stream operator for the TRU data handler
/// \param stream Stream to print on
/// \param data TRU data to be streamed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const TRUDataHandler& data);

/// \brief Output stream operator of the PatchIndexException
/// \param stream Stream to print on
/// \param error Error to be streamed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const TRUDataHandler::PatchIndexException& error);

} // namespace o2::emcal
#endif
