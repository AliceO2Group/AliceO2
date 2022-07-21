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

#ifndef ALICEO2_EMCAL_TRIGGERMAPPINGERRORS_H
#define ALICEO2_EMCAL_TRIGGERMAPPINGERRORS_H

#include <exception>
#include <string>

namespace o2
{

namespace emcal
{

/// \class TRUIndexException
/// \brief Error handling of faulty TRU indices
/// \ingroup EMCALbase
class TRUIndexException : public std::exception
{
 public:
  /// \brief Constructor
  /// \param truindex Index of the TRU
  TRUIndexException(unsigned int truindex) : std::exception(), mTRUIndex(truindex), mErrorMessage()
  {
    mErrorMessage = "Invalid TRU Index: " + std::to_string(truindex);
  }

  /// \brief Destructor
  ~TRUIndexException() noexcept final = default;

  /// \brief Get error message
  /// \return Error message of the exception
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get the index of the TRU raising the exception
  /// \return Index of the TRU
  unsigned int getTRUIndex() const noexcept { return mTRUIndex; }

 private:
  std::string mErrorMessage; ///< Buffer for the error message
  unsigned int mTRUIndex;    ///< Index of the TRU
};

/// \class FastORIndexException
/// \brief Error handling of faulty FastOR indices
/// \ingroup EMCALbase
class FastORIndexException : public std::exception
{
 public:
  /// \brief Constructor
  /// \param fastorindex Index of the FastOR
  FastORIndexException(unsigned int fastorindex) : std::exception(), mFastORIndex(fastorindex), mErrorMessage()
  {
    mErrorMessage = "Invalid FastOR Index: " + std::to_string(fastorindex);
  }

  /// \brief Destructor
  ~FastORIndexException() noexcept final = default;

  /// \brief Get error message
  /// \return Error message of the exception
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get the index of the FastOR raising the exception
  /// \return Index of the FastOR
  unsigned int getFastORIndex() const noexcept { return mFastORIndex; }

 private:
  std::string mErrorMessage; ///< Buffer for the error message
  unsigned int mFastORIndex; ///< Index of the FastOR
};

/// \class FastORPositionExceptionTRU
/// \brief Handling of invalid positions of a FastOR within a TRU
/// \ingroup EMCALbase
class FastORPositionExceptionTRU : public std::exception
{
 public:
  /// \brief Constructor
  /// \param truID Index of the TRU
  /// \param etaColumn Column of the FastOR with the faulty position in eta direction
  /// \param phiRow Row of the FastOR with the faulty position in row direction
  FastORPositionExceptionTRU(unsigned int truID, unsigned int etaColumn, unsigned int phiRow) : std::exception(),
                                                                                                mErrorMessage(),
                                                                                                mTRUID(truID),
                                                                                                mEtaColumn(etaColumn),
                                                                                                mPhiRow(phiRow)
  {
    mErrorMessage = "Invalid FastOR position in TRU " + std::to_string(truID) + ": eta = " + std::to_string(etaColumn) + ", phi = " + std::to_string(phiRow) + ")";
  }

  /// \brief Destructor
  ~FastORPositionExceptionTRU() noexcept final = default;

  /// \brief Get error message
  /// \return Error message of the exception
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get the TRU ID for which the position is invalid
  /// \return TRU ID
  unsigned int getTRUID() const noexcept { return mTRUID; }

  /// \brief Get the column in eta of the FastOR with the invalid position
  /// \return Column of the FastOR
  unsigned int getFastOREtaColumn() const noexcept { return mEtaColumn; }

  /// \brief Get the row in phi of the FastOR with the invalid position
  /// \return Row of the FastOR
  unsigned int getFastORPhiRow() const noexcept { return mPhiRow; }

 private:
  std::string mErrorMessage; ///< Buffer for the error message
  unsigned int mTRUID;       ///< ID of the TRU
  unsigned int mEtaColumn;   ///< Column of the FastOR in eta direction
  unsigned int mPhiRow;      ///< Row of the FastOR in phi direction
};

/// \class FastORPositionExceptionSupermodule
/// \brief Handling of invalid positions of a FastOR within a supermodule
/// \ingroup EMCALbase
class FastORPositionExceptionSupermodule : public std::exception
{
 public:
  /// \brief Constructor
  /// \param supermoduleID Index of the supermodule
  /// \param etaColumn Column of the FastOR with the faulty position in eta direction
  /// \param phiRow Row of the FastOR with the faulty position in row direction
  FastORPositionExceptionSupermodule(unsigned int supermoduleID, unsigned int etaColumn, unsigned int phiRow) : std::exception(),
                                                                                                                mErrorMessage(),
                                                                                                                mSupermoduleID(supermoduleID),
                                                                                                                mEtaColumn(etaColumn),
                                                                                                                mPhiRow(phiRow)
  {
    mErrorMessage = "Invalid FastOR position in supermodule " + std::to_string(supermoduleID) + ": eta = " + std::to_string(etaColumn) + ", phi = " + std::to_string(phiRow) + ")";
  }

  /// \brief Destructor
  ~FastORPositionExceptionSupermodule() noexcept final = default;

  /// \brief Get error message
  /// \return Error message of the exception
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get the supermodule ID for which the position is invalid
  /// \return Supermodule ID
  unsigned int getSupermoduleID() const noexcept { return mSupermoduleID; }

  /// \brief Get the column in eta of the FastOR with the invalid position
  /// \return Column of the FastOR
  unsigned int getFastOREtaColumn() const noexcept { return mEtaColumn; }

  /// \brief Get the row in phi of the FastOR with the invalid position
  /// \return Row of the FastOR
  unsigned int getFastORPhiRow() const noexcept { return mPhiRow; }

 private:
  std::string mErrorMessage;   ///< Buffer for error message
  unsigned int mSupermoduleID; ///< ID of the supermodule
  unsigned int mEtaColumn;     ///< Column of the FastOR in eta direction
  unsigned int mPhiRow;        ///< Row of the FastOR in phi direction
};

/// \class FastORPositionExceptionEMCAL
/// \brief Handling of invalid positions of a FastOR in the detector
/// \ingroup EMCALbase
class FastORPositionExceptionEMCAL : public std::exception
{
 public:
  /// \brief Constructor
  /// \param etaColumn Column of the FastOR with the faulty position in eta direction
  /// \param phiRow Row of the FastOR with the faulty position in row direction
  FastORPositionExceptionEMCAL(unsigned int etaColumn, unsigned int phiRow) : std::exception(), mErrorMessage(), mEtaColumn(etaColumn), mPhiRow(phiRow)
  {
    mErrorMessage = "Invalid FastOR position: eta = " + std::to_string(etaColumn) + ", phi = " + std::to_string(phiRow) + ")";
  }

  /// \brief Destructor
  ~FastORPositionExceptionEMCAL() noexcept final = default;

  /// \brief Get error message
  /// \return Error message of the exception
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get the column in eta of the FastOR with the invalid position
  /// \return Column of the FastOR
  unsigned int getFastOREtaColumn() const noexcept { return mEtaColumn; }

  /// \brief Get the row in phi of the FastOR with the invalid position
  /// \return Row of the FastOR
  unsigned int getFastORPhiRow() const noexcept { return mPhiRow; }

 private:
  std::string mErrorMessage; ///< Buffer for error message
  unsigned int mEtaColumn;   ///< Column of the FastOR in eta direction
  unsigned int mPhiRow;      ///< Row of the FastOR in phi direction
};

/// \class PHOSRegionException
/// \brief Handling of invalid PHOS regions
/// \ingroup EMCALbase
class PHOSRegionException : public std::exception
{
 public:
  /// \brief Constructor
  /// \param phosregion Index of the PHOS region
  PHOSRegionException(unsigned int phosregion) : std::exception(), mErrorMessage(), mPHOSRegion(phosregion)
  {
    mErrorMessage = "Invalid PHOS region: " + std::to_string(phosregion);
  }

  /// \brief Destructor
  ~PHOSRegionException() noexcept final = default;

  /// \brief Get error message
  /// \return Error message of the exception
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get index of the PHOS region
  /// \return PHOS region index
  unsigned int getPHOSRegion() const noexcept { return mPHOSRegion; }

 private:
  std::string mErrorMessage; ///< Buffer for error message
  unsigned int mPHOSRegion;  ///< PHOS region
};

/// \class GeometryNotSetException
/// \brief Handling cases where the geometry is required but not defined
/// \ingroup EMCALbase
class GeometryNotSetException : public std::exception
{
 public:
  /// \brief Constructor
  GeometryNotSetException() = default;

  /// \brief Destructor
  ~GeometryNotSetException() noexcept final = default;

  /// \brief Access to error message
  /// \return Error message
  const char* what() const noexcept final
  {
    return "Geometry not available";
  }
};

/// \class L0sizeInvalidException
/// \brief Handlig access of L0 index mapping with invalid patch size
/// \ingroup EMCALbase
class L0sizeInvalidException : public std::exception
{
 public:
  /// \brief Constructor
  /// \param l0size Size of the L0 patch
  L0sizeInvalidException(unsigned int l0size) : std::exception(), mErrorMessage(), mL0size(l0size)
  {
    mErrorMessage = "L0 patch size invalid: " + std::to_string(l0size);
  }

  /// \brief Destructor
  ~L0sizeInvalidException() noexcept final = default;

  /// \brief Access to error message
  /// \return Error message
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get the size of the L0 patch
  /// \return Size of the L0 patch
  unsigned int getL0size() const noexcept { return mL0size; }

 private:
  std::string mErrorMessage; ///< Buffer for error message
  unsigned int mL0size;      ///< Size of the L0 patch
};

} // namespace emcal

} // namespace o2

#endif //  ALICEO2_EMCAL_TRIGGERMAPPINGERRORS_H