// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_GEOMETRYBASE_H
#define ALICEO2_EMCAL_GEOMETRYBASE_H

#include <string>

namespace o2
{
namespace emcal
{
enum EMCALSMType {
  NOT_EXISTENT = -1,
  EMCAL_STANDARD = 0,
  EMCAL_HALF = 1,
  EMCAL_THIRD = 2,
  DCAL_STANDARD = 3,
  DCAL_EXT = 4
}; // possible SM Type

enum AcceptanceType_t { EMCAL_ACCEPTANCE = 1,
                        DCAL_ACCEPTANCE = 2,
                        NON_ACCEPTANCE = 0 };

const std::string DEFAULT_GEOMETRY = "EMCAL_COMPLETE12SMV1_DCAL_8SM";

/// \class InvalidModuleException
/// \brief Error Handling when an invalid module ID (outside the limits) is called
class InvalidModuleException : public std::exception
{
 public:
  /// \brief Constructor
  /// \param nModule Module number raising the exception
  /// \param nMax Maximum amount of modules in setup
  InvalidModuleException(Int_t nModule, Int_t nMax) : std::exception(),
                                                      mModule(nModule),
                                                      mMax(nMax),
                                                      mMessage("Invalid Module [ " + std::to_string(mModule) + "|" + std::to_string(mMax) + "]")
  {
  }

  /// \brief Destructor
  ~InvalidModuleException() noexcept final = default;

  /// \brief Get ID of the module raising the exception
  /// \return ID of the module
  int GetModuleID() const noexcept { return mModule; }

  /// \brief Get number of modules
  /// \return Number of modules
  int GetMaxNumberOfModules() const noexcept { return mMax; }

  /// \brief Access to error message
  /// \return Error message for given exception
  const char* what() const noexcept final { return mMessage.c_str(); }

 private:
  Int_t mModule;        ///< Module ID raising the exception
  Int_t mMax;           ///< Max. Number of modules
  std::string mMessage; ///< Error message
};

/// \class InvalidPositionException
/// \brief Exception handling errors due to positions not in the EMCAL area
class InvalidPositionException : public std::exception
{
 public:
  /// \brief Constructor, setting the position raising the exception
  /// \param eta Eta coordinate of the position
  /// \param phi Phi coordinate of the position
  InvalidPositionException(double eta, double phi) : std::exception(),
                                                     mEta(eta),
                                                     mPhi(phi),
                                                     mMessage("Position phi (" + std::to_string(mPhi) + "), eta(" + std::to_string(mEta) + ") not im EMCAL")
  {
  }

  /// \brief Destructor
  ~InvalidPositionException() noexcept final = default;

  /// \brief Access to eta coordinate raising the exception
  /// \return Eta coordinate of the position
  double getEta() const noexcept { return mEta; }

  /// \brief Access to phi corrdinate raising the exception
  /// \return Phi coordinate of the position
  double getPhi() const noexcept { return mPhi; }

  /// \brief Access to error message of the exception
  /// \return Error message
  const char* what() const noexcept final { return mMessage.data(); }

 private:
  double mEta = 0.;     ///< Position in eta raising the exception
  double mPhi = 0.;     ///< Position in phi raising the exception
  std::string mMessage; ///< Error message
};

/// \class InvalidCellIDException
/// \brief Exception handling non-existing cell IDs
class InvalidCellIDException : public std::exception
{
 public:
  /// \brief Constructor, setting cell ID raising the exception
  /// \param cellID Cell ID raising the exception
  InvalidCellIDException(Int_t cellID) : std::exception(),
                                         mCellID(cellID),
                                         mMessage("Cell ID " + std::to_string(mCellID) + " outside limits.")
  {
  }

  /// \brief Destructor
  ~InvalidCellIDException() noexcept final = default;

  /// \brief Access to cell ID raising the exception
  /// \return Cell ID
  Int_t getCellID() const noexcept { return mCellID; }

  /// \brief Access to error message of the exception
  /// \return Error message
  const char* what() const noexcept final { return mMessage.data(); }

 private:
  Int_t mCellID;        ///< Cell ID raising the exception
  std::string mMessage; ///< error Message
};

/// \class InvalidSupermoduleTypeException
/// \brief Exception handling improper or uninitialized supermodule types
class InvalidSupermoduleTypeException : public std::exception
{
 public:
  /// \brief constructor
  InvalidSupermoduleTypeException() = default;

  /// \brief Destructor
  ~InvalidSupermoduleTypeException() noexcept final = default;

  /// \brief Access to error message of the exception
  const char* what() const noexcept final { return "Uknown SuperModule Type !!"; }
};

/// \class SupermoduleIndexException
/// \brief Handling error due to invalid supermodule
class SupermoduleIndexException : public std::exception
{
 public:
  /// \brief Constructor, initializing the exception
  /// \param supermodule Supermodule ID raising the exception
  /// \param maxSupermodules Max. number of supermodules in the geometry setup
  SupermoduleIndexException(int supermodule, int maxSupermodules) : std::exception(),
                                                                    mSupermoduleIndex(supermodule),
                                                                    mMaxSupermodules(maxSupermodules)
  {
    mMessage = "Invalid supermodule ID " + std::to_string(mSupermoduleIndex) + ", max " + std::to_string(mMaxSupermodules);
  }

  /// \brief Destructor
  ~SupermoduleIndexException() noexcept final = default;

  /// \brief Access to supermodule index raising the exception
  /// \return Supermodule index
  int getSupermodule() const noexcept { return mSupermoduleIndex; }

  /// \brief Access to maximum number of supermodules
  /// \return Max. number of supermodules
  int getMaxSupermodule() const noexcept { return mMaxSupermodules; }

  /// \brief Access to error message of the exception
  /// \return Error message
  const char* what() const noexcept final { return mMessage.data(); }

 private:
  int mSupermoduleIndex; ///< Supermodule index raising the exception
  int mMaxSupermodules;  ///< Max. number of supermodules
  std::string mMessage;  ///< Error message
};

} // namespace emcal
} // namespace o2

#endif
