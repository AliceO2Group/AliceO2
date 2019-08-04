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

#include <sstream>
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
  InvalidModuleException(Int_t nModule, Int_t nMax) : std::exception(), mModule(nModule), mMax(nMax), mMessage()
  {
    mMessage = "Invalid Module [ " + std::to_string(mModule) + "|" + std::to_string(mMax) + "]";
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

class InvalidPositionException : public std::exception
{
 public:
  InvalidPositionException(double eta, double phi) : std::exception(), mEta(eta), mPhi(phi)
  {
    std::stringstream msgbuilder;
    msgbuilder << "Position phi (" << mPhi << "), eta(" << mEta << ") not im EMCAL";
    mMessage = msgbuilder.str();
  }
  ~InvalidPositionException() noexcept final = default;

  double GetEta() const noexcept { return mEta; }
  double GetPhi() const noexcept { return mPhi; }

  const char* what() const noexcept final { return mMessage.c_str(); }

 private:
  double mEta = 0.;
  double mPhi = 0.;     ///< Position (eta, phi) raising the exception
  std::string mMessage; ///< Error message
};

class InvalidCellIDException : public std::exception
{
 public:
  InvalidCellIDException(Int_t cellID) : std::exception(), mCellID(cellID), mMessage()
  {
    std::stringstream msgbuilder;
    msgbuilder << "Cell ID " << mCellID << " outside limits.";
    mMessage = msgbuilder.str();
  }

  ~InvalidCellIDException() noexcept final = default;
  Int_t GetCellID() const noexcept { return mCellID; }
  const char* what() const noexcept final { return mMessage.c_str(); }

 private:
  Int_t mCellID;        ///< Cell ID raising the exception
  std::string mMessage; ///< error Message
};

class InvalidSupermoduleTypeException : public std::exception
{
 public:
  InvalidSupermoduleTypeException() = default;
  ~InvalidSupermoduleTypeException() noexcept final = default;
  const char* what() const noexcept final { return "Uknown SuperModule Type !!"; }
};
} // namespace emcal
} // namespace o2

#endif
