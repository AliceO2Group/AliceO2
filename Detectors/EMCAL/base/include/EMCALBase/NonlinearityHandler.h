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
#ifndef ALICEO2_EMCAL_NONLINEARITYHANDLER__H
#define ALICEO2_EMCAL_NONLINEARITYHANDLER__H

#include <exception>
#include <iosfwd>
#include <unordered_map>
#include <string>
#include <Rtypes.h>
#include "DataFormatsEMCAL/AnalysisCluster.h"

namespace o2::emcal
{

/// \class NonlinearityHandler
/// \brief Nonlinearity functions for energy correction
/// \ingroup EMCALbase
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Feb 27, 2023
///
/// Calculating a corrected cluster energy based on the raw cluster energy.
/// Several parameterisations are provided. The function is selected during
/// construction of the object. The corrected cluster energy is obtained via
/// the function getCorrectedClusterEnergy.
///
/// The correction for the shaper sturation must be applied at cell energy level.
/// Only one parameterisation for the shaper nonlinearity exists, for which the
/// parameterisation does not depend on the type of the cluster nonlinearity. The
/// function evaluateShaperCorrectionCellEnergy is static and can therefore be applied
/// without a cluster nonlinearity parameterisation.
///
/// based on nonlinearity implementation in AliEMCALRecoUtils
class NonlinearityHandler
{
 public:
  /// \class UninitException
  /// \brief Handling missing initialisation of the NonlinearityHanlder
  class UninitException : public std::exception
  {
   public:
    /// \brief Constructor
    UninitException() = default;

    /// \brief Destructor
    ~UninitException() noexcept final = default;

    /// \brief Provide error message
    /// \return Error message
    const char* what() const noexcept final
    {
      return "Nonlinearity handler not initialized";
    }
  };

  /// \enum NonlinType_t
  /// \brief Types of nonlinearity functions available
  enum class NonlinType_t {
    DATA_TESTBEAM_SHAPER,         ///< Data, testbeam nonlin for shaper correction, with energy rescaling
    DATA_TESTBEAM_SHAPER_WOSCALE, ///< Data, testbeam nonlin for shaper correction, without energy rescaling
    DATA_TESTBEAM_CORRECTED,      ///< Data, inital testbeam nonlin without shaper correction
    DATA_TESTBEAM_CORRECTED_V2,   ///< Data, inital testbeam nonlin without shaper correction, version 2
    DATA_TESTBEAM_CORRECTED_V3,   ///< Data, inital testbeam nonlin without shaper correction, version 3
    DATA_TESTBEAM_CORRECTED_V4,   ///< Data, inital testbeam nonlin without shaper correction, version 4
    MC_TESTBEAM_FINAL,            ///< MC, function corresponding to data testbeam nonlin for shaper correction, with energy rescaling
    MC_PI0,                       ///< MC, function corresponding to inital testbeam nonlin without shaper correction
    MC_PI0_V2,                    ///< MC, function corresponding to inital testbeam nonlin without shaper correction, version 2
    MC_PI0_V3,                    ///< MC, function corresponding to inital testbeam nonlin without shaper correction, version 3
    MC_PI0_V5,                    ///< MC, function corresponding to inital testbeam nonlin without shaper correction, version 5
    MC_PI0_V6,                    ///< MC, function corresponding to inital testbeam nonlin without shaper correction, version 6
    UNKNOWN                       ///< No function set
  };

  /// \brief Get type of a nonlinearity function from its name
  /// \param name Name of the nonlinearity function
  /// \return Nonlinearity function type
  static NonlinType_t getNonlinType(const std::string_view name);

  /// \brief Get name of the nonlinearity function from the function type
  /// \param nonlin Type of the nonlinearity function
  /// \return Name of the nonlinearity function
  static const char* getNonlinName(NonlinType_t nonlin);

  /// \brief Dummy constructor
  ///
  /// Non-linearity type not set, will fail when trying to evaluate
  /// for a certain energy. Use constructor with nonlinearity function
  /// specified instead. Only intended for constructing standard containers.
  NonlinearityHandler() = default;

  /// \brief Constructor, defining nonlinearity function
  /// \param nonlintype Type of t, he nonlinearity function
  ///
  /// Initializing all parameters and settings of the nonlinearity function.
  /// Nonlinearity correction at cluster level can be obtained using
  /// objects constructed by this.
  NonlinearityHandler(NonlinType_t nonlintype);

  /// \brief Destructor
  ~NonlinearityHandler() = default;

  /// \brief Set type of nonlinearity function
  /// \param nonlintype Type of nonlinearity function
  ///
  /// Updating also function parameters
  void setNonlinType(NonlinType_t nonlintype)
  {
    mNonlinearyFunction = nonlintype;
    initParams();
  }

  /// \brief Get corrected cluster energy for the selected nonlinearity parameterization
  /// \param energy Raw cluster energy
  /// \return Corrected cluster energy
  /// \throw UninitException in case the NonlinearityHandler is not configured
  double getCorrectedClusterEnergy(double energy) const;

  /// \brief Get corrected cluster energy for the selected nonlinearity parameterization
  /// \param energy Raw cluster energy
  /// \return Corrected cluster energy
  /// \throw UninitException in case the NonlinearityHandler is not configured
  double getCorrectedClusterEnergy(const AnalysisCluster& cluster) const { return getCorrectedClusterEnergy(cluster.E()); }

  /// \brief Get corrected energy at cell level for the shaper saturation at high energy
  /// \param energy Raw cell energy
  /// \param ecalibHG Finetuning of the high-gain energy scale
  static double evaluateShaperCorrectionCellEnergy(double energy, double ecalibHG = 1);

  /// \brief Print information about the nonlinearity function
  /// \param stream Stream to print the information on
  void printStream(std::ostream& stream) const;

 private:
  NonlinType_t mNonlinearyFunction = NonlinType_t::UNKNOWN; ///< Nonlinearity function
  bool mApplyScaleCorrection = false;                       ///< Scale correction
  std::array<double, 11> mNonlinearityParam;                ///< Storage for params used in the function evaluation

  /// \brief Classical model, before testbeam reanalysis, data and most MC parameterisations
  /// \param energy Raw cluster energy
  /// \return Corrected energy
  double evaluateTestbeamCorrected(double energy) const;

  /// \brief New model after testbeam reanalysis, data and MC
  /// \param energy Raw cluster energy
  /// \return Corrected energy
  double evaluateTestbeamShaper(double energy) const;

  /// \brief Model for nonlin based on MC pi0 analysis, initial version
  /// \param energy Raw cluster energy
  /// \return Corrected energy
  double evaluatePi0MC(double energy) const;

  /// \brief Model for nonlin based on MC pi0 analysis, version 2
  /// \param energy Raw cluster energy
  /// \return Corrected energy
  double evaluatePi0MCv2(double energy) const;

  /// \brief Initialise params of the nonlinearity function
  ///
  /// Initialising all params corresponding to the fits of the various
  /// nonlinearity functions.
  void initParams();

  ClassDefNV(NonlinearityHandler, 1)
};

/// \class NonlinearityFactory
/// \brief Creator and handler class of nonlinearity parameterizations
/// \ingroup EMCALbase
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Feb 27, 2023
///
/// Factory and manager of nonlinearty handlers. The class acts as singleton class.
/// NonlinearityHandlers can be constructed via the function getNonlinearity
/// either using either the symbolic form or using the name as a string representation.
/// In the second case an exception is thrown in case the request is done for a funciton
/// name which doesn't exist.
class NonlinearityFactory
{
 public:
  /// \class FunctionNotFoundExcpetion
  /// \brief Handling request of non-exisiting nonlinearity functions
  class FunctionNotFoundExcpetion : public std::exception
  {
   public:
    /// \brief Constructor
    /// \param Name of the nonlinearity function raising the exception
    FunctionNotFoundExcpetion(const std::string_view name) : mName(name), mMessage()
    {
      mMessage = "Nonlinearity funciton " + mName + " not found";
    }

    /// \brief Destructor
    ~FunctionNotFoundExcpetion() noexcept final = default;

    /// \brief Provide error message
    /// \return Error message
    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    /// \brief Getting the name of the function raising the exception
    /// \return Name of the function raising the exception
    const std::string_view getNonlinName() const noexcept { return mName.data(); }

   private:
    std::string mName;    ///< Name of the requested function
    std::string mMessage; ///< Error message
  };

  /// \class NonlinInitError
  /// \brief Handling errors of initialisation of a certain nonlinearity function
  class NonlinInitError : public std::exception
  {
   public:
    /// \brief Constructor
    NonlinInitError() {}

    /// \brief Destructor
    ~NonlinInitError() noexcept final = default;

    /// \brief Provide error message
    /// \return Error message
    const char* what() const noexcept final
    {
      return "Failed constructing nonlinearity handler";
    }
  };

  /// \brief Get instance of the nonlinearity factory
  /// \return Factory instance
  static NonlinearityFactory& getInstance()
  {
    static NonlinearityFactory currentInstance;
    return currentInstance;
  }

  /// \brief Get nonlinearity handler for the given nonlinearty type
  /// \return Nonlineary handler for the given nonlinearity function
  /// \throw NonlinInitError in case the object could not be constructed
  ///
  /// Internally caching existing nonlinearty handlers. Only constructing
  /// handler in case the handler is not yet existing.
  NonlinearityHandler& getNonlinearity(NonlinearityHandler::NonlinType_t nonlintype);

  /// \brief Get nonlinearity handler for the given nonlinearty name
  /// \return Nonlineary handler for the given nonlinearity function based on its name
  /// \throw FunctionNotFoundExcpetion in case the name of the function is unknown
  /// \throw NonlinInitError in case the object could not be constructed
  ///
  /// Internally caching existing nonlinearty handlers. Only constructing
  /// handler in case the handler is not yet existing.
  NonlinearityHandler& getNonlinearity(const std::string_view nonlintype);

 private:
  /// \brief Constructor
  ///
  /// Initialising lookup of function types for function names
  NonlinearityFactory() { initNonlinNames(); }

  /// \brief Destructor
  ~NonlinearityFactory() = default;

  /// \brief Initialise lookup table with string representation of nonlinearity functions
  void initNonlinNames();

  /// \brief Find type of nonlinearity function from a function name (string representation)
  /// \param nonlinName Name of the nonlinearity function
  /// \return Type of the nonlinearity function
  /// \throw FunctionNotFoundExcpetion in case no function type can be found for the given name
  NonlinearityHandler::NonlinType_t getNonlinType(const std::string_view nonlinName) const;

  std::unordered_map<NonlinearityHandler::NonlinType_t, NonlinearityHandler> mHandlers; ///< Map with nonlinearity handlers for given nonlinearity functions
  std::unordered_map<std::string, NonlinearityHandler::NonlinType_t> mNonlinNames;      ///< Lookup table with nonlinearity types for given nonlinearity names

  ClassDefNV(NonlinearityFactory, 1)
};

/// \brief Output streaming operator for the NonlinearityHander
/// \param in Stream to print on
/// \param handler NonlinearityHander to be displayed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& in, const NonlinearityHandler& handler);

} // namespace o2::emcal

#endif // ALICEO2_EMCAL_NONLINEARITYHANDLER__H