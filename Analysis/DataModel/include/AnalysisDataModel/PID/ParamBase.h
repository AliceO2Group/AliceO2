// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   ParamBase.h
/// \author Nicolo' Jacazio
///
/// \brief Set of utilities to handle the parametrization of the PID response for each detector
/// These are the basic storage elements to be kept in the CCDB
///

#ifndef O2_FRAMEWORK_PARAMBASE_H_
#define O2_FRAMEWORK_PARAMBASE_H_

// ROOT includes
#include "TNamed.h"

namespace o2::pid
{
/// Variable to use for the pid input/output i.e. float, double et cetera
using pidvar_t = float;

/// \brief Class to handle the parameters of a given detector response
class Parameters : public TNamed
{
 public:
  /// Default constructor
  Parameters() = default;

  /// Parametric constructor
  /// \param size Number of parameters in the container
  Parameters(unsigned int size) : mPar(std::vector<pidvar_t>(size)){};

  /// Parametric constructor
  /// \param size Number of parameters in the container
  Parameters(const TString name, unsigned int size) : TNamed(name, name), mPar(std::vector<pidvar_t>(size)){};

  /// Parametric constructor
  /// \param params Parameters to initialize the container
  Parameters(const TString name, const std::vector<pidvar_t> params) : TNamed(name, name), mPar{} { SetParameters(params); };

  /// Default destructor
  ~Parameters() override = default;

  /// Setter for the parameter at position iparam
  /// \param iparam index in the array of the parameters
  /// \param value value of the parameter at position iparam
  void SetParameter(const unsigned int iparam, const pidvar_t value) { mPar[iparam] = value; }

  /// Setter for the parameter, using an array
  /// \param param array with parameters
  void SetParameters(const pidvar_t* params) { std::copy(params, params + mPar.size(), mPar.begin()); }

  /// Setter for the parameter, using a vector
  /// \param params vector with parameters
  void SetParameters(const std::vector<pidvar_t> params);

  /// Setter for the parameter, using a parameter object
  /// \param params parameter object with parameters
  void SetParameters(const Parameters params) { SetParameters(params.mPar); };

  /// Setter for the parameter, using a parameter pointer
  /// \param params pointer to parameter object with parameters
  void SetParameters(const Parameters* params) { SetParameters(params->mPar); };

  /// Printer of the parameter values
  void Print(Option_t* option = "") const override;

  /// Loader from file
  /// \param FileName name of the input file
  /// \param ParamName name of the input object
  void LoadParamFromFile(const TString FileName, const TString ParamName);

  /// Getter for the parameters
  /// \return returns an array of parameters
  const pidvar_t* GetParameters() const { return mPar.data(); }

  /// Getter for the size of the parameter
  /// \return returns the size of the parameter array
  unsigned int size() const { return mPar.size(); }

  /// Getter of the parameter at position i
  /// \param i index of the parameter to get
  /// \return returns the parameter value at position i
  pidvar_t operator[](const unsigned int i) const { return mPar[i]; }

 private:
  /// Vector of the parameter
  std::vector<pidvar_t> mPar;

  ClassDef(Parameters, 1); // Container for parameter of parametrizations
};

/// \brief Class to handle the parameters and the parametrization of a given detector response
class Parametrization : public TNamed
{
 public:
  /// Default constructor
  Parametrization() : TNamed("DefaultParametrization", "DefaultParametrization"), mParameters{0} {};

  /// Parametric constructor
  /// \param name Name (and title) of the parametrization
  /// \param size Number of parameters of the parametrization
  Parametrization(TString name, unsigned int size) : TNamed(name, name), mParameters(name + "Parameters", size){};

  /// Parametric constructor
  /// \param name Name (and title) of the parametrization
  /// \param params Parameters of the parametrization
  Parametrization(TString name, const std::vector<pidvar_t> params) : TNamed(name, name), mParameters{name + "Parameters", params} {};

  /// Default destructor
  ~Parametrization() override = default;

  /// Getter for parametrization values, to be reimplemented in the custom parametrization of the user
  /// \param x array of variables to use in order to compute the return value
  virtual pidvar_t operator()(const pidvar_t* x) const;

  /// Printer for the parametrization
  void Print(Option_t* option = "") const override;

  /// Loader from file
  /// \param FileName name of the input file
  /// \param ParamName name of the input object
  void LoadParamFromFile(const TString FileName, const TString ParamName);

  /// Setter for the parameter at position iparam
  /// \param iparam index in the array of the parameters
  /// \param value value of the parameter at position iparam
  void SetParameter(const unsigned int iparam, const pidvar_t value) { mParameters.SetParameter(iparam, value); }

  /// Setter for the parameter, using a vector, a parameter object or a pointer to a parameter object
  /// \param params vector, parameter object, pointer to parameter object with parameters
  template <typename T>
  void SetParameters(const T params)
  {
    mParameters.SetParameters(params);
  }

  /// Getter for the parameters
  Parameters GetParameters() const { return mParameters; }

  /// Getter for the parameters
  void GetParameters(Parameters*& parameters) { parameters = &mParameters; }

 protected:
  /// Parameters of the parametrization
  Parameters mParameters;

  ClassDef(Parametrization, 1); // Container for the parametrization of the response function
};

} // namespace o2::pid

#endif // O2_FRAMEWORK_PARAMBASE_H_
