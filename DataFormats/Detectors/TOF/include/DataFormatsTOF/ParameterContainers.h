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

/// \file   ParameterContainers.h
/// \author Francesco Noferini
/// \author Nicolò Jacazio nicolo.jacazio@cern.ch
/// @since  2022-11-08
/// \brief  Definitions of the containers for the general parameters

#ifndef O2_TOF_PARAMCONTAINER_H
#define O2_TOF_PARAMCONTAINER_H

#include "TNamed.h"
#include "TFile.h"
#include "Framework/Logger.h"
#include "map"

namespace o2
{
namespace tof
{
using paramvar_t = float;

template <int nPar>
class Parameters
{
 public:
  /// Default constructor
  Parameters(std::array<std::string, nPar> parNames, std::string name) : mName{name}, mPar{}, mParNames{parNames} {};

  /// Default destructor
  ~Parameters() = default;

  /// Setter for the parameter at position iparam
  /// \param iparam index in the array of the parameters
  /// \param value value of the parameter at position iparam
  void setParameter(const unsigned int iparam, const paramvar_t value) { mPar[iparam] = value; }

  /// Setter for the parameter, using an array
  /// \param param array with parameters
  void setParameters(const paramvar_t* params) { std::copy(params, params + mPar.size(), mPar.begin()); }

  /// Setter for the parameter, using a vector
  /// \param params vector with parameters
  void setParameters(const std::array<paramvar_t, nPar> params)
  {
    for (int i = 0; i < nPar; i++) {
      mPar[i] = params[i];
    }
  }

  /// Setter for the parameter, using a parameter object
  /// \param params parameter object with parameters
  void setParameters(const Parameters<nPar> params) { setParameters(params.mPar); };

  /// Setter for the parameter, using a parameter pointer
  /// \param params pointer to parameter object with parameters
  void setParameters(const Parameters<nPar>* params) { setParameters(params->mPar); };

  /// Printer of the parameter values
  void print() const
  {
    LOG(info) << "Parameters '" << mName << "'";
    for (int i = 0; i < nPar; i++) {
      LOG(info) << "Parameter " << i << "/" << nPar - 1 << " is " << mPar[i];
    }
  }

  /// Adds the parameters to the metadata
  void addToMetadata(std::map<std::string, std::string>& metadata) const
  {
    for (int i = 0; i < nPar; i++) {
      metadata[Form("p%i", i)] = Form("%f", mPar[i]);
    }
  }

  /// Loader from file
  /// \param FileName name of the input file
  /// \param ParamName name of the input object
  void loadParamFromFile(const TString FileName, const TString ParamName)
  {
    TFile f(FileName, "READ");
    if (!f.Get(ParamName)) {
      LOG(fatal) << "Did not find parameters " << ParamName << " in file " << FileName;
    }
    LOG(info) << "Loading parameters " << ParamName << " from TFile " << FileName;
    Parameters<nPar>* p;
    f.GetObject(ParamName, p);
    if (!p) {
      LOG(fatal) << "Could not get parameters " << ParamName << " from file";
      f.ls();
    }
    f.Close();
    setParameters(p);
    print();
  }

  /// Getter for the parameters
  /// \return returns an array of parameters
  const paramvar_t* getParameters() const { return mPar.to_array(); }

  /// Getter for the parameters
  /// \return returns an array of parameters
  const paramvar_t getParameter(int i) const { return mPar[i]; }

  /// Getter for the parameters
  /// \return returns an array of parameters
  const std::string getParameterName(int i) const { return mParNames[i]; }

  /// Getter for the parameters
  /// \return returns an array of parameters
  const std::string getName() const { return mName; }

  /// Getter for the size of the parameter
  /// \return returns the size of the parameter array
  static int size() { return nPar; }

  /// Getter of the parameter at position i
  /// \param i index of the parameter to get
  /// \return returns the parameter value at position i
  paramvar_t operator[](const unsigned int i) const { return mPar[i]; }

 private:
  /// Array of the parameter
  std::array<paramvar_t, nPar> mPar;
  const std::array<std::string, nPar> mParNames;
  std::string mName;
};

/// \brief Class container to hold different parameters meant to be stored on the CCDB
class ParameterCollection : public TNamed
{
 public:
  /// Default constructor
  ParameterCollection(TString name = "DefaultParameters") : TNamed(name, name), mParameters{} {};

  /// Default destructor
  ~ParameterCollection() override = default;

  /// @brief Checks if the container has a particular key e.g. a pass
  /// @return true if found, false if not
  bool hasKey(const std::string& key) const { return (mParameters.find(key) != mParameters.end()); }

  /// @brief Function to load the parameters from the this container into the array based for the asked key, e.g. pass or version
  ///        Parameters that are not found in storage are kept unchanged in the array.
  /// @tparam ParType type of the parameter container
  /// @param p parameter list to configure from the stored information
  /// @param key key to look for in the stored information e.g. pass
  /// @return true if found and configured false if not fully configured
  template <typename ParType>
  bool retrieveParameters(ParType& p, const std::string& key) const
  {
    if (!hasKey(key)) { // Can't find the required key. Can't load parameters to the object
      return false;
    }

    const auto& toGet = mParameters.at(key);
    for (int i = 0; i < p.size(); i++) {
      const auto& name = p.getParameterName(i);
      if (toGet.find(name) == toGet.end()) {
        LOG(debug) << "Did not find parameter '" << name << "' in collection, keeping preexisting";
        continue;
      }
      LOG(debug) << "Found parameter '" << name << "' in collection, keeping preexisting";
      p.setParameter(i, toGet.at(name));
    }
    return true;
  }

  /// @brief Function to add a single parameter conatiner based on the asked key, e.g. pass or version
  /// @param value parameter to add to the stored information
  /// @param pass key to look for in the stored information e.g. pass
  /// @return true if found and configured false if not fully configured
  bool addParameter(const std::string& pass, const std::string& parName, float value);

  /// @return the size of the container i.e. the number of stored keys (or passes)
  int getSize(const std::string& pass) const;

  /// @brief Function to push the parameters from the sub container into the collection and store it under a given key
  /// @tparam ParType type of the parameter container
  /// @param p parameter list to store
  /// @param key store key
  /// @return true if modified and false if a new key is added
  template <typename ParType>
  bool storeParameters(const ParType& p, const std::string& key)
  {
    const bool alreadyPresent = hasKey(key);
    if (alreadyPresent) {
      LOG(debug) << "Changing parametrization corresponding to key " << key << " from size " << mParameters[key].size() << " to " << p.getName() << " of size " << p.size();
    } else {
      mParameters[key] = std::unordered_map<std::string, paramvar_t>{};
      LOG(debug) << "Adding new parametrization corresponding to key " << key << ": " << p.getName() << " of size " << p.size();
    }
    for (int i = 0; i < p.size(); i++) {
      mParameters[key][p.getParameterName(i)] = p[i];
    }
    return alreadyPresent;
  }

  /// @brief getter for the parameters stored in the container matching to a pass
  const auto& getPars(const std::string& pass) const { return mParameters.at(pass); }

  /// @brief printing function for the content of the pass
  /// @param pass pass to print
  void print(const std::string& pass) const;

  /// @brief printing function for the full content of the container
  void print() const;

  /// @brief Getter of the full map of parameters stored in the container
  /// @return returns the full map of parameters
  const auto& getFullMap() { return mParameters; }

  /// Loader from file
  /// \param FileName name of the input file
  /// \param ParamName name of the input object
  void loadParamFromFile(const TString FileName, const TString ParamName)
  {
    TFile f(FileName, "READ");
    if (!f.Get(ParamName)) {
      LOG(fatal) << "Did not find parameters " << ParamName << " in file " << FileName;
    }
    LOG(info) << "Loading parameters " << ParamName << " from TFile " << FileName;
    ParameterCollection* p;
    f.GetObject(ParamName, p);
    if (!p) {
      LOG(fatal) << "Could not get parameters " << ParamName << " from file";
      f.ls();
    }
    f.Close();

    for (const auto& pass : p->mParameters) {
      for (const auto& par : pass.second) {
        addParameter(pass.first, par.first, par.second);
      }
    }
    print();
  }

 private:
  /// Array of the parameter
  std::unordered_map<std::string, std::unordered_map<std::string, paramvar_t>> mParameters;

  ClassDefOverride(ParameterCollection, 1); // Container for containers of parameter of parametrizations. To be used as a manager, in help of CCDB
};

} // namespace tof
} // namespace o2

#endif // O2_TOF_PARAMCONTAINER_H
