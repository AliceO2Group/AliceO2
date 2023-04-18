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
/// \brief  Implementation of the containers for the general parameters

#include "DataFormatsTOF/ParameterContainers.h"

// ClassImp(o2::tof::Parameters);
using namespace o2::tof;

bool ParameterCollection::addParameter(const std::string& pass, const std::string& parName, float value)
{
  const bool alreadyPresent = hasKey(pass);
  if (alreadyPresent) {
    LOG(debug) << "Changing parametrization corresponding to key " << pass << " from size " << mParameters[pass].size() << " to " << parName;
  } else {
    mParameters[pass] = std::unordered_map<std::string, paramvar_t>{};
    LOG(debug) << "Adding new parametrization corresponding to key " << pass << ": " << parName;
  }
  mParameters[pass][parName] = value;
  return true;
}

int ParameterCollection::getSize(const std::string& pass) const
{
  if (!hasKey(pass)) {
    return -1;
  }
  return mParameters.at(pass).size();
}

void ParameterCollection::print() const
{
  for (const auto& [pass, pars] : mParameters) {
    print(pass);
  }
}

void ParameterCollection::print(const std::string& pass) const
{
  const auto& size = getSize(pass);
  if (size < 0) {
    LOG(info) << "empty pass: " << pass;
    return;
  }
  LOG(info) << "Pass \"" << pass << "\" with size " << size;
  for (const auto& [par, value] : mParameters.at(pass)) {
    LOG(info) << "par name = " << par << ", value = " << value;
  }
}
