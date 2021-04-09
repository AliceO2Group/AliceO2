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
/// \file   ParamBase.cxx
/// \author Nicolo' Jacazio
/// \since 07/08/2020
/// \brief Set of utilities to handle the parametrization of the PID response for each detector
///        These are the basic storage elements to be kept in the CCDB
///

#include "AnalysisDataModel/PID/ParamBase.h"
#include "Framework/Logger.h"
#include "TFile.h"

namespace o2::pid
{

void Parameters::SetParameters(const std::vector<pidvar_t> params)
{
  if (mPar.size() != params.size()) {
    LOG(fatal) << "Updating parametrization size!";
  }
  mPar.assign(params.begin(), params.end());
}

void Parameters::Print(Option_t* options) const
{
  LOG(info) << "Parameters '" << fName << "'";
  for (unsigned int i = 0; i < size(); i++) {
    LOG(info) << "Parameter " << i << "/" << size() - 1 << " is " << mPar[i];
  }
};

void Parameters::LoadParamFromFile(const TString FileName, const TString ParamName)
{
  TFile f(FileName, "READ");
  if (!f.Get(ParamName)) {
    LOG(fatal) << "Did not find parameters " << ParamName << " in file " << FileName;
  }
  LOG(info) << "Loading parameters " << ParamName << " from TFile " << FileName;
  Parameters* p;
  f.GetObject(ParamName, p);
  f.Close();
  SetParameters(p);
  Print();
}

pidvar_t Parametrization::operator()(const pidvar_t* x) const
{
  LOG(fatal) << "Parametrization " << fName << " is not implemented!";
  return -999.999f;
}

void Parametrization::Print(Option_t* options) const
{
  LOG(info) << "Parametrization '" << fName << "'";
  mParameters.Print(options);
};

void Parametrization::LoadParamFromFile(const TString FileName, const TString ParamName)
{
  TFile f(FileName, "READ");
  if (!f.Get(ParamName)) {
    LOG(fatal) << "Did not find parameters " << ParamName << " in file " << FileName;
  }
  LOG(info) << "Loading parameters " << ParamName << " from TFile " << FileName;
  Parametrization* p;
  f.GetObject(ParamName, p);
  f.Close();
  SetParameters(p->mParameters);
  Print();
}

} // namespace o2::pid