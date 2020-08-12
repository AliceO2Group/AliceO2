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

// Root includes
#include "TFile.h"

// O2 includes
#include "PIDBase/ParamBase.h"
#include "PIDBase/DetectorResponse.h"
#include "Framework/Logger.h"

namespace o2::pid
{

const std::array<TString, DetectorResponse::kNParams> DetectorResponse::ParamName = {{"Signal", "Sigma"}};

void DetectorResponse::LoadParamFromFile(const TString fname, const TString pname, const Param_t ptype)
{
  TFile f(fname, "READ");
  if (!f.Get(pname)) {
    LOG(fatal) << "Did not find parametrization " << pname << " in file " << fname;
  }
  LOG(info) << "Loading parametrization " << pname << " from TFile " << fname;
  f.GetObject(pname, mParam[ptype]);
  f.Close();
  mParam[ptype]->Print();
  mParam[ptype]->PrintParametrization();
}

void DetectorResponse::SetParameters(const Param_t ptype, std::vector<pidvar_t> p)
{
  if (!mParam[ptype]) {
    const TString pname = ParamName[ptype] + "_default_param";
    LOG(info) << "Creating new parametrization " << pname << " of size " << p.size();
    mParam[ptype] = new Parametrization(pname, p.size());
    mParam[ptype]->Print();
  }
  mParam[ptype]->SetParameters(p);
}

} // namespace o2::pid