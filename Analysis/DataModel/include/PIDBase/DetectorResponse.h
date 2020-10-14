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
/// \file   DetectorResponse.h
/// \author Nicolo' Jacazio
/// \since  2020-07-30
/// \brief  Handler for any detector (or other entity) response.
///         This provides the basic quantities computed by any response i.e. expected values, resolutions and Nsigmas
///

#ifndef O2_ANALYSIS_PID_DETECTORRESPONSE_H_
#define O2_ANALYSIS_PID_DETECTORRESPONSE_H_

#include <array>
#include <vector>
#include "Framework/Logger.h"
// ROOT includes
#include "Rtypes.h"
#include "TMath.h"
#include "TFile.h"

// O2 includes
#include "ReconstructionDataFormats/PID.h"
#include "PIDBase/ParamBase.h"

namespace o2::pid
{
/// \brief Class to handle the general detector response
class DetectorResponse
{
 public:
  DetectorResponse() = default;
  virtual ~DetectorResponse() = default;

  /// Enumeration of the different types of parametrizations
  enum Param_t { kSignal,
                 kSigma,
                 kNParams };

  static constexpr std::array<char const*, kNParams> ParamName = {{"Signal", "Sigma"}};

  /// Setter for the parametrization from input TFile
  /// \param fname File name used for input
  /// \param pname Name of the parametrization in the file
  /// \param ptype Type of the parametrization
  void LoadParamFromFile(const TString fname, const TString pname, const Param_t ptype);

  /// Setter for the parametrization
  /// \param ptype Type of the parametrization
  /// \param param Parametrization
  void LoadParam(const Param_t ptype, Parametrization* param) { mParam[ptype] = param; }

  /// Getter for the parametrizations
  Parametrization* GetParam(const Param_t ptype) const { return mParam[ptype]; }

  /// Setter for the parametrizations parameters, if the parametrization is not yet initialized a new parametrization is created without any implementation and just parameters
  /// \param ptype parametrization type
  /// \param p vector with parameters
  void SetParameters(const Param_t ptype, std::vector<pidvar_t> p);

  /// Getter for the value of the parametrization
  /// \param ptype parametrization type
  /// \param x array with parameters
  virtual pidvar_t operator()(const Param_t ptype, const pidvar_t* x) const { return mParam[ptype]->operator()(x); }

 private:
  /// Parametrizations for the expected signal and sigma
  std::array<Parametrization*, kNParams> mParam;
};

inline void DetectorResponse::LoadParamFromFile(const TString fname, const TString pname, const Param_t ptype)
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

inline void DetectorResponse::SetParameters(const DetectorResponse::Param_t ptype, std::vector<pidvar_t> p)
{
  if (!mParam[ptype]) {
    const std::string pname = std::string(ParamName[ptype]) + "_default_param";
    LOG(info) << "Creating new parametrization " << pname << " of size " << p.size();
    mParam[ptype] = new Parametrization(pname, p.size());
    mParam[ptype]->Print();
  }
  mParam[ptype]->SetParameters(p);
}

} // namespace o2::pid

#endif // O2_ANALYSIS_PID_DETECTORRESPONSE_H_
