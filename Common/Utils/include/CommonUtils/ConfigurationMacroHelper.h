// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - February 2020

#ifndef ALICEO2_CONF_CONFIGURATIONMACRO_H_
#define ALICEO2_CONF_CONFIGURATIONMACRO_H_

#include "FairLogger.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TGlobal.h"
#include "TFunction.h"
#include <string>

namespace o2
{
namespace conf
{

template <typename T>
T GetFromMacro(const std::string& file, const std::string& funcname, const std::string& type, const std::string& unique)
{

  /** tweak the string to get the required global function **/
  auto func = funcname;
  if (func.empty()) {
    auto size = file.size();
    auto firstindex = file.find_last_of("/") + 1;
    auto lastindex = file.find_last_of(".");
    func = file.substr(firstindex < size ? firstindex : 0,
                       lastindex < size ? lastindex - firstindex : size - firstindex) +
           "()";
  }
  auto gfunc = func.substr(0, func.find_first_of('('));

  /** load macro is global function is not already defined **/
  if (!gROOT->GetGlobalFunction(gfunc.c_str())) {
    if (gROOT->LoadMacro(file.c_str()) != 0) {
      LOG(FATAL) << "Cannot find " << file;
      return nullptr;
    }
    if (!gROOT->GetGlobalFunction(gfunc.c_str())) {
      LOG(FATAL) << "Global function '" << gfunc << "' not defined";
      return nullptr;
    }
  }

  /** check the return type matches the required one **/
  if (strcmp(gROOT->GetGlobalFunction(gfunc.c_str())->GetReturnTypeName(), type.c_str())) {
    LOG(INFO) << "Global function '" << gfunc << "' does not return a '" << type << "' type";
    return nullptr;
  }

  /** process function and retrieve pointer to the returned type **/
  gROOT->ProcessLine(Form("%s __%s__ = %s;", type.c_str(), unique.c_str(), func.c_str()));
  auto ptr = (T*)gROOT->GetGlobal(Form("__%s__", unique.c_str()))->GetAddress();

  /** success **/
  return *ptr;
}

} // namespace conf
} // namespace o2

#endif /* ALICEO2_CONF_CONFIGURATIONMACRO_H_ */
