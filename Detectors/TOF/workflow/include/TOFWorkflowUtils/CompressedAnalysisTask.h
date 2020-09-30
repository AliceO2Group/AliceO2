// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedAnalysisTask.h
/// @author Roberto Preghenella
/// @since  2020-09-04
/// @brief  TOF compressed data analysis task

#ifndef O2_TOF_COMPRESSEDANALYSISTASK
#define O2_TOF_COMPRESSEDANALYSISTASK

#include "Framework/Task.h"
#include "TOFWorkflowUtils/CompressedAnalysis.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TGlobal.h"
#include "TFunction.h"
#include <string>
#include <iostream>

using namespace o2::framework;

namespace o2
{
namespace tof
{

class CompressedAnalysisTask : public Task
{
 public:
  CompressedAnalysisTask() = default;
  ~CompressedAnalysisTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  CompressedAnalysis* mAnalysis = nullptr;
  bool mStatus = false;

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
        std::cout << "Cannot find " << file << std::endl;
        return nullptr;
      }
      if (!gROOT->GetGlobalFunction(gfunc.c_str())) {
        std::cout << "Global function '" << gfunc << "' not defined" << std::endl;
        return nullptr;
      }
    }

    /** check the return type matches the required one **/
    if (strcmp(gROOT->GetGlobalFunction(gfunc.c_str())->GetReturnTypeName(), type.c_str())) {
      std::cout << "Global function '" << gfunc << "' does not return a '" << type << "' type" << std::endl;
      return nullptr;
    }

    /** process function and retrieve pointer to the returned type **/
    gROOT->ProcessLine(Form("%s __%s__ = %s;", type.c_str(), unique.c_str(), func.c_str()));
    auto ptr = (T*)gROOT->GetGlobal(Form("__%s__", unique.c_str()))->GetAddress();

    /** success **/
    return *ptr;
  }
};

} // namespace tof
} // namespace o2

#endif /* O2_TOF_COMPRESSEDANALYSISTASK */
