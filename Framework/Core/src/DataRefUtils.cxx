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

#include <typeinfo>
#include "CCDB/CcdbApi.h"
#include "Framework/DataRefUtils.h"
#include <TMemFile.h>
#include <TError.h>
#include "Framework/RuntimeError.h"

namespace o2::framework
{
// Adapted from CcdbApi private method interpretAsTMemFileAndExtract
// If the former is moved to public, throws on error and could be changed to
// not require a mutex we could use it.
void* DataRefUtils::decodeCCDB(DataRef const& ref, std::type_info const& tinfo)
{
  void* result = nullptr;
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  auto* dh = o2::header::get<o2::header::DataHeader*>(ref.header);
  TMemFile memFile("name", const_cast<char*>(ref.payload), dh->payloadSize, "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (memFile.IsZombie()) {
    return nullptr;
  }
  TClass* tcl = TClass::GetClass(tinfo);
  result = ccdb::CcdbApi::extractFromTFile(memFile, tcl);
  if (!result) {
    throw runtime_error_f("Couldn't retrieve object corresponding to %s from TFile", tcl->GetName());
  }
  memFile.Close();
  return result;
}
} // namespace o2::framework
