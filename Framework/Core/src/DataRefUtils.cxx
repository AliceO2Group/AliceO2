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

#include <span>
#include <typeinfo>
#include <cstring>
#include "Framework/DataRefUtils.h"
#include "Framework/RuntimeError.h"
#include "Framework/Logger.h"
#include <TMemFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TError.h>
#include <algorithm>
#if defined(_LIBCPP_VERSION)
#include <TList.h>
#include <TStreamerElement.h>
#include <TStreamerInfo.h>
#endif

namespace o2::framework
{

namespace
{
void* extractFromTFile(TFile& file, TClass const* cl, const char* what)
{
  if (!cl) {
    return nullptr;
  }
#if defined(_LIBCPP_VERSION)
  DataRefUtils::registerLibstdcxxStdArray(file, cl);
#endif
  auto object = file.GetObjectChecked(what, cl);
  if (!object) {
    // it could be that object was stored with previous convention
    // where the classname was taken as key
    std::string objectName(cl->GetName());
    objectName.erase(std::find_if(objectName.rbegin(), objectName.rend(), [](unsigned char ch) {
                       return !std::isspace(ch);
                     }).base(),
                     objectName.end());
    objectName.erase(objectName.begin(), std::find_if(objectName.begin(), objectName.end(), [](unsigned char ch) {
                       return !std::isspace(ch);
                     }));

    object = file.GetObjectChecked(objectName.c_str(), cl);
    LOG(warn) << "Did not find object under expected name " << what;
    if (!object) {
      return nullptr;
    }
    LOG(warn) << "Found object under deprecated name " << cl->GetName();
  }
  auto result = object;
  // We need to handle some specific cases as ROOT ties them deeply
  // to the file they are contained in
  if (cl->InheritsFrom("TObject")) {
    // make a clone
    // detach from the file
    auto tree = dynamic_cast<TTree*>((TObject*)object);
    if (tree) {
      tree->LoadBaskets(0x1L << 32); // make tree memory based
      tree->SetDirectory(nullptr);
      result = tree;
    } else {
      auto h = dynamic_cast<TH1*>((TObject*)object);
      if (h) {
        h->SetDirectory(nullptr);
        result = h;
      }
    }
  }
  return result;
}
} // namespace

#if defined(_LIBCPP_VERSION)
// The only member of std::array<T,N> is T _M_elems[N] in libstdc++, T __elems_[N] in libc++, so
// objects written on Linux carry a StreamerInfo with a checksum the libc++ class does not have.
// When opening the file, ROOT drops that StreamerInfo as an alias of the in-memory one
// (TStreamerInfo::BuildCheck), and the read fails with "Could not find the StreamerInfo with a
// checksum", leaving the array zero-filled. Register the on-file layout as a version of its own,
// under the in-memory member name, so that the payload, T[N] for both, is read as usual.
void DataRefUtils::registerLibstdcxxStdArray(TFile& file, TClass const* cl)
{
  auto* infos = std::strncmp(cl->GetName(), "array<", 6) == 0 ? file.GetStreamerInfoCache() : nullptr;
  auto* onfile = infos ? dynamic_cast<TStreamerInfo*>(infos->FindObject(cl->GetName())) : nullptr;
  if (!onfile || onfile->GetNelement() != 1 || std::strcmp(onfile->GetElement(0)->GetName(), "_M_elems") != 0) {
    return;
  }
  // Exists before BuildCheck, so that the on-file layout gets a version of its own
  auto* current = static_cast<TStreamerInfo*>(const_cast<TClass*>(cl)->GetStreamerInfo());
  auto* info = static_cast<TStreamerInfo*>(cl->FindStreamerInfo(onfile->GetCheckSum()));
  if (!info) {
    info = static_cast<TStreamerInfo*>(onfile->Clone());
    info->BuildCheck(&file);
  }
  // Also when ROOT registered it itself: it then reads nothing, without any error
  if (std::strcmp(info->GetElement(0)->GetName(), current->GetElement(0)->GetName()) != 0) {
    info->GetElement(0)->SetName(current->GetElement(0)->GetName());
    info->Clear("build");
    info->BuildOld();
  }
}
#endif

// Adapted from CcdbApi private method interpretAsTMemFileAndExtract
// If the former is moved to public, throws on error and could be changed to
// not require a mutex we could use it.
void* DataRefUtils::decodeCCDB(DataRef const& ref, std::type_info const& tinfo)
{
  void* result = nullptr;
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  auto* dh = o2::header::get<o2::header::DataHeader*>(ref.header);
  const char* buff = const_cast<char*>(ref.payload);
  size_t flSize = dh->payloadSize;
  // does it have a flattened headers map attached in the end?
  constexpr char FlatHeaderAnnot[] = "$HEADER$";
  constexpr int Offset = sizeof(int) + sizeof(FlatHeaderAnnot);
  int headerSize = 0;
  LOGP(debug, "DHPayloadSize={}>{} Ref:{}/{} Cmp {}:{}", dh->payloadSize, Offset, dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>(), std::string{buff + dh->payloadSize - sizeof(FlatHeaderAnnot)}, std::string{FlatHeaderAnnot});

  if (dh->payloadSize >= Offset &&
      !std::strncmp(buff + dh->payloadSize - sizeof(FlatHeaderAnnot), FlatHeaderAnnot, sizeof(FlatHeaderAnnot))) {
    headerSize = *reinterpret_cast<const int*>(buff + dh->payloadSize - Offset);
  }
  if (headerSize < 0) {
    LOGP(fatal, "Anomalous flattened header size {} extracted for CCDB object {}/{}", headerSize, dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>());
  }
  TMemFile memFile("name", const_cast<char*>(ref.payload), dh->payloadSize - headerSize, "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (memFile.IsZombie()) {
    return nullptr;
  }

  TClass* tcl = TClass::GetClass(tinfo);
  result = extractFromTFile(memFile, tcl, "ccdb_object");
  if (!result) {
    throw runtime_error_f("Couldn't retrieve object corresponding to %s from TFile", tcl->GetName());
  }
  memFile.Close();
  return result;
}

std::map<std::string, std::string> DataRefUtils::extractCCDBHeaders(DataRef const& ref)
{
  auto* dh = o2::header::get<o2::header::DataHeader*>(ref.header);
  const char* buff = const_cast<char*>(ref.payload);
  // does it have a flattened headers map attached in the end?
  constexpr char FlatHeaderAnnot[] = "$HEADER$";
  constexpr int Offset = sizeof(int) + sizeof(FlatHeaderAnnot);
  int headerSize = 0, ss0 = 0;
  std::map<std::string, std::string> res;
  if (dh->payloadSize >= Offset && !std::strncmp(buff + dh->payloadSize - sizeof(FlatHeaderAnnot), FlatHeaderAnnot, sizeof(FlatHeaderAnnot))) {
    headerSize = *reinterpret_cast<const int*>(buff + dh->payloadSize - Offset);
  } else { // header was not added
    LOGP(warn, "CCDB headers were not added to condition object blob, returning dummy header map");
    return res;
  }

  if (headerSize < 0) {
    LOGP(fatal, "Anomalous flattened header size {} extracted for CCDB object {}/{}", headerSize, dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>());
  }

  buff += dh->payloadSize - headerSize; // jump to the start of flattened header
  headerSize -= Offset;
  const char* str0 = &buff[ss0++];
  while (ss0 < headerSize) {
    if (buff[ss0++] == 0) {
      if (!str0) {
        str0 = &buff[ss0]; // new key string is found
      } else {
        res.emplace(std::string(str0), std::string(&buff[ss0])); // new value string found, add key value to the map
        LOGP(debug, "Header{} {}:{}", res.size(), std::string(str0), std::string(&buff[ss0]));
        str0 = nullptr;
      }
    }
  }
  return res;
}

std::span<const char> DataRefUtils::getCCDBPayloadBlob(DataRef const& ref)
{
  auto* dh = o2::header::get<o2::header::DataHeader*>(ref.header);
  const char* buff = ref.payload;
  size_t payloadSize = dh->payloadSize;
  constexpr char FlatHeaderAnnot[] = "$HEADER$";
  constexpr size_t Offset = sizeof(int) + sizeof(FlatHeaderAnnot);
  int headerSize = 0;
  if (payloadSize >= Offset &&
      !std::strncmp(buff + payloadSize - sizeof(FlatHeaderAnnot), FlatHeaderAnnot, sizeof(FlatHeaderAnnot))) {
    headerSize = *reinterpret_cast<const int*>(buff + payloadSize - Offset);
  }
  return {buff, payloadSize - headerSize};
}

} // namespace o2::framework
