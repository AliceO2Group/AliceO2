// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TOBJECTWRAPPER_H
#define O2_TOBJECTWRAPPER_H

#include <TClass.h>
#include <TObject.h>
#include <cxxabi.h>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <typeinfo>

namespace o2
{
// anonymous namespace to prevent usage outside of this file
namespace
{
/// utility function to demangle cxx type names
std::string demangle(const char* name)
{
  int status = -4; // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
}
} // end anonymous namespace

/// A wrapper class to easily promote any type to a TObject
/// does not take ownership of wrapped object and should not be used
/// in tight loops since construction expensive
template <typename T>
class TObjectWrapper : public TObject
{
 public:
  TObjectWrapper(T* obj) : mObj(obj), TObject()
  {
    // make sure that a dictionary for this wrapper exists
    auto& t = typeid(*this);
    std::string message("Need dicionary for type ");
    auto typestring = demangle(t.name());
    message.append(typestring);
    message.append("\n");
    auto hasdict = TClass::HasDictionarySelection(typestring.c_str());
    if (!hasdict) {
      throw std::runtime_error(message);
    }
  }

  TObjectWrapper() : TObjectWrapper(nullptr)
  {
  }

  void setObj(T* obj)
  {
    mObj = obj;
  }

  T* getObj() const
  {
    return mObj;
  }

  ~TObjectWrapper() override = default;

 private:
  T* mObj;

  ClassDefOverride(TObjectWrapper, 1);
};
} // namespace o2

#endif
