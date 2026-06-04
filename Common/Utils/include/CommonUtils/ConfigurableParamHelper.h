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

//first version 8/2018, Sandro Wenzel

#ifndef COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_
#define COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_

#include "CommonUtils/ConfigurableParam.h"
#include "TClass.h"
#include <memory>
#include <type_traits>
#include <typeinfo>
#include "TFile.h"

namespace o2
{
namespace conf
{

// ----------------------------------------------------------------

// Utility structure for passing around ConfigurableParam data member info
// (where value is the string representation)
struct ParamDataMember {
  std::string name;
  std::string value;
  std::string provenance;

  std::string toString(std::string const& prefix, bool showProv, size_t padding = 0) const;
};

// ----------------------------------------------------------------

// just a (non-templated) helper with exclusively private functions
// used by ConfigurableParamHelper
class _ParamHelper
{
 private:
  static std::vector<ParamDataMember>* getDataMembersImpl(std::string const& mainkey, TClass* cl, void*,
                                                          std::map<std::string, ConfigurableParam::EParamProvenance> const* provmap, size_t virtualoffset);

  static void fillKeyValuesImpl(std::string const& mainkey, TClass* cl, void*, boost::property_tree::ptree*,
                                std::map<std::string, std::pair<std::type_info const&, void*>>*,
                                EnumRegistry*, size_t offset);

  static void printWarning(std::type_info const&);

  static void assignmentImpl(std::string const& mainkey, TClass* cl, void* to, void* from,
                             std::map<std::string, ConfigurableParam::EParamProvenance>* provmap, size_t offset);
  static void syncCCDBandRegistry(std::string const& mainkey, TClass* cl, void* to, void* from,
                                  std::map<std::string, ConfigurableParam::EParamProvenance>* provmap, size_t offset);

  static void outputMembersImpl(std::ostream& out, std::string const& mainkey, std::vector<ParamDataMember> const* members, bool showProv, bool useLogger, bool withPadding = false, bool showHash = false);
  static void printMembersImpl(std::string const& mainkey, std::vector<ParamDataMember> const* members, bool showProv, bool useLogger, bool withPadding, bool showHash);

  static size_t getHashImpl(std::string const& mainkey, std::vector<ParamDataMember> const* members);

  template <typename P>
  friend class ConfigurableParamHelper;

  template <typename Base, typename P>
  friend class ConfigurableParamPromoter;
};

// ----------------------------------------------------------------

// implementer (and checker) for concrete ConfigurableParam classes P
template <typename P>
class ConfigurableParamHelper : virtual public ConfigurableParam
{
 public:
  using ConfigurableParam::ConfigurableParam;
  static const P& Instance()
  {
    return P::sInstance;
  }

  // ----------------------------------------------------------------

  std::string getName() const final
  {
    return P::sKey;
  }

  // ----------------------------------------------------------------
  // get the provenace of the member with given key
  EParamProvenance getMemberProvenance(const std::string& key) const final
  {
    return getProvenance(getName() + '.' + key);
  }

  // ----------------------------------------------------------------

  // one of the key methods, using introspection to print itself
  void printKeyValues(bool showProv = true, bool useLogger = false, bool withPadding = true, bool showHash = true) const final
  {
    if (!isInitialized()) {
      initialize();
    }
    auto members = std::unique_ptr<std::vector<ParamDataMember>>(getDataMembers());
    _ParamHelper::printMembersImpl(getName(), members.get(), showProv, useLogger, withPadding, showHash);
  }

  //
  size_t getHash() const final
  {
    auto members = std::unique_ptr<std::vector<ParamDataMember>>(getDataMembers());
    return _ParamHelper::getHashImpl(getName(), members.get());
  }

  // ----------------------------------------------------------------

  void output(std::ostream& out) const final
  {
    auto members = std::unique_ptr<std::vector<ParamDataMember>>(getDataMembers());
    _ParamHelper::outputMembersImpl(out, getName(), members.get(), true, false);
  }

  // ----------------------------------------------------------------

  // Grab the list of ConfigurableParam data members
  // Returns a nullptr if the TClass of the P template class cannot be created.
  std::vector<ParamDataMember>* getDataMembers() const
  {
    // just a helper line to make sure P::sInstance is looked-up
    // and that compiler complains about missing static sInstance of type P
    // volatile void* ptr = (void*)&P::sInstance;
    // static assert on type of sInstance:
    static_assert(std::is_same<decltype(P::sInstance), P>::value,
                  "static instance must of same type as class");

    // obtain the TClass for P and delegate further
    auto cl = TClass::GetClass(typeid(P));
    if (!cl) {
      _ParamHelper::printWarning(typeid(P));
      return nullptr;
    }

    return _ParamHelper::getDataMembersImpl(getName(), cl, (void*)this, sValueProvenanceMap, 0);
  }

  // ----------------------------------------------------------------

  // fills the data structures with the initial default values
  void putKeyValues(boost::property_tree::ptree* tree) final
  {
    auto cl = TClass::GetClass(typeid(P));
    if (!cl) {
      _ParamHelper::printWarning(typeid(P));
      return;
    }
    _ParamHelper::fillKeyValuesImpl(getName(), cl, (void*)this, tree, sKeyToStorageMap, sEnumRegistry, 0);
  }

  // ----------------------------------------------------------------

  void initFrom(TFile* file) final
  {
    // switch off auto registering since the readback object is
    // only a "temporary" singleton
    setRegisterMode(false);
    P* readback = nullptr;
    file->GetObject(getName().c_str(), readback);
    if (readback != nullptr) {
      _ParamHelper::assignmentImpl(getName(), TClass::GetClass(typeid(P)), (void*)this, (void*)readback,
                                   sValueProvenanceMap, 0);
      delete readback;
    }
    setRegisterMode(true);
  }

  // ----------------------------------------------------------------

  void syncCCDBandRegistry(void* externalobj) final
  {
    // We may be getting an external copy from CCDB which is passed as externalobj.
    // The task of this function is to
    // a) update the internal registry with fields coming from CCDB
    //    but only if keys have not been modified via RT == command line / ini file
    // b) update the external object with with fields having RT provenance
    //
    setRegisterMode(false);
    _ParamHelper::syncCCDBandRegistry(getName(), TClass::GetClass(typeid(P)), (void*)this, (void*)externalobj,
                                      sValueProvenanceMap, 0);
    setRegisterMode(true);
  }

  // ----------------------------------------------------------------

  void serializeTo(TFile* file) const final
  {
    file->WriteObjectAny((void*)this, TClass::GetClass(typeid(P)), getName().c_str());
  }
};

// Promotes a simple struct Base to a configurable parameter class
// Aka implements all interfaces for a ConfigurableParam P, which shares or
// takes the fields from a Base struct
template <typename P, typename Base>
class ConfigurableParamPromoter : public Base, virtual public ConfigurableParam
{
 public:
  using ConfigurableParam::ConfigurableParam;

  static const P& Instance()
  {
    return P::sInstance;
  }

  // extracts a copy of the underlying data struct
  Base detach() const
  {
    static_assert(std::copyable<Base>, "Base type must be copyable.");
    return static_cast<Base>(*this);
  }

  // ----------------------------------------------------------------
  std::string getName() const final
  {
    return P::sKey;
  }

  // ----------------------------------------------------------------
  // get the provenace of the member with given key
  EParamProvenance getMemberProvenance(const std::string& key) const final
  {
    return getProvenance(getName() + '.' + key);
  }

  // ----------------------------------------------------------------

  // one of the key methods, using introspection to print itself
  void printKeyValues(bool showProv = true, bool useLogger = false, bool withPadding = true, bool showHash = true) const final
  {
    if (!isInitialized()) {
      initialize();
    }
    auto members = std::unique_ptr<std::vector<ParamDataMember>>(getDataMembers());
    _ParamHelper::printMembersImpl(getName(), members.get(), showProv, useLogger, withPadding, showHash);
  }

  //
  size_t getHash() const final
  {
    auto members = std::unique_ptr<std::vector<ParamDataMember>>(getDataMembers());
    return _ParamHelper::getHashImpl(getName(), members.get());
  }

  // ----------------------------------------------------------------

  void output(std::ostream& out) const final
  {
    auto members = std::unique_ptr<std::vector<ParamDataMember>>(getDataMembers());
    _ParamHelper::outputMembersImpl(out, getName(), members.get(), true, false);
  }

  // ----------------------------------------------------------------

  // Grab the list of ConfigurableParam data members
  // Returns a nullptr if the TClass of the P template class cannot be created.
  std::vector<ParamDataMember>* getDataMembers() const
  {
    // just a helper line to make sure P::sInstance is looked-up
    // and that compiler complains about missing static sInstance of type P
    // volatile void* ptr = (void*)&P::sInstance;
    // static assert on type of sInstance:
    static_assert(std::is_same<decltype(P::sInstance), P>::value,
                  "static instance must of same type as class");

    // obtain the TClass for the Base type and delegate further
    auto cl = TClass::GetClass(typeid(Base));
    if (!cl) {
      _ParamHelper::printWarning(typeid(Base));
      return nullptr;
    }

    // we need to put an offset of 8 bytes since internally this is using data members of the Base class
    // which doesn't account for the virtual table of P
    return _ParamHelper::getDataMembersImpl(getName(), cl, (void*)this, sValueProvenanceMap, 8);
  }

  // ----------------------------------------------------------------

  // fills the data structures with the initial default values
  void putKeyValues(boost::property_tree::ptree* tree) final
  {
    auto cl = TClass::GetClass(typeid(Base));
    if (!cl) {
      _ParamHelper::printWarning(typeid(Base));
      return;
    }
    _ParamHelper::fillKeyValuesImpl(getName(), cl, (void*)this, tree, sKeyToStorageMap, sEnumRegistry, 8);
  }

  // ----------------------------------------------------------------

  void initFrom(TFile* file) final
  {
    // switch off auto registering since the readback object is
    // only a "temporary" singleton
    setRegisterMode(false);
    P* readback = nullptr;
    file->GetObject(getName().c_str(), readback);
    if (readback != nullptr) {
      _ParamHelper::assignmentImpl(getName(), TClass::GetClass(typeid(Base)), (void*)this, (void*)readback,
                                   sValueProvenanceMap, 8);
      delete readback;
    }
    setRegisterMode(true);
  }

  // ----------------------------------------------------------------

  void syncCCDBandRegistry(void* externalobj) final
  {
    // We may be getting an external copy from CCDB which is passed as externalobj.
    // The task of this function is to
    // a) update the internal registry with fields coming from CCDB
    //    but only if keys have not been modified via RT == command line / ini file
    // b) update the external object with with fields having RT provenance
    //
    setRegisterMode(false);
    _ParamHelper::syncCCDBandRegistry(getName(), TClass::GetClass(typeid(Base)), (void*)this, (void*)externalobj,
                                      sValueProvenanceMap, 8);
    setRegisterMode(true);
  }

  // ----------------------------------------------------------------

  void serializeTo(TFile* file) const final
  {
    file->WriteObjectAny((void*)this, TClass::GetClass(typeid(P)), getName().c_str());
  }
};

} // namespace conf
} // namespace o2

#endif /* COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_ */
