// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//first version 8/2018, Sandro Wenzel

#ifndef COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_
#define COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_

#include "SimConfig/ConfigurableParam.h"
#include <sstream>
#include <iostream>
#include "TClass.h"
#include <type_traits>
#include <typeinfo>
#include "TFile.h"

namespace o2
{
namespace conf
{
// Utility structure for passing around ConfigurableParam data member info
// (where value is the string representation)
struct paramDataMember {
  std::string name;
  const char* value;
  std::string provenance;

  std::string toString(bool showProv) const
  {
    std::string nil = "<null>";
    std::string val = ((value == nullptr) ? nil : std::string(value));

    std::ostringstream out;
    out << name << " : " << val;

    if (showProv) {
      std::string prov = (provenance == "" ? nil : provenance);
      out << "\t\t[ " + prov + " ]";
    }

    out << "\n";
    return out.str();
  }

  std::ostream& operator<<(std::ostream& out)
  {
    out << this->toString(false);
    return out;
  }
};

// just a (non-templated) helper with exclusively private functions
// used by ConfigurableParamHelper
class _ParamHelper
{
 private:
  static std::vector<paramDataMember>* getDataMembersImpl(std::string mainkey, TClass* cl, void*,
                                                          std::map<std::string, ConfigurableParam::EParamProvenance> const* provmap);

  static void fillKeyValuesImpl(std::string mainkey, TClass* cl, void*, boost::property_tree::ptree*,
                                std::map<std::string, std::pair<std::type_info const&, void*>>*);

  static void printWarning(std::type_info const&);

  static void assignmentImpl(std::string mainkey, TClass* cl, void* to, void* from,
                             std::map<std::string, ConfigurableParam::EParamProvenance>* provmap);

  template <typename P>
  friend class ConfigurableParamHelper;
};

// ----------------------------------------------------------------
// ----------------------------------------------------------------
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

  // one of the key methods, using introspection to print itself
  void printKeyValues(bool showprov) const final
  {
    auto members = getDataMembers();
    if (members == nullptr) {
      return;
    }

    for (auto& member : *members) {
      std::cout << member.toString(showprov);
    }
  }

  // ----------------------------------------------------------------

  void output(std::ostream& out) const final
  {
    auto members = getDataMembers();
    if (members == nullptr) {
      return;
    }

    for (auto& member : *members) {
      out << member.toString(true);
    }
  }

  // ----------------------------------------------------------------

  // Grab the list of ConfigurableParam data members
  // Returns a nullptr if the TClass of the P template class cannot be created.
  std::vector<paramDataMember>* getDataMembers() const
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

    return _ParamHelper::getDataMembersImpl(getName(), cl, (void*)this, sValueProvenanceMap);
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
    _ParamHelper::fillKeyValuesImpl(getName(), cl, (void*)this, tree, sKeyToStorageMap);
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
                                   sValueProvenanceMap);
      delete readback;
    }
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
