// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingConditionCustom.cxx
/// \brief Implementation of DataSamplingConditionCustom
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataSamplingCondition.h"
#include "Framework/DataSamplingConditionFactory.h"
#include "Headers/DataHeader.h"

#include <fairlogger/Logger.h>

#include <TClass.h>
#include <TROOT.h>
#include <TSystem.h>

namespace o2
{
namespace framework
{

/// \brief A DataSamplingCondition which makes decisions based on payload size.
class DataSamplingConditionCustom : public DataSamplingCondition
{

 public:
  /// \brief Constructor.
  DataSamplingConditionCustom() : DataSamplingCondition(){};
  /// \brief Default destructor
  ~DataSamplingConditionCustom() override = default;

  /// \brief Instantiates and configures a custom condition based on configuration.
  ///
  /// \param config - it should include 'moduleName', full 'className' with namespaces. Optionally it can contain
  /// custom parameters for the loaded condition.
  void configure(const boost::property_tree::ptree& config) override
  {
    std::string libraryName = "lib" + config.get<std::string>("moduleName");
    std::string className = config.get<std::string>("className");

    int libLoaded = gSystem->Load(libraryName.c_str(), "", true);
    if (libLoaded < 0) {
      throw std::runtime_error("Failed to load the library: " + libraryName);
    }

    // it does not seem to be documented anywhere, but this pointer to
    // a dictionary should not be deleted - it results in segfaults.
    TClass* dictionary = TClass::GetClass(className.c_str());
    if (!dictionary) {
      throw std::runtime_error("Failed to load the dictionary of the class: " + className + " from the library: " + libraryName);
    }

    mCondition.reset(static_cast<DataSamplingCondition*>(dictionary->New()));
    if (mCondition == nullptr) {
      throw std::runtime_error("Failed to instantiate the class: " + className + " from the library: " + libraryName);
    }

    mCondition->configure(config);
  }

  /// \brief Invokes decide() of a custom condition
  bool decide(const o2::framework::DataRef& dataRef) override
  {
    return mCondition->decide(dataRef);
  }

 private:
  std::unique_ptr<DataSamplingCondition> mCondition;
};

std::unique_ptr<DataSamplingCondition> DataSamplingConditionFactory::createDataSamplingConditionCustom()
{
  return std::make_unique<DataSamplingConditionCustom>();
}

} // namespace framework
} // namespace o2
