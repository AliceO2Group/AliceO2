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

#ifndef ALICEO2_DATASAMPLINGPOLICY_H
#define ALICEO2_DATASAMPLINGPOLICY_H

/// \file DataSamplingPolicy.h
/// \brief A declaration of O2 Data Sampling Policy
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Headers/DataHeader.h"
#include "Framework/InputSpec.h"
#include "Framework/Output.h"
#include "Framework/OutputSpec.h"
#include "DataSampling/DataSamplingCondition.h"
#include "Framework/DataSpecUtils.h"

#include <boost/property_tree/ptree_fwd.hpp>

namespace o2::utilities
{

/// A class representing certain policy of sampling data.
///
/// This class stores information about specified sampling policy - data headers and conditions of sampling.
/// For given InputSpec, it can provide corresponding Output to pass data further.
class DataSamplingPolicy
{
 private:
  using PathVectorBase = std::vector<std::pair<framework::InputSpec, framework::OutputSpec>>;
  class PathMap : public PathVectorBase
  {
   public:
    PathMap() = default;
    ~PathMap() = default;
    const PathVectorBase::const_iterator find(const framework::ConcreteDataMatcher& input) const
    {
      return std::find_if(begin(), end(), [input](const auto& el) {
        return framework::DataSpecUtils::match(el.first, input);
      });
    }
  };

 public:
  /// \brief Configures a policy using structured configuration entry.
  static DataSamplingPolicy fromConfiguration(const boost::property_tree::ptree&);

  /// \brief Constructor.
  DataSamplingPolicy(std::string name);
  /// \brief Destructor
  ~DataSamplingPolicy() = default;
  /// \brief Move constructor
  DataSamplingPolicy(DataSamplingPolicy&&) = default;
  /// \brief Move assignment
  DataSamplingPolicy& operator=(DataSamplingPolicy&&) = default;

  /// \brief Adds a new association between inputs and outputs.
  void registerPath(const framework::InputSpec&, const framework::OutputSpec&);
  /// \brief Adds a new association between inputs and outputs.
  //  void registerPolicy(framework::InputSpec&&, framework::OutputSpec&&);
  /// \brief Adds a new sampling condition.
  void registerCondition(std::unique_ptr<DataSamplingCondition>&&);
  /// \brief Sets a raw fair::mq::Channel. Deprecated, do not use.
  void setFairMQOutputChannel(std::string);

  /// \brief Returns true if this policy requires data with given InputSpec.
  const framework::OutputSpec* match(const framework::ConcreteDataMatcher& input) const;
  /// \brief Returns true if user-defined conditions of sampling are fulfilled.
  bool decide(const o2::framework::DataRef&);
  /// \brief Returns Output for given InputSpec to pass data forward.
  framework::Output prepareOutput(const framework::ConcreteDataMatcher& input, framework::Lifetime lifetime = framework::Lifetime::Timeframe) const;

  const std::string& getName() const;
  const PathMap& getPathMap() const;
  // optional fairmq channel to send stuff outside of DPL
  const std::string& getFairMQOutputChannel() const;
  std::string getFairMQOutputChannelName() const;
  uint32_t getTotalAcceptedMessages() const;
  uint32_t getTotalEvaluatedMessages() const;

  static header::DataOrigin createPolicyDataOrigin();
  static header::DataDescription createPolicyDataDescription(std::string policyName, size_t id);

 private:
  std::string mName;
  PathMap mPaths;
  std::vector<std::unique_ptr<DataSamplingCondition>> mConditions;
  std::string mFairMQOutputChannel;

  // stats
  uint32_t mTotalAcceptedMessages = 0;
  uint32_t mTotalEvaluatedMessages = 0;
};

} // namespace o2::utilities

#endif // ALICEO2_DATASAMPLINGPOLICY_H
