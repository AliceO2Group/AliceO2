// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <mutex>
#include <boost/property_tree/ptree.hpp>

#include "Headers/DataHeader.h"
#include "Framework/InputSpec.h"
#include "Framework/Output.h"
#include "Framework/OutputSpec.h"
#include "Framework/DataSamplingCondition.h"

namespace o2
{
namespace framework
{

/// A class representing certain policy of sampling data.
///
/// This class stores information about specified sampling policy - data headers and conditions of sampling.
/// For given InputSpec, it can provide corresponding Output to pass data further.
class DataSamplingPolicy
{
 private:
  struct inputSpecHasher {
    size_t operator()(const InputSpec& i) const
    {
      return (static_cast<size_t>(i.description.itg[0]) << 32 |
             static_cast<size_t>(i.description.itg[1])) ^
               static_cast<size_t>(i.origin.itg[0]);
    }
  };
  struct inputSpecEqual {
    bool operator()(const InputSpec& a, const InputSpec& b) const
    {
      // -1 means 'match all subSpec'
      if (a.subSpec == -1 || b.subSpec == -1) {
        return a.description == b.description && a.origin == b.origin;
      } else {
        return a.description == b.description && a.origin == b.origin && a.subSpec == b.subSpec;
      }
    }
  };
  using PathMap = std::unordered_map<InputSpec, OutputSpec, inputSpecHasher, inputSpecEqual>;

 public:
  /// \brief Constructor.
  DataSamplingPolicy();
  /// \brief Constructor.
  DataSamplingPolicy(const boost::property_tree::ptree&);
  /// \brief Destructor
  ~DataSamplingPolicy();

  /// \brief Configures a policy using structured configuration entry.
  void configure(const boost::property_tree::ptree&);
  /// \brief Returns true if this policy requires data with given InputSpec.
  bool match(const InputSpec&) const;
  /// \brief Returns true if user-defined conditions of sampling are fulfilled.
  bool decide(const o2::framework::DataRef&);
  /// \brief Returns Output for given InputSpec to pass data forward.
  const Output prepareOutput(const InputSpec&) const;

  const std::string& getName() const;
  //  const std::vector<InputSpec>& getInputs() const;
  const PathMap& getPathMap() const;
  const header::DataHeader::SubSpecificationType getSubSpec() const;
  // optional fairmq channel to send stuff outside of DPL
  const std::string& getFairMQOutputChannel() const;
  std::string getFairMQOutputChannelName() const;

  static header::DataOrigin policyDataOrigin();
  static header::DataDescription policyDataDescription(std::string policyName, size_t id);

 private:
  std::string mName;
  PathMap mPaths;
  header::DataHeader::SubSpecificationType mSubSpec;
  std::vector<std::unique_ptr<DataSamplingCondition>> mConditions;
  std::string mFairMQOutputChannel;

  std::mutex mDecisionMutex;
};

} // namespace framework
} // namespace o2

#endif // ALICEO2_DATASAMPLINGPOLICY_H
