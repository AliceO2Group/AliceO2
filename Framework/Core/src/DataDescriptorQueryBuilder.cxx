// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataDescriptorMatcher.h"

#include <memory>
#include <regex>
#include <string>
#include <vector>

using namespace o2::framework::data_matcher;

namespace o2
{
namespace framework
{

DataDescriptorQuery DataDescriptorQueryBuilder::buildFromKeepConfig(std::string const& config)
{
  static const std::regex specTokenRE(R"re((\w{1,4})/(\w{1,16})/(\d*))re");
  static const std::regex delimiter(",");

  std::sregex_token_iterator iter(config.begin(),
                                  config.end(),
                                  delimiter,
                                  -1);
  std::sregex_token_iterator end;

  std::unique_ptr<DataDescriptorMatcher> result;

  for (; iter != end; ++iter) {
    std::smatch m;
    auto s = iter->str();
    std::regex_match(s, m, specTokenRE);
    std::unique_ptr<DataDescriptorMatcher> next;
    auto newNode = std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{ m[1] },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{ m[2] },
        SubSpecificationTypeValueMatcher{ m[3] }));
    if (result.get() == nullptr) {
      result = std::move(newNode);
    } else {
      next = std::move(std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Or,
                                                               std::move(result),
                                                               std::move(newNode)));
      result = std::move(next);
    }
  }

  return std::move(DataDescriptorQuery{ {}, std::move(result) });
}

} // namespace framework
} // namespace o2
