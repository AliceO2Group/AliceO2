// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataSpecUtils.h"
#include "Framework/DataDescriptorMatcher.h"
#include <cstring>
#include <cinttypes>

namespace o2
{
namespace framework
{
namespace
{
std::string join(ConcreteDataMatcher const& matcher, std::string const sep = "_")
{
  return matcher.origin.as<std::string>() + sep + matcher.description.as<std::string>() + sep + std::to_string(matcher.subSpec);
}
}

using namespace data_matcher;

bool DataSpecUtils::match(const InputSpec& spec,
                          const o2::header::DataOrigin& origin,
                          const o2::header::DataDescription& description,
                          const o2::header::DataHeader::SubSpecificationType& subSpec)
{
  ConcreteDataMatcher target{ origin, description, subSpec };
  return match(spec, target);
}

std::string DataSpecUtils::describe(InputSpec const& spec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return join(*concrete, "/");
  } else if (auto matcher = std::get_if<DataDescriptorMatcher>(&spec.matcher)) {
    std::ostringstream ss;
    ss << "<matcher query: " << *matcher << ">";
    return ss.str();
  }
  throw std::runtime_error("Unhandled InputSpec kind.");
}

void DataSpecUtils::describe(char* buffer, size_t size, InputSpec const& spec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    char origin[5];
    origin[4] = 0;
    char description[17];
    description[16] = 0;
    snprintf(buffer, size, "%s/%s/%" PRIu32, (strncpy(origin, concrete->origin.str, 4), origin),
             (strncpy(description, concrete->description.str, 16), description), concrete->subSpec);
  } else if (auto matcher = std::get_if<DataDescriptorMatcher>(&spec.matcher)) {
    std::ostringstream ss;
    ss << "<matcher query: " << *matcher << ">";
    strncpy(buffer, ss.str().c_str(),  size - 1);
  } else {
    throw std::runtime_error("Unsupported InputSpec");
  }
}

std::string DataSpecUtils::label(InputSpec const& spec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return join(*concrete, "_");
  } else if (auto matcher = std::get_if<DataDescriptorMatcher>(&spec.matcher)) {
    std::ostringstream ss;
    ss << *matcher;
    std::hash<std::string> hash_fn;
    auto s = ss.str();
    auto result = std::to_string(hash_fn(s));

    return result;
  }
  throw std::runtime_error("Unsupported InputSpec");
}

std::string DataSpecUtils::restEndpoint(InputSpec const& spec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return std::string("/") + join(*concrete, "/");
  } else {
    throw std::runtime_error("Unsupported InputSpec kind");
  }
}

void DataSpecUtils::updateMatchingSubspec(InputSpec& spec, header::DataHeader::SubSpecificationType subSpec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    concrete->subSpec = subSpec;
  } else {
    throw std::runtime_error("Unsupported InputSpec kind");
  }
}

void DataSpecUtils::updateMatchingSubspec(OutputSpec& spec, header::DataHeader::SubSpecificationType subSpec)
{
  spec.subSpec = subSpec;
}

bool DataSpecUtils::validate(InputSpec const& spec)
{
  using namespace header;
  if (spec.binding.empty()) {
    return false;
  }
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return (concrete->description != DataDescription("")) && (concrete->origin != DataOrigin(""));
  }
  return true;
}

bool DataSpecUtils::match(InputSpec const& spec, ConcreteDataMatcher const& target)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return *concrete == target;
  } else if (auto matcher = std::get_if<DataDescriptorMatcher>(&spec.matcher)) {
    data_matcher::VariableContext context;
    return matcher->match(target, context);
  } else {
    throw std::runtime_error("Unsupported InputSpec");
  }
}

bool DataSpecUtils::match(InputSpec const& input, OutputSpec const& output)
{
  auto concrete = DataSpecUtils::asConcreteDataMatcher(output);
  return DataSpecUtils::match(input, concrete);
}

ConcreteDataMatcher DataSpecUtils::asConcreteDataMatcher(InputSpec const& spec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return ConcreteDataMatcher{ concrete->origin, concrete->description, concrete->subSpec };
  }
  throw std::runtime_error("Unsupported matching pattern");
}

ConcreteDataMatcher DataSpecUtils::asConcreteDataMatcher(OutputSpec const& spec)
{
  return ConcreteDataMatcher{spec.origin, spec.description, spec.subSpec};
}

} // namespace framework
} // namespace o2
