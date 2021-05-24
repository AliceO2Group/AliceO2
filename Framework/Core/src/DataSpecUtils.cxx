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
#include "Framework/DataMatcherWalker.h"
#include "Framework/VariantHelpers.h"
#include "Framework/Logger.h"
#include "Framework/RuntimeError.h"

#include <cstring>
#include <cinttypes>

namespace o2::framework
{

namespace
{
std::string join(ConcreteDataMatcher const& matcher, std::string const& sep = "_")
{
  return matcher.origin.as<std::string>() + sep + matcher.description.as<std::string>() + sep + std::to_string(matcher.subSpec);
}

std::string join(ConcreteDataTypeMatcher const& matcher, std::string const& sep = "_")
{
  return matcher.origin.as<std::string>() + sep + matcher.description.as<std::string>();
}
} // namespace

using namespace data_matcher;

bool DataSpecUtils::match(const InputSpec& spec,
                          const o2::header::DataOrigin& origin,
                          const o2::header::DataDescription& description,
                          const o2::header::DataHeader::SubSpecificationType& subSpec)
{
  ConcreteDataMatcher target{origin, description, subSpec};
  return match(spec, target);
}

bool DataSpecUtils::match(const OutputSpec& spec,
                          const o2::header::DataOrigin& origin,
                          const o2::header::DataDescription& description,
                          const o2::header::DataHeader::SubSpecificationType& subSpec)
{
  ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);

  return std::visit(overloaded{[&dataType, &origin, &description, &subSpec](ConcreteDataMatcher const& matcher) -> bool {
                                 return dataType.origin == origin &&
                                        dataType.description == description &&
                                        matcher.subSpec == subSpec;
                               },
                               [&dataType, &origin, &description](ConcreteDataTypeMatcher const& matcher) {
                                 return dataType.origin == origin &&
                                        dataType.description == description;
                               }},
                    spec.matcher);
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
  throw runtime_error("Unhandled InputSpec kind.");
}

std::string DataSpecUtils::describe(OutputSpec const& spec)
{
  return std::visit([](auto const& concrete) {
    return join(concrete, "/");
  },
                    spec.matcher);
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
    strncpy(buffer, ss.str().c_str(), size - 1);
  } else {
    throw runtime_error("Unsupported InputSpec");
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
  throw runtime_error("Unsupported InputSpec");
}

std::string DataSpecUtils::label(OutputSpec const& spec)
{
  // FIXME: unite with the InputSpec one...
  return std::visit(overloaded{[](auto const& matcher) {
                      return join(matcher, "_");
                    }},
                    spec.matcher);
}

std::string DataSpecUtils::restEndpoint(InputSpec const& spec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return std::string("/") + join(*concrete, "/");
  } else {
    throw runtime_error("Unsupported InputSpec kind");
  }
}

void DataSpecUtils::updateMatchingSubspec(InputSpec& spec, header::DataHeader::SubSpecificationType subSpec)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    concrete->subSpec = subSpec;
  } else {
    // FIXME: this will only work for the cases in which we do have a dataType defined.
    auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
    spec.matcher = ConcreteDataMatcher(dataType.origin, dataType.description, subSpec);
  }
}

void DataSpecUtils::updateMatchingSubspec(OutputSpec& spec, header::DataHeader::SubSpecificationType subSpec)
{
  std::visit(overloaded{
               [&subSpec](ConcreteDataMatcher& concrete) {
                 concrete.subSpec = subSpec;
               },
               [&spec, &subSpec](ConcreteDataTypeMatcher& dataType) {
                 spec.matcher = ConcreteDataMatcher{
                   dataType.origin,
                   dataType.description,
                   subSpec};
               },
             },
             spec.matcher);
}

bool DataSpecUtils::validate(InputSpec const& spec)
{
  using namespace header;
  if (spec.binding.empty()) {
    return false;
  }
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return (concrete->description != DataDescription("")) &&
           (concrete->description != header::gDataDescriptionInvalid) &&
           (concrete->origin != DataOrigin("")) &&
           (concrete->origin != header::gDataOriginInvalid);
  }
  return true;
}

bool DataSpecUtils::validate(OutputSpec const& spec)
{
  using namespace header;
  auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
  return (dataType.description != DataDescription("")) &&
         (dataType.description != header::gDataDescriptionInvalid) &&
         (dataType.origin != DataOrigin("")) &&
         (dataType.origin != header::gDataOriginInvalid);
}

bool DataSpecUtils::match(InputSpec const& spec, ConcreteDataTypeMatcher const& target)
{
  return std::visit(overloaded{
                      [](ConcreteDataMatcher const& matcher) {
                        // We return false because the matcher is more
                        // qualified (has subSpec) than the target.
                        return false;
                      },
                      [&target](DataDescriptorMatcher const& matcher) {
                        // FIXME: to do it properly we should actually make sure that the
                        // matcher is invariant for changes of SubSpecification. Maybe it's
                        // enough to check that none of the nodes actually match on SubSpec.
                        ConcreteDataMatcher concreteExample{
                          target.origin,
                          target.description,
                          static_cast<header::DataHeader::SubSpecificationType>(0xffffffff)};
                        data_matcher::VariableContext context;
                        return matcher.match(concreteExample, context);
                      }},
                    spec.matcher);
}

bool DataSpecUtils::match(InputSpec const& spec, ConcreteDataMatcher const& target)
{
  if (auto concrete = std::get_if<ConcreteDataMatcher>(&spec.matcher)) {
    return *concrete == target;
  } else if (auto matcher = std::get_if<DataDescriptorMatcher>(&spec.matcher)) {
    data_matcher::VariableContext context;
    return matcher->match(target, context);
  } else {
    throw runtime_error("Unsupported InputSpec");
  }
}

bool DataSpecUtils::match(OutputSpec const& spec, ConcreteDataMatcher const& target)
{
  return std::visit(overloaded{
                      [&target](ConcreteDataMatcher const& concrete) {
                        return concrete == target;
                      },
                      [&target](ConcreteDataTypeMatcher const& concrete) {
                        return concrete.origin == target.origin &&
                               concrete.description == target.description;
                      }},
                    spec.matcher);
}

bool DataSpecUtils::match(OutputSpec const& left, OutputSpec const& right)
{
  if (auto leftConcrete = std::get_if<ConcreteDataMatcher>(&left.matcher)) {
    return match(right, *leftConcrete);
  } else if (auto rightConcrete = std::get_if<ConcreteDataMatcher>(&right.matcher)) {
    return match(left, *rightConcrete);
  } else {
    // both sides are ConcreteDataTypeMatcher without subspecification, we simply specify 0
    // this is ignored in the mathing since also left hand object is ConcreteDataTypeMatcher
    ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(right);
    return match(left, dataType.origin, dataType.description, 0);
  }
}

bool DataSpecUtils::match(InputSpec const& input, OutputSpec const& output)
{
  return std::visit([&input](auto const& concrete) -> bool {
    return DataSpecUtils::match(input, concrete);
  },
                    output.matcher);
}

bool DataSpecUtils::partialMatch(OutputSpec const& output, header::DataOrigin const& origin)
{
  auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(output);
  return dataType.origin == origin;
}

bool DataSpecUtils::partialMatch(InputSpec const& input, header::DataOrigin const& origin)
{
  return DataSpecUtils::asConcreteOrigin(input) == origin;
}

bool DataSpecUtils::partialMatch(InputSpec const& input, header::DataDescription const& description)
{
  try {
    return DataSpecUtils::asConcreteDataDescription(input) == description;
  } catch (...) {
    return false;
  }
}

bool DataSpecUtils::partialMatch(OutputSpec const& output, header::DataDescription const& description)
{
  try {
    return DataSpecUtils::asConcreteDataTypeMatcher(output).description == description;
  } catch (...) {
    return false;
  }
}

struct MatcherInfo {
  header::DataOrigin origin = header::gDataOriginInvalid;                // Whether or not we found an origins (should be a bad query!)
  header::DataDescription description = header::gDataDescriptionInvalid; // Whether or not we found a description
  header::DataHeader::SubSpecificationType subSpec = 0;                  // Whether or not we found a description
  bool hasOrigin = false;
  bool hasDescription = false;
  bool hasSubSpec = false;
  bool hasUniqueOrigin = false;      // Whether the matcher involves a unique origin
  bool hasUniqueDescription = false; // Whether the matcher involves a unique origin
  bool hasUniqueSubSpec = false;
  bool hasError = false;
};

MatcherInfo extractMatcherInfo(DataDescriptorMatcher const& top)
{
  using namespace data_matcher;
  using ops = DataDescriptorMatcher::Op;

  MatcherInfo state;
  auto nodeWalker = overloaded{
    [&state](EdgeActions::EnterNode action) {
      if (state.hasError) {
        return VisitNone;
      }
      // For now we do not support extracting a type from things
      // which have an OR, so we reset all the uniqueness attributes.
      if (action.node->getOp() == ops::Or || action.node->getOp() == ops::Xor) {
        state.hasError = true;
        return VisitNone;
      }
      if (action.node->getOp() == ops::Just) {
        return VisitLeft;
      }
      return VisitBoth;
    },
    [](auto) { return VisitNone; }};

  auto leafWalker = overloaded{
    [&state](OriginValueMatcher const& valueMatcher) {
      // FIXME: If we found already more than one data origin, it means
      // we are ANDing two incompatible origins.  Until we support OR,
      // this is an error.
      // In case we find a ContextRef, it means we have
      // a wildcard, so there is not a unique origin.
      if (state.hasOrigin) {
        state.hasError = false;
        return;
      }
      state.hasOrigin = true;

      valueMatcher.visit(overloaded{
        [&state](std::string const& s) {
          strncpy(state.origin.str, s.data(), 4);
          state.hasUniqueOrigin = true;
        },
        [&state](auto) { state.hasUniqueOrigin = false; }});
    },
    [&state](DescriptionValueMatcher const& valueMatcher) {
      if (state.hasDescription) {
        state.hasError = true;
        return;
      }
      state.hasDescription = true;
      valueMatcher.visit(overloaded{
        [&state](std::string const& s) {
          strncpy(state.description.str, s.data(), 16);
          state.hasUniqueDescription = true;
        },
        [&state](auto) { state.hasUniqueDescription = false; }});
    },
    [&state](SubSpecificationTypeValueMatcher const& valueMatcher) {
      if (state.hasSubSpec) {
        state.hasError = true;
        return;
      }
      state.hasSubSpec = true;
      valueMatcher.visit(overloaded{
        [&state](uint32_t const& data) {
          state.subSpec = data;
          state.hasUniqueSubSpec = true;
        },
        [&state](auto) { state.hasUniqueSubSpec = false; }});
    },
    [](auto t) {}};
  DataMatcherWalker::walk(top, nodeWalker, leafWalker);
  return state;
}

ConcreteDataMatcher DataSpecUtils::asConcreteDataMatcher(InputSpec const& spec)
{
  return std::visit(overloaded{[](ConcreteDataMatcher const& concrete) {
                                 return concrete;
                               },
                               [&binding = spec.binding](DataDescriptorMatcher const& matcher) {
                                 auto info = extractMatcherInfo(matcher);
                                 if (info.hasOrigin && info.hasUniqueOrigin &&
                                     info.hasDescription && info.hasDescription &&
                                     info.hasSubSpec && info.hasUniqueSubSpec) {
                                   return ConcreteDataMatcher{info.origin, info.description, info.subSpec};
                                 }
                                 throw std::runtime_error("Cannot convert " + binding + " to ConcreteDataMatcher");
                               }},
                    spec.matcher);
}

ConcreteDataMatcher DataSpecUtils::asConcreteDataMatcher(OutputSpec const& spec)
{
  return std::get<ConcreteDataMatcher>(spec.matcher);
}

ConcreteDataTypeMatcher DataSpecUtils::asConcreteDataTypeMatcher(OutputSpec const& spec)
{
  return std::visit([](auto const& concrete) {
    return ConcreteDataTypeMatcher{concrete.origin, concrete.description};
  },
                    spec.matcher);
}

ConcreteDataTypeMatcher DataSpecUtils::asConcreteDataTypeMatcher(InputSpec const& spec)
{
  return std::visit(overloaded{
                      [](auto const& concrete) {
                        return ConcreteDataTypeMatcher{concrete.origin, concrete.description};
                      },
                      [](DataDescriptorMatcher const& matcher) {
                        auto state = extractMatcherInfo(matcher);
                        if (state.hasUniqueOrigin && state.hasUniqueDescription) {
                          return ConcreteDataTypeMatcher{state.origin, state.description};
                        }
                        throw runtime_error("Could not extract data type from query");
                      }},
                    spec.matcher);
}

header::DataOrigin DataSpecUtils::asConcreteOrigin(InputSpec const& spec)
{
  return std::visit(overloaded{
                      [](auto const& concrete) {
                        return concrete.origin;
                      },
                      [](DataDescriptorMatcher const& matcher) {
                        auto state = extractMatcherInfo(matcher);
                        if (state.hasUniqueOrigin) {
                          return state.origin;
                        }
                        throw runtime_error("Could not extract data type from query");
                      }},
                    spec.matcher);
}

header::DataDescription DataSpecUtils::asConcreteDataDescription(InputSpec const& spec)
{
  return std::visit(overloaded{
                      [](auto const& concrete) {
                        return concrete.description;
                      },
                      [](DataDescriptorMatcher const& matcher) {
                        auto state = extractMatcherInfo(matcher);
                        if (state.hasUniqueDescription) {
                          return state.description;
                        }
                        throw runtime_error("Could not extract data type from query");
                      }},
                    spec.matcher);
}

OutputSpec DataSpecUtils::asOutputSpec(InputSpec const& spec)
{
  return std::visit(overloaded{
                      [&spec](ConcreteDataMatcher const& concrete) {
                        return OutputSpec{{spec.binding}, concrete, spec.lifetime};
                      },
                      [&spec](DataDescriptorMatcher const& matcher) {
                        auto state = extractMatcherInfo(matcher);
                        if (state.hasUniqueOrigin && state.hasUniqueDescription && state.hasUniqueSubSpec) {
                          return OutputSpec{{spec.binding}, ConcreteDataMatcher{state.origin, state.description, state.subSpec}, spec.lifetime};
                        } else if (state.hasUniqueOrigin && state.hasUniqueDescription) {
                          return OutputSpec{{spec.binding}, ConcreteDataTypeMatcher{state.origin, state.description}, spec.lifetime};
                        }

                        throw runtime_error_f("Could not extract neither ConcreteDataMatcher nor ConcreteDataTypeMatcher from query %s", describe(spec).c_str());
                      }},
                    spec.matcher);
}

DataDescriptorMatcher DataSpecUtils::dataDescriptorMatcherFrom(ConcreteDataMatcher const& concrete)
{
  DataDescriptorMatcher matchEverything{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{concrete.origin.as<std::string>()},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{concrete.description.as<std::string>()},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{concrete.subSpec},
        std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Just,
                                                StartTimeValueMatcher{ContextRef{0}})))};
  return std::move(matchEverything);
}

DataDescriptorMatcher DataSpecUtils::dataDescriptorMatcherFrom(ConcreteDataTypeMatcher const& dataType)
{
  auto timeDescriptionMatcher = std::make_unique<DataDescriptorMatcher>(
    DataDescriptorMatcher::Op::And,
    DescriptionValueMatcher{dataType.description.as<std::string>()},
    StartTimeValueMatcher(ContextRef{0}));
  return std::move(DataDescriptorMatcher(
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{dataType.origin.as<std::string>()},
    std::move(timeDescriptionMatcher)));
}

DataDescriptorMatcher DataSpecUtils::dataDescriptorMatcherFrom(header::DataOrigin const& origin)
{
  char buf[5] = {0, 0, 0, 0, 0};
  strncpy(buf, origin.str, 4);
  DataDescriptorMatcher matchOnlyOrigin{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{buf},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ContextRef{1}},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ContextRef{2}},
        std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Just,
                                                StartTimeValueMatcher{ContextRef{0}})))};
  return std::move(matchOnlyOrigin);
}

DataDescriptorMatcher DataSpecUtils::dataDescriptorMatcherFrom(header::DataDescription const& description)
{
  char buf[17] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  strncpy(buf, description.str, 16);
  DataDescriptorMatcher matchOnlyOrigin{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ContextRef{1}},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{buf},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ContextRef{2}},
        std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Just,
                                                StartTimeValueMatcher{ContextRef{0}})))};
  return std::move(matchOnlyOrigin);
}

InputSpec DataSpecUtils::matchingInput(OutputSpec const& spec)
{
  return std::visit(overloaded{
                      [&spec](ConcreteDataMatcher const& concrete) -> InputSpec {
                        return InputSpec{
                          spec.binding.value,
                          concrete.origin,
                          concrete.description,
                          concrete.subSpec,
                          spec.lifetime};
                      },
                      [&spec](ConcreteDataTypeMatcher const& dataType) -> InputSpec {
                        auto&& matcher = DataSpecUtils::dataDescriptorMatcherFrom(dataType);
                        return InputSpec{
                          spec.binding.value,
                          std::move(matcher)};
                      }},
                    spec.matcher);
}

std::optional<header::DataOrigin> DataSpecUtils::getOptionalOrigin(InputSpec const& spec)
{
  // FIXME: try to address at least a few cases.
  return std::visit(overloaded{
                      [](ConcreteDataMatcher const& concrete) -> std::optional<header::DataOrigin> {
                        return std::make_optional(concrete.origin);
                      },
                      [](DataDescriptorMatcher const& matcher) -> std::optional<header::DataOrigin> {
                        auto state = extractMatcherInfo(matcher);
                        if (state.hasUniqueOrigin) {
                          return std::make_optional(state.origin);
                        } else if (state.hasError) {
                          throw runtime_error("Could not extract origin from query");
                        }
                        return {};
                      }},
                    spec.matcher);
}

std::optional<header::DataDescription> DataSpecUtils::getOptionalDescription(InputSpec const& spec)
{
  // FIXME: try to address at least a few cases.
  return std::visit(overloaded{
                      [](ConcreteDataMatcher const& concrete) -> std::optional<header::DataDescription> {
                        return std::make_optional(concrete.description);
                      },
                      [](DataDescriptorMatcher const& matcher) -> std::optional<header::DataDescription> {
                        auto state = extractMatcherInfo(matcher);
                        if (state.hasUniqueDescription) {
                          return std::make_optional(state.description);
                        } else if (state.hasError) {
                          throw runtime_error("Could not extract description from query");
                        }
                        return {};
                      }},
                    spec.matcher);
}

std::optional<header::DataHeader::SubSpecificationType> DataSpecUtils::getOptionalSubSpec(OutputSpec const& spec)
{
  return std::visit(overloaded{
                      [](ConcreteDataMatcher const& concrete) -> std::optional<header::DataHeader::SubSpecificationType> {
                        return std::make_optional(concrete.subSpec);
                      },
                      [](ConcreteDataTypeMatcher const&) -> std::optional<header::DataHeader::SubSpecificationType> {
                        return {};
                      }},
                    spec.matcher);
}

std::optional<header::DataHeader::SubSpecificationType> DataSpecUtils::getOptionalSubSpec(InputSpec const& spec)
{
  // FIXME: try to address at least a few cases.
  return std::visit(overloaded{
                      [](ConcreteDataMatcher const& concrete) -> std::optional<header::DataHeader::SubSpecificationType> {
                        return std::make_optional(concrete.subSpec);
                      },
                      [](DataDescriptorMatcher const& matcher) -> std::optional<header::DataHeader::SubSpecificationType> {
                        auto state = extractMatcherInfo(matcher);
                        if (state.hasUniqueSubSpec) {
                          return std::make_optional(state.subSpec);
                        } else if (state.hasError) {
                          throw runtime_error("Could not extract subSpec from query");
                        }
                        return {};
                      }},
                    spec.matcher);
}

bool DataSpecUtils::includes(const InputSpec& left, const InputSpec& right)
{
  return std::visit(
    overloaded{
      [&left](ConcreteDataMatcher const& rightMatcher) {
        return match(left, rightMatcher);
      },
      [&left](DataDescriptorMatcher const& rightMatcher) {
        auto rightInfo = extractMatcherInfo(rightMatcher);
        return std::visit(
          overloaded{
            [&rightInfo](ConcreteDataMatcher const& leftMatcher) {
              return rightInfo.hasUniqueOrigin && rightInfo.origin == leftMatcher.origin &&
                     rightInfo.hasUniqueDescription && rightInfo.description == leftMatcher.description &&
                     rightInfo.hasUniqueSubSpec && rightInfo.subSpec == leftMatcher.subSpec;
            },
            [&rightInfo](DataDescriptorMatcher const& leftMatcher) {
              auto leftInfo = extractMatcherInfo(leftMatcher);
              return (!leftInfo.hasOrigin || (rightInfo.hasOrigin && leftInfo.origin == rightInfo.origin)) &&
                     (!leftInfo.hasDescription || (rightInfo.hasDescription && leftInfo.description == rightInfo.description)) &&
                     (!leftInfo.hasSubSpec || (rightInfo.hasSubSpec && leftInfo.subSpec == rightInfo.subSpec));
            }},
          left.matcher);
      }},
    right.matcher);
}

} // namespace o2::framework
