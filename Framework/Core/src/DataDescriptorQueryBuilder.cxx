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

#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/InputSpec.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <iostream>

using namespace o2::framework::data_matcher;

namespace o2::framework
{

enum QueryBuilderState {
  IN_BEGIN_QUERY,
  IN_BEGIN_MATCHER,
  IN_BEGIN_BINDING,
  IN_END_BINDING,
  IN_BEGIN_ORIGIN,
  IN_END_ORIGIN,
  IN_BEGIN_DESCRIPTION,
  IN_END_DESCRIPTION,
  IN_BEGIN_SUBSPEC,
  IN_END_SUBSPEC,
  IN_BEGIN_TIMEMODULO,
  IN_END_TIMEMODULO,
  IN_END_MATCHER,
  IN_END_QUERY,
  IN_STRING,
  IN_NUMBER,
  IN_NEGATION,
  IN_ERROR,
  IN_BEGIN_ATTRIBUTES,
  IN_END_ATTRIBUTES,
  IN_BEGIN_KEY,
  IN_END_KEY,
  IN_BEGIN_VALUE,
  IN_END_VALUE
};

std::vector<InputSpec> DataDescriptorQueryBuilder::parse(char const* config)
{
  std::vector<QueryBuilderState> states;
  states.push_back(IN_BEGIN_QUERY);

  std::vector<InputSpec> result;
  char const* next = config;
  char const* cur = config;
  char const* expectedSeparators;
  char* endptr;
  std::string errorString;
  std::unique_ptr<DataDescriptorMatcher> currentMatcher;
  std::vector<data_matcher::Node> nodes;
  std::optional<std::string> currentBinding;
  std::optional<std::string> currentOrigin;
  std::optional<std::string> currentDescription;
  std::optional<std::string> currentKey;
  std::optional<std::string> currentValue;
  std::optional<header::DataHeader::SubSpecificationType> currentSubSpec;
  std::optional<uint64_t> currentTimeModulo;
  Lifetime currentLifetime = Lifetime::Timeframe;
  size_t currentNumber;

  auto error = [&errorString, &states](std::string const& s) {
    errorString = s;
    states.push_back(IN_ERROR);
  };

  auto pushState = [&states](QueryBuilderState state) {
    states.push_back(state);
  };

  auto checkModifier = [&states, &cur, &next, &error](QueryBuilderState state, const char* mod) {
    const char* modifier = nullptr;
    if (mod[0] != '\0' && (modifier = strpbrk(cur, mod)) != nullptr) {
      switch (*modifier) {
        case '!':
          states.push_back(IN_NEGATION);
          break;
        default:
          error("invalid modifier '" + std::string(modifier, 1) + "'");
          return;
      }
      next = ++cur;
    }
    states.push_back(state);
  };

  auto token = [&states, &expectedSeparators](QueryBuilderState state, char const* sep) {
    states.push_back(state);
    expectedSeparators = sep;
  };

  auto assignLastStringMatch = [&next, &cur, &error, &pushState](std::string const& what, int maxSize, std::optional<std::string>& s, QueryBuilderState nextState) {
    if ((next - cur == 0) || (next - cur > maxSize)) {
      error(what + " needs to be between 1 and " + std::to_string(maxSize) + " char long");
      return false;
    }
    s = std::string(cur, next - cur);
    if (memchr("\0;", *next, 2)) {
      pushState(IN_END_MATCHER);
      cur = next;
      return true;
    }
    pushState(nextState);
    cur = next + 1;
    return true;
  };

  auto assignLastNumericMatch = [&next, &cur, &error, &pushState, &currentNumber](std::string const&, auto& value,
                                                                                  QueryBuilderState nextState) {
    if ((next - cur == 0)) {
      error("number expected");
      return false;
    }
    value = currentNumber;
    if (memchr("\0;", *next, 2)) {
      pushState(IN_END_MATCHER);
      cur = next;
      return true;
    }
    pushState(nextState);
    cur = next + 1;
    return true;
  };

  std::vector<ConfigParamSpec> attributes;

  auto buildMatchingTree = [&nodes](std::string const& binding, std::vector<ConfigParamSpec> attributes) -> InputSpec {
    auto lastMatcher =
      std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Just,
                                              StartTimeValueMatcher(ContextRef{0}));
    for (size_t ni = 0, ne = nodes.size(); ni < ne; ++ni) {
      auto& node = nodes[nodes.size() - 1 - ni];
      auto tmp = std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::And,
                                                         std::move(node),
                                                         std::move(lastMatcher));
      assert(lastMatcher.get() == nullptr);
      lastMatcher = std::move(tmp);
    }
    Lifetime lifetime = Lifetime::Timeframe;
    for (auto& attribute : attributes) {
      if (attribute.name == "lifetime" && attribute.defaultValue.get<std::string>() == "condition") {
        lifetime = Lifetime::Condition;
      }
    }
    return InputSpec{binding, std::move(*lastMatcher.release()), lifetime, attributes};
  };

  auto pushMatcher = [&nodes, &states](auto&& matcher) {
    if (states.empty() == false && states.back() == IN_NEGATION) {
      states.pop_back();
      auto notMatcher = std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Xor,
                                                                std::move(matcher),
                                                                data_matcher::ConstantValueMatcher{true});
      nodes.push_back(std::move(notMatcher));
    } else {
      nodes.push_back(std::move(matcher));
    }
  };

  while (states.empty() == false) {
    auto const state = states.back();
    states.pop_back();

    switch (state) {
      case IN_STRING: {
        next = strpbrk(cur, expectedSeparators);
        if (next == nullptr) {
          next = cur + strlen(cur);
        }
      } break;
      case IN_NUMBER: {
        currentNumber = strtoll(cur, &endptr, 0);
        if (endptr == cur) {
          error("Expected a number");
        }
        next = endptr;
      } break;
      case IN_BEGIN_QUERY: {
        (*cur == 0) ? pushState(IN_END_QUERY)
                    : pushState(IN_BEGIN_MATCHER);
      } break;
      case IN_BEGIN_MATCHER: {
        nodes.clear();
        attributes.clear();
        pushState(IN_BEGIN_BINDING);
      } break;
      case IN_BEGIN_BINDING: {
        pushState(IN_END_BINDING);
        token(IN_STRING, ":/;,");
      } break;
      case IN_END_BINDING: {
        // We are at the end of the string already.
        // This is really an origin...
        if (strchr("\0/;", *next)) {
          pushState(IN_END_ORIGIN);
          continue;
        }
        if (next - cur == 0) {
          error("empty binding string");
          continue;
        }
        currentBinding = std::string(cur, next - cur);
        pushState(IN_BEGIN_ORIGIN);
        cur = next + 1;
      } break;
      case IN_BEGIN_ORIGIN: {
        pushState(IN_END_ORIGIN);
        token(IN_STRING, "/;?");
      } break;
      case IN_END_ORIGIN: {
        if (*next == '/' && assignLastStringMatch("origin", 4, currentOrigin, IN_BEGIN_DESCRIPTION)) {
          nodes.push_back(OriginValueMatcher{*currentOrigin});
        } else if (*next == ';' && assignLastStringMatch("origin", 4, currentOrigin, IN_END_MATCHER)) {
          nodes.push_back(OriginValueMatcher{*currentOrigin});
        } else if (*next == '?' && assignLastStringMatch("origin", 4, currentOrigin, IN_BEGIN_ATTRIBUTES)) {
          nodes.push_back(OriginValueMatcher{*currentOrigin});
        } else if (*next == '\0' && assignLastStringMatch("origin", 4, currentOrigin, IN_END_MATCHER)) {
          nodes.push_back(OriginValueMatcher{*currentOrigin});
        } else {
          error("origin needs to be between 1 and 4 char long");
        }
      } break;
      case IN_BEGIN_DESCRIPTION: {
        pushState(IN_END_DESCRIPTION);
        token(IN_STRING, "/;?");
      } break;
      case IN_END_DESCRIPTION: {
        if (*next == '/' && assignLastStringMatch("description", 16, currentDescription, IN_BEGIN_SUBSPEC)) {
          nodes.push_back(DescriptionValueMatcher{*currentDescription});
        } else if (*next == ';' && assignLastStringMatch("description", 16, currentDescription, IN_END_MATCHER)) {
          nodes.push_back(DescriptionValueMatcher{*currentDescription});
        } else if (*next == '?' && assignLastStringMatch("description", 16, currentDescription, IN_BEGIN_ATTRIBUTES)) {
          nodes.push_back(DescriptionValueMatcher{*currentDescription});
        } else if (*next == '\0' && assignLastStringMatch("description", 16, currentDescription, IN_END_MATCHER)) {
          nodes.push_back(DescriptionValueMatcher{*currentDescription});
        } else {
          error("description needs to be between 1 and 16 char long");
        }
      } break;
      case IN_BEGIN_SUBSPEC: {
        checkModifier(IN_END_SUBSPEC, "!");
        token(IN_NUMBER, ";%?");
      } break;
      case IN_END_SUBSPEC: {
        if (*next == '%' && assignLastNumericMatch("subspec", currentSubSpec, IN_BEGIN_TIMEMODULO)) {
        } else if (*next == '?' && assignLastNumericMatch("subspec", currentSubSpec, IN_BEGIN_ATTRIBUTES)) {
        } else if (*next == ';' && assignLastNumericMatch("subspec", currentSubSpec, IN_END_MATCHER)) {
        } else if (*next == '\0' && assignLastNumericMatch("subspec", currentSubSpec, IN_END_MATCHER)) {
        } else {
          error("Expected a number");
          break;
        }
        auto backup = states.back();
        states.pop_back();
        pushMatcher(SubSpecificationTypeValueMatcher{*currentSubSpec});
        states.push_back(backup);
      } break;
      case IN_BEGIN_TIMEMODULO: {
        pushState(IN_END_TIMEMODULO);
        token(IN_NUMBER, ";?");
      } break;
      case IN_END_TIMEMODULO: {
        assignLastNumericMatch("timemodulo", currentTimeModulo, IN_ERROR);
      } break;
      case IN_END_MATCHER: {
        if (*cur == ';' && *(cur + 1) == '\0') {
          error("Remove trailing ;");
          continue;
        }
        result.push_back(buildMatchingTree(*currentBinding, attributes));
        if (*cur == '\0') {
          pushState(IN_END_QUERY);
        } else if (*cur == ';') {
          cur += 1;
          pushState(IN_BEGIN_MATCHER);
        } else {
          error("Unexpected character " + std::string(cur, 1));
        }
      } break;
      case IN_BEGIN_ATTRIBUTES: {
        pushState(IN_BEGIN_KEY);
      } break;
      case IN_BEGIN_KEY: {
        pushState(IN_END_KEY);
        token(IN_STRING, "=");
      } break;
      case IN_END_KEY: {
        if (*next == '=') {
          assignLastStringMatch("key", 1000, currentKey, IN_BEGIN_VALUE);
        } else {
          error("missing value for attribute key");
        }
      } break;
      case IN_BEGIN_VALUE: {
        pushState(IN_END_VALUE);
        token(IN_STRING, "&;");
      } break;
      case IN_END_VALUE: {
        if (*next == '&') {
          assignLastStringMatch("value", 1000, currentValue, IN_BEGIN_KEY);
          if (*currentKey == "lifetime" && currentValue == "condition") {
            currentLifetime = Lifetime::Condition;
          }
          if (*currentKey == "ccdb-run-dependent" && (currentValue != "false" && currentValue != "0")) {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::Bool, true, {}});
          } else if (*currentKey == "ccdb-run-dependent" && (currentValue == "false" || currentValue == "0")) {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::Bool, false, {}});
          } else if (*currentKey == "ccdb-run-dependent") {
            error("ccdb-run-dependent can only be true or false");
          } else {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::String, *currentValue, {}});
          }
        } else if (*next == ';') {
          assignLastStringMatch("value", 1000, currentValue, IN_END_ATTRIBUTES);
          if (*currentKey == "lifetime" && currentValue == "condition") {
            currentLifetime = Lifetime::Condition;
          }
          if (*currentKey == "ccdb-run-dependent" && (currentValue != "false" && currentValue != "0")) {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::Bool, true, {}});
          } else if (*currentKey == "ccdb-run-dependent" && (currentValue == "false" || currentValue == "0")) {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::Bool, false, {}});
          } else if (*currentKey == "ccdb-run-dependent") {
            error("ccdb-run-dependent can only be true or false");
          } else {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::String, *currentValue, {}});
          }
        } else if (*next == '\0') {
          assignLastStringMatch("value", 1000, currentValue, IN_END_ATTRIBUTES);
          if (*currentKey == "lifetime" && currentValue == "condition") {
            currentLifetime = Lifetime::Condition;
          }
          if (*currentKey == "ccdb-run-dependent" && (currentValue != "false" && currentValue != "0")) {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::Bool, true, {}});
          } else if (*currentKey == "ccdb-run-dependent" && (currentValue == "false" || currentValue == "0")) {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::Bool, false, {}});
          } else if (*currentKey == "ccdb-run-dependent") {
            error("ccdb-run-dependent can only be true or false");
          } else {
            attributes.push_back(ConfigParamSpec{*currentKey, VariantType::String, *currentValue, {}});
          }
        } else {
          error("missing value for string value");
        }
      } break;
      case IN_END_ATTRIBUTES: {
        pushState(IN_END_MATCHER);
      } break;
      case IN_NEGATION: {
        error("property modifiers should have been handled before when inserting previous matcher");
      } break;
      case IN_ERROR: {
        throw std::runtime_error("Parse error: " + errorString);
      } break;
      case IN_END_QUERY: {
      } break;
      default: {
        error("Unhandled state");
      } break;
    }
  }
  return std::move(result);
}

DataDescriptorQuery DataDescriptorQueryBuilder::buildFromKeepConfig(std::string const& config)
{
  static const std::regex delim(",");

  std::sregex_token_iterator end;
  std::sregex_token_iterator iter(config.begin(),
                                  config.end(),
                                  delim,
                                  -1);

  std::unique_ptr<DataDescriptorMatcher> next, result;

  for (; iter != end; ++iter) {
    std::smatch m;
    auto s = iter->str();
    auto newNode = buildNode(s);

    if (result.get() == nullptr) {
      result = std::move(newNode);
    } else {
      next = std::move(std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Or,
                                                               std::move(result),
                                                               std::move(newNode)));
      result = std::move(next);
    }
  }

  return std::move(DataDescriptorQuery{{}, std::move(result)});
}

DataDescriptorQuery DataDescriptorQueryBuilder::buildFromExtendedKeepConfig(std::string const& config)
{
  static const std::regex delim1(",");
  static const std::regex delim2(":");

  std::sregex_token_iterator end;
  std::sregex_token_iterator iter1(config.begin(),
                                   config.end(),
                                   delim1,
                                   -1);

  std::unique_ptr<DataDescriptorMatcher> next, result;

  // looping over ','-separated items
  for (; iter1 != end; ++iter1) {
    auto s = iter1->str();

    // get first part of item
    std::sregex_token_iterator iter2(s.begin(),
                                     s.end(),
                                     delim2,
                                     -1);
    if (iter2 == end) {
      continue;
    }
    s = iter2->str();

    // create the corresponding DataDescriptorMatcher
    std::smatch m;
    auto newNode = buildNode(s);

    if (result.get() == nullptr) {
      result = std::move(newNode);
    } else {
      next = std::move(std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Or,
                                                               std::move(result),
                                                               std::move(newNode)));
      result = std::move(next);
    }
  }

  return std::move(DataDescriptorQuery{{}, std::move(result)});
}

std::unique_ptr<DataDescriptorMatcher> DataDescriptorQueryBuilder::buildNode(std::string const& nodeString)
{

  std::smatch m = getTokens(nodeString);

  std::unique_ptr<DataDescriptorMatcher> next;
  auto newNode = std::make_unique<DataDescriptorMatcher>(
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{m[1]},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{m[2]},
      SubSpecificationTypeValueMatcher{m[3]}));

  return newNode;
}

std::smatch DataDescriptorQueryBuilder::getTokens(std::string const& nodeString)
{

  static const std::regex specTokenRE(R"re((\w{1,4})/(\w{1,16})/(\d*))re");
  std::smatch m;

  std::regex_match(nodeString, m, specTokenRE);

  return m;
}

} // namespace o2::framework
