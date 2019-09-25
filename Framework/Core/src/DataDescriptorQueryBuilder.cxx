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
#include "Framework/InputSpec.h"

#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

using namespace o2::framework::data_matcher;

namespace o2
{
namespace framework
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
  IN_ERROR
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
  std::optional<header::DataHeader::SubSpecificationType> currentSubSpec;
  std::optional<uint64_t> currentTimeModulo;
  size_t currentNumber;

  auto error = [&errorString, &states](std::string const& s) {
    errorString = s;
    states.push_back(IN_ERROR);
  };

  auto pushState = [&states](QueryBuilderState state) {
    states.push_back(state);
  };

  auto token = [&states, &expectedSeparators](QueryBuilderState state, char const* sep) {
    states.push_back(state);
    expectedSeparators = sep;
  };

  auto assignLastStringMatch = [&next, &cur, &error, &pushState, &nodes](std::string const& what, size_t maxSize, std::optional<std::string>& s, QueryBuilderState nextState) {
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

  auto assignLastNumericMatch = [&next, &cur, &error, &pushState, &currentNumber](std::string const& what, auto& value,
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

  auto buildMatchingTree = [&nodes](std::string const& binding) -> InputSpec {
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
    return InputSpec{binding, std::move(*lastMatcher.release())};
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
        currentNumber = strtoll(cur, &endptr, 10);
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
        token(IN_STRING, "/;");
      } break;
      case IN_END_ORIGIN: {
        assignLastStringMatch("origin", 4, currentOrigin, IN_BEGIN_DESCRIPTION);
        nodes.push_back(OriginValueMatcher{*currentOrigin});
      } break;
      case IN_BEGIN_DESCRIPTION: {
        pushState(IN_END_DESCRIPTION);
        token(IN_STRING, "/;");
      } break;
      case IN_END_DESCRIPTION: {
        assignLastStringMatch("description", 16, currentDescription, IN_BEGIN_SUBSPEC);
        nodes.push_back(DescriptionValueMatcher{*currentDescription});
      } break;
      case IN_BEGIN_SUBSPEC: {
        pushState(IN_END_SUBSPEC);
        token(IN_NUMBER, ";%");
      } break;
      case IN_END_SUBSPEC: {
        assignLastNumericMatch("subspec", currentSubSpec, IN_BEGIN_TIMEMODULO);
        nodes.push_back(SubSpecificationTypeValueMatcher{*currentSubSpec});
      } break;
      case IN_BEGIN_TIMEMODULO: {
        pushState(IN_END_TIMEMODULO);
        token(IN_NUMBER, ";");
      } break;
      case IN_END_TIMEMODULO: {
        assignLastNumericMatch("timemodulo", currentTimeModulo, IN_ERROR);
      } break;
      case IN_END_MATCHER: {
        if (*cur == ';' && *(cur + 1) == '\0') {
          error("Remove trailing ;");
          continue;
        }
        result.push_back(buildMatchingTree(*currentBinding));
        if (*cur == '\0') {
          pushState(IN_END_QUERY);
        } else if (*cur == ';') {
          cur += 1;
          pushState(IN_BEGIN_MATCHER);
        } else {
          error("Unexpected character" + std::string(cur, 1));
        }
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
      OriginValueMatcher{m[1]},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{m[2]},
        SubSpecificationTypeValueMatcher{m[3]}));
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

} // namespace framework
} // namespace o2
