// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATAMATCHERWALKER_H_
#define O2_FRAMEWORK_DATAMATCHERWALKER_H_

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/VariantHelpers.h"

namespace o2::framework::data_matcher
{

struct EdgeActions {
  struct EnterNode {
    DataDescriptorMatcher const* node;
  };
  struct ExitNode {
    DataDescriptorMatcher const* node;
  };
  struct EnterLeft {
  };
  struct ExitLeft {
  };
  struct EnterRight {
  };
  struct ExitRight {
  };
};

enum ChildAction : int {
  VisitNone = 0,
  VisitLeft = 1,
  VisitRight = 2,
  VisitBoth = 3
};

using EdgeAction = std::variant<EdgeActions::EnterNode, EdgeActions::ExitNode>;

/// Helper class which holds methods which are useful
/// to navigate a DataDescriptorMatcher hierarchy
struct DataMatcherWalker {

  // Deep-first algorithm
  // @a top is the toplevel node in the tree.
  // @a edgeWalker is a visitor for an EdgeAction
  // @a leafWalker is a visitor for the DataDescriptorMatcher node
  template <typename EDGEWALKER, typename LEAFWALKER>
  static void walk(DataDescriptorMatcher const& top,
                   EDGEWALKER edgeWalker,
                   LEAFWALKER leafWalker)
  {
    std::vector<EdgeAction> matchers;
    matchers.push_back(EdgeActions::EnterNode{ &top });

    while (matchers.empty() == false) {
      EdgeAction action = matchers.back();
      matchers.pop_back();
      ChildAction childrenVisitor = std::visit(overloaded{
                                                 [&matchers, &edgeWalker](EdgeActions::EnterNode action) {
                                                   matchers.push_back(EdgeActions::ExitNode{ action.node });
                                                   return edgeWalker(action);
                                                 },
                                                 [&edgeWalker](EdgeActions::ExitNode action) {
                                                   edgeWalker(action);
                                                   return ChildAction::VisitNone;
                                                 } },
                                               action);

      if (childrenVisitor & ChildAction::VisitRight) {
        auto node = std::visit([](auto action) { return action.node; }, action);

        std::visit(overloaded{
                     [&matchers](std::unique_ptr<DataDescriptorMatcher> const& matcher) {
                       matchers.push_back(EdgeActions::EnterNode{ matcher.get() });
                     },
                     [edgeWalker, leafWalker](auto const& leaf) {
                       edgeWalker(EdgeActions::EnterRight{});
                       leafWalker(leaf);
                       edgeWalker(EdgeActions::ExitRight{});
                     } },
                   node->getRight());
      }

      if (childrenVisitor & ChildAction::VisitLeft) {
        auto node = std::visit([](auto action) { return action.node; }, action);

        std::visit(overloaded{
                     [&matchers](std::unique_ptr<DataDescriptorMatcher> const& matcher) {
                       matchers.push_back(EdgeActions::EnterNode{ matcher.get() });
                     },
                     [edgeWalker, leafWalker](auto const& leaf) {
                       edgeWalker(EdgeActions::EnterLeft{});
                       leafWalker(leaf);
                       edgeWalker(EdgeActions::ExitLeft{});
                     } },
                   node->getLeft());
      }
    }
  }
};

} // namespace o2::framework::data_matcher

#endif // O2_FRAMEWORK_DATAMATCHERWALKER_H_
