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
#ifndef O2_FRAMEWORK_CONDITION_H_
#define O2_FRAMEWORK_CONDITION_H_
#include <string>

namespace o2::framework
{

template <typename T>
struct Condition {
  std::string path;
  T* instance;
  using type = T;

  Condition(std::string path)
    : path(std::move(path))
  {
  }

  T* get()
  {
    return this->instance;
  }

  operator T()
  {
    return *this->instance;
  }

  T const* operator->() const
  {
    return this->instance;
  }
};

/// Can be used to group together a number of Configurables
/// to overcome the limit of 100 Configurables per task.
/// In order to do so you can do:
///
/// struct MyTask {
///   struct : ConditionGroup {
///     Condition<SomeConditionObject> someCondition{...};
///   } group;
/// };
///
/// and access it with
///
/// group.aCut;
struct ConditionGroup {
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONDITION_H_
