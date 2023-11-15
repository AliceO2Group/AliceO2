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
#ifndef O2_FRAMEWORK_BASICOPS_H_
#define O2_FRAMEWORK_BASICOPS_H_

namespace o2::framework
{
enum BasicOp : unsigned int {
  LogicalAnd, // 2-ar operations
  LogicalOr,
  Addition,
  Subtraction,
  Division,
  Multiplication,
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  LessThan,
  LessThanOrEqual,
  GreaterThan,
  GreaterThanOrEqual,
  Equal,
  NotEqual,
  Atan2, // 2-ar functions
  Power,
  Sqrt, // 1-ar functions
  Exp,
  Log,
  Log10,
  Sin,
  Cos,
  Tan,
  Asin,
  Acos,
  Atan,
  Abs,
  Round,
  BitwiseNot,
  Conditional // 3-ar functions
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_BASICOPS_H_
