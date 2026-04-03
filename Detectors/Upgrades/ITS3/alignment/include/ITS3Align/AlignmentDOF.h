// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ITS3_ALIGNMENT_DOF_H
#define O2_ITS3_ALIGNMENT_DOF_H

#include <cstdint>
#include <format>
#include <algorithm>
#include <vector>

// Generic set of DOF
class DOFSet
{
 public:
  enum class Type : uint8_t { RigidBody,
                              Legendre };
  virtual ~DOFSet() = default;
  virtual Type type() const = 0;
  int nDOFs() const { return static_cast<int>(mFree.size()); }
  virtual std::string dofName(int idx) const = 0;
  bool isFree(int idx) const { return mFree[idx]; }
  void setFree(int idx, bool f) { mFree[idx] = f; }
  void setAllFree(bool f) { std::fill(mFree.begin(), mFree.end(), f); }
  int nFreeDOFs() const
  {
    int n = 0;
    for (bool f : mFree) {
      n += f;
    }
    return n;
  }

 protected:
  DOFSet(int n) : mFree(n, true) {}
  std::vector<bool> mFree;
};

// Rigid body set
class RigidBodyDOFSet final : public DOFSet
{
 public:
  // indices for rigid body parameters in LOC frame
  enum RigidBodyDOF : uint8_t {
    TX = 0,
    TY,
    TZ,
    RX,
    RY,
    RZ,
    NDOF,
  };
  static constexpr const char* RigidBodyDOFNames[RigidBodyDOF::NDOF] = {"TX", "TY", "TZ", "RX", "RY", "RZ"};

  RigidBodyDOFSet() : DOFSet(NDOF) {}
  // mask: bitmask of free DOFs (bit i = DOF i is free)
  explicit RigidBodyDOFSet(uint8_t mask) : DOFSet(NDOF)
  {
    for (int i = 0; i < NDOF; ++i) {
      mFree[i] = (mask >> i) & 1;
    }
  }
  Type type() const override { return Type::RigidBody; }
  std::string dofName(int idx) const override { return RigidBodyDOFNames[idx]; }
  uint8_t mask() const
  {
    uint8_t m = 0;
    for (int i = 0; i < NDOF; ++i) {
      m |= (uint8_t(mFree[i]) << i);
    }
    return m;
  }
};

// Legendre DOFs
// Describing radial misplacement
class LegendreDOFSet final : public DOFSet
{
 public:
  explicit LegendreDOFSet(int order) : DOFSet((order + 1) * (order + 2) / 2), mOrder(order) {}
  Type type() const override { return Type::Legendre; }
  int order() const { return mOrder; }
  std::string dofName(int idx) const override
  {
    int i = 0;
    while ((i + 1) * (i + 2) / 2 <= idx) {
      ++i;
    }
    int j = idx - (i * (i + 1) / 2);
    return std::format("L({},{})", i, j);
  }

 private:
  int mOrder;
};

#endif
