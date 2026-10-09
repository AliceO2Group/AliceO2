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

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ITS3Align/AlignmentLabel.h"

struct DerivativeContext {
  int sensorID{-1};
  int layerID{-1};
  double measX{0.};
  double measAlpha{0.};
  double measZ{0.};
  double trkY{0.};
  double trkZ{0.};
  double snp{0.};
  double tgl{0.};
  double dydx{0.};
  double dzdx{0.};
};

// Generic set of DOF
class DOFSet
{
 public:
  enum class Type : uint8_t {
    RigidBody,
    Legendre,
    Inextensional
  };
  virtual ~DOFSet() = default;
  virtual Type type() const = 0;
  int nDOFs() const { return static_cast<int>(mFree.size()); }
  virtual std::string dofName(int idx) const = 0;
  virtual void fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const = 0;
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
  DOFSet(int n) : mFree(n, true) { GlobalLabel::checkDOFCount(n); }
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
  void fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const override;
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
  void fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const override;

 private:
  int mOrder;
};

// Deformation DOFs for an open cylindrical half-shell.
//
// Inextensional part. Vanishing linear membrane strains admit the general solution (u in the local (r, phi, z)
// directions) u_z = f(phi) u_phi = -(z/r) f'(phi) + g(phi) u_r = (z/r) f''(phi) - g'(phi) with two arbitrary
// one-dimensional functions f, g. Because the shell is open in phi these are expanded in Legendre polynomials of the
// normalised azimuth u in [-1, 1]: f(phi) = sum_k f_k P_k(u), g(phi) = sum_k g_k P_k(u).
//
// Extensional part (optional). The inextensional u_r is at most linear in z, so radial deformations with curvature
// along z lie outside it. They are added as strictly radial modes u_r += sum_{k,l} h_{k,l} P_k(u) P_l(v), l >= 1, with
// v the normalised axial coordinate. l = 0 is excluded because a z-independent radial field is already spanned by the g
// family.
//
// Flat index layout: [f_0, g_0, f_1, g_1, ..., f_K, g_K, h_{0,1} ... h_{0,Lz}, h_{1,1} ... h_{Kphi,Lz}]
//
// NOTE on degeneracies: f_0 is a rigid translation along the cylinder axis and g_0 a rigid rotation about it, i.e. they
// duplicate rigid-body DOFs of the same volume.
class InextensionalDOFSet final : public DOFSet
{
 public:
  explicit InextensionalDOFSet(int maxOrder, int extOrderPhi = -1, int extOrderZ = 0)
    : DOFSet(nDOFsFor(maxOrder, extOrderPhi, extOrderZ)),
      mMaxOrder(maxOrder),
      mExtOrderPhi(extOrderZ > 0 ? extOrderPhi : -1),
      mExtOrderZ(extOrderPhi >= 0 ? extOrderZ : 0)
  {
    if (maxOrder < 1) {
      // only k = 0 is left, which is equivalent to a rigid body motion
      throw std::invalid_argument("InextensionalDOFSet requires maxOrder >= 1");
    }
    // f_0 / g_0 are rigid: fixed unless explicitly freed
    setFree(fIdx(0), false);
    setFree(gIdx(0), false);
  }

  static int nDOFsFor(int maxOrder, int extOrderPhi, int extOrderZ)
  {
    int n = 2 * (maxOrder + 1);
    if (extOrderPhi >= 0 && extOrderZ > 0) {
      n += (extOrderPhi + 1) * extOrderZ;
    }
    return n;
  }

  Type type() const override { return Type::Inextensional; }
  int maxOrder() const { return mMaxOrder; }
  int extOrderPhi() const { return mExtOrderPhi; }
  int extOrderZ() const { return mExtOrderZ; }
  bool hasExtensional() const { return mExtOrderPhi >= 0 && mExtOrderZ > 0; }

  // number of inextensional DOFs (before the radial h modes)
  int nInextensional() const { return 2 * (mMaxOrder + 1); }

  // flat indices
  static int fIdx(int k) { return 2 * k; }
  static int gIdx(int k) { return (2 * k) + 1; }
  int hIdx(int k, int l) const { return nInextensional() + (k * mExtOrderZ) + (l - 1); }

  std::string dofName(int idx) const override
  {
    if (idx < nInextensional()) {
      return std::format("{}_{}", (idx % 2 == 0) ? "f" : "g", idx / 2);
    }
    const int e = idx - nInextensional();
    return std::format("h_{}_{}", e / mExtOrderZ, (e % mExtOrderZ) + 1);
  }
  void fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const override;

 private:
  int mMaxOrder;
  int mExtOrderPhi;
  int mExtOrderZ;
};

#endif
