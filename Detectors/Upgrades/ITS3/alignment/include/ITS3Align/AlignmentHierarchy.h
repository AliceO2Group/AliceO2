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

#ifndef O2_ITS3_ALIGNMENT_HIERARCHY_H
#define O2_ITS3_ALIGNMENT_HIERARCHY_H

#include <memory>
#include <compare>
#include <type_traits>
#include <map>
#include <utility>
#include <vector>
#include <ostream>
#include <string>
#include <format>
#include <algorithm>

#include <Eigen/Dense>

#include <TGeoMatrix.h>
#include <TGeoPhysicalNode.h>

namespace o2::its3::align
{
using Matrix36 = Eigen::Matrix<double, 3, 6>;
using Matrix66 = Eigen::Matrix<double, 6, 6>;

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

// return the rigid body derivatives
// trk has be at in the measurment frame
auto getRigidBodyDerivatives(const auto& trk)
{
  // calculate slopes
  const double tgl = trk.getTgl(), snp = trk.getSnp();
  const double csp = 1. / sqrt(1. + (tgl * tgl));
  const double u = trk.getY(), v = trk.getZ();
  const double uP = snp * csp, vP = tgl * csp;
  Matrix36 der;
  der.setZero();
  // columns: Tt,  Tu,  Tv,  Rt,    Ru,   Rv
  //          (X)  (Y)  (Z)  (RX)   (RY)  (RZ)
  der << uP, -1., 0., v, v * uP, -u * uP,
    vP, 0., -1., -u, v * vP, -u * vP;
  return der;
}

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

class RigidBodyDOFSet final : public DOFSet
{
 public:
  static constexpr int NDOF = RigidBodyDOF::NDOF;
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

class GlobalLabel
{
  // Millepede label is any positive integer [1....)
  // Layout: DOF(5) | CALIB(1) | ID(22) | SENS(1) | DET(2) = 31 usable bits (MSB reserved, GBL uses signed int)
 public:
  using T = uint32_t;
  static constexpr int DOF_BITS = 5;   // bits 0-4
  static constexpr int CALIB_BITS = 1; // bit 5: 0 = rigid body, 1 = calibration
  static constexpr int ID_BITS = 22;   // bits 6-27
  static constexpr int SENS_BITS = 1;  // bit 28
  static constexpr int TOTAL_BITS = sizeof(T) * 8;
  static constexpr int DET_BITS = TOTAL_BITS - (DOF_BITS + CALIB_BITS + ID_BITS + SENS_BITS) - 1; // one less bit since GBL uses int!
  static constexpr T bitMask(int b) noexcept
  {
    return (T(1) << b) - T(1);
  }
  static constexpr int DOF_SHIFT = 0;
  static constexpr T DOF_MAX = (T(1) << DOF_BITS) - T(1);
  static constexpr T DOF_MASK = DOF_MAX << DOF_SHIFT;
  static constexpr int CALIB_SHIFT = DOF_BITS;
  static constexpr T CALIB_MAX = (T(1) << CALIB_BITS) - T(1);
  static constexpr T CALIB_MASK = CALIB_MAX << CALIB_SHIFT;
  static constexpr int ID_SHIFT = DOF_BITS + CALIB_BITS;
  static constexpr T ID_MAX = (T(1) << ID_BITS) - T(1);
  static constexpr T ID_MASK = ID_MAX << ID_SHIFT;
  static constexpr int SENS_SHIFT = DOF_BITS + CALIB_BITS + ID_BITS;
  static constexpr T SENS_MAX = (T(1) << SENS_BITS) - T(1);
  static constexpr T SENS_MASK = SENS_MAX << SENS_SHIFT;
  static constexpr int DET_SHIFT = DOF_BITS + CALIB_BITS + ID_BITS + SENS_BITS;
  static constexpr T DET_MAX = (T(1) << DET_BITS) - T(1);
  static constexpr T DET_MASK = DET_MAX << DET_SHIFT;

  GlobalLabel(T det, T id, bool sens, bool calib = false)
    : mID((((id + 1) & ID_MAX) << ID_SHIFT) |
          ((det & DET_MAX) << DET_SHIFT) |
          ((T(sens) & SENS_MAX) << SENS_SHIFT) |
          ((T(calib) & CALIB_MAX) << CALIB_SHIFT))
  {
  }

  /// produce the raw Millepede label for a given DOF index (rigid body: calib=0 in label)
  constexpr T raw(T dof) const noexcept { return (mID & ~DOF_MASK) | ((dof & DOF_MAX) << DOF_SHIFT); }
  constexpr int rawGBL(T dof) const noexcept { return static_cast<int>(raw(dof)); }

  /// return a copy of this label with the CALIB bit set (for calibration DOFs on same volume)
  GlobalLabel asCalib() const noexcept
  {
    GlobalLabel c{*this};
    c.mID |= (T(1) << CALIB_SHIFT);
    return c;
  }

  constexpr T id() const noexcept { return ((mID >> ID_SHIFT) & ID_MAX) - 1; }
  constexpr T det() const noexcept { return (mID & DET_MASK) >> DET_SHIFT; }
  constexpr bool sens() const noexcept { return (mID & SENS_MASK) >> SENS_SHIFT; }
  constexpr bool calib() const noexcept { return (mID & CALIB_MASK) >> CALIB_SHIFT; }

  std::string asString() const
  {
    return std::format("Det:{} Id:{} Sens:{} Calib:{}", det(), id(), sens(), calib());
  }

  constexpr auto operator<=>(const GlobalLabel&) const noexcept = default;

 private:
  T mID{0};
};

class HierarchyConstraint
{
 public:
  HierarchyConstraint(std::string name, double value) : mName(std::move(name)), mValue(value) {}
  void add(uint32_t lab, double coeff)
  {
    mLabels.push_back(lab);
    mCoeff.push_back(coeff);
  }
  void write(std::ostream& os) const;
  auto getSize() const noexcept { return mLabels.size(); }

 private:
  std::string mName;             // name of the constraint
  double mValue{0.0};            // constraint value
  std::vector<uint32_t> mLabels; // parameter labels
  std::vector<double> mCoeff;    // their coefficients
};

// --- AlignableVolume ---

class AlignableVolume
{
 public:
  using Ptr = std::unique_ptr<AlignableVolume>;
  using SensorMapping = std::map<GlobalLabel, AlignableVolume*>;

  AlignableVolume(const AlignableVolume&) = delete;
  AlignableVolume(AlignableVolume&&) = delete;
  AlignableVolume& operator=(const AlignableVolume&) = delete;
  AlignableVolume& operator=(AlignableVolume&&) = delete;
  AlignableVolume(const char* symName, uint32_t label, uint32_t det, bool sens);
  AlignableVolume(const char* symName, GlobalLabel label);
  virtual ~AlignableVolume() = default;

  void finalise(uint8_t level = 0);

  // steering file output
  void writeRigidBodyConstraints(std::ostream& os) const;
  void writeParameters(std::ostream& os) const;
  void writeTree(std::ostream& os, int indent = 0) const;

  // tree-like
  auto getLevel() const noexcept { return mLevel; }
  bool isRoot() const noexcept { return mParent == nullptr; }
  bool isLeaf() const noexcept { return mChildren.empty(); }
  template <class T = AlignableVolume>
    requires std::derived_from<T, AlignableVolume>
  AlignableVolume* addChild(const char* symName, uint32_t label, uint32_t det, bool sens)
  {
    auto c = std::make_unique<T>(symName, label, det, sens);
    return setParent(std::move(c));
  }
  template <class T = AlignableVolume>
    requires std::derived_from<T, AlignableVolume>
  AlignableVolume* addChild(const char* symName, GlobalLabel lbl)
  {
    auto c = std::make_unique<T>(symName, lbl);
    return setParent(std::move(c));
  }

  // bfs traversal
  void traverse(const std::function<void(AlignableVolume*)>& visitor)
  {
    visitor(this);
    for (auto& c : mChildren) {
      c->traverse(visitor);
    }
  }

  std::string getSymName() const noexcept { return mSymName; }
  GlobalLabel getLabel() const noexcept { return mLabel; }
  AlignableVolume* getParent() const { return mParent; }
  size_t getNChildren() const noexcept { return mChildren.size(); }

  // DOF management
  void setRigidBody(std::unique_ptr<DOFSet> rb) { mRigidBody = std::move(rb); }
  void setCalib(std::unique_ptr<DOFSet> cal) { mCalib = std::move(cal); }
  DOFSet* getRigidBody() const { return mRigidBody.get(); }
  DOFSet* getCalib() const { return mCalib.get(); }
  void setPseudo(bool p) noexcept { mIsPseudo = p; }
  bool isPseudo() const noexcept { return mIsPseudo; }
  void setSensorId(int id) noexcept { mSensorId = id; }
  int getSensorId() const noexcept { return mSensorId; }
  // true if this volume participates in the hierarchy (has DOFs or is pseudo)
  bool isActive() const noexcept { return mRigidBody != nullptr || mIsPseudo; }

  // transformation matrices
  virtual void defineMatrixL2G() {}
  virtual void defineMatrixT2L() {}
  virtual void computeJacobianL2T(const double* pos, Matrix66& jac) const {};
  const TGeoHMatrix& getL2P() const { return mL2P; }
  const TGeoHMatrix& getT2L() const { return mT2L; }
  const Matrix66& getJL2P() const { return mJL2P; }
  const Matrix66& getJP2L() const { return mJP2L; }

 protected:
  /// matrices
  AlignableVolume* mParent{nullptr}; // parent
  TGeoPNEntry* mPNE{nullptr};        // physical entry
  TGeoPhysicalNode* mPN{nullptr};    // physical node
  TGeoHMatrix mL2G;                  // (LOC) -> (GLO)
  TGeoHMatrix mL2P;                  // (LOC) -> (PAR)
  Matrix66 mJL2P;                    // jac (LOC) -> (PAR)
  Matrix66 mJP2L;                    // jac (PAR) -> (LOC)
  TGeoHMatrix mT2L;                  // (TRK) -> (LOC)

 private:
  std::string mSymName;
  GlobalLabel mLabel;
  uint8_t mLevel{0};
  bool mIsPseudo{false};
  int mSensorId{-1};
  std::unique_ptr<DOFSet> mRigidBody;
  std::unique_ptr<DOFSet> mCalib;

  AlignableVolume* setParent(Ptr c)
  {
    c->mParent = this;
    mChildren.push_back(std::move(c));
    return mChildren.back().get();
  }
  std::vector<Ptr> mChildren; // children

  void init();
};

// apply DOF configuration from a JSON file to the hierarchy
void applyDOFConfig(AlignableVolume* root, const std::string& jsonPath);

// parse millepede.res and write result.json with fitted parameters for ITS3 half barrels
void writeMillepedeResults(AlignableVolume* root, const std::string& milleResPath, const std::string& outJsonPath, const std::string& injectedJsonPath = "");

} // namespace o2::its3::align

#endif
