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
#include <utility>
#include <vector>
#include <ostream>
#include <string>
#include <map>
#include <algorithm>

#include <Eigen/Dense>

#include <TGeoMatrix.h>
#include <TGeoPhysicalNode.h>

#include "ITS3Align/AlignmentLabel.h"
#include "ITS3Align/AlignmentDOF.h"

namespace o2::its3::align
{
using Matrix36 = Eigen::Matrix<double, 3, 6>;
using Matrix66 = Eigen::Matrix<double, 6, 6>;

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
