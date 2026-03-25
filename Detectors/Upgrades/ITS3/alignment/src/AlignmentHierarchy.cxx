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

#include <format>
#include <fstream>
#include <sstream>
#include <fnmatch.h>
#include <cmath>
#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <nlohmann/json.hpp>

#include "ITS3Align/AlignmentHierarchy.h"
#include "ITSBase/GeometryTGeo.h"
#include "Framework/Logger.h"
#include "MathUtils/Utils.h"

namespace o2::its3::align
{

void HierarchyConstraint::write(std::ostream& os) const
{
  os << "!!! " << mName << '\n';
  os << "Constraint " << mValue << '\n';
  for (size_t i{0}; i < mLabels.size(); ++i) {
    os << mLabels[i] << " " << mCoeff[i] << '\n';
  }
  os << '\n';
}

AlignableVolume::AlignableVolume(const char* symName, uint32_t label, uint32_t det, bool sens) : mSymName(symName), mLabel(det, label, sens)
{
  init();
}

AlignableVolume::AlignableVolume(const char* symName, GlobalLabel label) : mSymName(symName), mLabel(label)
{
  init();
}

void AlignableVolume::init()
{
  // check if this sym volume actually exists
  mPNE = gGeoManager->GetAlignableEntry(mSymName.c_str());
  if (mPNE == nullptr) {
    LOGP(fatal, "Symbolic volume '{}' has no corresponding alignable entry!", mSymName);
  }
  mPN = mPNE->GetPhysicalNode();
  if (mPN == nullptr) {
    LOGP(debug, "Adding physical node to {}", mSymName);
    mPN = gGeoManager->MakePhysicalNode(mPNE->GetTitle());
    if (mPN == nullptr) {
      LOGP(fatal, "Failed to make physical node for {}", mSymName);
    }
  }
}

void AlignableVolume::finalise(uint8_t level)
{
  if (level == 0 && !isRoot()) {
    LOGP(fatal, "Finalise should be called only from the root node!");
  }
  mLevel = level;
  if (!isLeaf()) {
    // depth first
    for (const auto& c : mChildren) {
      c->finalise(level + 1);
    }
    // auto-disable parent RB DOFs if no children are active
    if (mRigidBody) {
      int nActiveChildren = 0;
      for (const auto& c : mChildren) {
        if (c->isActive()) {
          ++nActiveChildren;
        }
      }
      if (!nActiveChildren) {
        for (int iDOF = 0; iDOF < mRigidBody->nDOFs(); ++iDOF) {
          if (mRigidBody->isFree(iDOF)) {
            LOGP(warn, "Auto-disabling DOF {} for {} since no active children",
                 mRigidBody->dofName(iDOF), mSymName);
            mRigidBody->setFree(iDOF, false);
          }
        }
      }
    }
  } else {
    // for sensors we need also to define the transformation from the measurment (TRK) to the local frame (LOC)
    // need to it with including possible pre-alignment to allow for iterative convergence
    // (TRK) is defined wrt global z-axis
    defineMatrixL2G();
    defineMatrixT2L();
  }
  if (!isRoot()) {
    // prepare the transformation matrices, e.g. from child frame to parent frame
    // this is not necessarily just one level transformation
    TGeoHMatrix mat = *mPN->GetMatrix(); // global matrix (including possible pre-alignment) from this volume to the global frame
    if (isLeaf()) {
      mat = mL2G; // for sensor volumes they might have redefined the L2G definition
    }
    auto inv = mParent->mPN->GetMatrix()->Inverse(); // global (including possible pre-alignment) from this volume to the global frame
    mat.MultiplyLeft(inv);                           // left mult. effectively subtracts the parent transformation which is included in the the childs
    mL2P = mat;                                      // now this is directly the child to the parent transformation (LOC) (including possible pre-alignment)

    // prepare jacobian from child to parent frame
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotL2P(mL2P.GetRotationMatrix());
    Eigen::Matrix3d rotInv = rotL2P.transpose(); // parent-to-child rotation
    const double* t = mL2P.GetTranslation();     // child origin in parent frame
    Eigen::Matrix3d skewT;
    skewT << 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0;
    mJL2P.setZero();
    mJL2P.topLeftCorner<3, 3>() = rotInv;
    mJL2P.topRightCorner<3, 3>() = -rotInv * skewT;
    mJL2P.bottomRightCorner<3, 3>() = rotInv;
    mJP2L = mJL2P.inverse();
  }
}

void AlignableVolume::writeRigidBodyConstraints(std::ostream& os) const
{
  if (isLeaf() || !mRigidBody) {
    // recurse even if this node has no RB DOFs
    for (const auto& c : mChildren) {
      c->writeRigidBodyConstraints(os);
    }
    return;
  }

  for (int iDOF = 0; iDOF < mRigidBody->nDOFs(); ++iDOF) {
    if (!mRigidBody->isFree(iDOF)) {
      continue;
    }
    double nActiveChildren = 0.;
    for (const auto& c : mChildren) {
      if (c->isActive()) {
        ++nActiveChildren;
      }
    }
    if (nActiveChildren == 0.) {
      LOGP(fatal, "{} has dof {} active but no active children!", mSymName, mRigidBody->dofName(iDOF));
    }
    const double invN = 1.0 / nActiveChildren;
    HierarchyConstraint con(std::format("DOF {} for {}", mRigidBody->dofName(iDOF), mSymName), 0.0);
    for (const auto& c : mChildren) {
      if (!c->mRigidBody) {
        continue;
      }
      for (int jDOF = 0; jDOF < c->mRigidBody->nDOFs(); ++jDOF) {
        if (!c->mRigidBody->isFree(jDOF)) {
          continue;
        }
        double coeff = invN * c->getJP2L()(iDOF, jDOF);
        if (std::abs(coeff) > 1e-16f) {
          con.add(c->getLabel().raw(jDOF), coeff);
        }
      }
    }

    if (con.getSize() > 1) {
      con.write(os);
    }
  }
  for (const auto& c : mChildren) {
    c->writeRigidBodyConstraints(os);
  }
}

void AlignableVolume::writeParameters(std::ostream& os) const
{
  if (isRoot()) {
    os << "Parameter\n";
  }
  if (mRigidBody) {
    for (int iDOF = 0; iDOF < mRigidBody->nDOFs(); ++iDOF) {
      os << std::format("{:<10} {:>+15g} {:>+15g} ! {} {} ",
                        mLabel.raw(iDOF), 0.0, (mRigidBody->isFree(iDOF) ? 0.0 : -1.0),
                        (mRigidBody->isFree(iDOF) ? 'V' : 'F'), mRigidBody->dofName(iDOF))
         << mSymName << '\n';
    }
  }
  if (mCalib) {
    auto calibLbl = mLabel.asCalib();
    for (int iDOF = 0; iDOF < mCalib->nDOFs(); ++iDOF) {
      os << std::format("{:<10} {:>+15g} {:>+15g} ! {} {} ",
                        calibLbl.raw(iDOF), 0.0, (mCalib->isFree(iDOF) ? 0.0 : -1.0),
                        (mCalib->isFree(iDOF) ? 'V' : 'F'), mCalib->dofName(iDOF))
         << mSymName << '\n';
    }
  }
  for (const auto& c : mChildren) {
    c->writeParameters(os);
  }
}

void AlignableVolume::writeTree(std::ostream& os, int indent) const
{
  os << std::string(static_cast<size_t>(indent * 2), ' ') << mSymName << (mLabel.sens() ? " (sens)" : " (pasv)");
  if (mIsPseudo) {
    os << " pseudo";
  } else {
    int nFreeDofs{0};
    if (mRigidBody && mRigidBody->nFreeDOFs()) {
      nFreeDofs += mRigidBody->nFreeDOFs();
      os << " RB[";
      for (int i = 0; i < mRigidBody->nDOFs(); ++i) {
        if (mRigidBody->isFree(i)) {
          os << " " << mRigidBody->dofName(i) << "(" << mLabel.raw(i) << ")";
        }
      }
      os << " ]";
    }
    if (mCalib && mCalib->nFreeDOFs()) {
      nFreeDofs += mCalib->nFreeDOFs();
      os << " CAL[";
      auto calibLbl = mLabel.asCalib();
      for (int i = 0; i < mCalib->nDOFs(); ++i) {
        if (mCalib->isFree(i)) {
          os << " " << mCalib->dofName(i) << "(" << calibLbl.raw(i) << ")";
        }
      }
      os << " ]";
    }
    if (!nFreeDofs) {
      os << " no DOFs";
    }
  }
  os << '\n';
  for (const auto& c : mChildren) {
    c->writeTree(os, indent + 2);
  }
}

void applyDOFConfig(AlignableVolume* root, const std::string& jsonPath)
{
  using json = nlohmann::json;
  std::ifstream f(jsonPath);
  if (!f.is_open()) {
    LOGP(fatal, "Cannot open DOF config file: {}", jsonPath);
  }
  auto data = json::parse(f);
  json rules = data.is_array() ? data : data.value("rules", json::array());

  static const std::map<std::string, int> rbNameToIdx = {
    {"TX", 0}, {"TY", 1}, {"TZ", 2}, {"RX", 3}, {"RY", 4}, {"RZ", 5}};

  auto matchPattern = [](const std::string& pattern, const std::string& sym) -> bool {
    if (fnmatch(pattern.c_str(), sym.c_str(), 0) == 0) {
      return true;
    }
    std::string prefixed = "*" + pattern;
    return fnmatch(prefixed.c_str(), sym.c_str(), 0) == 0;
  };

  if (data.is_object() && data.contains("defaults")) {
    json defRule = data["defaults"];
    defRule["match"] = "*";
    rules.insert(rules.begin(), defRule);
  }

  root->traverse([&](AlignableVolume* vol) {
    const std::string& sym = vol->getSymName();
    for (const auto& rule : rules) {
      const auto pattern = rule["match"].get<std::string>();
      if (!matchPattern(pattern, sym)) {
        continue;
      }
      // rigid body DOFs
      if (rule.contains("rigidBody")) {
        const auto& rb = rule["rigidBody"];
        if (rb.is_string()) {
          auto s = rb.get<std::string>();
          if (s == "all" || s == "free") {
            vol->setRigidBody(std::make_unique<RigidBodyDOFSet>());
          } else if (s == "fixed") {
            auto dofSet = std::make_unique<RigidBodyDOFSet>();
            dofSet->setAllFree(false);
            vol->setRigidBody(std::move(dofSet));
          }
        } else if (rb.is_array()) {
          auto dofSet = std::make_unique<RigidBodyDOFSet>();
          dofSet->setAllFree(false);
          for (const auto& name : rb) {
            auto it = rbNameToIdx.find(name.get<std::string>());
            if (it != rbNameToIdx.end()) {
              dofSet->setFree(it->second, true);
            }
          }
          vol->setRigidBody(std::move(dofSet));
        } else if (rb.is_object()) {
          auto dofs = rb.value("dofs", std::string("all"));
          bool fixed = rb.value("fixed", false);
          if (dofs == "all") {
            auto dofSet = std::make_unique<RigidBodyDOFSet>();
            if (fixed) {
              dofSet->setAllFree(false);
            }
            vol->setRigidBody(std::move(dofSet));
          } else if (rb["dofs"].is_array()) {
            auto dofSet = std::make_unique<RigidBodyDOFSet>();
            dofSet->setAllFree(false);
            for (const auto& name : rb["dofs"]) {
              auto it = rbNameToIdx.find(name.get<std::string>());
              if (it != rbNameToIdx.end()) {
                dofSet->setFree(it->second, !fixed);
              }
            }
            vol->setRigidBody(std::move(dofSet));
          }
        }
      }
      // calibration DOFs
      if (rule.contains("calib")) {
        const auto& cal = rule["calib"];
        auto calType = cal.value("type", std::string(""));
        if (calType == "legendre") {
          int order = cal.value("order", 3);
          auto dofSet = std::make_unique<LegendreDOFSet>(order);
          bool fixed = cal.value("fixed", false);
          if (fixed) {
            dofSet->setAllFree(false);
          }
          // fix/free individual coefficients by name or index
          if (cal.contains("free")) {
            dofSet->setAllFree(false);
            for (const auto& item : cal["free"]) {
              if (item.is_number_integer()) {
                dofSet->setFree(item.get<int>(), true);
              } else if (item.is_string()) {
                // match by name e.g. "L(1,0)"
                for (int k = 0; k < dofSet->nDOFs(); ++k) {
                  if (dofSet->dofName(k) == item.get<std::string>()) {
                    dofSet->setFree(k, true);
                  }
                }
              }
            }
          }
          if (cal.contains("fix")) {
            for (const auto& item : cal["fix"]) {
              if (item.is_number_integer()) {
                dofSet->setFree(item.get<int>(), false);
              } else if (item.is_string()) {
                for (int k = 0; k < dofSet->nDOFs(); ++k) {
                  if (dofSet->dofName(k) == item.get<std::string>()) {
                    dofSet->setFree(k, false);
                  }
                }
              }
            }
          }
          vol->setCalib(std::move(dofSet));
        }
      }
    }
  });
}

void writeMillepedeResults(AlignableVolume* root, const std::string& milleResPath, const std::string& outJsonPath, const std::string& injectedJsonPath)
{
  using json = nlohmann::json;

  // parse millepede.res: label fittedValue presigma [...]
  std::ifstream fin(milleResPath);
  if (!fin.is_open()) {
    LOGP(fatal, "Cannot open millepede result file: {}", milleResPath);
  }
  std::map<uint32_t, double> labelToValue;
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty() || line[0] == '!' || line[0] == '*') {
      continue;
    }
    if (line.find("Parameter") != std::string::npos) {
      continue;
    }
    std::istringstream iss(line);
    uint32_t label = 0;
    double value = NAN, presigma = NAN;
    if (!(iss >> label >> value >> presigma)) {
      continue;
    }
    if (presigma >= 0.0) { // skip fixed parameters
      labelToValue[label] = value;
    }
  }
  fin.close();
  LOGP(info, "Parsed {} not fixed parameters from {}", labelToValue.size(), milleResPath);

  // load injected misalignment if provided (same format as closure test input)
  // indexed by sensorID
  std::map<int, std::vector<double>> injRB;
  std::map<int, std::vector<std::vector<double>>> injMatrix;
  if (!injectedJsonPath.empty()) {
    std::ifstream injFile(injectedJsonPath);
    if (injFile.is_open()) {
      json injData = json::parse(injFile);
      for (const auto& item : injData) {
        int id = item["id"].get<int>();
        if (item.contains("rigidBody")) {
          injRB[id] = item["rigidBody"].get<std::vector<double>>();
        }
        if (item.contains("matrix")) {
          injMatrix[id] = item["matrix"].get<std::vector<std::vector<double>>>();
        }
      }
      LOGP(info, "Loaded injected misalignment for {} sensors", injData.size());
    } else {
      LOGP(warn, "Cannot open injected misalignment file: {}, writing absolute values", injectedJsonPath);
    }
  }

  // collect results per volume that has RB or calib DOFs
  json output = json::array();
  root->traverse([&](AlignableVolume* vol) {
    auto* rb = vol->getRigidBody();
    auto* cal = vol->getCalib();
    if ((!rb && !cal) || vol->isPseudo()) {
      return;
    }
    int id = vol->getSensorId();
    json entry;
    entry["symName"] = vol->getSymName();
    entry["id"] = id;
    bool write = false;

    // rigid body parameters
    if (rb && rb->nFreeDOFs()) {
      write = true;
      json rbArr = json::array();
      const auto& inj = injRB.contains(id) ? injRB[id] : std::vector<double>{};
      for (int i = 0; i < rb->nDOFs(); ++i) {
        uint32_t raw = vol->getLabel().raw(i);
        auto it = labelToValue.find(raw);
        double fitted = it != labelToValue.end() ? it->second : 0.0;
        double ref = i < static_cast<int>(inj.size()) ? inj[i] : 0.0;
        rbArr.push_back(fitted - ref);
      }
      entry["rigidBody"] = rbArr;
    }

    // calibration (Legendre) parameters
    if (cal && cal->nFreeDOFs() && cal->type() == DOFSet::Type::Legendre) {
      write = true;
      auto* leg = dynamic_cast<const LegendreDOFSet*>(cal);
      int order = leg->order();
      auto calibLbl = vol->getLabel().asCalib();
      const auto& inj = injMatrix.contains(id) ? injMatrix[id] : std::vector<std::vector<double>>{};
      json matrix = json::array();
      int idx = 0;
      for (int i = 0; i <= order; ++i) {
        json row = json::array();
        for (int j = 0; j <= i; ++j) {
          uint32_t raw = calibLbl.raw(idx);
          auto it = labelToValue.find(raw);
          double fitted = it != labelToValue.end() ? it->second : 0.0;
          double ref = (i < static_cast<int>(inj.size()) && j < static_cast<int>(inj[i].size())) ? inj[i][j] : 0.0;
          row.push_back(fitted - ref);
          ++idx;
        }
        matrix.push_back(row);
      }
      entry["matrix"] = matrix;
    }
    if (write) {
      output.push_back(entry);
    }
  });

  std::ofstream fout(outJsonPath);
  if (!fout.is_open()) {
    LOGP(fatal, "Cannot open output file: {}", outJsonPath);
  }
  fout << output.dump(2) << '\n';
  fout.close();
  LOGP(info, "Wrote millepede results to {}", outJsonPath);
}

} // namespace o2::its3::align
