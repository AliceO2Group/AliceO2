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

/// \file GRPGeomHelper.h
/// \brief Helper for geometry and GRP related CCDB requests
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_GRPGEOM_HELPER
#define ALICEO2_GRPGEOM_HELPER

#include <vector>
#include <memory>
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DataFormatsParameters/GRPMagField.h"

namespace o2::framework
{
class ProcessingContext;
class ConcreteDataMatcher;
class InputSpec;
} // namespace o2::framework

namespace o2::detectors
{
class AlignParam;
}
namespace o2::parameters
{
class GRPECSObject;
class GRPLHCIFData;
class GRPMagField;
} // namespace o2::parameters

namespace o2
{
namespace base
{
class MatLayerCylSet;

/*

 // Helper class to request CCDB condition data from the processor specs definition
 // User should request wanted objects in the device spec defintion by calling

 std::vector<InputSpec> inputs;
 ...
 auto ccdbRequest = std::make_shared<GRPGeomRequest>(..., inputs);
 // and pass it to the Device class, which must do in the constructor or init method:
 GRPGeomRequest::instance()->setRequest(ccdbRequest);

 // Then user should call the method GRPGeomRequest::instance()->finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
 // from the device finaliseCCDB method and call the method GRPGeomRequest::instance()->checkUpdates(pc) in the beginning of the run method

 I.e the task should look like:
 class MyTask {
  public:
   MyTask(std::shared_ptr<GRPGeomRequest> req, ...) : mCCDBReq(req) {
     ...
   }
   void init(o2::framework::InitContext& ic) {
     GRPGeomHelper::instance().setRequest(mCCDBReq);
     ...
   }
   void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) {
     if (GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
       return;
     }
     ...
   }
   void run(ProcessingContext& pc) {
     GRPGeomHelper::instance().checkUpdates(pc);
     ...
   }
   protected:
     std::shared_ptr<GRPGeomRequest> mCCDBReq;
 }
*/

struct GRPGeomRequest {
  enum GeomRequest { None,
                     Aligned,
                     Ideal,
                     Alignments };

  bool askGRPECS = false;
  bool askGRPLHCIF = false;
  bool askGRPMagField = false;
  bool askMatLUT = false;
  bool askTime = false;       // need orbit reset time for precise timestamp calculation
  bool askGeomAlign = false;  // load aligned geometry
  bool askGeomIdeal = false;  // load ideal geometry
  bool askAlignments = false; // load detector alignments but don't apply them
  bool askOnceAllButField = false; // for all entries but field query only once
  bool needPropagatorD = false;    // init also PropagatorD

  GRPGeomRequest() = delete;
  GRPGeomRequest(bool orbitResetTime, bool GRPECS, bool GRPLHCIF, bool GRPMagField, bool askMatLUT, GeomRequest geom, std::vector<o2::framework::InputSpec>& inputs, bool askOnce = false, bool needPropD = false);
  void addInput(const o2::framework::InputSpec&& isp, std::vector<o2::framework::InputSpec>& inputs);
};

// Helper class to process and access GRPs and geometry objects.

class GRPGeomHelper
{

 public:
  static GRPGeomHelper& instance()
  {
    static GRPGeomHelper inst;
    return inst;
  }
  void setRequest(std::shared_ptr<GRPGeomRequest> req);
  bool finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj);
  void checkUpdates(o2::framework::ProcessingContext& pc) const;

  auto getAlignment(o2::detectors::DetID det) const { return mAlignments[det]; }
  auto getMatLUT() const { return mMatLUT; }
  auto getGRPECS() const { return mGRPECS; }
  auto getGRPLHCIF() const { return mGRPLHCIF; }
  auto getGRPMagField() const { return mGRPMagField; }
  auto getOrbitResetTimeMS() const { return mOrbitResetTimeMS; }
  static int getNHBFPerTF();

 private:
  GRPGeomHelper() = default;

  std::shared_ptr<GRPGeomRequest> mRequest;

  std::array<const std::vector<o2::detectors::AlignParam>*, o2::detectors::DetID::nDetectors> mAlignments{};
  const o2::base::MatLayerCylSet* mMatLUT = nullptr;
  const o2::parameters::GRPECSObject* mGRPECS = nullptr;
  const o2::parameters::GRPLHCIFData* mGRPLHCIF = nullptr;
  const o2::parameters::GRPMagField* mGRPMagField = nullptr;
  long mOrbitResetTimeMS = 0; // orbit reset time in milliseconds
};

} // namespace base
} // namespace o2

#endif
