// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "DetectorsBase/Detector.h"
#include <TVirtualMC.h> // for TVirtualMC, gMC
#include "DetectorsBase/MaterialManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Field/MagneticField.h"
#include "TString.h" // for TString
#include "TGeoManager.h"

using std::cout;
using std::endl;
using std::fstream;
using std::ios;
using std::ostream;

using namespace o2::base;
using namespace o2::detectors;

Float_t Detector::mDensityFactor = 1.0;

Detector::Detector() : FairDetector(), mMapMaterial(), mMapMedium() {}
Detector::Detector(const char* name, Bool_t Active)
  : FairDetector(name, Active, DetID(name)), mMapMaterial(), mMapMedium()
{
}

Detector::Detector(const Detector& rhs) = default;

Detector::~Detector() = default;

Detector& Detector::operator=(const Detector& rhs)
{
  // check assignment to self
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  FairDetector::operator=(rhs);

  return *this;
}

void Detector::Material(Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens, Float_t radl, Float_t absl,
                        Float_t* buf, Int_t nwbuf)
{
  auto& mgr = o2::base::MaterialManager::Instance();
  mgr.Material(GetName(), imat, name, a, z, dens, radl, absl, buf, nwbuf);
}

void Detector::Mixture(Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens, Int_t nlmat, Float_t* wmat)
{
  auto& mgr = o2::base::MaterialManager::Instance();
  mgr.Mixture(GetName(), imat, name, a, z, dens, nlmat, wmat);
}

void Detector::Medium(Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm,
                      Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil, Float_t stmin, Float_t* ubuf,
                      Int_t nbuf)
{
  auto& mgr = o2::base::MaterialManager::Instance();
  mgr.Medium(GetName(), numed, name, nmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin, ubuf, nbuf);
}

void Detector::SpecialCuts(Int_t numed, const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialCuts(GetName(), numed, parIDValMap);
}

void Detector::SpecialCut(Int_t numed, ECut parID, Float_t val)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialCut(GetName(), numed, parID, val);
}

void Detector::SpecialProcesses(Int_t numed, const std::initializer_list<std::pair<EProc, int>>& parIDValMap)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialProcesses(GetName(), numed, parIDValMap);
}

void Detector::SpecialProcess(Int_t numed, EProc parID, int val)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialProcess(GetName(), numed, parID, val);
}

void Detector::Matrix(Int_t& nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2, Float_t theta3,
                      Float_t phi3) const
{
  TVirtualMC::GetMC()->Matrix(nmat, theta1, phi1, theta2, phi2, theta3, phi3);
}

void Detector::defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan) {}
void Detector::setNumberOfWrapperVolumes(Int_t n) {}
void Detector::defineLayer(const Int_t nlay, const double phi0, const Double_t r, const Int_t nladd, const Int_t nmod,
                           const Double_t lthick, const Double_t dthick, const UInt_t dettypeID, const Int_t buildLevel)
{
}

void Detector::defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Int_t nladd, Int_t nmod, Double_t width,
                                Double_t tilt, Double_t lthick, Double_t dthick, UInt_t dettypeID, Int_t buildLevel)
{
}

void Detector::SetSpecialPhysicsCuts()
{
  // default implementation for physics cuts setting (might still be overriden by detectors)
  // we try to read an external text file supposed to be installed
  // in a standard directory
  // ${O2_ROOT}/share/Detectors/DETECTORNAME/simulation/data/simcuts.dat
  LOG(INFO) << "Setting special cuts for " << GetName();
  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputFile;
  if (aliceO2env) {
    inputFile = std::string(aliceO2env);
  }
  inputFile += "/share/Detectors/" + std::string(GetName()) + "/simulation/data/simcuts.dat";
  auto& matmgr = o2::base::MaterialManager::Instance();
  matmgr.loadCutsAndProcessesFromFile(GetName(), inputFile.c_str());

  // TODO:
  // foresee possibility to read from local (non-installed) file or
  // via command line
}

void Detector::initFieldTrackingParams(int& integration, float& maxfield)
{
  // set reasonable default values
  integration = 2;
  maxfield = 10;
  auto vmc = TVirtualMC::GetMC();
  if (vmc) {
    auto field = vmc->GetMagField();
    // see if we can query the o2 field
    if (auto o2field = dynamic_cast<o2::field::MagneticField*>(field)) {
      integration = o2field->Integral(); // default integration method?
      maxfield = o2field->Max();
      return;
    }
  }
  LOG(INFO) << "No magnetic field found; using default tracking values " << integration << " " << maxfield
            << " to initialize media\n";
}

TClonesArray* Detector::GetCollection(int) const
{
  LOG(WARNING) << "GetCollection interface no longer supported";
  LOG(WARNING) << "Use the GetHits function on invidiual detectors";
  return nullptr;
}

void Detector::addAlignableVolumes() const
{
  LOG(WARNING) << "Alignable volumes are not yet defined for " << GetName();
}

int Detector::registerSensitiveVolumeAndGetVolID(TGeoVolume const* vol)
{
  // register this volume with FairRoot
  this->FairModule::AddSensitiveVolume(const_cast<TGeoVolume*>(vol));
  // retrieve the VMC Monte Carlo ID for this volume
  const int volid = TVirtualMC::GetMC()->VolId(vol->GetName());
  if (volid <= 0) {
    LOG(ERROR) << "Could not retrieve VMC volume ID for " << vol->GetName();
  }
  return volid;
}

int Detector::registerSensitiveVolumeAndGetVolID(std::string const& name)
{
  // we need to fetch the TGeoVolume which is needed for FairRoot
  auto vol = gGeoManager->GetVolume(name.c_str());
  if (!vol) {
    LOG(ERROR) << "Volume " << name << " not found in geometry; Cannot register sensitive volume";
    return -1;
  }
  return registerSensitiveVolumeAndGetVolID(vol);
}

#include <FairMQMessage.h>
#include <FairMQParts.h>
#include <FairMQChannel.h>
namespace o2
{
namespace base
{
// this goes into the source
void attachMessageBufferToParts(FairMQParts& parts, FairMQChannel& channel, void* data, size_t size,
                                void (*free_func)(void* data, void* hint), void* hint)
{
  std::unique_ptr<FairMQMessage> message(channel.NewMessage(data, size, free_func, hint));
  parts.AddPart(std::move(message));
}
void attachDetIDHeaderMessage(int id, FairMQChannel& channel, FairMQParts& parts)
{
  std::unique_ptr<FairMQMessage> message(channel.NewSimpleMessage(id));
  parts.AddPart(std::move(message));
}
void attachShmMessage(void* hits_ptr, FairMQChannel& channel, FairMQParts& parts, bool* busy_ptr)
{
  struct shmcontext {
    int id;
    void* object_ptr;
    bool* busy_ptr;
  };

  auto& instance = o2::utils::ShmManager::Instance();
  shmcontext info{instance.getShmID(), hits_ptr, busy_ptr};
  LOG(DEBUG) << "-- SHM SEND --";
  LOG(INFO) << "-- OBJ PTR -- " << info.object_ptr << " ";
  assert(instance.isPointerOk(info.object_ptr));

  std::unique_ptr<FairMQMessage> message(channel.NewSimpleMessage(info));
  parts.AddPart(std::move(message));
}
void* decodeShmCore(FairMQParts& dataparts, int index, bool*& busy)
{
  auto rawmessage = std::move(dataparts.At(index));
  struct shmcontext {
    int id;
    void* object_ptr;
    bool* busy_ptr;
  };

  shmcontext* info = (shmcontext*)rawmessage->GetData();

  busy = info->busy_ptr;
  return info->object_ptr;
}

void* decodeTMessageCore(FairMQParts& dataparts, int index)
{
  class TMessageWrapper : public TMessage
  {
   public:
    TMessageWrapper(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
    ~TMessageWrapper() override = default;
  };
  auto rawmessage = std::move(dataparts.At(index));
  auto message = std::make_unique<TMessageWrapper>(rawmessage->GetData(), rawmessage->GetSize());
  return message.get()->ReadObjectAny(message.get()->GetClass());
}

} // namespace base
} // namespace o2
ClassImp(o2::base::Detector);
