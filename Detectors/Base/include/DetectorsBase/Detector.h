// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.h
/// \brief Definition of the Detector class

#ifndef ALICEO2_BASE_DETECTOR_H_
#define ALICEO2_BASE_DETECTOR_H_

#include <map>
#include <vector>
#include <initializer_list>
#include <memory>

#include "FairDetector.h"  // for FairDetector
#include "FairRootManager.h"
#include "DetectorsBase/MaterialManager.h"
#include "Rtypes.h"        // for Float_t, Int_t, Double_t, Detector::Class, etc
#include <cxxabi.h>
#include <typeinfo>
#include <type_traits>
#include <string>
#include <FairMQChannel.h>
#include <FairMQMessage.h>
#include <FairMQParts.h>
#include <TMessage.h>
#include "CommonUtils/ShmManager.h"
#include "CommonUtils/ShmAllocator.h"
#include <sys/shm.h>
#include <type_traits>
#include <unistd.h>
#include <cassert>

namespace o2 {
namespace Base {

/// This is the basic class for any AliceO2 detector module, whether it is
/// sensitive or not. Detector classes depend on this.
class Detector : public FairDetector
{

  public:
    Detector(const char* name, Bool_t Active);

    /// Default Constructor
    Detector();

    /// Default Destructor
    ~Detector() override;

    // Module composition
    void Material(Int_t imat, const char *name, Float_t a, Float_t z, Float_t dens, Float_t radl, Float_t absl,
                  Float_t *buf = nullptr, Int_t nwbuf = 0);

    void Mixture(Int_t imat, const char *name, Float_t *a, Float_t *z, Float_t dens, Int_t nlmat,
                 Float_t *wmat);

    void Medium(Int_t numed, const char *name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm,
                Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil, Float_t stmin, Float_t *ubuf = nullptr,
                Int_t nbuf = 0);

    /// Custom processes and transport cuts
    void SpecialCuts(Int_t numed, const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap);
    /// Set cut by name and value
    void SpecialCut(Int_t numed, ECut parID, Float_t val);

    void SpecialProcesses(Int_t numed, const std::initializer_list<std::pair<EProc, int>>& parIDValMap);
    /// Set process by name and value
    void SpecialProcess(Int_t numed, EProc parID, int val);

    /// Define a rotation matrix. angles are in degrees.
    /// \param nmat on output contains the number assigned to the rotation matrix
    /// \param theta1 polar angle for axis I
    /// \param theta2 polar angle for axis II
    /// \param theta3 polar angle for axis III
    /// \param phi1 azimuthal angle for axis I
    /// \param phi2 azimuthal angle for axis II
    /// \param phi3 azimuthal angle for axis III
    void Matrix(Int_t &nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2, Float_t theta3,
                Float_t phi3) const;

    static void setDensityFactor(Float_t density)
    {
      mDensityFactor = density;
    }

    static Float_t getDensityFactor()
    {
      return mDensityFactor;
    }

    /// declare alignable volumes of detector
    virtual void addAlignableVolumes() const;
    
    /// Sets per wrapper volume parameters
    virtual void defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan);

    /// Books arrays for wrapper volumes
    virtual void setNumberOfWrapperVolumes(Int_t n);

    virtual void defineLayer(Int_t nlay, Double_t phi0, Double_t r, Int_t nladd, Int_t nmod,
                             Double_t lthick = 0., Double_t dthick = 0., UInt_t detType = 0, Int_t buildFlag = 0);

    virtual void defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Int_t nladd, Int_t nmod,
                                  Double_t width, Double_t tilt, Double_t lthick = 0., Double_t dthick = 0.,
                                  UInt_t detType = 0, Int_t buildFlag = 0);

    // returns global material ID given a "local" material ID for this detector
    // returns -1 in case local ID not found
    int getMaterialID(int imat) const {
      auto& mgr = o2::Base::MaterialManager::Instance();
      return mgr.getMaterialID(GetName(), imat);
    }

    // returns global medium ID given a "local" medium ID for this detector
    // returns -1 in case local ID not found
    int getMediumID(int imed) const {
      auto& mgr = o2::Base::MaterialManager::Instance();
      return mgr.getMediumID(GetName(), imed);
    }

    // fill the medium index mapping into a standard vector
    // the vector gets sized properly and will be overridden
    void getMediumIDMappingAsVector(std::vector<int>& mapping) {
      auto& mgr = o2::Base::MaterialManager::Instance();
      mgr.getMediumIDMappingAsVector(GetName(), mapping);
    }

    // return the name augmented by extension
    std::string addNameTo(const char* ext) const
    {
      std::string s(GetName());
      return s + ext;
    }

    // returning the name of the branch (corresponding to probe)
    // returns zero length string when probe not defined
    virtual std::string getHitBranchNames(int probe) const = 0;

    // interface to update track indices of data objects
    // usually called by the Stack, at the end of an event, which might have changed
    // the track indices due to filtering
    // FIXME: make private friend of stack?
    virtual void updateHitTrackIndices(std::map<int, int> const&) = 0;

    // interfaces to attach properly encoded hit information to a FairMQ message
    // and to decode it
    virtual void attachHits(FairMQChannel&, FairMQParts&) = 0;
    virtual void fillHitBranch(TTree& tr, FairMQParts& parts, int& index) = 0;

    // hook which is called automatically to custom initialize the O2 detectors
    // all initialization not able to do in constructors should be done here
    // (typically the case for geometry related stuff, etc)
    virtual void InitializeO2Detector() = 0;

    // the original FairModule/Detector virtual Initialize function
    // calls individual customized initializations and makes sure that the mother Initialize
    // is called as well. Marked final for this reason!
    void Initialize() final
    {
      InitializeO2Detector();
      // make sure the basic initialization is also done
      FairDetector::Initialize();
    }

    // a second initialization method for stuff that should be initialized late
    // (in our case after forking off from the main simulation setup
    // ... for things that should be setup in each simulation worker separately)
    virtual void initializeLate() = 0;

    // The GetCollection interface is made final and deprecated since
    // we no longer support TClonesArrays
    [[deprecated("Use getHits API on concrete detectors!")]]
    TClonesArray* GetCollection(int iColl) const final;

    // static and reusable service function to set tracking parameters in relation to field
    // returns global integration mode (inhomogenety) for the field and the max field value
    // which is required for media creation
    static void initFieldTrackingParams(int &mode, float &maxfield);

  protected:
    Detector(const Detector &origin);

    Detector &operator=(const Detector &);

  private:
    /// Mapping of the ALICE internal material number to the one
    /// automatically assigned by geant/TGeo.
    /// This is required to easily being able to copy the geometry setup
    /// used in AliRoot
    std::map<int, int> mMapMaterial; //!< material mapping

    /// See comment for mMapMaterial
    std::map<int, int> mMapMedium;   //!< medium mapping

    static Float_t mDensityFactor; //! factor that is multiplied to all material densities (ONLY for
    // systematic studies)

    ClassDefOverride(Detector, 1) // Base class for ALICE Modules
};

/// utility function to demangle cxx type names
inline std::string demangle(const char* name)
{
  int status = -4; // some arbitrary value to eliminate compiler warnings
  std::unique_ptr<char, void (*)(void*)> res{ abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free };
  return (status == 0) ? res.get() : name;
}

template <typename Container>
void attachShmMessage(Container const& hits, FairMQChannel& channel, FairMQParts& parts, bool* busy_ptr)
{
  struct shmcontext {
    int id;
    void* object_ptr;
    bool* busy_ptr;
  };

  auto& instance = o2::utils::ShmManager::Instance();
  shmcontext info{ instance.getShmID(), (void*)&hits, busy_ptr };
  LOG(DEBUG) << "-- SHM SEND --";
  LOG(INFO) << "-- SENDING -- " << hits.size() << " HITS ";
  LOG(INFO) << "-- OBJ PTR -- " << info.object_ptr << " ";
  assert(instance.isPointerOk(info.object_ptr));

  std::unique_ptr<FairMQMessage> message(channel.NewSimpleMessage(info));
  parts.AddPart(std::move(message));
}

template <typename T>
T decodeShmMessage(FairMQParts& dataparts, int index, bool*& busy)
{
  auto rawmessage = std::move(dataparts.At(index));
  struct shmcontext {
    int id;
    void* object_ptr;
    bool* busy_ptr;
  };

  shmcontext* info = (shmcontext*)rawmessage->GetData();
  LOG(DEBUG) << " GOT SHMID " << info->id;

  busy = info->busy_ptr;
  return reinterpret_cast<T>(info->object_ptr);
}

template <typename Container>
void attachTMessage(Container const& hits, FairMQChannel& channel, FairMQParts& parts)
{
  TMessage* tmsg = new TMessage();
  tmsg->WriteObjectAny((void*)&hits, TClass::GetClass(typeid(hits)));
  auto free_tmessage = [](void* data, void* hint) { delete static_cast<TMessage*>(hint); };

  std::unique_ptr<FairMQMessage> message(channel.NewMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));
  parts.AddPart(std::move(message));
}

template <typename T>
T decodeTMessage(FairMQParts& dataparts, int index)
{
  // sanity check
  if (index >= dataparts.Size()) {
    return T(0);
  }
  class TMessageWrapper : public TMessage
  {
   public:
    TMessageWrapper(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
    ~TMessageWrapper() override = default;
  };
  auto rawmessage = std::move(dataparts.At(index));
  auto message = std::make_unique<TMessageWrapper>(rawmessage->GetData(), rawmessage->GetSize());
  return static_cast<T>(message.get()->ReadObjectAny(message.get()->GetClass()));
}

template <typename T>
void attachMetaMessage(T secret, FairMQChannel& channel, FairMQParts& parts)
{
  std::unique_ptr<FairMQMessage> message(channel.NewSimpleMessage(secret));
  parts.AddPart(std::move(message));
}

template <typename T>
TBranch* getOrMakeBranch(TTree& tree, const char* brname, T* ptr)
{
  if (auto br = tree.GetBranch(brname)) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree.Branch(brname, ptr);
}

// a trait to determine if we should use shared mem or serialize using TMessage
template <typename Det>
struct UseShm {
  static constexpr bool value = false;
};

// an implementation helper template which automatically implements
// common functionality for deriving classes via the CRT pattern
// (example: it implements the updateHitTrackIndices function and avoids
// code duplication, while at the same time avoiding virtual function calls)
template <typename Det>
class DetImpl : public o2::Base::Detector
{
 public:
  // offer same constructors as base
  using Detector::Detector;

  // default implementation for getHitBranchNames
  std::string getHitBranchNames(int probe) const override
  {
    if (probe == 0) {
      return addNameTo("Hit");
    }
    return std::string(); // empty string as undefined
  }

  // generic implementation for the updateHitTrackIndices interface
  // assumes Detectors have a GetHits(int) function that return some iterable
  // hits which are o2::BaseHits
  void updateHitTrackIndices(std::map<int, int> const& indexmapping) override
  {
    int probe = 0; // some Detectors have multiple hit vectors and we are probing
                   // them via a probe integer until we get a nullptr
    while (auto hits = static_cast<Det*>(this)->Det::getHits(probe++)) {
      for (auto& hit : *hits) {
        auto iter = indexmapping.find(hit.GetTrackID());
        hit.SetTrackID(iter->second);
      }
    }
  }

  void attachHits(FairMQChannel& channel, FairMQParts& parts) override
  {
    int probe = 0;
    // check if there is anything to be attached
    // at least the first hit index should return non nullptr
    if (static_cast<Det*>(this)->Det::getHits(0) == nullptr) {
      return;
    }

    attachMetaMessage(GetDetId(), channel, parts); // the DetId s are universal as they come from o2::detector::DetID

    while (auto hits = static_cast<Det*>(this)->Det::getHits(probe++)) {
      if (!UseShm<Det>::value || !o2::utils::ShmManager::Instance().isOperational()) {
        attachTMessage(*hits, channel, parts);
      } else {
        // this is the shared mem variant
        // we will just send the sharedmem ID and the offset inside
        *mShmBusy[mCurrentBuffer] = true;
        attachShmMessage(*hits, channel, parts, mShmBusy[mCurrentBuffer]);
      }
    }
  }

 public:
  void fillHitBranch(TTree& tr, FairMQParts& parts, int& index) override
  {
    int probe = 0;
    using Hit_t = decltype(static_cast<Det*>(this)->Det::getHits(probe));
    std::string name = static_cast<Det*>(this)->getHitBranchNames(probe++);
    while (name.size() > 0) {
      if (!UseShm<Det>::value || !o2::utils::ShmManager::Instance().isOperational()) {

        // for each branch name we extract/decode hits from the message parts ...
        auto hitsptr = decodeTMessage<Hit_t>(parts, index++);
        if (hitsptr) {
          // ... and fill the tree branch
          auto br = getOrMakeBranch(tr, name.c_str(), hitsptr);
          br->SetAddress(static_cast<void*>(&hitsptr));
          br->Fill();
          br->ResetAddress();
          delete hitsptr;
        }
      } else {
        // for each branch name we extract/decode hits from the message parts ...
        bool* busy;
        auto hitsptr = decodeShmMessage<Hit_t>(parts, index++, busy);
        LOG(INFO) << "GOT " << hitsptr->size() << " HITS ";
        // ... and fill the tree branch
        auto br = getOrMakeBranch(tr, name.c_str(), hitsptr);
        br->SetAddress(static_cast<void*>(&hitsptr));
        br->Fill();
        br->ResetAddress();

        // he we are done so unset the busy flag
        *busy = false;
      }
      // next name
      name = static_cast<Det*>(this)->getHitBranchNames(probe++);
    }
  }

  // implementing CloneModule (for G4-MT mode) automatically for each deriving
  // Detector class "Det"; calls copy constructor of Det
  FairModule* CloneModule() const final
  {
    return new Det(static_cast<const Det&>(*this));
  }

  void freeHitBuffers()
  {
    using Hit_t = decltype(static_cast<Det*>(this)->Det::getHits(0));
    if (UseShm<Det>::value) {
      for (int buffer = 0; buffer < NHITBUFFERS; ++buffer) {
        for (auto ptr : mCachedPtr[buffer]) {
          o2::utils::freeSimVector(static_cast<Hit_t>(ptr));
        }
      }
    }
  }

  template <typename Hit_t>
  bool setHits(int i, std::vector<Hit_t>* ptr)
  {
    if (i == 0) {
      static_cast<Det*>(this)->Det::mHits = ptr;
    }
    return false;
  }

  // creating a number of hit buffers (in shared mem) -- to which
  // detectors can write in round-robin fashion
  void createHitBuffers()
  {
    using VectorHit_t = decltype(static_cast<Det*>(this)->Det::getHits(0));
    using Hit_t = typename std::remove_pointer<VectorHit_t>::type::value_type;
    for (int buffer = 0; buffer < NHITBUFFERS; ++buffer) {
      int probe = 0;
      bool more{ false };
      do {
        auto ptr = o2::utils::createSimVector<Hit_t>();
        more = static_cast<Det*>(this)->Det::setHits(probe, ptr);
        mCachedPtr[buffer].emplace_back(ptr);
        probe++;
      } while (more);
    }
  }

  void initializeLate() final
  {
    if (!mInitialized) {
      if (UseShm<Det>::value) {
        static_cast<Det*>(this)->Det::createHitBuffers();
        for (int b = 0; b < NHITBUFFERS; ++b) {
          auto& instance = o2::utils::ShmManager::Instance();
          mShmBusy[b] = instance.hasSegment() ? (bool*)instance.getmemblock(sizeof(bool)) : new bool;
          *mShmBusy[b] = false;
        }
      }
      mInitialized = true;
      mCurrentBuffer = 0;
    }
  }

  void BeginEvent() final
  {
    if (UseShm<Det>::value) {
      mCurrentBuffer = (mCurrentBuffer + 1) % NHITBUFFERS;
      while (mShmBusy[mCurrentBuffer] != nullptr && *mShmBusy[mCurrentBuffer]) {
        // this should ideally never happen
        LOG(INFO) << " BUSY WAITING SIZE ";
        sleep(1);
      }

      using Hit_t = decltype(static_cast<Det*>(this)->Det::getHits(0));

      // now we have to clear the hits before writing again
      int probe = 0;
      for (auto bareptr : mCachedPtr[mCurrentBuffer]) {
        auto hits = static_cast<Hit_t>(bareptr);
        // assign ..
        static_cast<Det*>(this)->Det::setHits(probe, hits);
        hits->clear();
        probe++;
      }
    }
  }

  ~DetImpl() override
  {
    for (int i = 0; i < NHITBUFFERS; ++i) {
      if (mShmBusy[i]) {
        auto& instance = o2::utils::ShmManager::Instance();
        if (instance.hasSegment()) {
          instance.freememblock(mShmBusy[i]);
        } else {
          delete mShmBusy[i];
        }
      }
    }
    freeHitBuffers();
  }

 protected:
  static constexpr int NHITBUFFERS = 3;      // number of buffers for hits in order to allow async processing
                                             // in the hit merger without blocking nor copying the data
                                             // (like done in typical data aquisition systems)
  bool* mShmBusy[NHITBUFFERS] = { nullptr }; //! pointer to bool in shared mem indicating of IO busy
  std::vector<void*> mCachedPtr[NHITBUFFERS];
  int mCurrentBuffer = 0; // holding the current buffer information
  int mInitialized = false;
  ClassDefOverride(DetImpl, 0);
};
}
}

#endif
