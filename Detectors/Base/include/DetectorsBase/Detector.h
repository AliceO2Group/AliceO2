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

/// \file Detector.h
/// \brief Definition of the Detector class

#ifndef ALICEO2_BASE_DETECTOR_H_
#define ALICEO2_BASE_DETECTOR_H_

#include <map>
#include <tbb/concurrent_unordered_map.h>
#include <vector>
#include <initializer_list>
#include <memory>

#include "FairDetector.h" // for FairDetector
#include "FairRootManager.h"
#include "DetectorsBase/MaterialManager.h"
#include "Rtypes.h" // for Float_t, Int_t, Double_t, Detector::Class, etc
#include <cxxabi.h>
#include <typeinfo>
#include <type_traits>
#include <string>
#include <TMessage.h>
#include "CommonUtils/ShmManager.h"
#include "CommonUtils/ShmAllocator.h"
#include <sys/shm.h>
#include <type_traits>
#include <unistd.h>
#include <cassert>
#include <list>
#include <mutex>
#include <thread>

#include <fairmq/FwdDecls.h>

namespace o2
{
namespace base
{

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
  void Material(Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens, Float_t radl, Float_t absl,
                Float_t* buf = nullptr, Int_t nwbuf = 0);

  void Mixture(Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens, Int_t nlmat,
               Float_t* wmat);

  void Medium(Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm,
              Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil, Float_t stmin, Float_t* ubuf = nullptr,
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
  void Matrix(Int_t& nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2, Float_t theta3,
              Float_t phi3) const;

  static void setDensityFactor(Float_t density)
  {
    mDensityFactor = density;
  }

  static Float_t getDensityFactor()
  {
    return mDensityFactor;
  }

  /// implements interface of FairModule;
  /// generic implementation for O2 detectors
  void SetSpecialPhysicsCuts() override;

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
  int getMaterialID(int imat) const
  {
    auto& mgr = o2::base::MaterialManager::Instance();
    return mgr.getMaterialID(GetName(), imat);
  }

  // returns global medium ID given a "local" medium ID for this detector
  // returns -1 in case local ID not found
  int getMediumID(int imed) const
  {
    auto& mgr = o2::base::MaterialManager::Instance();
    return mgr.getMediumID(GetName(), imed);
  }

  // fill the medium index mapping into a standard vector
  // the vector gets sized properly and will be overridden
  void getMediumIDMappingAsVector(std::vector<int>& mapping)
  {
    auto& mgr = o2::base::MaterialManager::Instance();
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
  virtual void attachHits(fair::mq::Channel&, fair::mq::Parts&) = 0;
  virtual void fillHitBranch(TTree& tr, fair::mq::Parts& parts, int& index) = 0;
  virtual void collectHits(int eventID, fair::mq::Parts& parts, int& index) = 0;
  virtual void mergeHitEntriesAndFlush(int eventID,
                                       TTree& target,
                                       std::vector<int> const& trackoffsets,
                                       std::vector<int> const& nprimaries,
                                       std::vector<int> const& subevtsOrdered) = 0;

  // interface needed to merge together hit entries in TBranches (as used by hit merger process)
  // trackoffsets: a map giving the corresponding trackoffset to be applied to the trackID property when
  // merging
  virtual void mergeHitEntries(TTree& origin, TTree& target, std::vector<int> const& trackoffsets, std::vector<int> const& nprimaries, std::vector<int> const& subevtsOrdered) = 0;

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

  /// helper wrapper function to register a geometry volume given by name with FairRoot
  /// @returns The MonteCarlo ID for the volume
  int registerSensitiveVolumeAndGetVolID(std::string const& name);

  /// helper wrapper function to register a geometry volume given by TGeoVolume vol
  /// @returns The MonteCarlo ID for the volume
  int registerSensitiveVolumeAndGetVolID(TGeoVolume const* vol);

  // The GetCollection interface is made final and deprecated since
  // we no longer support TClonesArrays
  [[deprecated("Use getHits API on concrete detectors!")]] TClonesArray* GetCollection(int iColl) const final;

  // static and reusable service function to set tracking parameters in relation to field
  // returns global integration mode (inhomogenety) for the field and the max field value
  // which is required for media creation
  static void initFieldTrackingParams(int& mode, float& maxfield);

  /// set the DetID to HitBitIndex mapping. Succeeds if not already set.
  static void setDetId2HitBitIndex(std::vector<int> const& v) { Detector::sDetId2HitBitIndex = v; }
  static std::vector<int> const& getDetId2HitBitIndex() { return Detector::sDetId2HitBitIndex; }

 protected:
  Detector(const Detector& origin);

  Detector& operator=(const Detector&);

 private:
  /// Mapping of the ALICE internal material number to the one
  /// automatically assigned by geant/TGeo.
  /// This is required to easily being able to copy the geometry setup
  /// used in AliRoot
  std::map<int, int> mMapMaterial; //!< material mapping

  /// See comment for mMapMaterial
  std::map<int, int> mMapMedium; //!< medium mapping

  static Float_t mDensityFactor; //! factor that is multiplied to all material densities (ONLY for
  // systematic studies)
  static std::vector<int> sDetId2HitBitIndex; //! global lookup table keeping mapping of DetID to index in hit bit field (used in MCTrack)

  ClassDefOverride(Detector, 1); // Base class for ALICE Modules
};

/// utility function to demangle cxx type names
inline std::string demangle(const char* name)
{
  int status = -4; // some arbitrary value to eliminate compiler warnings
  std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
}

void attachShmMessage(void* hitsptr, fair::mq::Channel& channel, fair::mq::Parts& parts, bool* busy_ptr);
void* decodeShmCore(fair::mq::Parts& dataparts, int index, bool*& busy);

template <typename T>
T decodeShmMessage(fair::mq::Parts& dataparts, int index, bool*& busy)
{
  return reinterpret_cast<T>(decodeShmCore(dataparts, index, busy));
}

// this goes into the source
void attachMessageBufferToParts(fair::mq::Parts& parts, fair::mq::Channel& channel,
                                void* data, size_t size, void (*func_ptr)(void* data, void* hint), void* hint);

template <typename Container>
void attachTMessage(Container const& hits, fair::mq::Channel& channel, fair::mq::Parts& parts)
{
  TMessage* tmsg = new TMessage();
  tmsg->WriteObjectAny((void*)&hits, TClass::GetClass(typeid(hits)));
  attachMessageBufferToParts(
    parts, channel, tmsg->Buffer(), tmsg->BufferSize(),
    [](void* data, void* hint) { delete static_cast<TMessage*>(hint); }, tmsg);
}

void* decodeTMessageCore(fair::mq::Parts& dataparts, int index);
template <typename T>
T decodeTMessage(fair::mq::Parts& dataparts, int index)
{
  return static_cast<T>(decodeTMessageCore(dataparts, index));
}

void attachDetIDHeaderMessage(int id, fair::mq::Channel& channel, fair::mq::Parts& parts);

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
class DetImpl : public o2::base::Detector
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

  void attachHits(fair::mq::Channel& channel, fair::mq::Parts& parts) override
  {
    int probe = 0;
    // check if there is anything to be attached
    // at least the first hit index should return non nullptr
    if (static_cast<Det*>(this)->Det::getHits(0) == nullptr) {
      return;
    }

    attachDetIDHeaderMessage(GetDetId(), channel, parts); // the DetId s are universal as they come from o2::detector::DetID

    while (auto hits = static_cast<Det*>(this)->Det::getHits(probe++)) {
      if (!UseShm<Det>::value || !o2::utils::ShmManager::Instance().isOperational()) {
        attachTMessage(*hits, channel, parts);
      } else {
        // this is the shared mem variant
        // we will just send the sharedmem ID and the offset inside
        *mShmBusy[mCurrentBuffer] = true;
        attachShmMessage((void*)hits, channel, parts, mShmBusy[mCurrentBuffer]);
      }
    }
  }

  // this merges several entries from the TBranch brname from the origin TTree
  // into a single entry in a target TTree / same branch
  // (assuming T is typically a vector; merging is simply done by appending)
  // make this function a (static helper)
  template <typename T>
  void mergeAndAdjustHits(std::string const& brname, TTree& origin, TTree& target,
                          std::vector<int> const& trackoffsets, std::vector<int> const& nprimaries, std::vector<int> const& subevtsOrdered)
  {
    auto originbr = origin.GetBranch(brname.c_str());
    if (originbr) {
      auto targetdata = new T;
      T* incomingdata = nullptr;
      originbr->SetAddress(&incomingdata);

      T* filladdress = nullptr;
      if (origin.GetEntries() == 1) {
        originbr->GetEntry(0);
        filladdress = incomingdata;
      } else {
        Int_t entries = origin.GetEntries();
        Int_t nprimTot = 0;
        for (auto entry = 0; entry < entries; entry++) {
          nprimTot += nprimaries[entry];
        }
        // offset for pimary track index
        Int_t idelta0 = 0;
        // offset for secondary track index
        Int_t idelta1 = nprimTot;
        for (int entry = entries - 1; entry >= 0; --entry) {
          // proceed in the order of subevent Ids
          Int_t index = subevtsOrdered[entry];
          // numbe of primaries for this event
          Int_t nprim = nprimaries[index];
          idelta1 -= nprim;
          filladdress = targetdata;
          originbr->GetEntry(index);
          if (incomingdata) {
            // fix the trackIDs for this data
            for (auto& hit : *incomingdata) {
              const auto oldID = hit.GetTrackID();
              // offset depends on whether the trackis a primary or secondary
              Int_t offset = (oldID < nprim) ? idelta0 : idelta1;
              hit.SetTrackID(oldID + offset);
            }
            // this could be further generalized by using a policy for T
            std::copy(incomingdata->begin(), incomingdata->end(), std::back_inserter(*targetdata));
            delete incomingdata;
            incomingdata = nullptr;
          }
          // adjust offsets for next subevent
          idelta0 += nprim;
          idelta1 += trackoffsets[index];
        } // subevent loop
      }
      // fill target for this event
      auto targetbr = o2::base::getOrMakeBranch(target, brname.c_str(), &filladdress);
      targetbr->SetAddress(&filladdress);
      targetbr->Fill();
      targetbr->ResetAddress();
      targetdata->clear();
      if (incomingdata) {
        delete incomingdata;
        incomingdata = nullptr;
      }
      delete targetdata;
    }
  }

  // this merges several entries from temporary hit buffer into
  // into a single entry in a target TTree / same branch
  // (assuming T is typically a vector; merging is simply done by appending)
  template <typename T, typename L>
  void mergeAndAdjustHits(std::string const& brname, L& hitbuffervector, TTree& target,
                          std::vector<int> const& trackoffsets, std::vector<int> const& nprimaries,
                          std::vector<int> const& subevtsOrdered)
  {
    auto entries = hitbuffervector.size();

    auto targetdata = new T;  // used to collect data inside a single container
    T* filladdress = nullptr; // pointer used for final ROOT IO
    if (entries == 1) {
      filladdress = hitbuffervector[0].get();
      // nothing to do; we can directly do IO from the existing buffer
    } else {
      // here we need to do merging and index adjustment
      int nprimTot = 0;
      for (auto entry = 0; entry < entries; entry++) {
        nprimTot += nprimaries[entry];
      }
      // offset for pimary track index
      int idelta0 = 0;
      // offset for secondary track index
      int idelta1 = nprimTot;
      filladdress = targetdata;
      for (int entry = entries - 1; entry >= 0; --entry) {
        // proceed in the order of subevent Ids
        int index = subevtsOrdered[entry];
        // number of primaries for this event
        int nprim = nprimaries[index];
        idelta1 -= nprim;

        // fetch correct data item
        auto incomingdata = hitbuffervector[index].get();
        if (incomingdata) {
          // fix the trackIDs for this data
          for (auto& hit : *incomingdata) {
            const auto oldID = hit.GetTrackID();
            // offset depends on whether the trackis a primary or secondary
            int offset = (oldID < nprim) ? idelta0 : idelta1;
            hit.SetTrackID(oldID + offset);
          }
          // this could be further generalized by using a policy for T
          std::copy(incomingdata->begin(), incomingdata->end(), std::back_inserter(*targetdata));
        }
        // adjust offsets for next subevent
        idelta0 += nprim;
        idelta1 += trackoffsets[index];
      } // subevent loop
    }
    // fill target for this event
    auto targetbr = o2::base::getOrMakeBranch(target, brname.c_str(), &filladdress);
    targetbr->SetAddress(&filladdress);
    targetbr->Fill();
    targetbr->ResetAddress();
    targetdata->clear();
    hitbuffervector.clear();
    hitbuffervector = L(); // swap with empty vector to release mem
    delete targetdata;
  }

  void mergeHitEntries(TTree& origin, TTree& target, std::vector<int> const& trackoffsets, std::vector<int> const& nprimaries, std::vector<int> const& subevtsOrdered) final
  {
    // loop over hit containers / different branches
    // adjust trackID in hits on the go
    int probe = 0;
    using Hit_t = decltype(static_cast<Det*>(this)->Det::getHits(probe));
    std::string name = static_cast<Det*>(this)->getHitBranchNames(probe++);
    while (name.size() > 0) {
      mergeAndAdjustHits<typename std::remove_pointer<Hit_t>::type>(name, origin, target, trackoffsets, nprimaries, subevtsOrdered);
      // next name
      name = static_cast<Det*>(this)->getHitBranchNames(probe++);
    }
  }

  void mergeHitEntriesAndFlush(int eventID, TTree& target, std::vector<int> const& trackoffsets, std::vector<int> const& nprimaries, std::vector<int> const& subevtsOrdered) final
  {
    // loop over hit containers / different branches
    // adjust trackID in hits on the go
    int probe = 0;
    using Hit_t = typename std::remove_pointer<decltype(static_cast<Det*>(this)->Det::getHits(0))>::type;
    // remove buffered event from the hit store
    using Collector_t = tbb::concurrent_unordered_map<int, std::vector<std::vector<std::unique_ptr<Hit_t>>>>;
    auto hitbufferPtr = reinterpret_cast<Collector_t*>(mHitCollectorBufferPtr);
    auto iter = hitbufferPtr->find(eventID);
    if (iter == hitbufferPtr->end()) {
      LOG(error) << "No buffered hits available for event " << eventID;
      return;
    }

    std::string name = static_cast<Det*>(this)->getHitBranchNames(probe);
    while (name.size() > 0) {
      auto& vectorofHitBuffers = (*iter).second[probe];
      // flushing and buffer removal is done inside here:
      mergeAndAdjustHits<Hit_t>(name, vectorofHitBuffers, target, trackoffsets, nprimaries, subevtsOrdered);
      // next name
      probe++;
      name = static_cast<Det*>(this)->getHitBranchNames(probe);
    }
  }

 public:
  /// Collect Hits available as incoming message (shared mem or not)
  /// inside this process for later streaming to output. A function needed
  /// by the hit-merger process (not for direct use by users)
  void collectHits(int eventID, fair::mq::Parts& parts, int& index) override
  {
    using Hit_t = typename std::remove_pointer<decltype(static_cast<Det*>(this)->Det::getHits(0))>::type;
    using Collector_t = tbb::concurrent_unordered_map<int, std::vector<std::vector<std::unique_ptr<Hit_t>>>>;
    static Collector_t hitcollector; // note: we can't put this as member because
    // decltype type deduction doesn't seem to work for class members; so we use a static member
    // and will use some pointer member to communicate this data to other functions
    mHitCollectorBufferPtr = (char*)&hitcollector;

    int probe = 0;
    bool* busy = nullptr;
    using HitPtr_t = decltype(static_cast<Det*>(this)->Det::getHits(probe));
    std::string name = static_cast<Det*>(this)->getHitBranchNames(probe);

    auto copyToBuffer = [this, eventID](HitPtr_t hitdata, Collector_t& collectbuffer, int probe) {
      std::vector<std::vector<std::unique_ptr<Hit_t>>>* hitvector = nullptr;
      {
        auto eventIter = collectbuffer.find(eventID);
        if (eventIter == collectbuffer.end()) {
          // key insertion and traversal are thread-safe with tbb so no need
          // to protect
          collectbuffer[eventID] = std::vector<std::vector<std::unique_ptr<Hit_t>>>();
        }
        hitvector = &(collectbuffer[eventID]);
      }
      if (probe >= hitvector->size()) {
        hitvector->resize(probe + 1);
      }
      // add empty hit bucket to list for this event and probe
      (*hitvector)[probe].emplace_back(new Hit_t());
      // copy the data into this bucket
      *((*hitvector)[probe].back()) = *hitdata;
    };

    while (name.size() > 0) {
      if (!UseShm<Det>::value || !o2::utils::ShmManager::Instance().isOperational()) {
        // for each branch name we extract/decode hits from the message parts ...
        auto hitsptr = decodeTMessage<HitPtr_t>(parts, index++);
        if (hitsptr) {
          // ... and copy them to the buffer
          copyToBuffer(hitsptr, hitcollector, probe);
          delete hitsptr;
        }
      } else {
        // for each branch name we extract/decode hits from the message parts ...
        auto hitsptr = decodeShmMessage<HitPtr_t>(parts, index++, busy);
        // ... and copy them to the buffer
        copyToBuffer(hitsptr, hitcollector, probe);
      }
      // next name
      probe++;
      name = static_cast<Det*>(this)->getHitBranchNames(probe);
    }
    // there is only one busy flag per detector so we need to clear it only
    // at the end (after all branches have been treated)
    if (busy) {
      *busy = false;
    }
  }

  void fillHitBranch(TTree& tr, fair::mq::Parts& parts, int& index) override
  {
    int probe = 0;
    bool* busy = nullptr;
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
        auto hitsptr = decodeShmMessage<Hit_t>(parts, index++, busy);
        // ... and fill the tree branch
        auto br = getOrMakeBranch(tr, name.c_str(), hitsptr);
        br->SetAddress(static_cast<void*>(&hitsptr));
        br->Fill();
        br->ResetAddress();
      }
      // next name
      name = static_cast<Det*>(this)->getHitBranchNames(probe++);
    }
    // there is only one busy flag per detector so we need to clear it only
    // at the end (after all branches have been treated)
    if (busy) {
      *busy = false;
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

  // default implementation for setting hits
  // always returns false indicating that there is no other
  // component to assign to apart from i == 0
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
      bool more{false};
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
        LOG(info) << " BUSY WAITING SIZE ";
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
  static constexpr int NHITBUFFERS = 3;    // number of buffers for hits in order to allow async processing
                                           // in the hit merger without blocking nor copying the data
                                           // (like done in typical data aquisition systems)
  bool* mShmBusy[NHITBUFFERS] = {nullptr}; //! pointer to bool in shared mem indicating of IO busy
  std::vector<void*> mCachedPtr[NHITBUFFERS];
  int mCurrentBuffer = 0; // holding the current buffer information
  int mInitialized = false;

  char* mHitCollectorBufferPtr = nullptr; //! pointer to hit (collector) buffer location (strictly internal)

  ClassDefOverride(DetImpl, 0);
};
} // namespace base
} // namespace o2

#endif
