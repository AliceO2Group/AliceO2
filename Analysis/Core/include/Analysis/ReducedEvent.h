
#ifndef REDUCEDEVENT_H
#define REDUCEDEVENT_H

#include <TObject.h>
#include <TClonesArray.h>

class ReducedTrack;

//_________________________________________________________________________
class ReducedEvent : public TObject {

 public:
  ReducedEvent();
  virtual ~ReducedEvent();
  
  // getters
  uint64_t      EventTag()              const {return fEventTag;}
  bool          TestEventTag(short bit) const {return (bit<8*sizeof(uint64_t) ? (fEventTag&(uint64_t(1)<<bit)) : false);}
  float         Vertex(short axis)      const {return (axis>=0 && axis<=2 ? fVtx[axis] : 0);}
  float         CentVZERO()             const {return fCentVZERO;}
  ReducedTrack* GetTrack(int i)         const {return (fTracks && i>=0 && i<fTracks->GetEntries() ? (ReducedTrack*)fTracks->At(i) : 0x0);}
  TClonesArray* GetTracks()             const {return fTracks;}
  
  // setters
  void SetEventTags(uint64_t tag)      {fEventTag = tag;}
  void SetEventTag(short iflag)      {if (iflag>=8*sizeof(uint64_t)) return; fEventTag |= (uint64_t(1)<<iflag);}
  void UnsetEventTag(short iflag)    {if (iflag>=8*sizeof(uint64_t)) return; if(TestEventTag(iflag)) fEventTag^=(uint64_t(1)<<iflag);}
  void SetVertex(short axis, float value) {if(axis>=0 && axis<3) fVtx[axis] = value;}
  
  void ClearEvent();
    
 protected:
  uint64_t fEventTag;         // Event tags to be used either during analysis or to filter events
  float    fVtx[3];           // global event vertex vector in cm
  float    fCentVZERO;        // centrality; 0-V0M, 1-CL1, 2-TRK, 3-ZEMvsZDC, 4-V0A, 5-V0C, 6-ZNA
      
  TClonesArray* fTracks;            //->   array containing particles
  static TClonesArray* fgTracks;    //       global tracks

  ReducedEvent& operator= (const ReducedEvent &c);
  ReducedEvent(const ReducedEvent &c);

  ClassDef(ReducedEvent, 1);
};

#endif
