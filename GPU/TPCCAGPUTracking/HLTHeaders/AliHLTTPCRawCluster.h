// $Id$
#ifndef ALIHLTTPCRAWCLUSTER_H
#define ALIHLTTPCRAWCLUSTER_H

/**
 * @struct AliHLTTPCRawCluster
 * Primitive data of a TPC cluster in raw coordinates. The plan is to store the
 * data in a compressed format by limiting the resolution of the float values.
 * @ingroup alihlt_tpc_datastructs
 */
struct AliHLTTPCRawCluster {
  AliHLTTPCRawCluster()
    : fPadRow(0)
    , fFlags(0)
    , fPad(0.)
    , fTime(0.)
    , fSigmaPad2(0.)
    , fSigmaTime2(0.)
    , fCharge(0)
    , fQMax(0)
  {}

  AliHLTTPCRawCluster(short PadRow,
		      float Pad,
		      float Time,
		      float SigmaPad2,
		      float SigmaTime2,
		      unsigned short Charge,
		      unsigned short QMax,
		      unsigned short Flags
		      )
    : fPadRow(PadRow)
    , fFlags(Flags)
    , fPad(Pad)
    , fTime(Time)
    , fSigmaPad2(SigmaPad2)
    , fSigmaTime2(SigmaTime2)
    , fCharge(Charge)
    , fQMax(QMax)
  {}

  AliHLTTPCRawCluster(const AliHLTTPCRawCluster& other)
    : fPadRow(other.fPadRow)
    , fFlags(other.fFlags)
    , fPad(other.fPad)
    , fTime(other.fTime)
    , fSigmaPad2(other.fSigmaPad2)
    , fSigmaTime2(other.fSigmaTime2)
    , fCharge(other.fCharge)
    , fQMax(other.fQMax)
  {}

  AliHLTTPCRawCluster& operator=(const AliHLTTPCRawCluster& other) {
    if (this==&other) return *this;
    this->~AliHLTTPCRawCluster();
    new (this) AliHLTTPCRawCluster(other);
    return *this;
  }

  void Clear() {
    this->~AliHLTTPCRawCluster();
    new (this) AliHLTTPCRawCluster;
  }

  short fPadRow;
  unsigned short fFlags; //Flags: (1 << 0): Split in pad direction
                         //       (1 << 1): Split in time direction
                         //       (1 << 2): Edge Cluster
                         //During cluster merging, flags are OR'd
  float fPad;
  float fTime;
  float fSigmaPad2;
  float fSigmaTime2;
  unsigned short fCharge;
  unsigned short fQMax;

  int   GetPadRow()  const {return fPadRow;}
  float GetPad()     const {return fPad;}
  float GetTime()    const {return fTime;}
  float GetSigmaPad2() const {return fSigmaPad2;}
  float GetSigmaTime2() const {return fSigmaTime2;}
  int   GetCharge()  const {return fCharge;}
  int   GetQMax()    const {return fQMax;}
  bool  GetFlagSplitPad() const {return (fFlags & (1 << 0));}
  bool  GetFlagSplitTime() const {return (fFlags & (1 << 1));}
  bool  GetFlagSplitAny() const {return (fFlags & 3);}
  bool  GetFlagEdge() const {return (fFlags & (1 << 2));}
  bool  GetFlagSplitAnyOrEdge() const {return (fFlags & 7);}
  unsigned short GetFlags() const {return(fFlags);}

  void SetPadRow(short padrow)  {fPadRow=padrow;}
  void SetPad(float pad)     {fPad=pad;}
  void SetTime(float time)    {fTime=time;}
  void SetSigmaPad2(float sigmaPad2) {fSigmaPad2=sigmaPad2;}
  void SetSigmaTime2(float sigmaTime2) {fSigmaTime2=sigmaTime2;}
  void SetCharge(unsigned short charge)  {fCharge=charge;}
  void SetQMax(unsigned short qmax)    {fQMax=qmax;}

  void ClearFlags() {fFlags = 0;}
  void SetFlags(unsigned short flags) {fFlags = flags;}
  void SetFlagSplitPad() {fFlags |= (1 << 0);}
  void SetFlagSplitTime() {fFlags |= (1 << 1);}
  void SetFlagEdge() {fFlags |= (1 << 2);}
};
typedef struct AliHLTTPCRawCluster AliHLTTPCRawCluster;

struct AliHLTTPCRawClusterData
{
  unsigned int fVersion; // version number
  unsigned int fCount;   // number of clusters
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTPCRawCluster  fClusters[1]; // array of clusters
#else
  AliHLTTPCRawCluster  fClusters[0]; // array of clusters
#endif
};
typedef struct AliHLTTPCRawClusterData AliHLTTPCRawClusterData;

#endif
