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

/// \file PadPEM.h
/// \brief Pads representation and transformation
///
/// \author Gilles Grasseau, Subatech

#ifndef O2_MCH_PADSPEM_H_
#define O2_MCH_PADSPEM_H_

#include "MCHClustering/ClusterConfig.h"

namespace o2
{
namespace mch
{
// Neigbours
static const int MaxNeighbors =
  25 + 13; // 5*5 neigbours + the center pad itself + separator (-1)
           // 5x5 neighbours + 50 %
template <class T>
inline static T getNeighborListOf(T neigh, PadIdx_t i)
{
  return &neigh[MaxNeighbors * i];
};

inline static PadIdx_t* getTheFirtsNeighborOf(PadIdx_t* neigh, PadIdx_t i)
{
  return &neigh[MaxNeighbors * i];
};
inline static PadIdx_t getTheFirstNeighborOf(PadIdx_t* neigh, PadIdx_t i)
{
  return neigh[MaxNeighbors * i];
};

class Pads
{
 public:
  enum padMode {
    xydxdyMode = 0x0,  ///< x, y, dx, dy pad coordinates
    xyInfSupMode = 0x1 ///< xInf=x, xSup=dx, yInf=y, ySup=dy pad coordinates
  };
  static constexpr double epsilonGeometry =
    1.0e-04; // Uncertainty on pad location (in cm)
  // Representation mode  (see padMode)
  int mode = xydxdyMode;

  // Utilities
  static void printNeighbors(const PadIdx_t* neigh, int N);
  static void printPads(const char* title, const Pads& pads);
  static inline int getNbrOfPads(const Pads* pads)
  {
    return (pads == nullptr) ? 0 : pads->nPads;
  };

  // Allocation constructor
  Pads(int N, int chId, int mode = xydxdyMode);
  // Build a new set of pads with different coordinates
  // xydxdy mode or xyInfSup
  Pads(const Pads& pads, int mode_);
  // Build a pads set from those selected by "mask"
  // Used to extract sub-clusters
  Pads(const Pads& pads, const Groups_t* mask);
  // Concatenate the 2 pads sets
  Pads(const Pads* pads1, const Pads* pads2, int mode);
  // Main constructor
  Pads(const double* x_, const double* y_, const double* dx_, const double* dy_,
       const double* q_, const short* cathode, const Mask_t* saturate_,
       short cathID, int chId, PadIdx_t* mapCathPadIdxToPadIdx, int nAllPads);
  Pads(const double* x_, const double* y_, const double* dx_,
       const double* dy_, const double* q_, const short* cathode,
       const Mask_t* saturate_, int chId, int nAllPads);
  // Take the ownership of coordinates (x, y, dx, dy)
  Pads(double* x_, double* y_, double* dx_, double* dy_, int chId, int nPads_);
  inline int getNbrOfPads() const { return nPads; };
  inline const double* getX() const { return x; };
  inline const double* getY() const { return y; };
  inline const double* getDX() const { return dx; };
  inline const double* getDY() const { return dy; };
  inline const double* getXInf() const { return x; };
  inline const double* getYInf() const { return y; };
  inline const double* getXSup() const { return dx; };
  inline const double* getYSup() const { return dy; };
  inline const double* getCharges() const { return q; };
  inline const Mask_t* getSaturates() const { return saturate; };
  inline const Mask_t* getCathodes() const { return cath; };
  inline double getTotalCharge() const { return totalCharge; };
  inline int getChamberId() const { return chamberId; };
  void setCharges(double c);
  void setCharges(double* q_, int n);
  // Remove pads whos charge is less than qCut
  int removePads(double qCut);
  // Charges normalization
  void normalizeCharges();
  // Split each pads in 4 smaller pads with the same sizes
  Pads* refinePads();
  // Add zero-charged pads to the neighboring of the pads (cathode cluster)
  Pads* addBoundaryPads();
  // Building Neighbors
  PadIdx_t* buildFirstNeighbors();
  // Building K-Neighbors
  PadIdx_t* buildKFirstsNeighbors(int kernelSize);
  // Extract local maximima
  Pads* extractLocalMax();
  // Extract local maximima, with of without a neighboring
  // Obsolete
  Pads* clipOnLocalMax(bool extractLocalMax);
  // Groups
  int addIsolatedPadInGroups(Mask_t* cathToGrp, Mask_t* grpToGrp, int nGroups);
  ~Pads();

 private:
  // Mode xydxdy
  double* x = nullptr;
  double* y = nullptr;
  double* dx = nullptr;
  double* dy = nullptr;
  // Mode xyInfSupMode
  // xInf = x, xSup = dx
  // yInf = y, ySup = dx
  Mask_t* cath = nullptr;
  Mask_t* saturate = nullptr;
  double* q = nullptr;
  double totalCharge = 0;
  int nPads = 0;
  int chamberId = -1;
  PadIdx_t* neighbors = nullptr;
  //
  // Memory allocation/deallocation
  void allocate();
  void release();
  // Utilities
  void removePad(int index);
  PadIdx_t* buildFirstNeighbors(double* X, double* Y, double* DX, double* DY,
                                int N);
  void setToZero();
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PADSPEM_H_
