// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.h
/// \brief Cluster structure for TPC
#ifndef ALICEO2_TPC_CLUSTER_H
#define ALICEO2_TPC_CLUSTER_H

#include <Rtypes.h>
#include <CommonDataFormat/TimeStamp.h>
#include <boost/serialization/base_object.hpp> // for base_object

namespace boost
{
namespace serialization
{
class access;
}
} // namespace boost

namespace o2
{
namespace TPC
{
using ClusterTimeStamp = o2::dataformats::TimeStampWithError<float, float>;

/// \class Cluster
/// \brief Cluster class for the TPC
///
class Cluster : public ClusterTimeStamp
{
 public:
  /// Default constructor
  Cluster();

  /// Constructor, initializing all values
  /// \param cru CRU
  /// \param row Row
  /// \param q Total charge of cluster
  /// \param qmax Maximum charge in a single cell (pad, time)
  /// \param padmean Mean position of cluster in pad direction
  /// \param padsigma Sigma of cluster in pad direction
  /// \param timemean Mean position of cluster in time direction
  /// \param timesigma Sigma of cluster in time direction
  Cluster(short cru, short row, float q, float qmax, float padmean, float padsigma, float timemean, float timesigma);

  /// Destructor
  ~Cluster() = default;

  /// Copy Constructor
  Cluster(const Cluster& other);

  /// Set parameters of cluster
  /// \param cru CRU
  /// \param row Row
  /// \param q Total charge of cluster
  /// \param qmax Maximum charge in a single cell (pad, time)
  /// \param padmean Mean position of cluster in pad direction
  /// \param padsigma Sigma of cluster in pad direction
  /// \param timemean Mean position of cluster in time direction
  /// \param timesigma Sigma of cluster in time direction
  void setParameters(short cru, short row, float q, float qmax, float padmean, float padsigma, float timemean,
                     float timesigma);

  int getCRU() const { return mCRU; }
  int getRow() const { return mRow; }
  float getQ() const { return mQ; }
  float getQmax() const { return mQmax; }
  float getPadMean() const { return mPadMean; }
  float getTimeMean() const { return getTimeStamp(); }
  float getPadSigma() const { return mPadSigma; }
  float getTimeSigma() const { return getTimeStampError(); }

  /// Print function: Print basic digit information on the  output stream
  /// \param output Stream to put the digit on
  /// \return The output stream
  friend std::ostream& operator<<(std::ostream& out, const Cluster& c) { return c.print(out); };

 protected:
  std::ostream& print(std::ostream& out) const;

 private:
  friend class boost::serialization::access;

  short mCRU;
  short mRow;
  float mQ;
  float mQmax;
  float mPadMean;
  float mPadSigma;

  ClassDefNV(Cluster, 1);
};
//________________________________________________________________________
inline Cluster::Cluster() : Cluster(-1, -1, -1, -1, -1, -1, -1, -1) {}

//________________________________________________________________________
inline Cluster::Cluster(short cru, short row, float q, float qmax, float padmean, float padsigma, float timemean,
                        float timesigma)
  : ClusterTimeStamp(timemean, timesigma),
    mCRU(cru),
    mRow(row),
    mQ(q),
    mQmax(qmax),
    mPadMean(padmean),
    mPadSigma(padsigma)
{
}

//________________________________________________________________________
inline Cluster::Cluster(const Cluster& other)
  : ClusterTimeStamp(other),
    mCRU(other.mCRU),
    mRow(other.mRow),
    mQ(other.mQ),
    mQmax(other.mQmax),
    mPadMean(other.mPadMean),
    mPadSigma(other.mPadSigma)
{
}

//________________________________________________________________________
inline void Cluster::setParameters(short cru, short row, float q, float qmax, float padmean, float padsigma,
                                   float timemean, float timesigma)
{
  mCRU = cru;
  mRow = row;
  mQ = q;
  mQmax = qmax;
  mPadMean = padmean;
  mPadSigma = padsigma;
  setTimeStamp(timemean);
  setTimeStampError(timesigma);
}

} // namespace TPC
} // namespace o2

#endif /* ALICEO2_TPC_CLUSTER_H */
