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

#include "hyperTracker.h"
namespace o2
{
    namespace tracking
    {

        hyperTracker::hyperTracker(const TrackITS &motherTrack, const V0 &v0, const std::vector<ITSCluster> &motherClusters, o2::its::GeometryTGeo *gman, DCAFitter2 &mFitterV0)
            : hyperTrack{motherTrack}, hyperClusters{motherClusters}, geomITS{gman}, mFitterV0{mFitterV0}
        {

            auto posTrack = v0.getProng(0);
            auto negTrack = v0.getProng(1);
            auto alphaV0 = calcV0alpha(v0);
            alphaV0 > 0 ? posTrack.setAbsCharge(2) : negTrack.setAbsCharge(2);

            auto isRecr = recreateV0(posTrack, negTrack, v0.getProngID(0), v0.getProngID(1));
            if (!isRecr)
            {
                LOG(INFO) << "V0 regeneration not successful, using default one";
                hypV0 = v0;
            }
            setNclusMatching(motherClusters.size());
        }

        hyperTracker::hyperTracker(const TrackITS &motherTrack, const V0 &v0, const std::vector<ITSCluster> &motherClusters, o2::its::GeometryTGeo *gman)
            : hyperTrack{motherTrack}, hypV0{v0}, hyperClusters{motherClusters}, geomITS{gman}
        {
            setNclusMatching(motherClusters.size());
        }

        double hyperTracker::getMatchingChi2()
        {
            auto &outerClus = hyperClusters[0];
            float alpha = geomITS->getSensorRefAlpha(outerClus.getSensorID()), x = outerClus.getX();

            auto p0 = hypV0.getProng(0);
            auto p1 = hypV0.getProng(1);

            if (hypV0.rotate(alpha))
            {
                if (hypV0.propagateTo(x, mBz))
                {
                    std::cout << "Pred chi2 outermost Cluster: " << hypV0.getPredictedChi2(outerClus) << std::endl;
                    std::cout << "Pred chi2 V0-ITStrack: " << hypV0.getPredictedChi2(hyperTrack.getParamOut()) << std::endl;
                    return hypV0.getPredictedChi2(hyperTrack.getParamOut());
                }
            }
            return -1;
        }

        bool hyperTracker::process()
        {
            int isProcessed = 0;
            bool tryDaughter = true;

            for (auto &clus : hyperClusters)
            {

                if (updateV0topology(clus, tryDaughter) == 1)
                {
                    tryDaughter = false;
                    isProcessed++;
                    LOG(INFO) << "Attach cluster to V0 for layer: " << geomITS->getLayer(clus.getSensorID());
                    continue;
                }
                else if (updateV0topology(clus, tryDaughter) == 2)
                {
                    recreateV0(hypV0.getProng(0), hypV0.getProng(1), hypV0.getProngID(0), hypV0.getProngID(1));
                    isProcessed++;
                    LOG(INFO) << "Attach cluster to He3 for layer: " << geomITS->getLayer(clus.getSensorID());
                    continue;
                }
                else
                {
                    break;
                }
            }
            return isProcessed >= nClusMatching;
        }

        int hyperTracker::updateV0topology(const ITSCluster &clus, bool tryDaughter)
        {
            int isUpdated = 0;

            if (propagateToClus(clus, hypV0))
            {
                if (hypV0.getPredictedChi2(clus) < mMaxChi2)
                {
                    hypV0.update(clus);
                    isUpdated++;
                }
            }

            if (!isUpdated && tryDaughter)
            {
                auto alphaArm = calcV0alpha(hypV0);
                auto &he3track = alphaArm > 0 ? hypV0.getProng(0) : hypV0.getProng(1);

                if (propagateToClus(clus, he3track))
                {
                    if (he3track.getPredictedChi2(clus) < mMaxChi2)
                    {
                        he3track.update(clus);
                        isUpdated += 2;
                    }
                }
            }
            return isUpdated;
        }

        bool hyperTracker::propagateToClus(const ITSCluster &clus, o2::track::TrackParCov &track)
        {
            float alpha = geomITS->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
            int layer{geomITS->getLayer(clus.getSensorID())};
            float thick = layer < 3 ? 0.005 : 0.01;

            if (track.rotate(alpha))
            {
                if (track.propagateTo(x, mBz))
                {
                    constexpr float radl = 9.36f; // Radiation length of Si [cm]
                    constexpr float rho = 2.33f;  // Density of Si [g/cm^3]
                    return track.correctForMaterial(thick, thick * rho * radl);
                }
            }
            return false;
        }

        bool hyperTracker::recreateV0(const o2::track::TrackParCov &posTrack, const o2::track::TrackParCov &negTrack, const int posID, const int negID)
        {

            int cand = 0; //best V0 candidate
            int nCand;

            try
            {
                nCand = mFitterV0.process(posTrack, negTrack);
            }
            catch (std::runtime_error &e)
            {
                return false;
            }
            if (!nCand)
                return false;

            mFitterV0.propagateTracksToVertex();
            auto &propPos = mFitterV0.getTrack(0, 0);
            auto &propNeg = mFitterV0.getTrack(1, 0);

            const auto &v0XYZ = mFitterV0.getPCACandidatePos();
            std::array<float, 3> pP, pN;
            propPos.getPxPyPzGlo(pP);
            propNeg.getPxPyPzGlo(pN);
            std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};

            hypV0 = V0(v0XYZ, pV0, mFitterV0.calcPCACovMatrixFlat(cand), propPos, propNeg, posID, negID, o2::track::PID::HyperTriton);
            hypV0.setAbsCharge(1);

            return true;
        }

        double hyperTracker::calcV0alpha(const V0 &v0)
        {
            std::array<float, 3> fV0mom, fPmom, fNmom = {0, 0, 0};
            v0.getProng(0).getPxPyPzGlo(fPmom);
            v0.getProng(1).getPxPyPzGlo(fNmom);
            v0.getPxPyPzGlo(fV0mom);

            TVector3 momNeg(fNmom[0], fNmom[1], fNmom[2]);
            TVector3 momPos(fPmom[0], fPmom[1], fPmom[2]);
            TVector3 momTot(fV0mom[0], fV0mom[1], fV0mom[2]);

            Double_t lQlNeg = momNeg.Dot(momTot) / momTot.Mag();
            Double_t lQlPos = momPos.Dot(momTot) / momTot.Mag();

            return (lQlPos - lQlNeg) / (lQlPos + lQlNeg);
        }

    } // namespace tracking
} // namespace o2