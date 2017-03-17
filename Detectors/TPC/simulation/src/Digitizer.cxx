/// \file Digitizer.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Constants.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/SAMPAProcessing.h"

#include "TPCBase/Mapper.h"

#include "TClonesArray.h"
#include "TCollection.h"

#include "FairLogger.h"

ClassImp(AliceO2::TPC::Digitizer)

using namespace AliceO2::TPC;

Digitizer::Digitizer()
  : TObject()
  , mDigitContainer(nullptr)
{}

Digitizer::~Digitizer()
{
  delete mDigitContainer;
}

void Digitizer::init()
{
  /// @todo get rid of new? check with Mohammad
  mDigitContainer = new DigitContainer();
}

DigitContainer *Digitizer::Process(TClonesArray *points)
{
  /// @todo Containers?
  mDigitContainer->reset();
  const Mapper& mapper = Mapper::instance();

  // static_thread for thread savety?
  // avid multiple creation of the random lookup tables inside
  static GEMAmplification gemStack;
  static ElectronTransport electronTransport;
  static PadResponse padResp;

  static std::array<float, mNShapedPoints> signalArray;

  for(auto pointObject : *points) {
    Point *inputpoint = static_cast<Point *>(pointObject);

    const GlobalPosition3D posEle(inputpoint->GetX(), inputpoint->GetY(), inputpoint->GetZ());
    int nPrimaryElectrons = static_cast<int>(inputpoint->GetEnergyLoss()/WION);

    int MCEventID = inputpoint->GetEventID();
    int MCTrackID = inputpoint->GetTrackID();

    //Loop over electrons
    /// @todo can be vectorized?
    /// @todo split transport and signal formation in two separate loop?
    for(int iEle=0; iEle < nPrimaryElectrons; ++iEle) {

      // Drift and Diffusion
      const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle);

      /// @todo Time management in continuous mode (adding the time of the event?)
      const float driftTime = getTime(posEleDiff.getZ());

      // Attachment
      if(electronTransport.isElectronAttachment(driftTime)) continue;

      // remove electrons that end up outside the active volume
      /// @todo should go to mapper?
      if(fabs(posEleDiff.getZ()) > TPCLENGTH) continue;

      const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleDiff);
      if(!digiPadPos.isValid()) continue;

      const int nElectronsGEM = gemStack.getStackAmplification();

      // Loop over all individual pads with signal due to pad response function
      // PRF does not yet work as there is a massive discrepancy between the computed Pad Centre and the actual electron
      // position, therefore prf = 0;
      // for(int ipad = -2; ipad<3; ++ipad) {
      //   for(int irow = -2; irow<3; ++irow) {
      //     PadPos padPos(digiPadPos.getPadPos().getRow() + irow, digiPadPos.getPadPos().getPad() + ipad);
      DigitPos digiPos = digiPadPos;
      //     DigitPos digiPos(digiPadPos.getCRU(), padPos); /// @todo this is not at all optimal - in principle the next pad row could be in the next cru - to be changed!
      if (!digiPos.isValid()) continue;
      const float prfWeight = 1.f;
//      const float prfWeight = padResp.getPadResponse(posEleDiff, digiPos);
      if (prfWeight <= 0) continue;
      const int pad = digiPos.getPadPos().getPad();
      const int row = digiPos.getPadPos().getRow();
      const float ADCsignal = SAMPAProcessing::getADCvalue(nElectronsGEM * prfWeight);
      SAMPAProcessing::getShapedSignal(ADCsignal, driftTime, signalArray);

      for(float i=0; i<mNShapedPoints; ++i) {
        float time = driftTime + i * ZBINWIDTH;
        mDigitContainer->addDigit(MCEventID, MCTrackID, digiPos.getCRU().number(), getTimeBinFromTime(time), row, pad, signalArray[i]);
      }

      //         }
      //       }
      // end of loop over prf
    }
    // end of loop over electrons
  }
  // end of loop over points

  return mDigitContainer;
}
