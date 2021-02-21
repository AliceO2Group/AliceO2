#if !defined(__CLING__) || defined(__ROOTCLING__)
//#include "DataFormatsFT0/Digit.h" - later
// before run:
// o2-sim -n 10
// o2-sim-digitizer-wokflow -b
using namespace o2::steer;
#endif
void test()
{
        auto dctx = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
        const auto& irvec = dctx->getEventRecords(false); // use true to get hadronic and QED interactions together
        for (const auto& ir : irvec)
        {
            // create toy CT{ data
            //o2::ctp::CTPRawData data;
            ir.print();
        }
        //auto dctx = o2::steer::DigitizationContext::loadFromFile("ft0digits.root");
        //const auto& digs =
}
