<!-- doxy
\page refDetectorsMUONMCHRawEncoder Encoder
/doxy -->

Like the decoder, the encoder deals with two formats (Bare and UserLogic), both
of which in two different modes (charge sum and sample mode). Note that only
the chargesum mode is fully useable (but that should not be much of a
    limitation as we're yet to find a real use case for simulating the sample
    mode.

Generation of MCH raw data buffers is a two stage process : first we build
[payloads](Payload/) and then only we organize them in (RDH,payload) blocks
(see for instance how this is done by the [digit2raw](Digit/) program.

<!-- doxy
* \subpage refDetectorsMUONMCHRawEncoderPayload
* \subpage refDetectorsMUONMCHRawEncoderDigit
/doxy -->
