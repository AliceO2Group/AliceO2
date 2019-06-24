\page refFrameworkLogger Logger

## O2 Framework Logger

O2 wrapper around FairLogger. No other dependencies but FairLogger and fmt
allowed.

This introduces a new header file, Framework/Logger.h which should
be used for any AliceO2 logging needs.

In the simple case, this simply forwards the API provided by FairLogger.

In addition it provides a generic LOGF macro to produce log messages
using a printf like formatting API. When available FairLogger is
actually used to do the printing and fmt is used to do the formatting.

Extra convenience macros O2DEBUG, O2INFO, O2ERROR are provided to save
a few keystrokes.
