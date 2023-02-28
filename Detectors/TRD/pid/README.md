# Particle Identification with TRD
## Usage
Activate PID during tracking with the '--with-pid' flag.

    o2-trd-global-tracking --with-pid --policy ML

Specify a which algorithm (called policy) should be use.
Implemented are the following:

    - LQ1D
    - LQ2D
    - LQ3D
    - ML (every model, which is exported to the ONNX format):
        - XGB (XGBoost model)
        - NN (Pytorch model)
    - Dummy (returns only -1)
    - Test (one of the above)
    - Default (one of the above, gets picked if '--policy' is unspecified)

## Implementation details
### Tracking workflow
Every TRDTrack gets a PID value set (mSignal), which then gets propergated to the AO2D writer.

### Basic Interface
The base interface for PID is defined in [here](include/TRDPID/PIDBase.h).
The 'init' function is such that each policy can specify what if anything it needs from the CCDB.
For the 'process' each policy defines how a TRDTrack gets assigned a PID value.
Additionally, the base class implements how to get the _corrected charges_ from the tracklets.
_Corrected charges_ means z-row merged and calibrated charges.

### Classical Likelihood
The classical LQND policies ([here](include/TRDPID/LQND.h)) require an array of lookup tables (LUTs) from the ccdb.
$N$ stands for the dimension.
Then the electron likelihood for layer $i$ is defined as this:

$$L_i(e|Q_i)=\frac{P(Q_i|e)}{P(Q_i|e)+P(Q_i|\pi)}$$

From the charge $Q_i$ the LUTs give the corresponding $L_i$.
The _combined electron likelihood_ is obtained by this formula:

$$L(e|Q)=\frac{\prod_i L_i(e|Q_i)}{\prod_i L_i(e|Q_i) + \prod_i L_i(\pi|Q_i)}$$

where $L_i(\pi|Q_i)=1-L_i(e|Q_i)$.


Extension to higher dimensions is easy each tracklet has charges $Q_j$ which cover the integral of the pulse height spectrum in different slice ($j\in [0,1,2]$).
In our case $Q0$ covers the pulse height peak, $Q1$ the Transition Radiation peak and $Q2$ the plateau.
For each charge $j$ a LUT is available which gives the likelihood $L^e_j$.
For each layer $i$ the likelihood is then:

$$L_i(e|Q)=\frac{\prod_j L_{i,j}(e|Q_j)}{\prod_j L_{i,j}(e|Q_j) + \prod_j L_{i,j}(\pi|Q_j)}$$

The combined electron likelihood is then:

$$L(e|Q)=\frac{\prod_{i,j} L_{i,j}(e|Q_j)}{\prod_{i,j} L_{i,j}(e|Q_j) + \prod_{i,j} L_{i,j}(\pi|Q_j)}$$


### Machine Learning
The ML policies ([here](include/TRDPID/ML.h)) are uploaded to the CCDB in the ONNX file format (most python machine learning libraries support this standardized format).
In O2 we leverage the ONNXRuntime to use these formats and calculate a PID value.
The models can thus be trained in python which is very convenient.
The code should take care of most of the annoying stuff.
Policies just have to specify how to get the electron likelihood from the ONNXRuntime output (each python library varies in that somewhat).
The 'prepareModelInput' prepares the TRDTracks as input.
