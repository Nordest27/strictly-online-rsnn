#include "definitions.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
std::mt19937 hostRNG;
std::uniform_real_distribution<float> standardUniformDistribution(0.000000000e+00f, 1.000000000e+00f);
std::normal_distribution<float> standardNormalDistribution(0.000000000e+00f, 1.000000000e+00f);
std::exponential_distribution<float> standardExponentialDistribution(1.000000000e+00f);


// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
double customUpdateDaleInitTime = 0.0;
double customUpdateDaleInitTransposeTime = 0.0;
double customUpdateDaleInitRemapTime = 0.0;
double customUpdateDaleInitHostTime = 0.0;
double customUpdateDalePruneTime = 0.0;
double customUpdateDalePruneTransposeTime = 0.0;
double customUpdateDalePruneRemapTime = 0.0;
double customUpdateDalePruneHostTime = 0.0;
double customUpdateDaleRewireTime = 0.0;
double customUpdateDaleRewireTransposeTime = 0.0;
double customUpdateDaleRewireRemapTime = 0.0;
double customUpdateDaleRewireHostTime = 0.0;
double customUpdateGradientLearnTime = 0.0;
double customUpdateGradientLearnTransposeTime = 0.0;
double customUpdateGradientLearnRemapTime = 0.0;
double customUpdateGradientLearnHostTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
}  // extern "C"
void allocateMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
     {
        std::seed_seq seeds{2};
        hostRNG.seed(seeds);
    }
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
}

void stepTime(unsigned long long timestep, unsigned long long numRecordingTimesteps) {
    const float t = timestep * 1.000000000e+00f;
    updateSynapses(t);
    updateNeurons(t); 
}

