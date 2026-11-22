#include "definitions.h"
struct MergedNeuronInitGroup0
 {
    float* __restrict__ E;
    float* __restrict__ PRew;
    float* __restrict__ PrevVal;
    float* __restrict__ V;
    float* __restrict__ ValReg;
    float* __restrict__ actionTaken;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ reward;
    float* __restrict__ tdError;
    
}
;
struct MergedNeuronInitGroup1
 {
    float* __restrict__ E;
    float* __restrict__ PG;
    float* __restrict__ PRew;
    float* __restrict__ TdE;
    float* __restrict__ V;
    float* __restrict__ actionTaken;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ pre_E;
    float* __restrict__ pre_PG;
    float* __restrict__ pre_PRew;
    float* __restrict__ reward;
    float* __restrict__ tdError;
    
}
;
struct MergedNeuronInitGroup2
 {
    float* __restrict__ VE;
    float* __restrict__ PertEpsTraceDivSigma;
    float* __restrict__ PsiBarInSynWUMPost0;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ V;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ VR;
    float* __restrict__ ZFilterOutSynWUMPre0;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPreOutSyn1;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    float* __restrict__ A;
    float* __restrict__ PertEps;
    float* __restrict__ PRew;
    float* __restrict__ PR;
    float* __restrict__ PGEps;
    float* __restrict__ PG;
    float* __restrict__ Noise4;
    float* __restrict__ Noise3;
    float* __restrict__ Noise2;
    float* __restrict__ Noise1;
    float* __restrict__ LogSigma;
    float* __restrict__ FAvgInSynWUMPost0;
    float* __restrict__ Ebase;
    float* __restrict__ E;
    float* __restrict__ Beta;
    float* __restrict__ AbsTdInSynWUMPost0;
    
}
;
struct MergedNeuronInitGroup3
 {
    float* __restrict__ XMeanOutSynWUMPre0;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ PsiInSynWUMPost1;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ TdMeanInSynWUMPost1;
    float* __restrict__ V;
    float* __restrict__ VE;
    float* __restrict__ VR;
    float* __restrict__ PsiBarInSynWUMPost1;
    float* __restrict__ XTraceOutSynWUMPre0;
    float* __restrict__ ZFilterOutSynWUMPre1;
    float* __restrict__ ZFilterOutSynWUMPre2;
    float* __restrict__ ZFilterOutSynWUMPre3;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPreOutSyn1;
    float* __restrict__ outPreOutSyn2;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    float* __restrict__ Noise3;
    float* __restrict__ AbsTdInSynWUMPost0;
    float* __restrict__ AbsTdInSynWUMPost1;
    float* __restrict__ Beta;
    float* __restrict__ E;
    float* __restrict__ Ebase;
    float* __restrict__ FAvgInSynWUMPost0;
    float* __restrict__ FAvgInSynWUMPost1;
    float* __restrict__ LogSigma;
    float* __restrict__ Noise1;
    float* __restrict__ Noise2;
    float* __restrict__ A;
    float* __restrict__ Noise4;
    float* __restrict__ PG;
    float* __restrict__ PGEps;
    float* __restrict__ PR;
    float* __restrict__ PRew;
    float* __restrict__ PertEps;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ PertEpsTraceDivSigma;
    float* __restrict__ PsiBarInSynWUMPost0;
    
}
;
struct MergedNeuronInitGroup4
 {
    float* __restrict__ YMeanInSynWUMPost1;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ PsiInSynWUMPost2;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ TdMeanInSynWUMPost2;
    float* __restrict__ V;
    float* __restrict__ VE;
    float* __restrict__ VR;
    float* __restrict__ PsiBarInSynWUMPost2;
    float* __restrict__ YTraceInSynWUMPost1;
    float* __restrict__ ZFilterOutSynWUMPre0;
    float* __restrict__ ZFilterOutSynWUMPre1;
    float* __restrict__ ZFilterOutSynWUMPre2;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPreOutSyn1;
    float* __restrict__ outPreOutSyn2;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    float* __restrict__ Noise3;
    float* __restrict__ AbsTdInSynWUMPost0;
    float* __restrict__ AbsTdInSynWUMPost2;
    float* __restrict__ Beta;
    float* __restrict__ E;
    float* __restrict__ Ebase;
    float* __restrict__ FAvgInSynWUMPost0;
    float* __restrict__ FAvgInSynWUMPost2;
    float* __restrict__ LogSigma;
    float* __restrict__ Noise1;
    float* __restrict__ Noise2;
    float* __restrict__ A;
    float* __restrict__ Noise4;
    float* __restrict__ PG;
    float* __restrict__ PGEps;
    float* __restrict__ PR;
    float* __restrict__ PRew;
    float* __restrict__ PertEps;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ PertEpsTraceDivSigma;
    float* __restrict__ PsiBarInSynWUMPost0;
    
}
;
struct MergedNeuronInitGroup5
 {
    float* __restrict__ Input;
    float* __restrict__ Noise1;
    float* __restrict__ Noise2;
    float* __restrict__ Noise3;
    float* __restrict__ Noise4;
    float* __restrict__ ZFilterOutSynWUMPre0;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    
}
;
struct MergedSynapseInitGroup0
 {
    float* __restrict__ g;
    uint32_t numSrcNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup0
 {
    uint32_t* __restrict__ ind;
    uint32_t* __restrict__ rowLength;
    uint32_t numSrcNeurons;
    uint32_t numTrgNeurons;
    float probLogRecip;
    uint32_t rowStride;
    
}
;
struct MergedSynapseSparseInitGroup0
 {
    float* __restrict__ g;
    uint32_t* __restrict__ rowLength;
    
}
;
struct MergedSynapseSparseInitGroup1
 {
    float* __restrict__ DeltaG;
    float* __restrict__ RLTrace;
    float* __restrict__ g;
    uint32_t* __restrict__ rowLength;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    float sdg;
    
}
;
struct MergedSynapseSparseInitGroup2
 {
    float* __restrict__ DeltaG;
    float* __restrict__ LogSynSig;
    float* __restrict__ NoiseTrace;
    float* __restrict__ RLNoiseGtrace;
    float* __restrict__ RLNoisePtrace;
    float* __restrict__ SynSig;
    float* __restrict__ eFiltered;
    float* __restrict__ epsilonA;
    float* __restrict__ g;
    uint32_t* __restrict__ rowLength;
    float meang;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    float sdg;
    
}
;
struct MergedCustomWUUpdateSparseInitGroup0
 {
    float* __restrict__ InitDone;
    float* __restrict__ M;
    float* __restrict__ S;
    float* __restrict__ Step;
    float* __restrict__ Surprise;
    float* __restrict__ VarInit;
    uint32_t* __restrict__ rowLength;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedCustomConnectivityUpdatePreInitGroup0
 {
    uint32_t* __restrict__ NumPruned;
    uint32_t numNeurons;
    
}
;
static MergedNeuronInitGroup0 mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, float* E, float* PRew, float* PrevVal, float* V, float* ValReg, float* actionTaken, float* outPostInSyn0, float* reward, float* tdError) {
    mergedNeuronInitGroup0[idx].E = E;
    mergedNeuronInitGroup0[idx].PRew = PRew;
    mergedNeuronInitGroup0[idx].PrevVal = PrevVal;
    mergedNeuronInitGroup0[idx].V = V;
    mergedNeuronInitGroup0[idx].ValReg = ValReg;
    mergedNeuronInitGroup0[idx].actionTaken = actionTaken;
    mergedNeuronInitGroup0[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup0[idx].reward = reward;
    mergedNeuronInitGroup0[idx].tdError = tdError;
}
static MergedNeuronInitGroup1 mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, float* E, float* PG, float* PRew, float* TdE, float* V, float* actionTaken, float* outPostInSyn0, float* outPreOutSyn0, float* pre_E, float* pre_PG, float* pre_PRew, float* reward, float* tdError) {
    mergedNeuronInitGroup1[idx].E = E;
    mergedNeuronInitGroup1[idx].PG = PG;
    mergedNeuronInitGroup1[idx].PRew = PRew;
    mergedNeuronInitGroup1[idx].TdE = TdE;
    mergedNeuronInitGroup1[idx].V = V;
    mergedNeuronInitGroup1[idx].actionTaken = actionTaken;
    mergedNeuronInitGroup1[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup1[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronInitGroup1[idx].pre_E = pre_E;
    mergedNeuronInitGroup1[idx].pre_PG = pre_PG;
    mergedNeuronInitGroup1[idx].pre_PRew = pre_PRew;
    mergedNeuronInitGroup1[idx].reward = reward;
    mergedNeuronInitGroup1[idx].tdError = tdError;
}
static MergedNeuronInitGroup2 mergedNeuronInitGroup2[2];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, float* VE, float* PertEpsTraceDivSigma, float* PsiBarInSynWUMPost0, float* PsiInSynWUMPost0, float* RefracTime, float* Sigma, float* TdE, float* TdMeanInSynWUMPost0, float* V, float* PertEpsTrace, float* VR, float* ZFilterOutSynWUMPre0, float* outPostInSyn0, float* outPreOutSyn0, float* outPreOutSyn1, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* A, float* PertEps, float* PRew, float* PR, float* PGEps, float* PG, float* Noise4, float* Noise3, float* Noise2, float* Noise1, float* LogSigma, float* FAvgInSynWUMPost0, float* Ebase, float* E, float* Beta, float* AbsTdInSynWUMPost0) {
    mergedNeuronInitGroup2[idx].VE = VE;
    mergedNeuronInitGroup2[idx].PertEpsTraceDivSigma = PertEpsTraceDivSigma;
    mergedNeuronInitGroup2[idx].PsiBarInSynWUMPost0 = PsiBarInSynWUMPost0;
    mergedNeuronInitGroup2[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronInitGroup2[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup2[idx].Sigma = Sigma;
    mergedNeuronInitGroup2[idx].TdE = TdE;
    mergedNeuronInitGroup2[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronInitGroup2[idx].V = V;
    mergedNeuronInitGroup2[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronInitGroup2[idx].VR = VR;
    mergedNeuronInitGroup2[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronInitGroup2[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup2[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronInitGroup2[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronInitGroup2[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronInitGroup2[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronInitGroup2[idx].A = A;
    mergedNeuronInitGroup2[idx].PertEps = PertEps;
    mergedNeuronInitGroup2[idx].PRew = PRew;
    mergedNeuronInitGroup2[idx].PR = PR;
    mergedNeuronInitGroup2[idx].PGEps = PGEps;
    mergedNeuronInitGroup2[idx].PG = PG;
    mergedNeuronInitGroup2[idx].Noise4 = Noise4;
    mergedNeuronInitGroup2[idx].Noise3 = Noise3;
    mergedNeuronInitGroup2[idx].Noise2 = Noise2;
    mergedNeuronInitGroup2[idx].Noise1 = Noise1;
    mergedNeuronInitGroup2[idx].LogSigma = LogSigma;
    mergedNeuronInitGroup2[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronInitGroup2[idx].Ebase = Ebase;
    mergedNeuronInitGroup2[idx].E = E;
    mergedNeuronInitGroup2[idx].Beta = Beta;
    mergedNeuronInitGroup2[idx].AbsTdInSynWUMPost0 = AbsTdInSynWUMPost0;
}
static MergedNeuronInitGroup3 mergedNeuronInitGroup3[1];
void pushMergedNeuronInitGroup3ToDevice(unsigned int idx, float* XMeanOutSynWUMPre0, float* PsiInSynWUMPost0, float* PsiInSynWUMPost1, float* RefracTime, float* Sigma, float* TdE, float* TdMeanInSynWUMPost0, float* TdMeanInSynWUMPost1, float* V, float* VE, float* VR, float* PsiBarInSynWUMPost1, float* XTraceOutSynWUMPre0, float* ZFilterOutSynWUMPre1, float* ZFilterOutSynWUMPre2, float* ZFilterOutSynWUMPre3, float* outPostInSyn0, float* outPreOutSyn0, float* outPreOutSyn1, float* outPreOutSyn2, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* Noise3, float* AbsTdInSynWUMPost0, float* AbsTdInSynWUMPost1, float* Beta, float* E, float* Ebase, float* FAvgInSynWUMPost0, float* FAvgInSynWUMPost1, float* LogSigma, float* Noise1, float* Noise2, float* A, float* Noise4, float* PG, float* PGEps, float* PR, float* PRew, float* PertEps, float* PertEpsTrace, float* PertEpsTraceDivSigma, float* PsiBarInSynWUMPost0) {
    mergedNeuronInitGroup3[idx].XMeanOutSynWUMPre0 = XMeanOutSynWUMPre0;
    mergedNeuronInitGroup3[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronInitGroup3[idx].PsiInSynWUMPost1 = PsiInSynWUMPost1;
    mergedNeuronInitGroup3[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup3[idx].Sigma = Sigma;
    mergedNeuronInitGroup3[idx].TdE = TdE;
    mergedNeuronInitGroup3[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronInitGroup3[idx].TdMeanInSynWUMPost1 = TdMeanInSynWUMPost1;
    mergedNeuronInitGroup3[idx].V = V;
    mergedNeuronInitGroup3[idx].VE = VE;
    mergedNeuronInitGroup3[idx].VR = VR;
    mergedNeuronInitGroup3[idx].PsiBarInSynWUMPost1 = PsiBarInSynWUMPost1;
    mergedNeuronInitGroup3[idx].XTraceOutSynWUMPre0 = XTraceOutSynWUMPre0;
    mergedNeuronInitGroup3[idx].ZFilterOutSynWUMPre1 = ZFilterOutSynWUMPre1;
    mergedNeuronInitGroup3[idx].ZFilterOutSynWUMPre2 = ZFilterOutSynWUMPre2;
    mergedNeuronInitGroup3[idx].ZFilterOutSynWUMPre3 = ZFilterOutSynWUMPre3;
    mergedNeuronInitGroup3[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup3[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronInitGroup3[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronInitGroup3[idx].outPreOutSyn2 = outPreOutSyn2;
    mergedNeuronInitGroup3[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronInitGroup3[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronInitGroup3[idx].Noise3 = Noise3;
    mergedNeuronInitGroup3[idx].AbsTdInSynWUMPost0 = AbsTdInSynWUMPost0;
    mergedNeuronInitGroup3[idx].AbsTdInSynWUMPost1 = AbsTdInSynWUMPost1;
    mergedNeuronInitGroup3[idx].Beta = Beta;
    mergedNeuronInitGroup3[idx].E = E;
    mergedNeuronInitGroup3[idx].Ebase = Ebase;
    mergedNeuronInitGroup3[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronInitGroup3[idx].FAvgInSynWUMPost1 = FAvgInSynWUMPost1;
    mergedNeuronInitGroup3[idx].LogSigma = LogSigma;
    mergedNeuronInitGroup3[idx].Noise1 = Noise1;
    mergedNeuronInitGroup3[idx].Noise2 = Noise2;
    mergedNeuronInitGroup3[idx].A = A;
    mergedNeuronInitGroup3[idx].Noise4 = Noise4;
    mergedNeuronInitGroup3[idx].PG = PG;
    mergedNeuronInitGroup3[idx].PGEps = PGEps;
    mergedNeuronInitGroup3[idx].PR = PR;
    mergedNeuronInitGroup3[idx].PRew = PRew;
    mergedNeuronInitGroup3[idx].PertEps = PertEps;
    mergedNeuronInitGroup3[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronInitGroup3[idx].PertEpsTraceDivSigma = PertEpsTraceDivSigma;
    mergedNeuronInitGroup3[idx].PsiBarInSynWUMPost0 = PsiBarInSynWUMPost0;
}
static MergedNeuronInitGroup4 mergedNeuronInitGroup4[1];
void pushMergedNeuronInitGroup4ToDevice(unsigned int idx, float* YMeanInSynWUMPost1, float* PsiInSynWUMPost0, float* PsiInSynWUMPost2, float* RefracTime, float* Sigma, float* TdE, float* TdMeanInSynWUMPost0, float* TdMeanInSynWUMPost2, float* V, float* VE, float* VR, float* PsiBarInSynWUMPost2, float* YTraceInSynWUMPost1, float* ZFilterOutSynWUMPre0, float* ZFilterOutSynWUMPre1, float* ZFilterOutSynWUMPre2, float* outPostInSyn0, float* outPreOutSyn0, float* outPreOutSyn1, float* outPreOutSyn2, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* Noise3, float* AbsTdInSynWUMPost0, float* AbsTdInSynWUMPost2, float* Beta, float* E, float* Ebase, float* FAvgInSynWUMPost0, float* FAvgInSynWUMPost2, float* LogSigma, float* Noise1, float* Noise2, float* A, float* Noise4, float* PG, float* PGEps, float* PR, float* PRew, float* PertEps, float* PertEpsTrace, float* PertEpsTraceDivSigma, float* PsiBarInSynWUMPost0) {
    mergedNeuronInitGroup4[idx].YMeanInSynWUMPost1 = YMeanInSynWUMPost1;
    mergedNeuronInitGroup4[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronInitGroup4[idx].PsiInSynWUMPost2 = PsiInSynWUMPost2;
    mergedNeuronInitGroup4[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup4[idx].Sigma = Sigma;
    mergedNeuronInitGroup4[idx].TdE = TdE;
    mergedNeuronInitGroup4[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronInitGroup4[idx].TdMeanInSynWUMPost2 = TdMeanInSynWUMPost2;
    mergedNeuronInitGroup4[idx].V = V;
    mergedNeuronInitGroup4[idx].VE = VE;
    mergedNeuronInitGroup4[idx].VR = VR;
    mergedNeuronInitGroup4[idx].PsiBarInSynWUMPost2 = PsiBarInSynWUMPost2;
    mergedNeuronInitGroup4[idx].YTraceInSynWUMPost1 = YTraceInSynWUMPost1;
    mergedNeuronInitGroup4[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronInitGroup4[idx].ZFilterOutSynWUMPre1 = ZFilterOutSynWUMPre1;
    mergedNeuronInitGroup4[idx].ZFilterOutSynWUMPre2 = ZFilterOutSynWUMPre2;
    mergedNeuronInitGroup4[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup4[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronInitGroup4[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronInitGroup4[idx].outPreOutSyn2 = outPreOutSyn2;
    mergedNeuronInitGroup4[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronInitGroup4[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronInitGroup4[idx].Noise3 = Noise3;
    mergedNeuronInitGroup4[idx].AbsTdInSynWUMPost0 = AbsTdInSynWUMPost0;
    mergedNeuronInitGroup4[idx].AbsTdInSynWUMPost2 = AbsTdInSynWUMPost2;
    mergedNeuronInitGroup4[idx].Beta = Beta;
    mergedNeuronInitGroup4[idx].E = E;
    mergedNeuronInitGroup4[idx].Ebase = Ebase;
    mergedNeuronInitGroup4[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronInitGroup4[idx].FAvgInSynWUMPost2 = FAvgInSynWUMPost2;
    mergedNeuronInitGroup4[idx].LogSigma = LogSigma;
    mergedNeuronInitGroup4[idx].Noise1 = Noise1;
    mergedNeuronInitGroup4[idx].Noise2 = Noise2;
    mergedNeuronInitGroup4[idx].A = A;
    mergedNeuronInitGroup4[idx].Noise4 = Noise4;
    mergedNeuronInitGroup4[idx].PG = PG;
    mergedNeuronInitGroup4[idx].PGEps = PGEps;
    mergedNeuronInitGroup4[idx].PR = PR;
    mergedNeuronInitGroup4[idx].PRew = PRew;
    mergedNeuronInitGroup4[idx].PertEps = PertEps;
    mergedNeuronInitGroup4[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronInitGroup4[idx].PertEpsTraceDivSigma = PertEpsTraceDivSigma;
    mergedNeuronInitGroup4[idx].PsiBarInSynWUMPost0 = PsiBarInSynWUMPost0;
}
static MergedNeuronInitGroup5 mergedNeuronInitGroup5[1];
void pushMergedNeuronInitGroup5ToDevice(unsigned int idx, float* Input, float* Noise1, float* Noise2, float* Noise3, float* Noise4, float* ZFilterOutSynWUMPre0, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0) {
    mergedNeuronInitGroup5[idx].Input = Input;
    mergedNeuronInitGroup5[idx].Noise1 = Noise1;
    mergedNeuronInitGroup5[idx].Noise2 = Noise2;
    mergedNeuronInitGroup5[idx].Noise3 = Noise3;
    mergedNeuronInitGroup5[idx].Noise4 = Noise4;
    mergedNeuronInitGroup5[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronInitGroup5[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronInitGroup5[idx].spkSynSpike0 = spkSynSpike0;
}
static MergedSynapseInitGroup0 mergedSynapseInitGroup0[5];
void pushMergedSynapseInitGroup0ToDevice(unsigned int idx, float* g, uint32_t numSrcNeurons) {
    mergedSynapseInitGroup0[idx].g = g;
    mergedSynapseInitGroup0[idx].numSrcNeurons = numSrcNeurons;
}
static MergedSynapseConnectivityInitGroup0 mergedSynapseConnectivityInitGroup0[8];
void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, uint32_t* ind, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t numTrgNeurons, float probLogRecip, uint32_t rowStride) {
    mergedSynapseConnectivityInitGroup0[idx].ind = ind;
    mergedSynapseConnectivityInitGroup0[idx].rowLength = rowLength;
    mergedSynapseConnectivityInitGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseConnectivityInitGroup0[idx].numTrgNeurons = numTrgNeurons;
    mergedSynapseConnectivityInitGroup0[idx].probLogRecip = probLogRecip;
    mergedSynapseConnectivityInitGroup0[idx].rowStride = rowStride;
}
static MergedSynapseSparseInitGroup0 mergedSynapseSparseInitGroup0[1];
void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, float* g, uint32_t* rowLength) {
    mergedSynapseSparseInitGroup0[idx].g = g;
    mergedSynapseSparseInitGroup0[idx].rowLength = rowLength;
}
static MergedSynapseSparseInitGroup1 mergedSynapseSparseInitGroup1[8];
void pushMergedSynapseSparseInitGroup1ToDevice(unsigned int idx, float* DeltaG, float* RLTrace, float* g, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t rowStride, float sdg) {
    mergedSynapseSparseInitGroup1[idx].DeltaG = DeltaG;
    mergedSynapseSparseInitGroup1[idx].RLTrace = RLTrace;
    mergedSynapseSparseInitGroup1[idx].g = g;
    mergedSynapseSparseInitGroup1[idx].rowLength = rowLength;
    mergedSynapseSparseInitGroup1[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseSparseInitGroup1[idx].rowStride = rowStride;
    mergedSynapseSparseInitGroup1[idx].sdg = sdg;
}
static MergedSynapseSparseInitGroup2 mergedSynapseSparseInitGroup2[7];
void pushMergedSynapseSparseInitGroup2ToDevice(unsigned int idx, float* DeltaG, float* LogSynSig, float* NoiseTrace, float* RLNoiseGtrace, float* RLNoisePtrace, float* SynSig, float* eFiltered, float* epsilonA, float* g, uint32_t* rowLength, float meang, uint32_t numSrcNeurons, uint32_t rowStride, float sdg) {
    mergedSynapseSparseInitGroup2[idx].DeltaG = DeltaG;
    mergedSynapseSparseInitGroup2[idx].LogSynSig = LogSynSig;
    mergedSynapseSparseInitGroup2[idx].NoiseTrace = NoiseTrace;
    mergedSynapseSparseInitGroup2[idx].RLNoiseGtrace = RLNoiseGtrace;
    mergedSynapseSparseInitGroup2[idx].RLNoisePtrace = RLNoisePtrace;
    mergedSynapseSparseInitGroup2[idx].SynSig = SynSig;
    mergedSynapseSparseInitGroup2[idx].eFiltered = eFiltered;
    mergedSynapseSparseInitGroup2[idx].epsilonA = epsilonA;
    mergedSynapseSparseInitGroup2[idx].g = g;
    mergedSynapseSparseInitGroup2[idx].rowLength = rowLength;
    mergedSynapseSparseInitGroup2[idx].meang = meang;
    mergedSynapseSparseInitGroup2[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseSparseInitGroup2[idx].rowStride = rowStride;
    mergedSynapseSparseInitGroup2[idx].sdg = sdg;
}
static MergedCustomWUUpdateSparseInitGroup0 mergedCustomWUUpdateSparseInitGroup0[9];
void pushMergedCustomWUUpdateSparseInitGroup0ToDevice(unsigned int idx, float* InitDone, float* M, float* S, float* Step, float* Surprise, float* VarInit, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedCustomWUUpdateSparseInitGroup0[idx].InitDone = InitDone;
    mergedCustomWUUpdateSparseInitGroup0[idx].M = M;
    mergedCustomWUUpdateSparseInitGroup0[idx].S = S;
    mergedCustomWUUpdateSparseInitGroup0[idx].Step = Step;
    mergedCustomWUUpdateSparseInitGroup0[idx].Surprise = Surprise;
    mergedCustomWUUpdateSparseInitGroup0[idx].VarInit = VarInit;
    mergedCustomWUUpdateSparseInitGroup0[idx].rowLength = rowLength;
    mergedCustomWUUpdateSparseInitGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedCustomWUUpdateSparseInitGroup0[idx].rowStride = rowStride;
}
static MergedCustomConnectivityUpdatePreInitGroup0 mergedCustomConnectivityUpdatePreInitGroup0[7];
void pushMergedCustomConnectivityUpdatePreInitGroup0ToDevice(unsigned int idx, uint32_t* NumPruned, uint32_t numNeurons) {
    mergedCustomConnectivityUpdatePreInitGroup0[idx].NumPruned = NumPruned;
    mergedCustomConnectivityUpdatePreInitGroup0[idx].numNeurons = numNeurons;
}
void initializeHost() {
}
void initialize() {
    // ------------------------------------------------------------------------
    // Neuron groups
     {
        // merged neuron init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup0[g]; 
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->E[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->tdError[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->reward[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->actionTaken[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PRew[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ValReg[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PrevVal[i] = initVal;
                }
            }
            for (unsigned int i = 0; i < ((1u)); i++) {
                group->outPostInSyn0[i] = 0.000000000e+00f;
            }
        }
    }
     {
        // merged neuron init group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup1[g]; 
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->E[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->tdError[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->reward[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->actionTaken[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PRew[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PG[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->pre_PG[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->pre_E[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((4u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->pre_PRew[i] = initVal;
                }
            }
            for (unsigned int i = 0; i < ((4u)); i++) {
                group->outPostInSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((4u)); i++) {
                group->outPreOutSyn0[i] = 0.000000000e+00f;
            }
        }
    }
     {
        // merged neuron init group 2
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedNeuronInitGroup2[g]; 
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->A[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (1.740000000e-02f);
                    group->Beta[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTrace[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTraceDivSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEps[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Sigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (-4.605170186e+00f);
                    group->LogSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->E[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Ebase[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise3[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise4[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PG[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PRew[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PGEps[i] = initVal;
                }
            }
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->spkSynSpike0[i] = 0;
            }
            group->spkCntSynSpike0[0] = 0;
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->outPostInSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->outPreOutSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->outPreOutSyn1[i] = 0.000000000e+00f;
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiBarInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->AbsTdInSynWUMPost0[i] = initVal;
                }
            }
        }
    }
     {
        // merged neuron init group 3
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup3[g]; 
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->A[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (1.740000000e-02f);
                    group->Beta[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTrace[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTraceDivSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEps[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Sigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (-4.605170186e+00f);
                    group->LogSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->E[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Ebase[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise3[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise4[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PG[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PRew[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PGEps[i] = initVal;
                }
            }
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->spkSynSpike0[i] = 0;
            }
            group->spkCntSynSpike0[0] = 0;
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->outPostInSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->outPreOutSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->outPreOutSyn1[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((48u)); i++) {
                group->outPreOutSyn2[i] = 0.000000000e+00f;
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->XTraceOutSynWUMPre0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (2.050416649e-01f);
                    group->XMeanOutSynWUMPre0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre3[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiBarInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->AbsTdInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiBarInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdMeanInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->AbsTdInSynWUMPost1[i] = initVal;
                }
            }
        }
    }
     {
        // merged neuron init group 4
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup4[g]; 
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->A[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (1.740000000e-02f);
                    group->Beta[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTrace[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTraceDivSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEps[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Sigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (-4.605170186e+00f);
                    group->LogSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->E[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Ebase[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise3[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise4[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PG[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PRew[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PGEps[i] = initVal;
                }
            }
            for (unsigned int i = 0; i < ((108u)); i++) {
                group->spkSynSpike0[i] = 0;
            }
            group->spkCntSynSpike0[0] = 0;
            for (unsigned int i = 0; i < ((108u)); i++) {
                group->outPostInSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((108u)); i++) {
                group->outPreOutSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((108u)); i++) {
                group->outPreOutSyn1[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < ((108u)); i++) {
                group->outPreOutSyn2[i] = 0.000000000e+00f;
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiBarInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->AbsTdInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->YTraceInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (2.050416649e-01f);
                    group->YMeanInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiInSynWUMPost2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgInSynWUMPost2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiBarInSynWUMPost2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdMeanInSynWUMPost2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((108u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->AbsTdInSynWUMPost2[i] = initVal;
                }
            }
        }
    }
     {
        // merged neuron init group 5
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup5[g]; 
             {
                for (unsigned int i = 0; i < ((1200u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Input[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1200u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1200u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1200u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise3[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((1200u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise4[i] = initVal;
                }
            }
            for (unsigned int i = 0; i < ((1200u)); i++) {
                group->spkSynSpike0[i] = 0;
            }
            group->spkCntSynSpike0[0] = 0;
             {
                for (unsigned int i = 0; i < ((1200u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre0[i] = initVal;
                }
            }
        }
    }
    // ------------------------------------------------------------------------
    // Synapse groups
     {
        // merged synapse init group 0
        for(unsigned int g = 0; g < 5; g++) {
            const auto *group = &mergedSynapseInitGroup0[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    for (unsigned int j = 0; j < (1u); j++) {
                        const unsigned int idSyn = (i * (1u)) + j;
                        float initVal;
                        initVal = (1.000000000e+00f);
                        group->g[idSyn] = initVal;
                    }
                }
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom update groups
    // ------------------------------------------------------------------------
    // Custom connectivity presynaptic update groups
     {
        // merged custom connectivity presynaptic init group 0
        for(unsigned int g = 0; g < 7; g++) {
            const auto *group = &mergedCustomConnectivityUpdatePreInitGroup0[g]; 
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    uint32_t initVal;
                    initVal = (0.000000000e+00f);
                    group->NumPruned[i] = initVal;
                }
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom connectivity postsynaptic update groups
    // ------------------------------------------------------------------------
    // Custom WU update groups
    // ------------------------------------------------------------------------
    // Synapse sparse connectivity
     {
        // merged synapse connectivity init group 0
        for(unsigned int g = 0; g < 8; g++) {
            const auto *group = &mergedSynapseConnectivityInitGroup0[g]; 
            for (unsigned int i = 0; i < group->numSrcNeurons; i++) {
                int32_t _prevJ = -1;
                while(true)
                 {
                    const float _u = standardUniformDistribution(hostRNG);
                    _prevJ += (1 + (int32_t)(log(_u) * group->probLogRecip));
                    if(_prevJ < group->numTrgNeurons)
                     {
                         {
                            const unsigned int idPost = _prevJ + 0;
                            const unsigned int idSyn = (i * group->rowStride) + group->rowLength[i];
                            group->ind[idSyn] = idPost;
                            group->rowLength[i]++;
                        }
                        ;
                    }
                    else
                     {
                        break;
                    }
                    
                }
                
            }
        }
    }
}

void initializeSparse() {
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
     {
        // merged sparse synapse init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedSynapseSparseInitGroup0[g]; 
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < (48u); i++) {
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * (27u)) + j;
                        float initVal;
                        initVal = (-1.039012269e-01f) + (standardNormalDistribution(hostRNG) * (1.731687115e-02f));
                        group->g[idSyn] = initVal;
                    }
                }
            }
        }
    }
     {
        // merged sparse synapse init group 1
        for(unsigned int g = 0; g < 8; g++) {
            const auto *group = &mergedSynapseSparseInitGroup1[g]; 
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f) + (standardNormalDistribution(hostRNG) * group->sdg);
                        group->g[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->DeltaG[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->RLTrace[idSyn] = initVal;
                    }
                }
            }
        }
    }
     {
        // merged sparse synapse init group 2
        for(unsigned int g = 0; g < 7; g++) {
            const auto *group = &mergedSynapseSparseInitGroup2[g]; 
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = group->meang + (standardNormalDistribution(hostRNG) * group->sdg);
                        group->g[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->DeltaG[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->epsilonA[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->eFiltered[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->LogSynSig[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->SynSig[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->NoiseTrace[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->RLNoisePtrace[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->RLNoiseGtrace[idSyn] = initVal;
                    }
                }
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom sparse WU update groups
     {
        // merged custom sparse WU update group 0
        for(unsigned int g = 0; g < 9; g++) {
            const auto *group = &mergedCustomWUUpdateSparseInitGroup0[g]; 
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->M[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->S[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->Step[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->Surprise[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->VarInit[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * group->rowStride) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->InitDone[idSyn] = initVal;
                    }
                }
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom connectivity update sparse init groups
}
