#include "definitions.h"
struct MergedNeuronInitGroup0
 {
    float* __restrict__ E;
    float* __restrict__ EpMode;
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
    float* __restrict__ V;
    float* __restrict__ PR;
    float* __restrict__ PRew;
    float* __restrict__ PertEps;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ PertEpsTraceDivSigma;
    float* __restrict__ PsiBarInSynWUMPost0;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ SpkCount;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ PGEps;
    float* __restrict__ VE;
    float* __restrict__ VR;
    float* __restrict__ ZDecInSynWUMPost0;
    float* __restrict__ ZDecMeanInSynWUMPost0;
    float* __restrict__ ZFilterOutSynWUMPre0;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPostInSyn1;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPreOutSyn1;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    float* __restrict__ IPVBar;
    float* __restrict__ Beta;
    float* __restrict__ ContrDInSynWUMPost0;
    float* __restrict__ ContrMeanInSynWUMPost0;
    float* __restrict__ ContrRInSynWUMPost0;
    float* __restrict__ E;
    float* __restrict__ Ebase;
    float* __restrict__ FAvgInSynWUMPost0;
    float* __restrict__ FAvgTraceInSynWUMPost0;
    float* __restrict__ HebbMeanInSynWUMPost0;
    float* __restrict__ HebbZInSynWUMPost0;
    float* __restrict__ IPBias;
    float* __restrict__ IPPsiBar;
    float* __restrict__ A;
    float* __restrict__ InhMean;
    float* __restrict__ InhNow;
    float* __restrict__ InhVar;
    float* __restrict__ LocMeanInSynWUMPost0;
    float* __restrict__ LocRInSynWUMPost0;
    float* __restrict__ LogSigma;
    float* __restrict__ Noise1;
    float* __restrict__ Noise2;
    float* __restrict__ Noise3;
    float* __restrict__ Noise4;
    float* __restrict__ PG;
    
}
;
struct MergedNeuronInitGroup3
 {
    float* __restrict__ VE;
    float* __restrict__ PertEpsTraceDivSigma;
    float* __restrict__ PoolBias;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ SpkCount;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ V;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ VR;
    float* __restrict__ ZFilterOutSynWUMPre0;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPostInSyn1;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPreOutSyn1;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    float* __restrict__ Noise1;
    float* __restrict__ Beta;
    float* __restrict__ E;
    float* __restrict__ Ebase;
    float* __restrict__ FAvgInSynWUMPost0;
    float* __restrict__ IPBias;
    float* __restrict__ IPPsiBar;
    float* __restrict__ IPVBar;
    float* __restrict__ LogSigma;
    float* __restrict__ A;
    float* __restrict__ Noise2;
    float* __restrict__ Noise3;
    float* __restrict__ Noise4;
    float* __restrict__ PG;
    float* __restrict__ PGEps;
    float* __restrict__ PR;
    float* __restrict__ PRew;
    float* __restrict__ PertEps;
    
}
;
struct MergedNeuronInitGroup4
 {
    float* __restrict__ PsiBarInSynWUMPost1;
    float* __restrict__ TdMeanInSynWUMPost1;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ TdE;
    float* __restrict__ SpkCount;
    float* __restrict__ Sigma;
    float* __restrict__ RefracTime;
    float* __restrict__ PsiInSynWUMPost1;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ V;
    float* __restrict__ PsiBarInSynWUMPost0;
    float* __restrict__ PertEpsTraceDivSigma;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ PertEps;
    float* __restrict__ PRew;
    float* __restrict__ PR;
    float* __restrict__ A;
    float* __restrict__ ZFilterOutSynWUMPre1;
    uint32_t* __restrict__ spkSynSpike0;
    uint32_t* __restrict__ spkCntSynSpike0;
    float* __restrict__ outPreOutSyn2;
    float* __restrict__ outPreOutSyn1;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPostInSyn1;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ ZFilterOutSynWUMPre2;
    float* __restrict__ PGEps;
    float* __restrict__ ZFilterOutSynWUMPre0;
    float* __restrict__ ZDecMeanInSynWUMPost1;
    float* __restrict__ ZDecMeanInSynWUMPost0;
    float* __restrict__ ZDecInSynWUMPost1;
    float* __restrict__ ZDecInSynWUMPost0;
    float* __restrict__ VR;
    float* __restrict__ VE;
    float* __restrict__ E;
    float* __restrict__ PG;
    float* __restrict__ HebbMeanInSynWUMPost1;
    float* __restrict__ HebbMeanInSynWUMPost0;
    float* __restrict__ FAvgTraceInSynWUMPost1;
    float* __restrict__ FAvgTraceInSynWUMPost0;
    float* __restrict__ FAvgInSynWUMPost1;
    float* __restrict__ FAvgInSynWUMPost0;
    float* __restrict__ Ebase;
    float* __restrict__ HebbZInSynWUMPost0;
    float* __restrict__ ContrRInSynWUMPost1;
    float* __restrict__ ContrRInSynWUMPost0;
    float* __restrict__ ContrMeanInSynWUMPost1;
    float* __restrict__ ContrMeanInSynWUMPost0;
    float* __restrict__ ContrDInSynWUMPost1;
    float* __restrict__ ContrDInSynWUMPost0;
    float* __restrict__ Beta;
    float* __restrict__ LocMeanInSynWUMPost1;
    float* __restrict__ Noise4;
    float* __restrict__ Noise3;
    float* __restrict__ Noise2;
    float* __restrict__ Noise1;
    float* __restrict__ LogSigma;
    float* __restrict__ LocRInSynWUMPost1;
    float* __restrict__ LocRInSynWUMPost0;
    float* __restrict__ HebbZInSynWUMPost1;
    float* __restrict__ LocMeanInSynWUMPost0;
    float* __restrict__ InhVar;
    float* __restrict__ InhNow;
    float* __restrict__ InhMean;
    float* __restrict__ IPVBar;
    float* __restrict__ IPPsiBar;
    float* __restrict__ IPBias;
    uint32_t numNeurons;
    
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
    float* __restrict__ DeltaG;
    float* __restrict__ RLTrace;
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
    float* __restrict__ RLTrace;
    float* __restrict__ SynSig;
    float* __restrict__ eFiltered;
    float* __restrict__ epsilonA;
    uint32_t* __restrict__ rowLength;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedSynapseSparseInitGroup3
 {
    float* __restrict__ DeltaG;
    float* __restrict__ LogSynSig;
    float* __restrict__ NoiseTrace;
    float* __restrict__ RLNoiseGtrace;
    float* __restrict__ RLNoisePtrace;
    float* __restrict__ SynSig;
    float* __restrict__ eFiltered;
    float* __restrict__ epsilonA;
    uint32_t* __restrict__ rowLength;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
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
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, float* E, float* EpMode, float* PRew, float* PrevVal, float* V, float* ValReg, float* actionTaken, float* outPostInSyn0, float* reward, float* tdError) {
    mergedNeuronInitGroup0[idx].E = E;
    mergedNeuronInitGroup0[idx].EpMode = EpMode;
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
static MergedNeuronInitGroup2 mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, float* V, float* PR, float* PRew, float* PertEps, float* PertEpsTrace, float* PertEpsTraceDivSigma, float* PsiBarInSynWUMPost0, float* PsiInSynWUMPost0, float* RefracTime, float* Sigma, float* SpkCount, float* TdE, float* TdMeanInSynWUMPost0, float* PGEps, float* VE, float* VR, float* ZDecInSynWUMPost0, float* ZDecMeanInSynWUMPost0, float* ZFilterOutSynWUMPre0, float* outPostInSyn0, float* outPostInSyn1, float* outPreOutSyn0, float* outPreOutSyn1, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* IPVBar, float* Beta, float* ContrDInSynWUMPost0, float* ContrMeanInSynWUMPost0, float* ContrRInSynWUMPost0, float* E, float* Ebase, float* FAvgInSynWUMPost0, float* FAvgTraceInSynWUMPost0, float* HebbMeanInSynWUMPost0, float* HebbZInSynWUMPost0, float* IPBias, float* IPPsiBar, float* A, float* InhMean, float* InhNow, float* InhVar, float* LocMeanInSynWUMPost0, float* LocRInSynWUMPost0, float* LogSigma, float* Noise1, float* Noise2, float* Noise3, float* Noise4, float* PG) {
    mergedNeuronInitGroup2[idx].V = V;
    mergedNeuronInitGroup2[idx].PR = PR;
    mergedNeuronInitGroup2[idx].PRew = PRew;
    mergedNeuronInitGroup2[idx].PertEps = PertEps;
    mergedNeuronInitGroup2[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronInitGroup2[idx].PertEpsTraceDivSigma = PertEpsTraceDivSigma;
    mergedNeuronInitGroup2[idx].PsiBarInSynWUMPost0 = PsiBarInSynWUMPost0;
    mergedNeuronInitGroup2[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronInitGroup2[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup2[idx].Sigma = Sigma;
    mergedNeuronInitGroup2[idx].SpkCount = SpkCount;
    mergedNeuronInitGroup2[idx].TdE = TdE;
    mergedNeuronInitGroup2[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronInitGroup2[idx].PGEps = PGEps;
    mergedNeuronInitGroup2[idx].VE = VE;
    mergedNeuronInitGroup2[idx].VR = VR;
    mergedNeuronInitGroup2[idx].ZDecInSynWUMPost0 = ZDecInSynWUMPost0;
    mergedNeuronInitGroup2[idx].ZDecMeanInSynWUMPost0 = ZDecMeanInSynWUMPost0;
    mergedNeuronInitGroup2[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronInitGroup2[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup2[idx].outPostInSyn1 = outPostInSyn1;
    mergedNeuronInitGroup2[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronInitGroup2[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronInitGroup2[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronInitGroup2[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronInitGroup2[idx].IPVBar = IPVBar;
    mergedNeuronInitGroup2[idx].Beta = Beta;
    mergedNeuronInitGroup2[idx].ContrDInSynWUMPost0 = ContrDInSynWUMPost0;
    mergedNeuronInitGroup2[idx].ContrMeanInSynWUMPost0 = ContrMeanInSynWUMPost0;
    mergedNeuronInitGroup2[idx].ContrRInSynWUMPost0 = ContrRInSynWUMPost0;
    mergedNeuronInitGroup2[idx].E = E;
    mergedNeuronInitGroup2[idx].Ebase = Ebase;
    mergedNeuronInitGroup2[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronInitGroup2[idx].FAvgTraceInSynWUMPost0 = FAvgTraceInSynWUMPost0;
    mergedNeuronInitGroup2[idx].HebbMeanInSynWUMPost0 = HebbMeanInSynWUMPost0;
    mergedNeuronInitGroup2[idx].HebbZInSynWUMPost0 = HebbZInSynWUMPost0;
    mergedNeuronInitGroup2[idx].IPBias = IPBias;
    mergedNeuronInitGroup2[idx].IPPsiBar = IPPsiBar;
    mergedNeuronInitGroup2[idx].A = A;
    mergedNeuronInitGroup2[idx].InhMean = InhMean;
    mergedNeuronInitGroup2[idx].InhNow = InhNow;
    mergedNeuronInitGroup2[idx].InhVar = InhVar;
    mergedNeuronInitGroup2[idx].LocMeanInSynWUMPost0 = LocMeanInSynWUMPost0;
    mergedNeuronInitGroup2[idx].LocRInSynWUMPost0 = LocRInSynWUMPost0;
    mergedNeuronInitGroup2[idx].LogSigma = LogSigma;
    mergedNeuronInitGroup2[idx].Noise1 = Noise1;
    mergedNeuronInitGroup2[idx].Noise2 = Noise2;
    mergedNeuronInitGroup2[idx].Noise3 = Noise3;
    mergedNeuronInitGroup2[idx].Noise4 = Noise4;
    mergedNeuronInitGroup2[idx].PG = PG;
}
static MergedNeuronInitGroup3 mergedNeuronInitGroup3[1];
void pushMergedNeuronInitGroup3ToDevice(unsigned int idx, float* VE, float* PertEpsTraceDivSigma, float* PoolBias, float* PsiInSynWUMPost0, float* RefracTime, float* Sigma, float* SpkCount, float* TdE, float* TdMeanInSynWUMPost0, float* V, float* PertEpsTrace, float* VR, float* ZFilterOutSynWUMPre0, float* outPostInSyn0, float* outPostInSyn1, float* outPreOutSyn0, float* outPreOutSyn1, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* Noise1, float* Beta, float* E, float* Ebase, float* FAvgInSynWUMPost0, float* IPBias, float* IPPsiBar, float* IPVBar, float* LogSigma, float* A, float* Noise2, float* Noise3, float* Noise4, float* PG, float* PGEps, float* PR, float* PRew, float* PertEps) {
    mergedNeuronInitGroup3[idx].VE = VE;
    mergedNeuronInitGroup3[idx].PertEpsTraceDivSigma = PertEpsTraceDivSigma;
    mergedNeuronInitGroup3[idx].PoolBias = PoolBias;
    mergedNeuronInitGroup3[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronInitGroup3[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup3[idx].Sigma = Sigma;
    mergedNeuronInitGroup3[idx].SpkCount = SpkCount;
    mergedNeuronInitGroup3[idx].TdE = TdE;
    mergedNeuronInitGroup3[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronInitGroup3[idx].V = V;
    mergedNeuronInitGroup3[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronInitGroup3[idx].VR = VR;
    mergedNeuronInitGroup3[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronInitGroup3[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup3[idx].outPostInSyn1 = outPostInSyn1;
    mergedNeuronInitGroup3[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronInitGroup3[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronInitGroup3[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronInitGroup3[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronInitGroup3[idx].Noise1 = Noise1;
    mergedNeuronInitGroup3[idx].Beta = Beta;
    mergedNeuronInitGroup3[idx].E = E;
    mergedNeuronInitGroup3[idx].Ebase = Ebase;
    mergedNeuronInitGroup3[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronInitGroup3[idx].IPBias = IPBias;
    mergedNeuronInitGroup3[idx].IPPsiBar = IPPsiBar;
    mergedNeuronInitGroup3[idx].IPVBar = IPVBar;
    mergedNeuronInitGroup3[idx].LogSigma = LogSigma;
    mergedNeuronInitGroup3[idx].A = A;
    mergedNeuronInitGroup3[idx].Noise2 = Noise2;
    mergedNeuronInitGroup3[idx].Noise3 = Noise3;
    mergedNeuronInitGroup3[idx].Noise4 = Noise4;
    mergedNeuronInitGroup3[idx].PG = PG;
    mergedNeuronInitGroup3[idx].PGEps = PGEps;
    mergedNeuronInitGroup3[idx].PR = PR;
    mergedNeuronInitGroup3[idx].PRew = PRew;
    mergedNeuronInitGroup3[idx].PertEps = PertEps;
}
static MergedNeuronInitGroup4 mergedNeuronInitGroup4[2];
void pushMergedNeuronInitGroup4ToDevice(unsigned int idx, float* PsiBarInSynWUMPost1, float* TdMeanInSynWUMPost1, float* TdMeanInSynWUMPost0, float* TdE, float* SpkCount, float* Sigma, float* RefracTime, float* PsiInSynWUMPost1, float* PsiInSynWUMPost0, float* V, float* PsiBarInSynWUMPost0, float* PertEpsTraceDivSigma, float* PertEpsTrace, float* PertEps, float* PRew, float* PR, float* A, float* ZFilterOutSynWUMPre1, uint32_t* spkSynSpike0, uint32_t* spkCntSynSpike0, float* outPreOutSyn2, float* outPreOutSyn1, float* outPreOutSyn0, float* outPostInSyn1, float* outPostInSyn0, float* ZFilterOutSynWUMPre2, float* PGEps, float* ZFilterOutSynWUMPre0, float* ZDecMeanInSynWUMPost1, float* ZDecMeanInSynWUMPost0, float* ZDecInSynWUMPost1, float* ZDecInSynWUMPost0, float* VR, float* VE, float* E, float* PG, float* HebbMeanInSynWUMPost1, float* HebbMeanInSynWUMPost0, float* FAvgTraceInSynWUMPost1, float* FAvgTraceInSynWUMPost0, float* FAvgInSynWUMPost1, float* FAvgInSynWUMPost0, float* Ebase, float* HebbZInSynWUMPost0, float* ContrRInSynWUMPost1, float* ContrRInSynWUMPost0, float* ContrMeanInSynWUMPost1, float* ContrMeanInSynWUMPost0, float* ContrDInSynWUMPost1, float* ContrDInSynWUMPost0, float* Beta, float* LocMeanInSynWUMPost1, float* Noise4, float* Noise3, float* Noise2, float* Noise1, float* LogSigma, float* LocRInSynWUMPost1, float* LocRInSynWUMPost0, float* HebbZInSynWUMPost1, float* LocMeanInSynWUMPost0, float* InhVar, float* InhNow, float* InhMean, float* IPVBar, float* IPPsiBar, float* IPBias, uint32_t numNeurons) {
    mergedNeuronInitGroup4[idx].PsiBarInSynWUMPost1 = PsiBarInSynWUMPost1;
    mergedNeuronInitGroup4[idx].TdMeanInSynWUMPost1 = TdMeanInSynWUMPost1;
    mergedNeuronInitGroup4[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronInitGroup4[idx].TdE = TdE;
    mergedNeuronInitGroup4[idx].SpkCount = SpkCount;
    mergedNeuronInitGroup4[idx].Sigma = Sigma;
    mergedNeuronInitGroup4[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup4[idx].PsiInSynWUMPost1 = PsiInSynWUMPost1;
    mergedNeuronInitGroup4[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronInitGroup4[idx].V = V;
    mergedNeuronInitGroup4[idx].PsiBarInSynWUMPost0 = PsiBarInSynWUMPost0;
    mergedNeuronInitGroup4[idx].PertEpsTraceDivSigma = PertEpsTraceDivSigma;
    mergedNeuronInitGroup4[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronInitGroup4[idx].PertEps = PertEps;
    mergedNeuronInitGroup4[idx].PRew = PRew;
    mergedNeuronInitGroup4[idx].PR = PR;
    mergedNeuronInitGroup4[idx].A = A;
    mergedNeuronInitGroup4[idx].ZFilterOutSynWUMPre1 = ZFilterOutSynWUMPre1;
    mergedNeuronInitGroup4[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronInitGroup4[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronInitGroup4[idx].outPreOutSyn2 = outPreOutSyn2;
    mergedNeuronInitGroup4[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronInitGroup4[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronInitGroup4[idx].outPostInSyn1 = outPostInSyn1;
    mergedNeuronInitGroup4[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronInitGroup4[idx].ZFilterOutSynWUMPre2 = ZFilterOutSynWUMPre2;
    mergedNeuronInitGroup4[idx].PGEps = PGEps;
    mergedNeuronInitGroup4[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronInitGroup4[idx].ZDecMeanInSynWUMPost1 = ZDecMeanInSynWUMPost1;
    mergedNeuronInitGroup4[idx].ZDecMeanInSynWUMPost0 = ZDecMeanInSynWUMPost0;
    mergedNeuronInitGroup4[idx].ZDecInSynWUMPost1 = ZDecInSynWUMPost1;
    mergedNeuronInitGroup4[idx].ZDecInSynWUMPost0 = ZDecInSynWUMPost0;
    mergedNeuronInitGroup4[idx].VR = VR;
    mergedNeuronInitGroup4[idx].VE = VE;
    mergedNeuronInitGroup4[idx].E = E;
    mergedNeuronInitGroup4[idx].PG = PG;
    mergedNeuronInitGroup4[idx].HebbMeanInSynWUMPost1 = HebbMeanInSynWUMPost1;
    mergedNeuronInitGroup4[idx].HebbMeanInSynWUMPost0 = HebbMeanInSynWUMPost0;
    mergedNeuronInitGroup4[idx].FAvgTraceInSynWUMPost1 = FAvgTraceInSynWUMPost1;
    mergedNeuronInitGroup4[idx].FAvgTraceInSynWUMPost0 = FAvgTraceInSynWUMPost0;
    mergedNeuronInitGroup4[idx].FAvgInSynWUMPost1 = FAvgInSynWUMPost1;
    mergedNeuronInitGroup4[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronInitGroup4[idx].Ebase = Ebase;
    mergedNeuronInitGroup4[idx].HebbZInSynWUMPost0 = HebbZInSynWUMPost0;
    mergedNeuronInitGroup4[idx].ContrRInSynWUMPost1 = ContrRInSynWUMPost1;
    mergedNeuronInitGroup4[idx].ContrRInSynWUMPost0 = ContrRInSynWUMPost0;
    mergedNeuronInitGroup4[idx].ContrMeanInSynWUMPost1 = ContrMeanInSynWUMPost1;
    mergedNeuronInitGroup4[idx].ContrMeanInSynWUMPost0 = ContrMeanInSynWUMPost0;
    mergedNeuronInitGroup4[idx].ContrDInSynWUMPost1 = ContrDInSynWUMPost1;
    mergedNeuronInitGroup4[idx].ContrDInSynWUMPost0 = ContrDInSynWUMPost0;
    mergedNeuronInitGroup4[idx].Beta = Beta;
    mergedNeuronInitGroup4[idx].LocMeanInSynWUMPost1 = LocMeanInSynWUMPost1;
    mergedNeuronInitGroup4[idx].Noise4 = Noise4;
    mergedNeuronInitGroup4[idx].Noise3 = Noise3;
    mergedNeuronInitGroup4[idx].Noise2 = Noise2;
    mergedNeuronInitGroup4[idx].Noise1 = Noise1;
    mergedNeuronInitGroup4[idx].LogSigma = LogSigma;
    mergedNeuronInitGroup4[idx].LocRInSynWUMPost1 = LocRInSynWUMPost1;
    mergedNeuronInitGroup4[idx].LocRInSynWUMPost0 = LocRInSynWUMPost0;
    mergedNeuronInitGroup4[idx].HebbZInSynWUMPost1 = HebbZInSynWUMPost1;
    mergedNeuronInitGroup4[idx].LocMeanInSynWUMPost0 = LocMeanInSynWUMPost0;
    mergedNeuronInitGroup4[idx].InhVar = InhVar;
    mergedNeuronInitGroup4[idx].InhNow = InhNow;
    mergedNeuronInitGroup4[idx].InhMean = InhMean;
    mergedNeuronInitGroup4[idx].IPVBar = IPVBar;
    mergedNeuronInitGroup4[idx].IPPsiBar = IPPsiBar;
    mergedNeuronInitGroup4[idx].IPBias = IPBias;
    mergedNeuronInitGroup4[idx].numNeurons = numNeurons;
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
static MergedSynapseConnectivityInitGroup0 mergedSynapseConnectivityInitGroup0[6];
void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, uint32_t* ind, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t numTrgNeurons, float probLogRecip, uint32_t rowStride) {
    mergedSynapseConnectivityInitGroup0[idx].ind = ind;
    mergedSynapseConnectivityInitGroup0[idx].rowLength = rowLength;
    mergedSynapseConnectivityInitGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseConnectivityInitGroup0[idx].numTrgNeurons = numTrgNeurons;
    mergedSynapseConnectivityInitGroup0[idx].probLogRecip = probLogRecip;
    mergedSynapseConnectivityInitGroup0[idx].rowStride = rowStride;
}
static MergedSynapseSparseInitGroup0 mergedSynapseSparseInitGroup0[2];
void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, float* DeltaG, float* RLTrace, uint32_t* rowLength) {
    mergedSynapseSparseInitGroup0[idx].DeltaG = DeltaG;
    mergedSynapseSparseInitGroup0[idx].RLTrace = RLTrace;
    mergedSynapseSparseInitGroup0[idx].rowLength = rowLength;
}
static MergedSynapseSparseInitGroup1 mergedSynapseSparseInitGroup1[6];
void pushMergedSynapseSparseInitGroup1ToDevice(unsigned int idx, float* DeltaG, float* RLTrace, float* g, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t rowStride, float sdg) {
    mergedSynapseSparseInitGroup1[idx].DeltaG = DeltaG;
    mergedSynapseSparseInitGroup1[idx].RLTrace = RLTrace;
    mergedSynapseSparseInitGroup1[idx].g = g;
    mergedSynapseSparseInitGroup1[idx].rowLength = rowLength;
    mergedSynapseSparseInitGroup1[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseSparseInitGroup1[idx].rowStride = rowStride;
    mergedSynapseSparseInitGroup1[idx].sdg = sdg;
}
static MergedSynapseSparseInitGroup2 mergedSynapseSparseInitGroup2[2];
void pushMergedSynapseSparseInitGroup2ToDevice(unsigned int idx, float* DeltaG, float* LogSynSig, float* NoiseTrace, float* RLTrace, float* SynSig, float* eFiltered, float* epsilonA, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseSparseInitGroup2[idx].DeltaG = DeltaG;
    mergedSynapseSparseInitGroup2[idx].LogSynSig = LogSynSig;
    mergedSynapseSparseInitGroup2[idx].NoiseTrace = NoiseTrace;
    mergedSynapseSparseInitGroup2[idx].RLTrace = RLTrace;
    mergedSynapseSparseInitGroup2[idx].SynSig = SynSig;
    mergedSynapseSparseInitGroup2[idx].eFiltered = eFiltered;
    mergedSynapseSparseInitGroup2[idx].epsilonA = epsilonA;
    mergedSynapseSparseInitGroup2[idx].rowLength = rowLength;
    mergedSynapseSparseInitGroup2[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseSparseInitGroup2[idx].rowStride = rowStride;
}
static MergedSynapseSparseInitGroup3 mergedSynapseSparseInitGroup3[8];
void pushMergedSynapseSparseInitGroup3ToDevice(unsigned int idx, float* DeltaG, float* LogSynSig, float* NoiseTrace, float* RLNoiseGtrace, float* RLNoisePtrace, float* SynSig, float* eFiltered, float* epsilonA, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseSparseInitGroup3[idx].DeltaG = DeltaG;
    mergedSynapseSparseInitGroup3[idx].LogSynSig = LogSynSig;
    mergedSynapseSparseInitGroup3[idx].NoiseTrace = NoiseTrace;
    mergedSynapseSparseInitGroup3[idx].RLNoiseGtrace = RLNoiseGtrace;
    mergedSynapseSparseInitGroup3[idx].RLNoisePtrace = RLNoisePtrace;
    mergedSynapseSparseInitGroup3[idx].SynSig = SynSig;
    mergedSynapseSparseInitGroup3[idx].eFiltered = eFiltered;
    mergedSynapseSparseInitGroup3[idx].epsilonA = epsilonA;
    mergedSynapseSparseInitGroup3[idx].rowLength = rowLength;
    mergedSynapseSparseInitGroup3[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseSparseInitGroup3[idx].rowStride = rowStride;
}
static MergedCustomWUUpdateSparseInitGroup0 mergedCustomWUUpdateSparseInitGroup0[11];
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
static MergedCustomConnectivityUpdatePreInitGroup0 mergedCustomConnectivityUpdatePreInitGroup0[10];
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
             {
                for (unsigned int i = 0; i < ((1u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->EpMode[i] = initVal;
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
        for(unsigned int g = 0; g < 1; g++) {
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
                    group->SpkCount[i] = initVal;
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
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->InhNow[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->InhMean[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->InhVar[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->IPBias[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (4.731000000e-02f);
                    group->IPPsiBar[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->IPVBar[i] = initVal;
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
                group->outPostInSyn1[i] = 0.000000000e+00f;
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
                    group->FAvgTraceInSynWUMPost0[i] = initVal;
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
                    group->HebbZInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->HebbMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrDInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrRInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->LocRInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->LocMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZDecInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZDecMeanInSynWUMPost0[i] = initVal;
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
                    group->SpkCount[i] = initVal;
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
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->IPBias[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (4.731000000e-02f);
                    group->IPPsiBar[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->IPVBar[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < ((48u)); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PoolBias[i] = initVal;
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
                group->outPostInSyn1[i] = 0.000000000e+00f;
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
                    group->TdMeanInSynWUMPost0[i] = initVal;
                }
            }
        }
    }
     {
        // merged neuron init group 4
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedNeuronInitGroup4[g]; 
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->A[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (1.740000000e-02f);
                    group->Beta[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTrace[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEpsTraceDivSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PertEps[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Sigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (-4.605170186e+00f);
                    group->LogSigma[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->E[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Ebase[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise3[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->Noise4[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->SpkCount[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PG[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PRew[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VE[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->VR[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PGEps[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->InhNow[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->InhMean[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->InhVar[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->IPBias[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (4.731000000e-02f);
                    group->IPPsiBar[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->IPVBar[i] = initVal;
                }
            }
            for (unsigned int i = 0; i < (group->numNeurons); i++) {
                group->spkSynSpike0[i] = 0;
            }
            group->spkCntSynSpike0[0] = 0;
            for (unsigned int i = 0; i < (group->numNeurons); i++) {
                group->outPostInSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < (group->numNeurons); i++) {
                group->outPostInSyn1[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < (group->numNeurons); i++) {
                group->outPreOutSyn0[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < (group->numNeurons); i++) {
                group->outPreOutSyn1[i] = 0.000000000e+00f;
            }
            for (unsigned int i = 0; i < (group->numNeurons); i++) {
                group->outPreOutSyn2[i] = 0.000000000e+00f;
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZFilterOutSynWUMPre2[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgTraceInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiBarInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->HebbZInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->HebbMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrDInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrRInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->LocRInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->LocMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZDecInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZDecMeanInSynWUMPost0[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->FAvgTraceInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->PsiBarInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->TdMeanInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->HebbZInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->HebbMeanInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrDInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrRInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ContrMeanInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->LocRInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->LocMeanInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZDecInSynWUMPost1[i] = initVal;
                }
            }
             {
                for (unsigned int i = 0; i < (group->numNeurons); i++) {
                    float initVal;
                    initVal = (0.000000000e+00f);
                    group->ZDecMeanInSynWUMPost1[i] = initVal;
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
        for(unsigned int g = 0; g < 10; g++) {
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
        for(unsigned int g = 0; g < 6; g++) {
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
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedSynapseSparseInitGroup0[g]; 
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < (48u); i++) {
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * (1u)) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->DeltaG[idSyn] = initVal;
                    }
                }
                 {
                    for (unsigned int j = 0; j < group->rowLength[i]; j++) {
                        const unsigned int idSyn = (i * (1u)) + j;
                        float initVal;
                        initVal = (0.000000000e+00f);
                        group->RLTrace[idSyn] = initVal;
                    }
                }
            }
        }
    }
     {
        // merged sparse synapse init group 1
        for(unsigned int g = 0; g < 6; g++) {
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
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedSynapseSparseInitGroup2[g]; 
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < group->numSrcNeurons; i++) {
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
                        initVal = (-8.326000000e-01f);
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
                        group->RLTrace[idSyn] = initVal;
                    }
                }
            }
        }
    }
     {
        // merged sparse synapse init group 3
        for(unsigned int g = 0; g < 8; g++) {
            const auto *group = &mergedSynapseSparseInitGroup3[g]; 
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < group->numSrcNeurons; i++) {
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
                        initVal = (-8.326000000e-01f);
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
        for(unsigned int g = 0; g < 11; g++) {
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
