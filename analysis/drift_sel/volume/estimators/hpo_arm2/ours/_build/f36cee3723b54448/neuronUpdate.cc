#include "definitions.h"

struct MergedNeuronUpdateGroup0
 {
    float* __restrict__ E;
    float* __restrict__ EpMode;
    float* __restrict__ PrevVal;
    float* __restrict__ V;
    float* __restrict__ YTrue;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ reward;
    float* __restrict__ tdError;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    float* __restrict__ E;
    float* __restrict__ PG;
    float* __restrict__ PRew;
    float* __restrict__ TdE;
    float* __restrict__ V;
    float* __restrict__ YTrue;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ pre_E;
    float* __restrict__ pre_PG;
    float* __restrict__ pre_PRew;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    float* __restrict__ VE;
    float* __restrict__ PRew;
    float* __restrict__ PertEps;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ PsiBarInSynWUMPost0;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ SpkCount;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ V;
    float* __restrict__ PR;
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
    float* __restrict__ InhNow;
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
    float* __restrict__ InhMean;
    float* __restrict__ A;
    float* __restrict__ InhVar;
    float* __restrict__ LocMeanInSynWUMPost0;
    float* __restrict__ LocRInSynWUMPost0;
    float* __restrict__ LogSigma;
    float* __restrict__ Noise1;
    float* __restrict__ Noise2;
    float* __restrict__ Noise3;
    float* __restrict__ Noise4;
    float* __restrict__ PG;
    float* __restrict__ PGEps;
    
}
;
struct MergedNeuronUpdateGroup3
 {
    float* __restrict__ VE;
    float* __restrict__ A;
    float* __restrict__ PRew;
    float* __restrict__ PertEps;
    float* __restrict__ PertEpsTrace;
    float* __restrict__ PsiBarInSynWUMPost0;
    float* __restrict__ PsiBarInSynWUMPost1;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ PsiInSynWUMPost1;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ SpkCount;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ TdMeanInSynWUMPost1;
    float* __restrict__ V;
    float* __restrict__ PR;
    float* __restrict__ VR;
    float* __restrict__ ZDecInSynWUMPost0;
    float* __restrict__ ZDecInSynWUMPost1;
    float* __restrict__ ZDecMeanInSynWUMPost0;
    float* __restrict__ ZDecMeanInSynWUMPost1;
    float* __restrict__ ZFilterOutSynWUMPre0;
    float* __restrict__ ZFilterOutSynWUMPre1;
    float* __restrict__ ZFilterOutSynWUMPre2;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPostInSyn1;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPreOutSyn1;
    float* __restrict__ outPreOutSyn2;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    float* __restrict__ HebbMeanInSynWUMPost1;
    float* __restrict__ Beta;
    float* __restrict__ ContrDInSynWUMPost0;
    float* __restrict__ ContrDInSynWUMPost1;
    float* __restrict__ ContrMeanInSynWUMPost0;
    float* __restrict__ ContrMeanInSynWUMPost1;
    float* __restrict__ ContrRInSynWUMPost0;
    float* __restrict__ ContrRInSynWUMPost1;
    float* __restrict__ E;
    float* __restrict__ Ebase;
    float* __restrict__ FAvgInSynWUMPost0;
    float* __restrict__ FAvgInSynWUMPost1;
    float* __restrict__ FAvgTraceInSynWUMPost0;
    float* __restrict__ FAvgTraceInSynWUMPost1;
    float* __restrict__ HebbMeanInSynWUMPost0;
    float* __restrict__ PGEps;
    float* __restrict__ HebbZInSynWUMPost0;
    float* __restrict__ HebbZInSynWUMPost1;
    float* __restrict__ InhMean;
    float* __restrict__ InhNow;
    float* __restrict__ InhVar;
    float* __restrict__ LocMeanInSynWUMPost0;
    float* __restrict__ LocMeanInSynWUMPost1;
    float* __restrict__ LocRInSynWUMPost0;
    float* __restrict__ LocRInSynWUMPost1;
    float* __restrict__ LogSigma;
    float* __restrict__ Noise1;
    float* __restrict__ Noise2;
    float* __restrict__ Noise3;
    float* __restrict__ Noise4;
    float* __restrict__ PG;
    uint32_t numNeurons;
    
}
;
struct MergedNeuronUpdateGroup4
 {
    float* __restrict__ VR;
    float* __restrict__ PsiInSynWUMPost0;
    float* __restrict__ RefracTime;
    float* __restrict__ Sigma;
    float* __restrict__ SpkCount;
    float* __restrict__ TdE;
    float* __restrict__ TdMeanInSynWUMPost0;
    float* __restrict__ V;
    float* __restrict__ VE;
    float* __restrict__ PoolBias;
    float* __restrict__ ZFilterOutSynWUMPre0;
    float* __restrict__ outPostInSyn0;
    float* __restrict__ outPostInSyn1;
    float* __restrict__ outPreOutSyn0;
    float* __restrict__ outPreOutSyn1;
    uint32_t* __restrict__ spkCntSynSpike0;
    uint32_t* __restrict__ spkSynSpike0;
    float* __restrict__ A;
    float* __restrict__ PertEpsTrace;
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
    
}
;
struct MergedNeuronUpdateGroup5
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
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    uint32_t* __restrict__ spkCntSynSpike0;
    
}
;
static MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, float* E, float* EpMode, float* PrevVal, float* V, float* YTrue, float* outPostInSyn0, float* reward, float* tdError) {
    mergedNeuronUpdateGroup0[idx].E = E;
    mergedNeuronUpdateGroup0[idx].EpMode = EpMode;
    mergedNeuronUpdateGroup0[idx].PrevVal = PrevVal;
    mergedNeuronUpdateGroup0[idx].V = V;
    mergedNeuronUpdateGroup0[idx].YTrue = YTrue;
    mergedNeuronUpdateGroup0[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronUpdateGroup0[idx].reward = reward;
    mergedNeuronUpdateGroup0[idx].tdError = tdError;
}
static MergedNeuronUpdateGroup1 mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, float* E, float* PG, float* PRew, float* TdE, float* V, float* YTrue, float* outPostInSyn0, float* outPreOutSyn0, float* pre_E, float* pre_PG, float* pre_PRew) {
    mergedNeuronUpdateGroup1[idx].E = E;
    mergedNeuronUpdateGroup1[idx].PG = PG;
    mergedNeuronUpdateGroup1[idx].PRew = PRew;
    mergedNeuronUpdateGroup1[idx].TdE = TdE;
    mergedNeuronUpdateGroup1[idx].V = V;
    mergedNeuronUpdateGroup1[idx].YTrue = YTrue;
    mergedNeuronUpdateGroup1[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronUpdateGroup1[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronUpdateGroup1[idx].pre_E = pre_E;
    mergedNeuronUpdateGroup1[idx].pre_PG = pre_PG;
    mergedNeuronUpdateGroup1[idx].pre_PRew = pre_PRew;
}
static MergedNeuronUpdateGroup2 mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, float* VE, float* PRew, float* PertEps, float* PertEpsTrace, float* PsiBarInSynWUMPost0, float* PsiInSynWUMPost0, float* RefracTime, float* Sigma, float* SpkCount, float* TdE, float* TdMeanInSynWUMPost0, float* V, float* PR, float* VR, float* ZDecInSynWUMPost0, float* ZDecMeanInSynWUMPost0, float* ZFilterOutSynWUMPre0, float* outPostInSyn0, float* outPostInSyn1, float* outPreOutSyn0, float* outPreOutSyn1, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* InhNow, float* Beta, float* ContrDInSynWUMPost0, float* ContrMeanInSynWUMPost0, float* ContrRInSynWUMPost0, float* E, float* Ebase, float* FAvgInSynWUMPost0, float* FAvgTraceInSynWUMPost0, float* HebbMeanInSynWUMPost0, float* HebbZInSynWUMPost0, float* InhMean, float* A, float* InhVar, float* LocMeanInSynWUMPost0, float* LocRInSynWUMPost0, float* LogSigma, float* Noise1, float* Noise2, float* Noise3, float* Noise4, float* PG, float* PGEps) {
    mergedNeuronUpdateGroup2[idx].VE = VE;
    mergedNeuronUpdateGroup2[idx].PRew = PRew;
    mergedNeuronUpdateGroup2[idx].PertEps = PertEps;
    mergedNeuronUpdateGroup2[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronUpdateGroup2[idx].PsiBarInSynWUMPost0 = PsiBarInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].RefracTime = RefracTime;
    mergedNeuronUpdateGroup2[idx].Sigma = Sigma;
    mergedNeuronUpdateGroup2[idx].SpkCount = SpkCount;
    mergedNeuronUpdateGroup2[idx].TdE = TdE;
    mergedNeuronUpdateGroup2[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].V = V;
    mergedNeuronUpdateGroup2[idx].PR = PR;
    mergedNeuronUpdateGroup2[idx].VR = VR;
    mergedNeuronUpdateGroup2[idx].ZDecInSynWUMPost0 = ZDecInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].ZDecMeanInSynWUMPost0 = ZDecMeanInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronUpdateGroup2[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronUpdateGroup2[idx].outPostInSyn1 = outPostInSyn1;
    mergedNeuronUpdateGroup2[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronUpdateGroup2[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronUpdateGroup2[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronUpdateGroup2[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronUpdateGroup2[idx].InhNow = InhNow;
    mergedNeuronUpdateGroup2[idx].Beta = Beta;
    mergedNeuronUpdateGroup2[idx].ContrDInSynWUMPost0 = ContrDInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].ContrMeanInSynWUMPost0 = ContrMeanInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].ContrRInSynWUMPost0 = ContrRInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].E = E;
    mergedNeuronUpdateGroup2[idx].Ebase = Ebase;
    mergedNeuronUpdateGroup2[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].FAvgTraceInSynWUMPost0 = FAvgTraceInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].HebbMeanInSynWUMPost0 = HebbMeanInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].HebbZInSynWUMPost0 = HebbZInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].InhMean = InhMean;
    mergedNeuronUpdateGroup2[idx].A = A;
    mergedNeuronUpdateGroup2[idx].InhVar = InhVar;
    mergedNeuronUpdateGroup2[idx].LocMeanInSynWUMPost0 = LocMeanInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].LocRInSynWUMPost0 = LocRInSynWUMPost0;
    mergedNeuronUpdateGroup2[idx].LogSigma = LogSigma;
    mergedNeuronUpdateGroup2[idx].Noise1 = Noise1;
    mergedNeuronUpdateGroup2[idx].Noise2 = Noise2;
    mergedNeuronUpdateGroup2[idx].Noise3 = Noise3;
    mergedNeuronUpdateGroup2[idx].Noise4 = Noise4;
    mergedNeuronUpdateGroup2[idx].PG = PG;
    mergedNeuronUpdateGroup2[idx].PGEps = PGEps;
}
static MergedNeuronUpdateGroup3 mergedNeuronUpdateGroup3[2];
void pushMergedNeuronUpdateGroup3ToDevice(unsigned int idx, float* VE, float* A, float* PRew, float* PertEps, float* PertEpsTrace, float* PsiBarInSynWUMPost0, float* PsiBarInSynWUMPost1, float* PsiInSynWUMPost0, float* PsiInSynWUMPost1, float* RefracTime, float* Sigma, float* SpkCount, float* TdE, float* TdMeanInSynWUMPost0, float* TdMeanInSynWUMPost1, float* V, float* PR, float* VR, float* ZDecInSynWUMPost0, float* ZDecInSynWUMPost1, float* ZDecMeanInSynWUMPost0, float* ZDecMeanInSynWUMPost1, float* ZFilterOutSynWUMPre0, float* ZFilterOutSynWUMPre1, float* ZFilterOutSynWUMPre2, float* outPostInSyn0, float* outPostInSyn1, float* outPreOutSyn0, float* outPreOutSyn1, float* outPreOutSyn2, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* HebbMeanInSynWUMPost1, float* Beta, float* ContrDInSynWUMPost0, float* ContrDInSynWUMPost1, float* ContrMeanInSynWUMPost0, float* ContrMeanInSynWUMPost1, float* ContrRInSynWUMPost0, float* ContrRInSynWUMPost1, float* E, float* Ebase, float* FAvgInSynWUMPost0, float* FAvgInSynWUMPost1, float* FAvgTraceInSynWUMPost0, float* FAvgTraceInSynWUMPost1, float* HebbMeanInSynWUMPost0, float* PGEps, float* HebbZInSynWUMPost0, float* HebbZInSynWUMPost1, float* InhMean, float* InhNow, float* InhVar, float* LocMeanInSynWUMPost0, float* LocMeanInSynWUMPost1, float* LocRInSynWUMPost0, float* LocRInSynWUMPost1, float* LogSigma, float* Noise1, float* Noise2, float* Noise3, float* Noise4, float* PG, uint32_t numNeurons) {
    mergedNeuronUpdateGroup3[idx].VE = VE;
    mergedNeuronUpdateGroup3[idx].A = A;
    mergedNeuronUpdateGroup3[idx].PRew = PRew;
    mergedNeuronUpdateGroup3[idx].PertEps = PertEps;
    mergedNeuronUpdateGroup3[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronUpdateGroup3[idx].PsiBarInSynWUMPost0 = PsiBarInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].PsiBarInSynWUMPost1 = PsiBarInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].PsiInSynWUMPost1 = PsiInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].RefracTime = RefracTime;
    mergedNeuronUpdateGroup3[idx].Sigma = Sigma;
    mergedNeuronUpdateGroup3[idx].SpkCount = SpkCount;
    mergedNeuronUpdateGroup3[idx].TdE = TdE;
    mergedNeuronUpdateGroup3[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].TdMeanInSynWUMPost1 = TdMeanInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].V = V;
    mergedNeuronUpdateGroup3[idx].PR = PR;
    mergedNeuronUpdateGroup3[idx].VR = VR;
    mergedNeuronUpdateGroup3[idx].ZDecInSynWUMPost0 = ZDecInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].ZDecInSynWUMPost1 = ZDecInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].ZDecMeanInSynWUMPost0 = ZDecMeanInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].ZDecMeanInSynWUMPost1 = ZDecMeanInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronUpdateGroup3[idx].ZFilterOutSynWUMPre1 = ZFilterOutSynWUMPre1;
    mergedNeuronUpdateGroup3[idx].ZFilterOutSynWUMPre2 = ZFilterOutSynWUMPre2;
    mergedNeuronUpdateGroup3[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronUpdateGroup3[idx].outPostInSyn1 = outPostInSyn1;
    mergedNeuronUpdateGroup3[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronUpdateGroup3[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronUpdateGroup3[idx].outPreOutSyn2 = outPreOutSyn2;
    mergedNeuronUpdateGroup3[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronUpdateGroup3[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronUpdateGroup3[idx].HebbMeanInSynWUMPost1 = HebbMeanInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].Beta = Beta;
    mergedNeuronUpdateGroup3[idx].ContrDInSynWUMPost0 = ContrDInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].ContrDInSynWUMPost1 = ContrDInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].ContrMeanInSynWUMPost0 = ContrMeanInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].ContrMeanInSynWUMPost1 = ContrMeanInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].ContrRInSynWUMPost0 = ContrRInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].ContrRInSynWUMPost1 = ContrRInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].E = E;
    mergedNeuronUpdateGroup3[idx].Ebase = Ebase;
    mergedNeuronUpdateGroup3[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].FAvgInSynWUMPost1 = FAvgInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].FAvgTraceInSynWUMPost0 = FAvgTraceInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].FAvgTraceInSynWUMPost1 = FAvgTraceInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].HebbMeanInSynWUMPost0 = HebbMeanInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].PGEps = PGEps;
    mergedNeuronUpdateGroup3[idx].HebbZInSynWUMPost0 = HebbZInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].HebbZInSynWUMPost1 = HebbZInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].InhMean = InhMean;
    mergedNeuronUpdateGroup3[idx].InhNow = InhNow;
    mergedNeuronUpdateGroup3[idx].InhVar = InhVar;
    mergedNeuronUpdateGroup3[idx].LocMeanInSynWUMPost0 = LocMeanInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].LocMeanInSynWUMPost1 = LocMeanInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].LocRInSynWUMPost0 = LocRInSynWUMPost0;
    mergedNeuronUpdateGroup3[idx].LocRInSynWUMPost1 = LocRInSynWUMPost1;
    mergedNeuronUpdateGroup3[idx].LogSigma = LogSigma;
    mergedNeuronUpdateGroup3[idx].Noise1 = Noise1;
    mergedNeuronUpdateGroup3[idx].Noise2 = Noise2;
    mergedNeuronUpdateGroup3[idx].Noise3 = Noise3;
    mergedNeuronUpdateGroup3[idx].Noise4 = Noise4;
    mergedNeuronUpdateGroup3[idx].PG = PG;
    mergedNeuronUpdateGroup3[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup4 mergedNeuronUpdateGroup4[1];
void pushMergedNeuronUpdateGroup4ToDevice(unsigned int idx, float* VR, float* PsiInSynWUMPost0, float* RefracTime, float* Sigma, float* SpkCount, float* TdE, float* TdMeanInSynWUMPost0, float* V, float* VE, float* PoolBias, float* ZFilterOutSynWUMPre0, float* outPostInSyn0, float* outPostInSyn1, float* outPreOutSyn0, float* outPreOutSyn1, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0, float* A, float* PertEpsTrace, float* PertEps, float* PRew, float* PR, float* PGEps, float* PG, float* Noise4, float* Noise3, float* Noise2, float* Noise1, float* LogSigma, float* FAvgInSynWUMPost0, float* Ebase, float* E, float* Beta) {
    mergedNeuronUpdateGroup4[idx].VR = VR;
    mergedNeuronUpdateGroup4[idx].PsiInSynWUMPost0 = PsiInSynWUMPost0;
    mergedNeuronUpdateGroup4[idx].RefracTime = RefracTime;
    mergedNeuronUpdateGroup4[idx].Sigma = Sigma;
    mergedNeuronUpdateGroup4[idx].SpkCount = SpkCount;
    mergedNeuronUpdateGroup4[idx].TdE = TdE;
    mergedNeuronUpdateGroup4[idx].TdMeanInSynWUMPost0 = TdMeanInSynWUMPost0;
    mergedNeuronUpdateGroup4[idx].V = V;
    mergedNeuronUpdateGroup4[idx].VE = VE;
    mergedNeuronUpdateGroup4[idx].PoolBias = PoolBias;
    mergedNeuronUpdateGroup4[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronUpdateGroup4[idx].outPostInSyn0 = outPostInSyn0;
    mergedNeuronUpdateGroup4[idx].outPostInSyn1 = outPostInSyn1;
    mergedNeuronUpdateGroup4[idx].outPreOutSyn0 = outPreOutSyn0;
    mergedNeuronUpdateGroup4[idx].outPreOutSyn1 = outPreOutSyn1;
    mergedNeuronUpdateGroup4[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronUpdateGroup4[idx].spkSynSpike0 = spkSynSpike0;
    mergedNeuronUpdateGroup4[idx].A = A;
    mergedNeuronUpdateGroup4[idx].PertEpsTrace = PertEpsTrace;
    mergedNeuronUpdateGroup4[idx].PertEps = PertEps;
    mergedNeuronUpdateGroup4[idx].PRew = PRew;
    mergedNeuronUpdateGroup4[idx].PR = PR;
    mergedNeuronUpdateGroup4[idx].PGEps = PGEps;
    mergedNeuronUpdateGroup4[idx].PG = PG;
    mergedNeuronUpdateGroup4[idx].Noise4 = Noise4;
    mergedNeuronUpdateGroup4[idx].Noise3 = Noise3;
    mergedNeuronUpdateGroup4[idx].Noise2 = Noise2;
    mergedNeuronUpdateGroup4[idx].Noise1 = Noise1;
    mergedNeuronUpdateGroup4[idx].LogSigma = LogSigma;
    mergedNeuronUpdateGroup4[idx].FAvgInSynWUMPost0 = FAvgInSynWUMPost0;
    mergedNeuronUpdateGroup4[idx].Ebase = Ebase;
    mergedNeuronUpdateGroup4[idx].E = E;
    mergedNeuronUpdateGroup4[idx].Beta = Beta;
}
static MergedNeuronUpdateGroup5 mergedNeuronUpdateGroup5[1];
void pushMergedNeuronUpdateGroup5ToDevice(unsigned int idx, float* Input, float* Noise1, float* Noise2, float* Noise3, float* Noise4, float* ZFilterOutSynWUMPre0, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0) {
    mergedNeuronUpdateGroup5[idx].Input = Input;
    mergedNeuronUpdateGroup5[idx].Noise1 = Noise1;
    mergedNeuronUpdateGroup5[idx].Noise2 = Noise2;
    mergedNeuronUpdateGroup5[idx].Noise3 = Noise3;
    mergedNeuronUpdateGroup5[idx].Noise4 = Noise4;
    mergedNeuronUpdateGroup5[idx].ZFilterOutSynWUMPre0 = ZFilterOutSynWUMPre0;
    mergedNeuronUpdateGroup5[idx].spkCntSynSpike0 = spkCntSynSpike0;
    mergedNeuronUpdateGroup5[idx].spkSynSpike0 = spkSynSpike0;
}
static MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[5];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, uint32_t* spkCntSynSpike0) {
    mergedNeuronSpikeQueueUpdateGroup0[idx].spkCntSynSpike0 = spkCntSynSpike0;
}
void pushMergedNeuronUpdate0YTrueToDevice(unsigned int idx, float* value) {
    mergedNeuronUpdateGroup0[idx].YTrue = value;
}
void pushMergedNeuronUpdate1YTrueToDevice(unsigned int idx, float* value) {
    mergedNeuronUpdateGroup1[idx].YTrue = value;
}
void updateNeurons(float t) {
     {
        // merged neuron spike queue update group 0
        for(unsigned int g = 0; g < 5; g++) {
            const auto *group = &mergedNeuronSpikeQueueUpdateGroup0[g]; 
             {
                // spike queue update 0
                group->spkCntSynSpike0[0] = 0;
            }
        }
    }
     {
        // merged neuron update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup0[g]; 
            
            for(unsigned int i = 0; i < (1u); i++) {
                float Isyn = 0;
                float _lPrevVal = group->PrevVal[i];
                float _lEpMode = group->EpMode[i];
                float _lreward = group->reward[i];
                float _ltdError = group->tdError[i];
                float _lE = group->E[i];
                float _lV = group->V[i];
                 {
                    // postsynaptic model 0
                    float linSyn = group->outPostInSyn0[i];
                    Isyn += linSyn;
                    linSyn = 0;
                    group->outPostInSyn0[i] = linSyn;
                }
                // calculate membrane potential
                _lV = ((9.048374180e-01f) * _lV) + Isyn + (0.000000000e+00f);
                const uint32_t _timestep = ((uint32_t)round(t / 1.000000000e+00f)) % 1;
                const uint32_t _index = (0 * 1 * (1u)) + (_timestep * (1u)) + i;
                const float _yTrue = group->YTrue[_index];
                const float _boot = (_lEpMode > 0.5f && _lEpMode < 1.5f) ? 0.0f : _lV;
                _lE = (_lEpMode > 1.5f) ? 0.0f : _boot * 0.9900133645233035f + _lreward - _lPrevVal;
                _lPrevVal = _boot;
                _lreward *= 0.9261187281287935f;
                _ltdError = 0;
                group->PrevVal[i] = _lPrevVal;
                group->EpMode[i] = _lEpMode;
                group->reward[i] = _lreward;
                group->tdError[i] = _ltdError;
                group->E[i] = _lE;
                group->V[i] = _lV;
            }
        }
    }
     {
        // merged neuron update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup1[g]; 
            
            for(unsigned int i = 0; i < (4u); i++) {
                float Isyn = 0;
                float _ISynTdE = 0.000000000e+00f;
                float _lpre_PRew = group->pre_PRew[i];
                float _lpre_PG = group->pre_PG[i];
                float _lpre_E = group->pre_E[i];
                float _lPG = group->PG[i];
                float _lTdE = group->TdE[i];
                float _lPRew = group->PRew[i];
                float _lE = group->E[i];
                float _lV = group->V[i];
                 {
                    // postsynaptic model 0
                    float linSyn = group->outPostInSyn0[i];
                    Isyn += linSyn;
                    linSyn = 0;
                    group->outPostInSyn0[i] = linSyn;
                }
                 {
                    _ISynTdE += group->outPreOutSyn0[i];
                    group->outPreOutSyn0[i] = 0.000000000e+00f;
                }
                // calculate membrane potential
                _lV = ((9.048374180e-01f) * _lV) + Isyn + (0.000000000e+00f);
                const uint32_t _timestep = ((uint32_t)round(t / 1.000000000e+00f)) % 1;
                const uint32_t _index = (0 * 1 * (4u)) + (_timestep * (4u)) + i;
                const float _yTrue = group->YTrue[_index];
                _lTdE = _ISynTdE;
                _lPG = _lpre_PG;
                _lpre_PG = 0.0f;
                _lE = _lpre_E;
                _lpre_E = 0.0f;
                _lPRew = _lpre_PRew;
                _lpre_PRew = 0;
                group->pre_PRew[i] = _lpre_PRew;
                group->pre_PG[i] = _lpre_PG;
                group->pre_E[i] = _lpre_E;
                group->PG[i] = _lPG;
                group->TdE[i] = _lTdE;
                group->PRew[i] = _lPRew;
                group->E[i] = _lE;
                group->V[i] = _lV;
            }
        }
    }
     {
        // merged neuron update group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup2[g]; 
            
            for(unsigned int i = 0; i < (48u); i++) {
                float Isyn = 0;
                float _ISynFeedback = 0.000000000e+00f;
                float _ISynPolicyGradient = 0.000000000e+00f;
                float _ISynValueError = 0.000000000e+00f;
                float _ISynPolicyRegularisation = 0.000000000e+00f;
                float _ISynValueRegularisation = 0.000000000e+00f;
                float _ISynTdE = 0.000000000e+00f;
                float _ISynPolicyReward = 0.000000000e+00f;
                float _ISynPertEps = 0.000000000e+00f;
                float _ISynInh = 0.000000000e+00f;
                float _lInhVar = group->InhVar[i];
                float _lInhMean = group->InhMean[i];
                float _lInhNow = group->InhNow[i];
                float _lPGEps = group->PGEps[i];
                float _lVE = group->VE[i];
                float _lPRew = group->PRew[i];
                float _lSpkCount = group->SpkCount[i];
                float _lNoise4 = group->Noise4[i];
                float _lNoise3 = group->Noise3[i];
                float _lBeta = group->Beta[i];
                float _lTdE = group->TdE[i];
                float _lNoise2 = group->Noise2[i];
                float _lE = group->E[i];
                float _lPertEpsTrace = group->PertEpsTrace[i];
                float _lA = group->A[i];
                float _lPertEps = group->PertEps[i];
                float _lVR = group->VR[i];
                float _lEbase = group->Ebase[i];
                float _lV = group->V[i];
                float _lNoise1 = group->Noise1[i];
                float _lSigma = group->Sigma[i];
                float _lPR = group->PR[i];
                float _lLogSigma = group->LogSigma[i];
                float _lPG = group->PG[i];
                float _lRefracTime = group->RefracTime[i];
                 {
                    // postsynaptic model 0
                    float linSyn = group->outPostInSyn0[i];
                    Isyn += linSyn;
                    linSyn = 0;
                    group->outPostInSyn0[i] = linSyn;
                }
                 {
                    // postsynaptic model 1
                    float linSyn = group->outPostInSyn1[i];
                    _ISynInh += linSyn;
                    linSyn = 0;
                    group->outPostInSyn1[i] = linSyn;
                }
                 {
                    _ISynValueError += group->outPreOutSyn0[i];
                    group->outPreOutSyn0[i] = 0.000000000e+00f;
                }
                 {
                    _ISynTdE += group->outPreOutSyn1[i];
                    group->outPreOutSyn1[i] = 0.000000000e+00f;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                _lSigma = 0.0f * exp(_lLogSigma);
                _lPertEps = _lSigma * standardNormalDistribution(hostRNG);
                _lPertEpsTrace = _lPertEpsTrace * (9.048374180e-01f) + _lPertEps;
                _lV = (9.048374180e-01f) * _lV + Isyn + _ISynInh + 0.0f * _lPertEps;
                _lA *= (9.966722161e-01f);
                if(_lRefracTime > 0.0f)
                 {
                    _lRefracTime -= 1.000000000e+00f;
                }
                
                _lE = _ISynFeedback;
                _lEbase = _lEbase * 0.999f + _lE * 0.001f;
                _lNoise1 = standardNormalDistribution(hostRNG);
                _lNoise2 = standardUniformDistribution(hostRNG);
                _lNoise3 = standardNormalDistribution(hostRNG);
                _lNoise4 = standardNormalDistribution(hostRNG);
                _lPG = _ISynPolicyGradient;
                _lVE = _ISynValueError;
                _lPR = _ISynPolicyRegularisation;
                _lVR = _ISynValueRegularisation;
                _lTdE = _ISynTdE;
                _lPRew = _ISynPolicyReward;
                _lPGEps = _ISynPertEps;
                _lInhNow = fabs(_ISynInh);
                _lInhMean = 0.999000499833375f * _lInhMean + 0.000999500166624978f * _lInhNow;
                _lInhVar = 0.999000499833375f * _lInhVar + 0.000999500166624978f * (_lInhNow - _lInhMean) * (_lInhNow - _lInhMean);
                 {
                    // presynaptic weight update 0
                    float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                    _lZFilter *= (9.048374180e-01f);
                    group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                }
                 {
                    // postsynaptic weight update 0
                    float _lZDecMean = group->ZDecMeanInSynWUMPost0[i];
                    float _lLocMean = group->LocMeanInSynWUMPost0[i];
                    float _lFAvg = group->FAvgInSynWUMPost0[i];
                    float _lPsi = group->PsiInSynWUMPost0[i];
                    float _lHebbZ = group->HebbZInSynWUMPost0[i];
                    float _lTdMean = group->TdMeanInSynWUMPost0[i];
                    float _lContrD = group->ContrDInSynWUMPost0[i];
                    float _lContrMean = group->ContrMeanInSynWUMPost0[i];
                    float _lFAvgTrace = group->FAvgTraceInSynWUMPost0[i];
                    float _lLocR = group->LocRInSynWUMPost0[i];
                    float _lPsiBar = group->PsiBarInSynWUMPost0[i];
                    float _lContrR = group->ContrRInSynWUMPost0[i];
                    float _lHebbMean = group->HebbMeanInSynWUMPost0[i];
                    float _lZDec = group->ZDecInSynWUMPost0[i];
                    _lFAvg *= (9.980019987e-01f);
                    _lContrR = (_lTdE + _lPRew - _lTdMean) * _lContrD;
                    _lContrMean = 0.999f * _lContrMean + 0.001f * _lContrR;
                    _lContrD = (9.900133645e-01f) * _lContrD + (_lHebbZ - _lHebbMean) * (_lHebbZ - _lHebbMean);
                    _lHebbMean = 0.9999f * _lHebbMean + 0.0001f * _lHebbZ;
                    _lHebbZ *= (9.048374180e-01f);
                    _lFAvgTrace = (9.900133645e-01f) * _lFAvgTrace + (1.0f - (9.900133645e-01f)) * fabs(_lFAvg - (1.000000000e-02f));
                    if(_lRefracTime > 0.0f)
                     {
                        _lPsi = 0.0f;
                    }
                    else
                     {
                        _lPsi = (1.0f / (6.100000000e-01f)) * 0.3f * fmax(0.0f, 1.0f - fabs((_lV - ((6.100000000e-01f) + (_lBeta * _lA))) / (6.100000000e-01f)));
                    }
                    
                    _lPsiBar = 0.99f * _lPsiBar + 0.01f * _lPsi;
                    _lTdMean = 0.999f * _lTdMean + 0.001f * (_lTdE + _lPRew);
                    const float _thrL = (6.100000000e-01f) + _lBeta * _lA;
                    _lLocR = 0.0f - 0.005246f * (fmax(0.0f, _lV - _thrL) + fmax(0.0f, -_lV - _thrL)) + 0.01716f * (_lFAvgTrace - fabs(_lFAvg - (1.000000000e-02f))) - 0.01036f * (_lZDec - _lZDecMean) * (_lInhNow - _lInhMean) / (_lInhMean + 1e-6f);
                    _lLocMean = 0.999f * _lLocMean + 0.001f * _lLocR;
                    _lZDecMean = 0.999000499833375f * _lZDecMean + 0.000999500166624978f * _lZDec;
                    _lZDec = 0.0f;
                    group->ZDecMeanInSynWUMPost0[i] = _lZDecMean;
                    group->LocMeanInSynWUMPost0[i] = _lLocMean;
                    group->FAvgInSynWUMPost0[i] = _lFAvg;
                    group->PsiInSynWUMPost0[i] = _lPsi;
                    group->HebbZInSynWUMPost0[i] = _lHebbZ;
                    group->TdMeanInSynWUMPost0[i] = _lTdMean;
                    group->ContrDInSynWUMPost0[i] = _lContrD;
                    group->ContrMeanInSynWUMPost0[i] = _lContrMean;
                    group->FAvgTraceInSynWUMPost0[i] = _lFAvgTrace;
                    group->LocRInSynWUMPost0[i] = _lLocR;
                    group->PsiBarInSynWUMPost0[i] = _lPsiBar;
                    group->ContrRInSynWUMPost0[i] = _lContrR;
                    group->HebbMeanInSynWUMPost0[i] = _lHebbMean;
                    group->ZDecInSynWUMPost0[i] = _lZDec;
                }
                // test for and register a true spike
                if ((_lV >= ((6.100000000e-01f) + (_lBeta * _lA)) && _lRefracTime <= 0.0f)) {
                     {
                        // presynaptic weight update 0
                        float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                        _lZFilter += 1.0f;
                        group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                    }
                     {
                        // postsynaptic weight update 0
                        float _lFAvg = group->FAvgInSynWUMPost0[i];
                        float _lHebbZ = group->HebbZInSynWUMPost0[i];
                        float _lZDec = group->ZDecInSynWUMPost0[i];
                        _lFAvg += (1.0f - (9.980019987e-01f));
                        _lHebbZ += 1.0f;
                        _lZDec = 1.0f;
                        group->FAvgInSynWUMPost0[i] = _lFAvg;
                        group->HebbZInSynWUMPost0[i] = _lHebbZ;
                        group->ZDecInSynWUMPost0[i] = _lZDec;
                    }
                     {
                        group->spkSynSpike0[group->spkCntSynSpike0[0]++] = i;
                    }
                    // spike reset code
                    _lV -= ((6.100000000e-01f) - (0.000000000e+00f));
                    _lA += 1.0f;
                    _lRefracTime = (3.000000000e+00f);
                    _lSpkCount += 1.0f;
                }
                group->InhVar[i] = _lInhVar;
                group->InhMean[i] = _lInhMean;
                group->InhNow[i] = _lInhNow;
                group->PGEps[i] = _lPGEps;
                group->VE[i] = _lVE;
                group->PRew[i] = _lPRew;
                group->SpkCount[i] = _lSpkCount;
                group->Noise4[i] = _lNoise4;
                group->Noise3[i] = _lNoise3;
                group->Beta[i] = _lBeta;
                group->TdE[i] = _lTdE;
                group->Noise2[i] = _lNoise2;
                group->E[i] = _lE;
                group->PertEpsTrace[i] = _lPertEpsTrace;
                group->A[i] = _lA;
                group->PertEps[i] = _lPertEps;
                group->VR[i] = _lVR;
                group->Ebase[i] = _lEbase;
                group->V[i] = _lV;
                group->Noise1[i] = _lNoise1;
                group->Sigma[i] = _lSigma;
                group->PR[i] = _lPR;
                group->LogSigma[i] = _lLogSigma;
                group->PG[i] = _lPG;
                group->RefracTime[i] = _lRefracTime;
            }
        }
    }
     {
        // merged neuron update group 3
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedNeuronUpdateGroup3[g]; 
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                float Isyn = 0;
                float _ISynFeedback = 0.000000000e+00f;
                float _ISynPolicyGradient = 0.000000000e+00f;
                float _ISynValueError = 0.000000000e+00f;
                float _ISynPolicyRegularisation = 0.000000000e+00f;
                float _ISynValueRegularisation = 0.000000000e+00f;
                float _ISynTdE = 0.000000000e+00f;
                float _ISynPolicyReward = 0.000000000e+00f;
                float _ISynPertEps = 0.000000000e+00f;
                float _ISynInh = 0.000000000e+00f;
                float _lInhVar = group->InhVar[i];
                float _lInhMean = group->InhMean[i];
                float _lInhNow = group->InhNow[i];
                float _lPGEps = group->PGEps[i];
                float _lVE = group->VE[i];
                float _lPRew = group->PRew[i];
                float _lSpkCount = group->SpkCount[i];
                float _lNoise4 = group->Noise4[i];
                float _lNoise3 = group->Noise3[i];
                float _lBeta = group->Beta[i];
                float _lTdE = group->TdE[i];
                float _lNoise2 = group->Noise2[i];
                float _lE = group->E[i];
                float _lPertEpsTrace = group->PertEpsTrace[i];
                float _lA = group->A[i];
                float _lPertEps = group->PertEps[i];
                float _lVR = group->VR[i];
                float _lEbase = group->Ebase[i];
                float _lV = group->V[i];
                float _lNoise1 = group->Noise1[i];
                float _lSigma = group->Sigma[i];
                float _lPR = group->PR[i];
                float _lLogSigma = group->LogSigma[i];
                float _lPG = group->PG[i];
                float _lRefracTime = group->RefracTime[i];
                 {
                    // postsynaptic model 0
                    float linSyn = group->outPostInSyn0[i];
                    Isyn += linSyn;
                    linSyn = 0;
                    group->outPostInSyn0[i] = linSyn;
                }
                 {
                    // postsynaptic model 1
                    float linSyn = group->outPostInSyn1[i];
                    _ISynInh += linSyn;
                    linSyn = 0;
                    group->outPostInSyn1[i] = linSyn;
                }
                 {
                    _ISynValueError += group->outPreOutSyn0[i];
                    group->outPreOutSyn0[i] = 0.000000000e+00f;
                }
                 {
                    _ISynPolicyGradient += group->outPreOutSyn1[i];
                    group->outPreOutSyn1[i] = 0.000000000e+00f;
                }
                 {
                    _ISynTdE += group->outPreOutSyn2[i];
                    group->outPreOutSyn2[i] = 0.000000000e+00f;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                _lSigma = 0.0f * exp(_lLogSigma);
                _lPertEps = _lSigma * standardNormalDistribution(hostRNG);
                _lPertEpsTrace = _lPertEpsTrace * (9.048374180e-01f) + _lPertEps;
                _lV = (9.048374180e-01f) * _lV + Isyn + _ISynInh + 0.0f * _lPertEps;
                _lA *= (9.966722161e-01f);
                if(_lRefracTime > 0.0f)
                 {
                    _lRefracTime -= 1.000000000e+00f;
                }
                
                _lE = _ISynFeedback;
                _lEbase = _lEbase * 0.999f + _lE * 0.001f;
                _lNoise1 = standardNormalDistribution(hostRNG);
                _lNoise2 = standardUniformDistribution(hostRNG);
                _lNoise3 = standardNormalDistribution(hostRNG);
                _lNoise4 = standardNormalDistribution(hostRNG);
                _lPG = _ISynPolicyGradient;
                _lVE = _ISynValueError;
                _lPR = _ISynPolicyRegularisation;
                _lVR = _ISynValueRegularisation;
                _lTdE = _ISynTdE;
                _lPRew = _ISynPolicyReward;
                _lPGEps = _ISynPertEps;
                _lInhNow = fabs(_ISynInh);
                _lInhMean = 0.999000499833375f * _lInhMean + 0.000999500166624978f * _lInhNow;
                _lInhVar = 0.999000499833375f * _lInhVar + 0.000999500166624978f * (_lInhNow - _lInhMean) * (_lInhNow - _lInhMean);
                 {
                    // presynaptic weight update 0
                    float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                    _lZFilter *= (9.048374180e-01f);
                    group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                }
                 {
                    // presynaptic weight update 1
                    float _lZFilter = group->ZFilterOutSynWUMPre1[i];
                    _lZFilter *= (9.048374180e-01f);
                    group->ZFilterOutSynWUMPre1[i] = _lZFilter;
                }
                 {
                    // presynaptic weight update 2
                    float _lZFilter = group->ZFilterOutSynWUMPre2[i];
                    _lZFilter *= (9.048374180e-01f);
                    group->ZFilterOutSynWUMPre2[i] = _lZFilter;
                }
                 {
                    // postsynaptic weight update 0
                    float _lZDecMean = group->ZDecMeanInSynWUMPost0[i];
                    float _lLocMean = group->LocMeanInSynWUMPost0[i];
                    float _lFAvg = group->FAvgInSynWUMPost0[i];
                    float _lPsi = group->PsiInSynWUMPost0[i];
                    float _lHebbZ = group->HebbZInSynWUMPost0[i];
                    float _lTdMean = group->TdMeanInSynWUMPost0[i];
                    float _lContrD = group->ContrDInSynWUMPost0[i];
                    float _lContrMean = group->ContrMeanInSynWUMPost0[i];
                    float _lFAvgTrace = group->FAvgTraceInSynWUMPost0[i];
                    float _lLocR = group->LocRInSynWUMPost0[i];
                    float _lPsiBar = group->PsiBarInSynWUMPost0[i];
                    float _lContrR = group->ContrRInSynWUMPost0[i];
                    float _lHebbMean = group->HebbMeanInSynWUMPost0[i];
                    float _lZDec = group->ZDecInSynWUMPost0[i];
                    _lFAvg *= (9.980019987e-01f);
                    _lContrR = (_lTdE + _lPRew - _lTdMean) * _lContrD;
                    _lContrMean = 0.999f * _lContrMean + 0.001f * _lContrR;
                    _lContrD = (9.900133645e-01f) * _lContrD + (_lHebbZ - _lHebbMean) * (_lHebbZ - _lHebbMean);
                    _lHebbMean = 0.9999f * _lHebbMean + 0.0001f * _lHebbZ;
                    _lHebbZ *= (9.048374180e-01f);
                    _lFAvgTrace = (9.900133645e-01f) * _lFAvgTrace + (1.0f - (9.900133645e-01f)) * fabs(_lFAvg - (1.000000000e-02f));
                    if(_lRefracTime > 0.0f)
                     {
                        _lPsi = 0.0f;
                    }
                    else
                     {
                        _lPsi = (1.0f / (6.100000000e-01f)) * 0.3f * fmax(0.0f, 1.0f - fabs((_lV - ((6.100000000e-01f) + (_lBeta * _lA))) / (6.100000000e-01f)));
                    }
                    
                    _lPsiBar = 0.99f * _lPsiBar + 0.01f * _lPsi;
                    _lTdMean = 0.999f * _lTdMean + 0.001f * (_lTdE + _lPRew);
                    const float _thrL = (6.100000000e-01f) + _lBeta * _lA;
                    _lLocR = 0.0f - 0.005246f * (fmax(0.0f, _lV - _thrL) + fmax(0.0f, -_lV - _thrL)) + 0.01716f * (_lFAvgTrace - fabs(_lFAvg - (1.000000000e-02f))) - 0.01036f * (_lZDec - _lZDecMean) * (_lInhNow - _lInhMean) / (_lInhMean + 1e-6f);
                    _lLocMean = 0.999f * _lLocMean + 0.001f * _lLocR;
                    _lZDecMean = 0.999000499833375f * _lZDecMean + 0.000999500166624978f * _lZDec;
                    _lZDec = 0.0f;
                    group->ZDecMeanInSynWUMPost0[i] = _lZDecMean;
                    group->LocMeanInSynWUMPost0[i] = _lLocMean;
                    group->FAvgInSynWUMPost0[i] = _lFAvg;
                    group->PsiInSynWUMPost0[i] = _lPsi;
                    group->HebbZInSynWUMPost0[i] = _lHebbZ;
                    group->TdMeanInSynWUMPost0[i] = _lTdMean;
                    group->ContrDInSynWUMPost0[i] = _lContrD;
                    group->ContrMeanInSynWUMPost0[i] = _lContrMean;
                    group->FAvgTraceInSynWUMPost0[i] = _lFAvgTrace;
                    group->LocRInSynWUMPost0[i] = _lLocR;
                    group->PsiBarInSynWUMPost0[i] = _lPsiBar;
                    group->ContrRInSynWUMPost0[i] = _lContrR;
                    group->HebbMeanInSynWUMPost0[i] = _lHebbMean;
                    group->ZDecInSynWUMPost0[i] = _lZDec;
                }
                 {
                    // postsynaptic weight update 1
                    float _lZDecMean = group->ZDecMeanInSynWUMPost1[i];
                    float _lLocMean = group->LocMeanInSynWUMPost1[i];
                    float _lFAvg = group->FAvgInSynWUMPost1[i];
                    float _lPsi = group->PsiInSynWUMPost1[i];
                    float _lHebbZ = group->HebbZInSynWUMPost1[i];
                    float _lTdMean = group->TdMeanInSynWUMPost1[i];
                    float _lContrD = group->ContrDInSynWUMPost1[i];
                    float _lContrMean = group->ContrMeanInSynWUMPost1[i];
                    float _lFAvgTrace = group->FAvgTraceInSynWUMPost1[i];
                    float _lLocR = group->LocRInSynWUMPost1[i];
                    float _lPsiBar = group->PsiBarInSynWUMPost1[i];
                    float _lContrR = group->ContrRInSynWUMPost1[i];
                    float _lHebbMean = group->HebbMeanInSynWUMPost1[i];
                    float _lZDec = group->ZDecInSynWUMPost1[i];
                    _lFAvg *= (9.980019987e-01f);
                    _lContrR = (_lTdE + _lPRew - _lTdMean) * _lContrD;
                    _lContrMean = 0.999f * _lContrMean + 0.001f * _lContrR;
                    _lContrD = (9.900133645e-01f) * _lContrD + (_lHebbZ - _lHebbMean) * (_lHebbZ - _lHebbMean);
                    _lHebbMean = 0.9999f * _lHebbMean + 0.0001f * _lHebbZ;
                    _lHebbZ *= (9.048374180e-01f);
                    _lFAvgTrace = (9.900133645e-01f) * _lFAvgTrace + (1.0f - (9.900133645e-01f)) * fabs(_lFAvg - (1.000000000e-02f));
                    if(_lRefracTime > 0.0f)
                     {
                        _lPsi = 0.0f;
                    }
                    else
                     {
                        _lPsi = (1.0f / (6.100000000e-01f)) * 0.3f * fmax(0.0f, 1.0f - fabs((_lV - ((6.100000000e-01f) + (_lBeta * _lA))) / (6.100000000e-01f)));
                    }
                    
                    _lPsiBar = 0.99f * _lPsiBar + 0.01f * _lPsi;
                    _lTdMean = 0.999f * _lTdMean + 0.001f * (_lTdE + _lPRew);
                    const float _thrL = (6.100000000e-01f) + _lBeta * _lA;
                    _lLocR = 0.0f - 0.005246f * (fmax(0.0f, _lV - _thrL) + fmax(0.0f, -_lV - _thrL)) + 0.01716f * (_lFAvgTrace - fabs(_lFAvg - (1.000000000e-02f))) - 0.01036f * (_lZDec - _lZDecMean) * (_lInhNow - _lInhMean) / (_lInhMean + 1e-6f);
                    _lLocMean = 0.999f * _lLocMean + 0.001f * _lLocR;
                    _lZDecMean = 0.999000499833375f * _lZDecMean + 0.000999500166624978f * _lZDec;
                    _lZDec = 0.0f;
                    group->ZDecMeanInSynWUMPost1[i] = _lZDecMean;
                    group->LocMeanInSynWUMPost1[i] = _lLocMean;
                    group->FAvgInSynWUMPost1[i] = _lFAvg;
                    group->PsiInSynWUMPost1[i] = _lPsi;
                    group->HebbZInSynWUMPost1[i] = _lHebbZ;
                    group->TdMeanInSynWUMPost1[i] = _lTdMean;
                    group->ContrDInSynWUMPost1[i] = _lContrD;
                    group->ContrMeanInSynWUMPost1[i] = _lContrMean;
                    group->FAvgTraceInSynWUMPost1[i] = _lFAvgTrace;
                    group->LocRInSynWUMPost1[i] = _lLocR;
                    group->PsiBarInSynWUMPost1[i] = _lPsiBar;
                    group->ContrRInSynWUMPost1[i] = _lContrR;
                    group->HebbMeanInSynWUMPost1[i] = _lHebbMean;
                    group->ZDecInSynWUMPost1[i] = _lZDec;
                }
                // test for and register a true spike
                if ((_lV >= ((6.100000000e-01f) + (_lBeta * _lA)) && _lRefracTime <= 0.0f)) {
                     {
                        // presynaptic weight update 0
                        float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                        _lZFilter += 1.0f;
                        group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                    }
                     {
                        // presynaptic weight update 1
                        float _lZFilter = group->ZFilterOutSynWUMPre1[i];
                        _lZFilter += 1.0f;
                        group->ZFilterOutSynWUMPre1[i] = _lZFilter;
                    }
                     {
                        // presynaptic weight update 2
                        float _lZFilter = group->ZFilterOutSynWUMPre2[i];
                        _lZFilter += 1.0f;
                        group->ZFilterOutSynWUMPre2[i] = _lZFilter;
                    }
                     {
                        // postsynaptic weight update 0
                        float _lFAvg = group->FAvgInSynWUMPost0[i];
                        float _lHebbZ = group->HebbZInSynWUMPost0[i];
                        float _lZDec = group->ZDecInSynWUMPost0[i];
                        _lFAvg += (1.0f - (9.980019987e-01f));
                        _lHebbZ += 1.0f;
                        _lZDec = 1.0f;
                        group->FAvgInSynWUMPost0[i] = _lFAvg;
                        group->HebbZInSynWUMPost0[i] = _lHebbZ;
                        group->ZDecInSynWUMPost0[i] = _lZDec;
                    }
                     {
                        // postsynaptic weight update 1
                        float _lFAvg = group->FAvgInSynWUMPost1[i];
                        float _lHebbZ = group->HebbZInSynWUMPost1[i];
                        float _lZDec = group->ZDecInSynWUMPost1[i];
                        _lFAvg += (1.0f - (9.980019987e-01f));
                        _lHebbZ += 1.0f;
                        _lZDec = 1.0f;
                        group->FAvgInSynWUMPost1[i] = _lFAvg;
                        group->HebbZInSynWUMPost1[i] = _lHebbZ;
                        group->ZDecInSynWUMPost1[i] = _lZDec;
                    }
                     {
                        group->spkSynSpike0[group->spkCntSynSpike0[0]++] = i;
                    }
                    // spike reset code
                    _lV -= ((6.100000000e-01f) - (0.000000000e+00f));
                    _lA += 1.0f;
                    _lRefracTime = (3.000000000e+00f);
                    _lSpkCount += 1.0f;
                }
                group->InhVar[i] = _lInhVar;
                group->InhMean[i] = _lInhMean;
                group->InhNow[i] = _lInhNow;
                group->PGEps[i] = _lPGEps;
                group->VE[i] = _lVE;
                group->PRew[i] = _lPRew;
                group->SpkCount[i] = _lSpkCount;
                group->Noise4[i] = _lNoise4;
                group->Noise3[i] = _lNoise3;
                group->Beta[i] = _lBeta;
                group->TdE[i] = _lTdE;
                group->Noise2[i] = _lNoise2;
                group->E[i] = _lE;
                group->PertEpsTrace[i] = _lPertEpsTrace;
                group->A[i] = _lA;
                group->PertEps[i] = _lPertEps;
                group->VR[i] = _lVR;
                group->Ebase[i] = _lEbase;
                group->V[i] = _lV;
                group->Noise1[i] = _lNoise1;
                group->Sigma[i] = _lSigma;
                group->PR[i] = _lPR;
                group->LogSigma[i] = _lLogSigma;
                group->PG[i] = _lPG;
                group->RefracTime[i] = _lRefracTime;
            }
        }
    }
     {
        // merged neuron update group 4
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup4[g]; 
            
            for(unsigned int i = 0; i < (48u); i++) {
                float Isyn = 0;
                float _ISynFeedback = 0.000000000e+00f;
                float _ISynPolicyGradient = 0.000000000e+00f;
                float _ISynValueError = 0.000000000e+00f;
                float _ISynPolicyRegularisation = 0.000000000e+00f;
                float _ISynValueRegularisation = 0.000000000e+00f;
                float _ISynTdE = 0.000000000e+00f;
                float _ISynPolicyReward = 0.000000000e+00f;
                float _ISynPertEps = 0.000000000e+00f;
                float _ISynInh = 0.000000000e+00f;
                float _lPGEps = group->PGEps[i];
                float _lVE = group->VE[i];
                float _lPRew = group->PRew[i];
                float _lSpkCount = group->SpkCount[i];
                float _lNoise4 = group->Noise4[i];
                float _lNoise3 = group->Noise3[i];
                float _lBeta = group->Beta[i];
                float _lPoolBias = group->PoolBias[i];
                float _lTdE = group->TdE[i];
                float _lNoise2 = group->Noise2[i];
                float _lE = group->E[i];
                float _lPertEpsTrace = group->PertEpsTrace[i];
                float _lA = group->A[i];
                float _lPertEps = group->PertEps[i];
                float _lVR = group->VR[i];
                float _lEbase = group->Ebase[i];
                float _lV = group->V[i];
                float _lNoise1 = group->Noise1[i];
                float _lSigma = group->Sigma[i];
                float _lPR = group->PR[i];
                float _lLogSigma = group->LogSigma[i];
                float _lPG = group->PG[i];
                float _lRefracTime = group->RefracTime[i];
                 {
                    // postsynaptic model 0
                    float linSyn = group->outPostInSyn0[i];
                    Isyn += linSyn;
                    linSyn = 0;
                    group->outPostInSyn0[i] = linSyn;
                }
                 {
                    // postsynaptic model 1
                    float linSyn = group->outPostInSyn1[i];
                    _ISynInh += linSyn;
                    linSyn = 0;
                    group->outPostInSyn1[i] = linSyn;
                }
                 {
                    _ISynPolicyGradient += group->outPreOutSyn0[i];
                    group->outPreOutSyn0[i] = 0.000000000e+00f;
                }
                 {
                    _ISynTdE += group->outPreOutSyn1[i];
                    group->outPreOutSyn1[i] = 0.000000000e+00f;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                _lSigma = 0.0f * exp(_lLogSigma);
                _lPertEps = _lSigma * standardNormalDistribution(hostRNG);
                _lPertEpsTrace = _lPertEpsTrace * (9.048374180e-01f) + _lPertEps;
                _lV = (9.048374180e-01f) * _lV + Isyn + _ISynInh + 0.0f * _lPertEps + _lPoolBias;
                _lA *= (9.966722161e-01f);
                if(_lRefracTime > 0.0f)
                 {
                    _lRefracTime -= 1.000000000e+00f;
                }
                
                _lE = _ISynFeedback;
                _lEbase = _lEbase * 0.999f + _lE * 0.001f;
                _lNoise1 = standardNormalDistribution(hostRNG);
                _lNoise2 = standardUniformDistribution(hostRNG);
                _lNoise3 = standardNormalDistribution(hostRNG);
                _lNoise4 = standardNormalDistribution(hostRNG);
                _lPG = _ISynPolicyGradient;
                _lVE = _ISynValueError;
                _lPR = _ISynPolicyRegularisation;
                _lVR = _ISynValueRegularisation;
                _lTdE = _ISynTdE;
                _lPRew = _ISynPolicyReward;
                _lPGEps = _ISynPertEps;
                 {
                    // presynaptic weight update 0
                    float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                    _lZFilter *= (9.048374180e-01f);
                    group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                }
                 {
                    // postsynaptic weight update 0
                    float _lTdMean = group->TdMeanInSynWUMPost0[i];
                    float _lFAvg = group->FAvgInSynWUMPost0[i];
                    float _lPsi = group->PsiInSynWUMPost0[i];
                    _lFAvg *= (9.980019987e-01f);
                    if(_lRefracTime > 0.0f)
                     {
                        _lPsi = 0.0f;
                    }
                    else
                     {
                        _lPsi = (1.0f / (6.100000000e-01f)) * 0.3f * fmax(0.0f, 1.0f - fabs((_lV - ((6.100000000e-01f) + (_lBeta * _lA))) / (6.100000000e-01f)));
                    }
                    
                    _lTdMean = 0.999f * _lTdMean + 0.001f * (_lTdE + _lPRew);
                    group->TdMeanInSynWUMPost0[i] = _lTdMean;
                    group->FAvgInSynWUMPost0[i] = _lFAvg;
                    group->PsiInSynWUMPost0[i] = _lPsi;
                }
                // test for and register a true spike
                if ((_lV >= ((6.100000000e-01f) + (_lBeta * _lA)) && _lRefracTime <= 0.0f)) {
                     {
                        // presynaptic weight update 0
                        float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                        _lZFilter += 1.0f;
                        group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                    }
                     {
                        // postsynaptic weight update 0
                        float _lFAvg = group->FAvgInSynWUMPost0[i];
                        _lFAvg += (1.0f - (9.980019987e-01f));
                        group->FAvgInSynWUMPost0[i] = _lFAvg;
                    }
                     {
                        group->spkSynSpike0[group->spkCntSynSpike0[0]++] = i;
                    }
                    // spike reset code
                    _lV -= ((6.100000000e-01f) - (0.000000000e+00f));
                    _lA += 1.0f;
                    _lRefracTime = (3.000000000e+00f);
                    _lSpkCount += 1.0f;
                }
                group->PGEps[i] = _lPGEps;
                group->VE[i] = _lVE;
                group->PRew[i] = _lPRew;
                group->SpkCount[i] = _lSpkCount;
                group->Noise4[i] = _lNoise4;
                group->Noise3[i] = _lNoise3;
                group->Beta[i] = _lBeta;
                group->PoolBias[i] = _lPoolBias;
                group->TdE[i] = _lTdE;
                group->Noise2[i] = _lNoise2;
                group->E[i] = _lE;
                group->PertEpsTrace[i] = _lPertEpsTrace;
                group->A[i] = _lA;
                group->PertEps[i] = _lPertEps;
                group->VR[i] = _lVR;
                group->Ebase[i] = _lEbase;
                group->V[i] = _lV;
                group->Noise1[i] = _lNoise1;
                group->Sigma[i] = _lSigma;
                group->PR[i] = _lPR;
                group->LogSigma[i] = _lLogSigma;
                group->PG[i] = _lPG;
                group->RefracTime[i] = _lRefracTime;
            }
        }
    }
     {
        // merged neuron update group 5
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup5[g]; 
            
            for(unsigned int i = 0; i < (1200u); i++) {
                float _lNoise4 = group->Noise4[i];
                float _lNoise3 = group->Noise3[i];
                float _lNoise2 = group->Noise2[i];
                float _lNoise1 = group->Noise1[i];
                const float _lInput = group->Input[i];
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                const bool _spike = standardUniformDistribution(hostRNG) >= exp(-fabs(_lInput) * 1.000000000e+00f);
                _lNoise1 = standardNormalDistribution(hostRNG);
                _lNoise2 = standardNormalDistribution(hostRNG);
                _lNoise3 = standardNormalDistribution(hostRNG);
                _lNoise4 = standardNormalDistribution(hostRNG);
                 {
                    // presynaptic weight update 0
                    float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                    _lZFilter *= (9.048374180e-01f);
                    group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                }
                // test for and register a true spike
                if ((_lInput > 0.0f && _spike)) {
                     {
                        // presynaptic weight update 0
                        float _lZFilter = group->ZFilterOutSynWUMPre0[i];
                        _lZFilter += 1.0f;
                        group->ZFilterOutSynWUMPre0[i] = _lZFilter;
                    }
                     {
                        group->spkSynSpike0[group->spkCntSynSpike0[0]++] = i;
                    }
                }
                group->Noise4[i] = _lNoise4;
                group->Noise3[i] = _lNoise3;
                group->Noise2[i] = _lNoise2;
                group->Noise1[i] = _lNoise1;
            }
        }
    }
}
