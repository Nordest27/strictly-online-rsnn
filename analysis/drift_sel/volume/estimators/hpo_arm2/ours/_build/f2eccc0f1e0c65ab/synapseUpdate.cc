#include "definitions.h"

struct MergedPresynapticUpdateGroup0
 {
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPost;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ srcSpk;
    uint32_t* __restrict__ srcSpkCnt;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPost;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ srcSpk;
    uint32_t* __restrict__ srcSpkCnt;
    
}
;
struct MergedPresynapticUpdateGroup2
 {
    float* __restrict__ LogSynSig;
    float* __restrict__ Noise1;
    float* __restrict__ NoiseTrace;
    float* __restrict__ SynSig;
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPost;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ srcSpk;
    uint32_t* __restrict__ srcSpkCnt;
    uint32_t rowStride;
    
}
;
struct MergedPresynapticUpdateGroup3
 {
    float* __restrict__ LogSynSig;
    float* __restrict__ Noise1;
    float* __restrict__ NoiseTrace;
    float* __restrict__ SynSig;
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPost;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ srcSpk;
    uint32_t* __restrict__ srcSpkCnt;
    uint32_t rowStride;
    
}
;
struct MergedPresynapticUpdateGroup4
 {
    float* __restrict__ LogSynSig;
    float* __restrict__ Noise1;
    float* __restrict__ NoiseTrace;
    float* __restrict__ SynSig;
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPost;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ srcSpk;
    uint32_t* __restrict__ srcSpkCnt;
    uint32_t rowStride;
    
}
;
struct MergedSynapseDynamicsGroup0
 {
    float* __restrict__ DeltaG;
    float* __restrict__ E_post;
    float* __restrict__ RLTrace;
    float* __restrict__ ValReg_post;
    float* __restrict__ ZFilter;
    uint32_t* __restrict__ ind;
    uint32_t* __restrict__ rowLength;
    
}
;
struct MergedSynapseDynamicsGroup1
 {
    float* __restrict__ DeltaG;
    float* __restrict__ E_post;
    float* __restrict__ PG_post;
    float* __restrict__ RLTrace;
    float* __restrict__ TdE_post;
    float* __restrict__ ZFilter;
    uint32_t* __restrict__ ind;
    uint32_t* __restrict__ rowLength;
    
}
;
struct MergedSynapseDynamicsGroup2
 {
    float* __restrict__ E_post;
    float* __restrict__ g;
    float* __restrict__ outPre;
    uint32_t numSrcNeurons;
    
}
;
struct MergedSynapseDynamicsGroup3
 {
    float* __restrict__ ContrMean;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ ind;
    float* __restrict__ epsilonA;
    float* __restrict__ eFiltered;
    float* __restrict__ ZFilter;
    float* __restrict__ TdMean;
    float* __restrict__ TdE_post;
    float* __restrict__ SynSig;
    float* __restrict__ RLNoisePtrace;
    float* __restrict__ RLNoiseGtrace;
    float* __restrict__ PsiBar;
    float* __restrict__ Psi;
    float* __restrict__ PRew_post;
    float* __restrict__ NoiseTrace;
    float* __restrict__ LogSynSig;
    float* __restrict__ LocR;
    float* __restrict__ LocMean;
    float* __restrict__ FAvg;
    float* __restrict__ DeltaG;
    float* __restrict__ ContrR;
    float* __restrict__ Beta_post;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedSynapseDynamicsGroup4
 {
    float* __restrict__ DeltaG;
    float* __restrict__ E_post;
    float* __restrict__ F_post;
    float* __restrict__ RLTrace;
    float* __restrict__ ValReg_post;
    float* __restrict__ ZFilter;
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPre;
    uint32_t* __restrict__ rowLength;
    uint32_t numSrcNeurons;
    
}
;
struct MergedSynapseDynamicsGroup5
 {
    float* __restrict__ DeltaG;
    float* __restrict__ E_post;
    float* __restrict__ F_post;
    float* __restrict__ PG_post;
    float* __restrict__ RLTrace;
    float* __restrict__ TdE_post;
    float* __restrict__ ZFilter;
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPre;
    uint32_t* __restrict__ rowLength;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedSynapseDynamicsGroup6
 {
    float* __restrict__ DeltaG;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ ind;
    float* __restrict__ epsilonA;
    float* __restrict__ eFiltered;
    float* __restrict__ ZFilter;
    float* __restrict__ VE_post;
    float* __restrict__ TdMean;
    float* __restrict__ TdE_post;
    float* __restrict__ SynSig;
    float* __restrict__ RLTrace;
    float* __restrict__ Psi;
    float* __restrict__ PRew_post;
    float* __restrict__ PG_post;
    float* __restrict__ NoiseTrace;
    float* __restrict__ LogSynSig;
    float* __restrict__ FAvg;
    float* __restrict__ Beta_post;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedSynapseDynamicsGroup7
 {
    float* __restrict__ ContrMean;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ ind;
    float* __restrict__ epsilonA;
    float* __restrict__ eFiltered;
    float* __restrict__ ZFilter;
    float* __restrict__ TdMean;
    float* __restrict__ TdE_post;
    float* __restrict__ SynSig;
    float* __restrict__ RLNoisePtrace;
    float* __restrict__ RLNoiseGtrace;
    float* __restrict__ PsiBar;
    float* __restrict__ Psi;
    float* __restrict__ PRew_post;
    float* __restrict__ NoiseTrace;
    float* __restrict__ LogSynSig;
    float* __restrict__ LocR;
    float* __restrict__ LocMean;
    float* __restrict__ FAvg;
    float* __restrict__ DeltaG;
    float* __restrict__ ContrR;
    float* __restrict__ Beta_post;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
static MergedPresynapticUpdateGroup0 mergedPresynapticUpdateGroup0[1];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* g, uint32_t* ind, float* outPost, uint32_t* rowLength, uint32_t* srcSpk, uint32_t* srcSpkCnt) {
    mergedPresynapticUpdateGroup0[idx].g = g;
    mergedPresynapticUpdateGroup0[idx].ind = ind;
    mergedPresynapticUpdateGroup0[idx].outPost = outPost;
    mergedPresynapticUpdateGroup0[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup0[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup0[idx].srcSpkCnt = srcSpkCnt;
}
static MergedPresynapticUpdateGroup1 mergedPresynapticUpdateGroup1[1];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* g, uint32_t* ind, float* outPost, uint32_t* rowLength, uint32_t* srcSpk, uint32_t* srcSpkCnt) {
    mergedPresynapticUpdateGroup1[idx].g = g;
    mergedPresynapticUpdateGroup1[idx].ind = ind;
    mergedPresynapticUpdateGroup1[idx].outPost = outPost;
    mergedPresynapticUpdateGroup1[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup1[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup1[idx].srcSpkCnt = srcSpkCnt;
}
static MergedPresynapticUpdateGroup2 mergedPresynapticUpdateGroup2[4];
void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, float* LogSynSig, float* Noise1, float* NoiseTrace, float* SynSig, float* g, uint32_t* ind, float* outPost, uint32_t* rowLength, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t rowStride) {
    mergedPresynapticUpdateGroup2[idx].LogSynSig = LogSynSig;
    mergedPresynapticUpdateGroup2[idx].Noise1 = Noise1;
    mergedPresynapticUpdateGroup2[idx].NoiseTrace = NoiseTrace;
    mergedPresynapticUpdateGroup2[idx].SynSig = SynSig;
    mergedPresynapticUpdateGroup2[idx].g = g;
    mergedPresynapticUpdateGroup2[idx].ind = ind;
    mergedPresynapticUpdateGroup2[idx].outPost = outPost;
    mergedPresynapticUpdateGroup2[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup2[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup2[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup2[idx].rowStride = rowStride;
}
static MergedPresynapticUpdateGroup3 mergedPresynapticUpdateGroup3[2];
void pushMergedPresynapticUpdateGroup3ToDevice(unsigned int idx, float* LogSynSig, float* Noise1, float* NoiseTrace, float* SynSig, float* g, uint32_t* ind, float* outPost, uint32_t* rowLength, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t rowStride) {
    mergedPresynapticUpdateGroup3[idx].LogSynSig = LogSynSig;
    mergedPresynapticUpdateGroup3[idx].Noise1 = Noise1;
    mergedPresynapticUpdateGroup3[idx].NoiseTrace = NoiseTrace;
    mergedPresynapticUpdateGroup3[idx].SynSig = SynSig;
    mergedPresynapticUpdateGroup3[idx].g = g;
    mergedPresynapticUpdateGroup3[idx].ind = ind;
    mergedPresynapticUpdateGroup3[idx].outPost = outPost;
    mergedPresynapticUpdateGroup3[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup3[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup3[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup3[idx].rowStride = rowStride;
}
static MergedPresynapticUpdateGroup4 mergedPresynapticUpdateGroup4[4];
void pushMergedPresynapticUpdateGroup4ToDevice(unsigned int idx, float* LogSynSig, float* Noise1, float* NoiseTrace, float* SynSig, float* g, uint32_t* ind, float* outPost, uint32_t* rowLength, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t rowStride) {
    mergedPresynapticUpdateGroup4[idx].LogSynSig = LogSynSig;
    mergedPresynapticUpdateGroup4[idx].Noise1 = Noise1;
    mergedPresynapticUpdateGroup4[idx].NoiseTrace = NoiseTrace;
    mergedPresynapticUpdateGroup4[idx].SynSig = SynSig;
    mergedPresynapticUpdateGroup4[idx].g = g;
    mergedPresynapticUpdateGroup4[idx].ind = ind;
    mergedPresynapticUpdateGroup4[idx].outPost = outPost;
    mergedPresynapticUpdateGroup4[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup4[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup4[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup4[idx].rowStride = rowStride;
}
static MergedSynapseDynamicsGroup0 mergedSynapseDynamicsGroup0[1];
void pushMergedSynapseDynamicsGroup0ToDevice(unsigned int idx, float* DeltaG, float* E_post, float* RLTrace, float* ValReg_post, float* ZFilter, uint32_t* ind, uint32_t* rowLength) {
    mergedSynapseDynamicsGroup0[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup0[idx].E_post = E_post;
    mergedSynapseDynamicsGroup0[idx].RLTrace = RLTrace;
    mergedSynapseDynamicsGroup0[idx].ValReg_post = ValReg_post;
    mergedSynapseDynamicsGroup0[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup0[idx].ind = ind;
    mergedSynapseDynamicsGroup0[idx].rowLength = rowLength;
}
static MergedSynapseDynamicsGroup1 mergedSynapseDynamicsGroup1[1];
void pushMergedSynapseDynamicsGroup1ToDevice(unsigned int idx, float* DeltaG, float* E_post, float* PG_post, float* RLTrace, float* TdE_post, float* ZFilter, uint32_t* ind, uint32_t* rowLength) {
    mergedSynapseDynamicsGroup1[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup1[idx].E_post = E_post;
    mergedSynapseDynamicsGroup1[idx].PG_post = PG_post;
    mergedSynapseDynamicsGroup1[idx].RLTrace = RLTrace;
    mergedSynapseDynamicsGroup1[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup1[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup1[idx].ind = ind;
    mergedSynapseDynamicsGroup1[idx].rowLength = rowLength;
}
static MergedSynapseDynamicsGroup2 mergedSynapseDynamicsGroup2[5];
void pushMergedSynapseDynamicsGroup2ToDevice(unsigned int idx, float* E_post, float* g, float* outPre, uint32_t numSrcNeurons) {
    mergedSynapseDynamicsGroup2[idx].E_post = E_post;
    mergedSynapseDynamicsGroup2[idx].g = g;
    mergedSynapseDynamicsGroup2[idx].outPre = outPre;
    mergedSynapseDynamicsGroup2[idx].numSrcNeurons = numSrcNeurons;
}
static MergedSynapseDynamicsGroup3 mergedSynapseDynamicsGroup3[4];
void pushMergedSynapseDynamicsGroup3ToDevice(unsigned int idx, float* ContrMean, uint32_t* rowLength, uint32_t* ind, float* epsilonA, float* eFiltered, float* ZFilter, float* TdMean, float* TdE_post, float* SynSig, float* RLNoisePtrace, float* RLNoiseGtrace, float* PsiBar, float* Psi, float* PRew_post, float* NoiseTrace, float* LogSynSig, float* LocR, float* LocMean, float* FAvg, float* DeltaG, float* ContrR, float* Beta_post, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseDynamicsGroup3[idx].ContrMean = ContrMean;
    mergedSynapseDynamicsGroup3[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup3[idx].ind = ind;
    mergedSynapseDynamicsGroup3[idx].epsilonA = epsilonA;
    mergedSynapseDynamicsGroup3[idx].eFiltered = eFiltered;
    mergedSynapseDynamicsGroup3[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup3[idx].TdMean = TdMean;
    mergedSynapseDynamicsGroup3[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup3[idx].SynSig = SynSig;
    mergedSynapseDynamicsGroup3[idx].RLNoisePtrace = RLNoisePtrace;
    mergedSynapseDynamicsGroup3[idx].RLNoiseGtrace = RLNoiseGtrace;
    mergedSynapseDynamicsGroup3[idx].PsiBar = PsiBar;
    mergedSynapseDynamicsGroup3[idx].Psi = Psi;
    mergedSynapseDynamicsGroup3[idx].PRew_post = PRew_post;
    mergedSynapseDynamicsGroup3[idx].NoiseTrace = NoiseTrace;
    mergedSynapseDynamicsGroup3[idx].LogSynSig = LogSynSig;
    mergedSynapseDynamicsGroup3[idx].LocR = LocR;
    mergedSynapseDynamicsGroup3[idx].LocMean = LocMean;
    mergedSynapseDynamicsGroup3[idx].FAvg = FAvg;
    mergedSynapseDynamicsGroup3[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup3[idx].ContrR = ContrR;
    mergedSynapseDynamicsGroup3[idx].Beta_post = Beta_post;
    mergedSynapseDynamicsGroup3[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseDynamicsGroup3[idx].rowStride = rowStride;
}
static MergedSynapseDynamicsGroup4 mergedSynapseDynamicsGroup4[3];
void pushMergedSynapseDynamicsGroup4ToDevice(unsigned int idx, float* DeltaG, float* E_post, float* F_post, float* RLTrace, float* ValReg_post, float* ZFilter, float* g, uint32_t* ind, float* outPre, uint32_t* rowLength, uint32_t numSrcNeurons) {
    mergedSynapseDynamicsGroup4[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup4[idx].E_post = E_post;
    mergedSynapseDynamicsGroup4[idx].F_post = F_post;
    mergedSynapseDynamicsGroup4[idx].RLTrace = RLTrace;
    mergedSynapseDynamicsGroup4[idx].ValReg_post = ValReg_post;
    mergedSynapseDynamicsGroup4[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup4[idx].g = g;
    mergedSynapseDynamicsGroup4[idx].ind = ind;
    mergedSynapseDynamicsGroup4[idx].outPre = outPre;
    mergedSynapseDynamicsGroup4[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup4[idx].numSrcNeurons = numSrcNeurons;
}
static MergedSynapseDynamicsGroup5 mergedSynapseDynamicsGroup5[3];
void pushMergedSynapseDynamicsGroup5ToDevice(unsigned int idx, float* DeltaG, float* E_post, float* F_post, float* PG_post, float* RLTrace, float* TdE_post, float* ZFilter, float* g, uint32_t* ind, float* outPre, uint32_t* rowLength, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseDynamicsGroup5[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup5[idx].E_post = E_post;
    mergedSynapseDynamicsGroup5[idx].F_post = F_post;
    mergedSynapseDynamicsGroup5[idx].PG_post = PG_post;
    mergedSynapseDynamicsGroup5[idx].RLTrace = RLTrace;
    mergedSynapseDynamicsGroup5[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup5[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup5[idx].g = g;
    mergedSynapseDynamicsGroup5[idx].ind = ind;
    mergedSynapseDynamicsGroup5[idx].outPre = outPre;
    mergedSynapseDynamicsGroup5[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup5[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseDynamicsGroup5[idx].rowStride = rowStride;
}
static MergedSynapseDynamicsGroup6 mergedSynapseDynamicsGroup6[2];
void pushMergedSynapseDynamicsGroup6ToDevice(unsigned int idx, float* DeltaG, uint32_t* rowLength, uint32_t* ind, float* epsilonA, float* eFiltered, float* ZFilter, float* VE_post, float* TdMean, float* TdE_post, float* SynSig, float* RLTrace, float* Psi, float* PRew_post, float* PG_post, float* NoiseTrace, float* LogSynSig, float* FAvg, float* Beta_post, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseDynamicsGroup6[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup6[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup6[idx].ind = ind;
    mergedSynapseDynamicsGroup6[idx].epsilonA = epsilonA;
    mergedSynapseDynamicsGroup6[idx].eFiltered = eFiltered;
    mergedSynapseDynamicsGroup6[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup6[idx].VE_post = VE_post;
    mergedSynapseDynamicsGroup6[idx].TdMean = TdMean;
    mergedSynapseDynamicsGroup6[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup6[idx].SynSig = SynSig;
    mergedSynapseDynamicsGroup6[idx].RLTrace = RLTrace;
    mergedSynapseDynamicsGroup6[idx].Psi = Psi;
    mergedSynapseDynamicsGroup6[idx].PRew_post = PRew_post;
    mergedSynapseDynamicsGroup6[idx].PG_post = PG_post;
    mergedSynapseDynamicsGroup6[idx].NoiseTrace = NoiseTrace;
    mergedSynapseDynamicsGroup6[idx].LogSynSig = LogSynSig;
    mergedSynapseDynamicsGroup6[idx].FAvg = FAvg;
    mergedSynapseDynamicsGroup6[idx].Beta_post = Beta_post;
    mergedSynapseDynamicsGroup6[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseDynamicsGroup6[idx].rowStride = rowStride;
}
static MergedSynapseDynamicsGroup7 mergedSynapseDynamicsGroup7[4];
void pushMergedSynapseDynamicsGroup7ToDevice(unsigned int idx, float* ContrMean, uint32_t* rowLength, uint32_t* ind, float* epsilonA, float* eFiltered, float* ZFilter, float* TdMean, float* TdE_post, float* SynSig, float* RLNoisePtrace, float* RLNoiseGtrace, float* PsiBar, float* Psi, float* PRew_post, float* NoiseTrace, float* LogSynSig, float* LocR, float* LocMean, float* FAvg, float* DeltaG, float* ContrR, float* Beta_post, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseDynamicsGroup7[idx].ContrMean = ContrMean;
    mergedSynapseDynamicsGroup7[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup7[idx].ind = ind;
    mergedSynapseDynamicsGroup7[idx].epsilonA = epsilonA;
    mergedSynapseDynamicsGroup7[idx].eFiltered = eFiltered;
    mergedSynapseDynamicsGroup7[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup7[idx].TdMean = TdMean;
    mergedSynapseDynamicsGroup7[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup7[idx].SynSig = SynSig;
    mergedSynapseDynamicsGroup7[idx].RLNoisePtrace = RLNoisePtrace;
    mergedSynapseDynamicsGroup7[idx].RLNoiseGtrace = RLNoiseGtrace;
    mergedSynapseDynamicsGroup7[idx].PsiBar = PsiBar;
    mergedSynapseDynamicsGroup7[idx].Psi = Psi;
    mergedSynapseDynamicsGroup7[idx].PRew_post = PRew_post;
    mergedSynapseDynamicsGroup7[idx].NoiseTrace = NoiseTrace;
    mergedSynapseDynamicsGroup7[idx].LogSynSig = LogSynSig;
    mergedSynapseDynamicsGroup7[idx].LocR = LocR;
    mergedSynapseDynamicsGroup7[idx].LocMean = LocMean;
    mergedSynapseDynamicsGroup7[idx].FAvg = FAvg;
    mergedSynapseDynamicsGroup7[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup7[idx].ContrR = ContrR;
    mergedSynapseDynamicsGroup7[idx].Beta_post = Beta_post;
    mergedSynapseDynamicsGroup7[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseDynamicsGroup7[idx].rowStride = rowStride;
}
void updateSynapses(float t) {
     {
        // merged synapse dynamics group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedSynapseDynamicsGroup0[g]; 
            for(unsigned int i = 0; i < (48u); i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * (1u)) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    group->DeltaG[idSyn] += group->RLTrace[idSyn] * group->E_post[idPost] + group->ZFilter[i] * group->ValReg_post[idPost];
                    group->RLTrace[idSyn] = (9.900133645e-01f) * group->RLTrace[idSyn] - group->ZFilter[i];
                }
            }
        }
    }
     {
        // merged synapse dynamics group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedSynapseDynamicsGroup1[g]; 
            for(unsigned int i = 0; i < (48u); i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * (1u)) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    group->DeltaG[idSyn] += group->TdE_post[idPost] * group->RLTrace[idSyn] + group->ZFilter[i] * group->E_post[idPost];
                    group->RLTrace[idSyn] = (9.900133645e-01f) * group->RLTrace[idSyn] + group->ZFilter[i] * group->PG_post[idPost];
                }
            }
        }
    }
     {
        // merged synapse dynamics group 2
        for(unsigned int g = 0; g < 5; g++) {
            const auto *group = &mergedSynapseDynamicsGroup2[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for (unsigned int j = 0; j < (1u); j++) {
                    const uint32_t idSyn = ((uint32_t)i * (1u)) + j;
                    group->outPre[i] += group->g[idSyn] * group->E_post[j];
                }
            }
        }
    }
     {
        // merged synapse dynamics group 3
        for(unsigned int g = 0; g < 4; g++) {
            const auto *group = &mergedSynapseDynamicsGroup3[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * group->rowStride) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    const float _reward = group->TdE_post[idPost] + group->PRew_post[idPost];
                    const float _epsA = group->epsilonA[idSyn];
                    group->SynSig[idSyn] = exp(group->LogSynSig[idSyn] - 5.0f);
                    const float _normXi = (1.0f - (9.048374180e-01f) * (9.048374180e-01f)) * group->NoiseTrace[idSyn] / (group->SynSig[idSyn] * group->SynSig[idSyn] + 1e-6f);
                    group->DeltaG[idSyn] += (((_reward - group->TdMean[idPost])) * group->RLNoisePtrace[idSyn]) + ((_reward - group->TdMean[idPost]) * group->RLNoiseGtrace[idSyn]) + (group->LocR[idPost] - group->LocMean[idPost]) * (-group->PsiBar[idPost] * group->ZFilter[i] * _normXi) + 0.2061f * (group->ContrR[idPost] - group->ContrMean[idPost]) * group->RLNoiseGtrace[idSyn];
                    const float _e = group->Psi[idPost] * group->ZFilter[i] - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->epsilonA[idSyn] = group->Psi[idPost] * group->ZFilter[i] + ((9.966722161e-01f) * _epsA) - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->eFiltered[idSyn] = (group->eFiltered[idSyn] * (9.048374180e-01f)) + _e;
                    group->DeltaG[idSyn] += (0.000000000e+00f) * (group->FAvg[idPost] - (1.000000000e-02f)) * group->eFiltered[idSyn];
                    group->RLNoisePtrace[idSyn] = (9.900133645e-01f) * group->RLNoisePtrace[idSyn] + (group->Psi[idPost] - group->PsiBar[idPost]) * group->ZFilter[i] * _normXi;
                    group->RLNoiseGtrace[idSyn] = (9.900133645e-01f) * group->RLNoiseGtrace[idSyn] + -group->PsiBar[idPost] * group->ZFilter[i] * _normXi;
                    group->NoiseTrace[idSyn] *= (9.048374180e-01f);
                }
            }
        }
    }
     {
        // merged synapse dynamics group 4
        for(unsigned int g = 0; g < 3; g++) {
            const auto *group = &mergedSynapseDynamicsGroup4[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * (1u)) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    group->DeltaG[idSyn] += group->RLTrace[idSyn] * group->E_post[idPost] + group->ZFilter[i] * group->ValReg_post[idPost];
                    group->RLTrace[idSyn] = (9.801264619e-01f) * group->RLTrace[idSyn] - group->ZFilter[i];
                    if((0.000000000e+00f) == 0)
                    group->outPre[i] += group->g[idSyn];else
                    group->outPre[i] += group->g[idSyn] * group->F_post[idPost];
                }
            }
        }
    }
     {
        // merged synapse dynamics group 5
        for(unsigned int g = 0; g < 3; g++) {
            const auto *group = &mergedSynapseDynamicsGroup5[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * group->rowStride) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    group->DeltaG[idSyn] += group->TdE_post[idPost] * group->RLTrace[idSyn] + group->ZFilter[i] * group->E_post[idPost];
                    group->RLTrace[idSyn] = (9.900133645e-01f) * group->RLTrace[idSyn] + group->ZFilter[i] * group->PG_post[idPost];
                    group->outPre[i] += group->g[idSyn] * group->F_post[idPost];
                }
            }
        }
    }
     {
        // merged synapse dynamics group 6
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedSynapseDynamicsGroup6[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * group->rowStride) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    const float _reward = group->TdE_post[idPost] + group->PRew_post[idPost];
                    const float _epsA = group->epsilonA[idSyn];
                    group->SynSig[idSyn] = exp(group->LogSynSig[idSyn] - 5.0f);
                    const float _normXi = (1.0f - (9.048374180e-01f) * (9.048374180e-01f)) * group->NoiseTrace[idSyn] / (group->SynSig[idSyn] * group->SynSig[idSyn] + 1e-6f);
                    group->DeltaG[idSyn] += ((_reward - group->TdMean[idPost]) * group->RLTrace[idSyn]);
                    const float _e = group->Psi[idPost] * group->ZFilter[i] - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->epsilonA[idSyn] = group->Psi[idPost] * group->ZFilter[i] + ((9.966722161e-01f) * _epsA) - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->eFiltered[idSyn] = (group->eFiltered[idSyn] * (9.048374180e-01f)) + _e;
                    group->DeltaG[idSyn] += (0.000000000e+00f) * (group->FAvg[idPost] - (1.000000000e-02f)) * group->eFiltered[idSyn];
                    group->RLTrace[idSyn] = (9.900133645e-01f) * group->RLTrace[idSyn] + group->eFiltered[idSyn] * (group->PG_post[idPost] - 0.1f * group->VE_post[idPost]);
                    group->NoiseTrace[idSyn] *= (9.048374180e-01f);
                }
            }
        }
    }
     {
        // merged synapse dynamics group 7
        for(unsigned int g = 0; g < 4; g++) {
            const auto *group = &mergedSynapseDynamicsGroup7[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * group->rowStride) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    const float _reward = group->TdE_post[idPost] + group->PRew_post[idPost];
                    const float _epsA = group->epsilonA[idSyn];
                    group->SynSig[idSyn] = exp(group->LogSynSig[idSyn] - 5.0f);
                    const float _normXi = (1.0f - (9.048374180e-01f) * (9.048374180e-01f)) * group->NoiseTrace[idSyn] / (group->SynSig[idSyn] * group->SynSig[idSyn] + 1e-6f);
                    group->DeltaG[idSyn] += (((_reward - group->TdMean[idPost])) * group->RLNoisePtrace[idSyn]) + ((_reward - group->TdMean[idPost]) * group->RLNoiseGtrace[idSyn]) + (group->LocR[idPost] - group->LocMean[idPost]) * (-group->PsiBar[idPost] * group->ZFilter[i] * _normXi) + 0.2061f * (group->ContrR[idPost] - group->ContrMean[idPost]) * group->RLNoiseGtrace[idSyn];
                    const float _e = group->Psi[idPost] * group->ZFilter[i] - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->epsilonA[idSyn] = group->Psi[idPost] * group->ZFilter[i] + ((9.966722161e-01f) * _epsA) - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->eFiltered[idSyn] = (group->eFiltered[idSyn] * (9.048374180e-01f)) + _e;
                    group->DeltaG[idSyn] += (0.000000000e+00f) * (group->FAvg[idPost] - (1.000000000e-02f)) * group->eFiltered[idSyn];
                    group->RLNoisePtrace[idSyn] = (9.900133645e-01f) * group->RLNoisePtrace[idSyn] + (group->Psi[idPost] - group->PsiBar[idPost]) * group->ZFilter[i] * _normXi;
                    group->RLNoiseGtrace[idSyn] = (9.900133645e-01f) * group->RLNoiseGtrace[idSyn] + -group->PsiBar[idPost] * group->ZFilter[i] * _normXi;
                    group->NoiseTrace[idSyn] *= (9.048374180e-01f);
                }
            }
        }
    }
     {
        // merged presynaptic update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedPresynapticUpdateGroup0[g]; 
            
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int idPre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[idPre];
                for (unsigned int j = 0; j < npost; j++) {
                    const uint32_t idSyn = ((uint32_t)idPre * (1u)) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->outPost[idPost] += group->g[idSyn];
                }
            }
        }
    }
     {
        // merged presynaptic update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedPresynapticUpdateGroup1[g]; 
            
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int idPre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[idPre];
                for (unsigned int j = 0; j < npost; j++) {
                    const uint32_t idSyn = ((uint32_t)idPre * (1u)) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->outPost[idPost] += group->g[idSyn];
                }
            }
        }
    }
     {
        // merged presynaptic update group 2
        for(unsigned int g = 0; g < 4; g++) {
            const auto *group = &mergedPresynapticUpdateGroup2[g]; 
            
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int idPre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[idPre];
                for (unsigned int j = 0; j < npost; j++) {
                    const uint32_t idSyn = ((uint32_t)idPre * group->rowStride) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->SynSig[idSyn] = exp(group->LogSynSig[idSyn] - 5.0f);
                    const float _eps = group->SynSig[idSyn] * group->Noise1[idPost];
                    group->outPost[idPost] += group->g[idSyn] + _eps;
                    group->NoiseTrace[idSyn] += _eps;
                }
            }
        }
    }
     {
        // merged presynaptic update group 3
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedPresynapticUpdateGroup3[g]; 
            
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int idPre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[idPre];
                for (unsigned int j = 0; j < npost; j++) {
                    const uint32_t idSyn = ((uint32_t)idPre * group->rowStride) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->SynSig[idSyn] = exp(group->LogSynSig[idSyn] - 5.0f);
                    const float _eps = group->SynSig[idSyn] * group->Noise1[idPost];
                    group->outPost[idPost] += group->g[idSyn] + _eps;
                    group->NoiseTrace[idSyn] += _eps;
                }
            }
        }
    }
     {
        // merged presynaptic update group 4
        for(unsigned int g = 0; g < 4; g++) {
            const auto *group = &mergedPresynapticUpdateGroup4[g]; 
            
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int idPre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[idPre];
                for (unsigned int j = 0; j < npost; j++) {
                    const uint32_t idSyn = ((uint32_t)idPre * group->rowStride) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->SynSig[idSyn] = exp(group->LogSynSig[idSyn] - 5.0f);
                    const float _eps = group->SynSig[idSyn] * group->Noise1[idPost];
                    group->outPost[idPost] += group->g[idSyn] + _eps;
                    group->NoiseTrace[idSyn] += _eps;
                }
            }
        }
    }
}
