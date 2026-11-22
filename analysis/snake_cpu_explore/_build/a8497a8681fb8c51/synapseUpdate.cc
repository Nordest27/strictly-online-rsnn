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
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    float* __restrict__ outPost;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ srcSpk;
    uint32_t* __restrict__ srcSpkCnt;
    
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
    float* __restrict__ InhAvg_post;
    float* __restrict__ InhTarget_post;
    float* __restrict__ XMean;
    float* __restrict__ XTrace;
    float* __restrict__ YMean;
    float* __restrict__ YTrace;
    float* __restrict__ g;
    uint32_t* __restrict__ ind;
    uint32_t* __restrict__ rowLength;
    
}
;
struct MergedSynapseDynamicsGroup3
 {
    float* __restrict__ E_post;
    float* __restrict__ g;
    float* __restrict__ outPre;
    uint32_t numSrcNeurons;
    
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
    
}
;
struct MergedSynapseDynamicsGroup6
 {
    float* __restrict__ Beta_post;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ ind;
    float* __restrict__ epsilonA;
    float* __restrict__ eFiltered;
    float* __restrict__ ZFilter;
    float* __restrict__ TdE_post;
    float* __restrict__ SynSig;
    float* __restrict__ RLNoisePtrace;
    float* __restrict__ RLNoiseGtrace;
    float* __restrict__ PsiTargetN;
    float* __restrict__ PsiBar;
    float* __restrict__ Psi;
    float* __restrict__ PRew_post;
    float* __restrict__ NoiseTrace;
    float* __restrict__ LogSynSig;
    float* __restrict__ FAvg;
    float* __restrict__ DeltaG;
    float* __restrict__ AbsTd;
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
static MergedPresynapticUpdateGroup2 mergedPresynapticUpdateGroup2[1];
void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, float* g, uint32_t* ind, float* outPost, uint32_t* rowLength, uint32_t* srcSpk, uint32_t* srcSpkCnt) {
    mergedPresynapticUpdateGroup2[idx].g = g;
    mergedPresynapticUpdateGroup2[idx].ind = ind;
    mergedPresynapticUpdateGroup2[idx].outPost = outPost;
    mergedPresynapticUpdateGroup2[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup2[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup2[idx].srcSpkCnt = srcSpkCnt;
}
static MergedPresynapticUpdateGroup3 mergedPresynapticUpdateGroup3[9];
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
static MergedSynapseDynamicsGroup2 mergedSynapseDynamicsGroup2[1];
void pushMergedSynapseDynamicsGroup2ToDevice(unsigned int idx, float* InhAvg_post, float* InhTarget_post, float* XMean, float* XTrace, float* YMean, float* YTrace, float* g, uint32_t* ind, uint32_t* rowLength) {
    mergedSynapseDynamicsGroup2[idx].InhAvg_post = InhAvg_post;
    mergedSynapseDynamicsGroup2[idx].InhTarget_post = InhTarget_post;
    mergedSynapseDynamicsGroup2[idx].XMean = XMean;
    mergedSynapseDynamicsGroup2[idx].XTrace = XTrace;
    mergedSynapseDynamicsGroup2[idx].YMean = YMean;
    mergedSynapseDynamicsGroup2[idx].YTrace = YTrace;
    mergedSynapseDynamicsGroup2[idx].g = g;
    mergedSynapseDynamicsGroup2[idx].ind = ind;
    mergedSynapseDynamicsGroup2[idx].rowLength = rowLength;
}
static MergedSynapseDynamicsGroup3 mergedSynapseDynamicsGroup3[5];
void pushMergedSynapseDynamicsGroup3ToDevice(unsigned int idx, float* E_post, float* g, float* outPre, uint32_t numSrcNeurons) {
    mergedSynapseDynamicsGroup3[idx].E_post = E_post;
    mergedSynapseDynamicsGroup3[idx].g = g;
    mergedSynapseDynamicsGroup3[idx].outPre = outPre;
    mergedSynapseDynamicsGroup3[idx].numSrcNeurons = numSrcNeurons;
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
void pushMergedSynapseDynamicsGroup5ToDevice(unsigned int idx, float* DeltaG, float* E_post, float* F_post, float* PG_post, float* RLTrace, float* TdE_post, float* ZFilter, float* g, uint32_t* ind, float* outPre, uint32_t* rowLength, uint32_t numSrcNeurons) {
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
}
static MergedSynapseDynamicsGroup6 mergedSynapseDynamicsGroup6[9];
void pushMergedSynapseDynamicsGroup6ToDevice(unsigned int idx, float* Beta_post, uint32_t* rowLength, uint32_t* ind, float* epsilonA, float* eFiltered, float* ZFilter, float* TdE_post, float* SynSig, float* RLNoisePtrace, float* RLNoiseGtrace, float* PsiTargetN, float* PsiBar, float* Psi, float* PRew_post, float* NoiseTrace, float* LogSynSig, float* FAvg, float* DeltaG, float* AbsTd, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseDynamicsGroup6[idx].Beta_post = Beta_post;
    mergedSynapseDynamicsGroup6[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup6[idx].ind = ind;
    mergedSynapseDynamicsGroup6[idx].epsilonA = epsilonA;
    mergedSynapseDynamicsGroup6[idx].eFiltered = eFiltered;
    mergedSynapseDynamicsGroup6[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup6[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup6[idx].SynSig = SynSig;
    mergedSynapseDynamicsGroup6[idx].RLNoisePtrace = RLNoisePtrace;
    mergedSynapseDynamicsGroup6[idx].RLNoiseGtrace = RLNoiseGtrace;
    mergedSynapseDynamicsGroup6[idx].PsiTargetN = PsiTargetN;
    mergedSynapseDynamicsGroup6[idx].PsiBar = PsiBar;
    mergedSynapseDynamicsGroup6[idx].Psi = Psi;
    mergedSynapseDynamicsGroup6[idx].PRew_post = PRew_post;
    mergedSynapseDynamicsGroup6[idx].NoiseTrace = NoiseTrace;
    mergedSynapseDynamicsGroup6[idx].LogSynSig = LogSynSig;
    mergedSynapseDynamicsGroup6[idx].FAvg = FAvg;
    mergedSynapseDynamicsGroup6[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup6[idx].AbsTd = AbsTd;
    mergedSynapseDynamicsGroup6[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseDynamicsGroup6[idx].rowStride = rowStride;
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
                    group->RLTrace[idSyn] = (9.771599684e-01f) * group->RLTrace[idSyn] - group->ZFilter[i];
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
                    const uint32_t idSyn = ((uint32_t)i * (4u)) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    group->DeltaG[idSyn] += group->TdE_post[idPost] * group->RLTrace[idSyn] + group->ZFilter[i] * group->E_post[idPost];
                    group->RLTrace[idSyn] = (9.925894760e-01f) * group->RLTrace[idSyn] + group->ZFilter[i] * group->PG_post[idPost];
                }
            }
        }
    }
     {
        // merged synapse dynamics group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedSynapseDynamicsGroup2[g]; 
            for(unsigned int i = 0; i < (48u); i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * (27u)) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    group->g[idSyn] = fmin(0.0f, group->g[idSyn] - (1.000000000e-04f) * (group->XTrace[i] - group->XMean[i]) * (group->YTrace[idPost] - group->YMean[idPost]));
                    if(group->InhTarget_post[idPost] > 0.0f)
                     {
                        group->g[idSyn] *= fmax(0.0f, 1.0f - (1.000000000e-03f) * (group->InhAvg_post[idPost] - group->InhTarget_post[idPost]) / group->InhTarget_post[idPost]);
                    }
                    
                }
            }
        }
    }
     {
        // merged synapse dynamics group 3
        for(unsigned int g = 0; g < 5; g++) {
            const auto *group = &mergedSynapseDynamicsGroup3[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for (unsigned int j = 0; j < (1u); j++) {
                    const uint32_t idSyn = ((uint32_t)i * (1u)) + j;
                    group->outPre[i] += group->g[idSyn] * group->E_post[j];
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
                    group->RLTrace[idSyn] = (9.699187010e-01f) * group->RLTrace[idSyn] - group->ZFilter[i];
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
                    const uint32_t idSyn = ((uint32_t)i * (4u)) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    group->DeltaG[idSyn] += group->TdE_post[idPost] * group->RLTrace[idSyn] + group->ZFilter[i] * group->E_post[idPost];
                    group->RLTrace[idSyn] = (9.925894760e-01f) * group->RLTrace[idSyn] + group->ZFilter[i] * group->PG_post[idPost];
                    group->outPre[i] += group->g[idSyn] * group->F_post[idPost];
                }
            }
        }
    }
     {
        // merged synapse dynamics group 6
        for(unsigned int g = 0; g < 9; g++) {
            const auto *group = &mergedSynapseDynamicsGroup6[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * group->rowStride) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    const float _reward = group->TdE_post[idPost] + group->PRew_post[idPost];
                    const float _epsA = group->epsilonA[idSyn];
                    group->SynSig[idSyn] = exp(group->LogSynSig[idSyn] - 5.0f);
                    const float _normXi = (1.0f - (9.048374180e-01f) * (9.048374180e-01f)) * group->NoiseTrace[idSyn] / (group->SynSig[idSyn] * group->SynSig[idSyn] + 1e-6f);
                    group->DeltaG[idSyn] += (_reward * group->RLNoisePtrace[idSyn]) + (_reward * group->RLNoiseGtrace[idSyn]) + 1.0f * group->AbsTd[idPost] * (group->PsiTargetN[idPost] > 0.0f ? (group->PsiBar[idPost] - group->PsiTargetN[idPost]) : 0.0f) * group->RLNoisePtrace[idSyn];
                    const float _e = group->Psi[idPost] * group->ZFilter[i] - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->epsilonA[idSyn] = group->Psi[idPost] * group->ZFilter[i] + ((9.966722161e-01f) * _epsA) - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->eFiltered[idSyn] = (group->eFiltered[idSyn] * (9.048374180e-01f)) + _e;
                    group->DeltaG[idSyn] += (1.000000000e-04f) * (group->FAvg[idPost] - (1.000000000e-02f)) * group->eFiltered[idSyn];
                    group->RLNoisePtrace[idSyn] = (9.925894760e-01f) * group->RLNoisePtrace[idSyn] + (group->Psi[idPost] - group->PsiBar[idPost]) * (group->ZFilter[i] - group->Beta_post[idPost] * _epsA) * _normXi;
                    group->RLNoiseGtrace[idSyn] = (9.925894760e-01f) * group->RLNoiseGtrace[idSyn] + -group->PsiBar[idPost] * (group->ZFilter[i] - group->Beta_post[idPost] * _epsA) * _normXi;
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
                    const uint32_t idSyn = ((uint32_t)idPre * (4u)) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->outPost[idPost] += group->g[idSyn];
                }
            }
        }
    }
     {
        // merged presynaptic update group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedPresynapticUpdateGroup2[g]; 
            
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int idPre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[idPre];
                for (unsigned int j = 0; j < npost; j++) {
                    const uint32_t idSyn = ((uint32_t)idPre * (27u)) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->outPost[idPost] += group->g[idSyn];
                }
            }
        }
    }
     {
        // merged presynaptic update group 3
        for(unsigned int g = 0; g < 9; g++) {
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
}
