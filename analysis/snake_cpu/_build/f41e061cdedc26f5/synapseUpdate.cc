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
struct MergedSynapseDynamicsGroup4
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
struct MergedSynapseDynamicsGroup5
 {
    float* __restrict__ DeltaG;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ ind;
    float* __restrict__ epsilonA;
    float* __restrict__ eFiltered;
    float* __restrict__ ZFilter;
    float* __restrict__ TdE_post;
    float* __restrict__ Sigma_post;
    float* __restrict__ RLNoiseTrace;
    float* __restrict__ PsiBar;
    float* __restrict__ Psi;
    float* __restrict__ PertEpsTrace_post;
    float* __restrict__ PRew_post;
    float* __restrict__ FAvg;
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
static MergedPresynapticUpdateGroup2 mergedPresynapticUpdateGroup2[10];
void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, float* g, uint32_t* ind, float* outPost, uint32_t* rowLength, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t rowStride) {
    mergedPresynapticUpdateGroup2[idx].g = g;
    mergedPresynapticUpdateGroup2[idx].ind = ind;
    mergedPresynapticUpdateGroup2[idx].outPost = outPost;
    mergedPresynapticUpdateGroup2[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup2[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup2[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup2[idx].rowStride = rowStride;
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
static MergedSynapseDynamicsGroup3 mergedSynapseDynamicsGroup3[3];
void pushMergedSynapseDynamicsGroup3ToDevice(unsigned int idx, float* DeltaG, float* E_post, float* F_post, float* RLTrace, float* ValReg_post, float* ZFilter, float* g, uint32_t* ind, float* outPre, uint32_t* rowLength, uint32_t numSrcNeurons) {
    mergedSynapseDynamicsGroup3[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup3[idx].E_post = E_post;
    mergedSynapseDynamicsGroup3[idx].F_post = F_post;
    mergedSynapseDynamicsGroup3[idx].RLTrace = RLTrace;
    mergedSynapseDynamicsGroup3[idx].ValReg_post = ValReg_post;
    mergedSynapseDynamicsGroup3[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup3[idx].g = g;
    mergedSynapseDynamicsGroup3[idx].ind = ind;
    mergedSynapseDynamicsGroup3[idx].outPre = outPre;
    mergedSynapseDynamicsGroup3[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup3[idx].numSrcNeurons = numSrcNeurons;
}
static MergedSynapseDynamicsGroup4 mergedSynapseDynamicsGroup4[3];
void pushMergedSynapseDynamicsGroup4ToDevice(unsigned int idx, float* DeltaG, float* E_post, float* F_post, float* PG_post, float* RLTrace, float* TdE_post, float* ZFilter, float* g, uint32_t* ind, float* outPre, uint32_t* rowLength, uint32_t numSrcNeurons) {
    mergedSynapseDynamicsGroup4[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup4[idx].E_post = E_post;
    mergedSynapseDynamicsGroup4[idx].F_post = F_post;
    mergedSynapseDynamicsGroup4[idx].PG_post = PG_post;
    mergedSynapseDynamicsGroup4[idx].RLTrace = RLTrace;
    mergedSynapseDynamicsGroup4[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup4[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup4[idx].g = g;
    mergedSynapseDynamicsGroup4[idx].ind = ind;
    mergedSynapseDynamicsGroup4[idx].outPre = outPre;
    mergedSynapseDynamicsGroup4[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup4[idx].numSrcNeurons = numSrcNeurons;
}
static MergedSynapseDynamicsGroup5 mergedSynapseDynamicsGroup5[10];
void pushMergedSynapseDynamicsGroup5ToDevice(unsigned int idx, float* DeltaG, uint32_t* rowLength, uint32_t* ind, float* epsilonA, float* eFiltered, float* ZFilter, float* TdE_post, float* Sigma_post, float* RLNoiseTrace, float* PsiBar, float* Psi, float* PertEpsTrace_post, float* PRew_post, float* FAvg, float* Beta_post, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedSynapseDynamicsGroup5[idx].DeltaG = DeltaG;
    mergedSynapseDynamicsGroup5[idx].rowLength = rowLength;
    mergedSynapseDynamicsGroup5[idx].ind = ind;
    mergedSynapseDynamicsGroup5[idx].epsilonA = epsilonA;
    mergedSynapseDynamicsGroup5[idx].eFiltered = eFiltered;
    mergedSynapseDynamicsGroup5[idx].ZFilter = ZFilter;
    mergedSynapseDynamicsGroup5[idx].TdE_post = TdE_post;
    mergedSynapseDynamicsGroup5[idx].Sigma_post = Sigma_post;
    mergedSynapseDynamicsGroup5[idx].RLNoiseTrace = RLNoiseTrace;
    mergedSynapseDynamicsGroup5[idx].PsiBar = PsiBar;
    mergedSynapseDynamicsGroup5[idx].Psi = Psi;
    mergedSynapseDynamicsGroup5[idx].PertEpsTrace_post = PertEpsTrace_post;
    mergedSynapseDynamicsGroup5[idx].PRew_post = PRew_post;
    mergedSynapseDynamicsGroup5[idx].FAvg = FAvg;
    mergedSynapseDynamicsGroup5[idx].Beta_post = Beta_post;
    mergedSynapseDynamicsGroup5[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseDynamicsGroup5[idx].rowStride = rowStride;
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
        for(unsigned int g = 0; g < 3; g++) {
            const auto *group = &mergedSynapseDynamicsGroup3[g]; 
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
        // merged synapse dynamics group 4
        for(unsigned int g = 0; g < 3; g++) {
            const auto *group = &mergedSynapseDynamicsGroup4[g]; 
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
        // merged synapse dynamics group 5
        for(unsigned int g = 0; g < 10; g++) {
            const auto *group = &mergedSynapseDynamicsGroup5[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const uint32_t idSyn = ((uint32_t)i * group->rowStride) + s;
                    const unsigned int idPost = group->ind[idSyn];
                    const float _reward = group->TdE_post[idPost] + group->PRew_post[idPost];
                    const float _epsA = group->epsilonA[idSyn];
                    const float _normXi = (1.0f - (9.048374180e-01f) * (9.048374180e-01f)) * group->PertEpsTrace_post[idPost] / (group->Sigma_post[idPost] * group->Sigma_post[idPost] + 1e-6f);
                    group->DeltaG[idSyn] += _reward * group->RLNoiseTrace[idSyn];
                    const float _e = group->Psi[idPost] * group->ZFilter[i] - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->epsilonA[idSyn] = group->Psi[idPost] * group->ZFilter[i] + ((9.966722161e-01f) * _epsA) - group->Psi[idPost] * group->Beta_post[idPost] * _epsA;
                    group->eFiltered[idSyn] = (group->eFiltered[idSyn] * (9.048374180e-01f)) + _e;
                    group->DeltaG[idSyn] += (1.000000000e-04f) * (group->FAvg[idPost] - (1.000000000e-02f)) * group->eFiltered[idSyn];
                    group->RLNoiseTrace[idSyn] = (9.925894760e-01f) * group->RLNoiseTrace[idSyn] + ((group->Psi[idPost]) - 2.0f * group->PsiBar[idPost]) * (group->ZFilter[i] - group->Beta_post[idPost] * _epsA) * _normXi;
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
        for(unsigned int g = 0; g < 10; g++) {
            const auto *group = &mergedPresynapticUpdateGroup2[g]; 
            
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int idPre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[idPre];
                for (unsigned int j = 0; j < npost; j++) {
                    const uint32_t idSyn = ((uint32_t)idPre * group->rowStride) + j;
                    const unsigned int idPost = group->ind[idSyn];
                    group->outPost[idPost] += group->g[idSyn];
                }
            }
        }
    }
}
