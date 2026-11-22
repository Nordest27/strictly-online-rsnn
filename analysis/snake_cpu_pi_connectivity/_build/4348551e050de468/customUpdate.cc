#include "definitions.h"

struct MergedCustomUpdateWUGroup0
 {
    float* __restrict__ Gradient;
    float* __restrict__ InitDone;
    float* __restrict__ M;
    float* __restrict__ S;
    float* __restrict__ Step;
    float* __restrict__ Surprise;
    float* __restrict__ VarInit;
    float* __restrict__ Variable;
    uint32_t* __restrict__ rowLength;
    float Alpha;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedCustomConnectivityUpdateGroup0
 {
    uint32_t* __restrict__ NumPruned;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ ind;
    float* __restrict__ g;
    float* __restrict__ dependentVar9;
    float* __restrict__ dependentVar8;
    float* __restrict__ dependentVar7;
    float* __restrict__ dependentVar6;
    float* __restrict__ dependentVar5;
    float* __restrict__ dependentVar4;
    float* __restrict__ dependentVar3;
    float* __restrict__ dependentVar2;
    float* __restrict__ dependentVar13;
    float* __restrict__ dependentVar12;
    float* __restrict__ dependentVar11;
    float* __restrict__ dependentVar10;
    float* __restrict__ dependentVar1;
    float* __restrict__ dependentVar0;
    uint32_t* __restrict__ Connectivity;
    int32_t Sign;
    uint32_t NumRowWords;
    uint32_t numSrcNeurons;
    uint32_t numTrgNeurons;
    uint32_t rowStride;
    
}
;
struct MergedCustomConnectivityUpdateGroup1
 {
    uint32_t* __restrict__ NumPruned;
    uint32_t* __restrict__ rowLength;
    uint32_t* __restrict__ ind;
    float* __restrict__ g;
    float* __restrict__ dependentVar9;
    float* __restrict__ dependentVar8;
    float* __restrict__ dependentVar7;
    float* __restrict__ dependentVar6;
    float* __restrict__ dependentVar5;
    float* __restrict__ dependentVar4;
    float* __restrict__ dependentVar3;
    float* __restrict__ dependentVar2;
    float* __restrict__ dependentVar13;
    float* __restrict__ dependentVar12;
    float* __restrict__ dependentVar11;
    float* __restrict__ dependentVar10;
    float* __restrict__ dependentVar1;
    float* __restrict__ dependentVar0;
    uint32_t* __restrict__ Connectivity;
    int32_t Sign;
    uint32_t NumRowWords;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedCustomConnectivityUpdateGroup2
 {
    uint32_t* __restrict__ Connectivity;
    uint32_t* __restrict__ ind;
    uint32_t* __restrict__ rowLength;
    int32_t AllowRecurrent;
    uint32_t NumRowWords;
    uint32_t numSrcNeurons;
    uint32_t numTrgNeurons;
    uint32_t rowStride;
    
}
;
static MergedCustomUpdateWUGroup0 mergedCustomUpdateWUGroup0[11];
void pushMergedCustomUpdateWUGroup0ToDevice(unsigned int idx, float* Gradient, float* InitDone, float* M, float* S, float* Step, float* Surprise, float* VarInit, float* Variable, uint32_t* rowLength, float Alpha, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedCustomUpdateWUGroup0[idx].Gradient = Gradient;
    mergedCustomUpdateWUGroup0[idx].InitDone = InitDone;
    mergedCustomUpdateWUGroup0[idx].M = M;
    mergedCustomUpdateWUGroup0[idx].S = S;
    mergedCustomUpdateWUGroup0[idx].Step = Step;
    mergedCustomUpdateWUGroup0[idx].Surprise = Surprise;
    mergedCustomUpdateWUGroup0[idx].VarInit = VarInit;
    mergedCustomUpdateWUGroup0[idx].Variable = Variable;
    mergedCustomUpdateWUGroup0[idx].rowLength = rowLength;
    mergedCustomUpdateWUGroup0[idx].Alpha = Alpha;
    mergedCustomUpdateWUGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedCustomUpdateWUGroup0[idx].rowStride = rowStride;
}
static MergedCustomConnectivityUpdateGroup0 mergedCustomConnectivityUpdateGroup0[9];
void pushMergedCustomConnectivityUpdateGroup0ToDevice(unsigned int idx, uint32_t* NumPruned, uint32_t* rowLength, uint32_t* ind, float* g, float* dependentVar9, float* dependentVar8, float* dependentVar7, float* dependentVar6, float* dependentVar5, float* dependentVar4, float* dependentVar3, float* dependentVar2, float* dependentVar13, float* dependentVar12, float* dependentVar11, float* dependentVar10, float* dependentVar1, float* dependentVar0, uint32_t* Connectivity, int32_t Sign, uint32_t NumRowWords, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride) {
    mergedCustomConnectivityUpdateGroup0[idx].NumPruned = NumPruned;
    mergedCustomConnectivityUpdateGroup0[idx].rowLength = rowLength;
    mergedCustomConnectivityUpdateGroup0[idx].ind = ind;
    mergedCustomConnectivityUpdateGroup0[idx].g = g;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar9 = dependentVar9;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar8 = dependentVar8;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar7 = dependentVar7;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar6 = dependentVar6;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar5 = dependentVar5;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar4 = dependentVar4;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar3 = dependentVar3;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar2 = dependentVar2;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar13 = dependentVar13;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar12 = dependentVar12;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar11 = dependentVar11;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar10 = dependentVar10;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar1 = dependentVar1;
    mergedCustomConnectivityUpdateGroup0[idx].dependentVar0 = dependentVar0;
    mergedCustomConnectivityUpdateGroup0[idx].Connectivity = Connectivity;
    mergedCustomConnectivityUpdateGroup0[idx].Sign = Sign;
    mergedCustomConnectivityUpdateGroup0[idx].NumRowWords = NumRowWords;
    mergedCustomConnectivityUpdateGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedCustomConnectivityUpdateGroup0[idx].numTrgNeurons = numTrgNeurons;
    mergedCustomConnectivityUpdateGroup0[idx].rowStride = rowStride;
}
static MergedCustomConnectivityUpdateGroup1 mergedCustomConnectivityUpdateGroup1[9];
void pushMergedCustomConnectivityUpdateGroup1ToDevice(unsigned int idx, uint32_t* NumPruned, uint32_t* rowLength, uint32_t* ind, float* g, float* dependentVar9, float* dependentVar8, float* dependentVar7, float* dependentVar6, float* dependentVar5, float* dependentVar4, float* dependentVar3, float* dependentVar2, float* dependentVar13, float* dependentVar12, float* dependentVar11, float* dependentVar10, float* dependentVar1, float* dependentVar0, uint32_t* Connectivity, int32_t Sign, uint32_t NumRowWords, uint32_t numSrcNeurons, uint32_t rowStride) {
    mergedCustomConnectivityUpdateGroup1[idx].NumPruned = NumPruned;
    mergedCustomConnectivityUpdateGroup1[idx].rowLength = rowLength;
    mergedCustomConnectivityUpdateGroup1[idx].ind = ind;
    mergedCustomConnectivityUpdateGroup1[idx].g = g;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar9 = dependentVar9;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar8 = dependentVar8;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar7 = dependentVar7;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar6 = dependentVar6;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar5 = dependentVar5;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar4 = dependentVar4;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar3 = dependentVar3;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar2 = dependentVar2;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar13 = dependentVar13;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar12 = dependentVar12;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar11 = dependentVar11;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar10 = dependentVar10;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar1 = dependentVar1;
    mergedCustomConnectivityUpdateGroup1[idx].dependentVar0 = dependentVar0;
    mergedCustomConnectivityUpdateGroup1[idx].Connectivity = Connectivity;
    mergedCustomConnectivityUpdateGroup1[idx].Sign = Sign;
    mergedCustomConnectivityUpdateGroup1[idx].NumRowWords = NumRowWords;
    mergedCustomConnectivityUpdateGroup1[idx].numSrcNeurons = numSrcNeurons;
    mergedCustomConnectivityUpdateGroup1[idx].rowStride = rowStride;
}
static MergedCustomConnectivityUpdateGroup2 mergedCustomConnectivityUpdateGroup2[9];
void pushMergedCustomConnectivityUpdateGroup2ToDevice(unsigned int idx, uint32_t* Connectivity, uint32_t* ind, uint32_t* rowLength, int32_t AllowRecurrent, uint32_t NumRowWords, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride) {
    mergedCustomConnectivityUpdateGroup2[idx].Connectivity = Connectivity;
    mergedCustomConnectivityUpdateGroup2[idx].ind = ind;
    mergedCustomConnectivityUpdateGroup2[idx].rowLength = rowLength;
    mergedCustomConnectivityUpdateGroup2[idx].AllowRecurrent = AllowRecurrent;
    mergedCustomConnectivityUpdateGroup2[idx].NumRowWords = NumRowWords;
    mergedCustomConnectivityUpdateGroup2[idx].numSrcNeurons = numSrcNeurons;
    mergedCustomConnectivityUpdateGroup2[idx].numTrgNeurons = numTrgNeurons;
    mergedCustomConnectivityUpdateGroup2[idx].rowStride = rowStride;
}
void pushMergedCustomUpdateWU0AlphaToDevice(unsigned int idx, float value) {
    mergedCustomUpdateWUGroup0[idx].Alpha = value;
}
void pushMergedCustomConnectivityUpdate0ConnectivityToDevice(unsigned int idx, uint32_t* value) {
    mergedCustomConnectivityUpdateGroup0[idx].Connectivity = value;
}
void pushMergedCustomConnectivityUpdate1ConnectivityToDevice(unsigned int idx, uint32_t* value) {
    mergedCustomConnectivityUpdateGroup1[idx].Connectivity = value;
}
void pushMergedCustomConnectivityUpdate2ConnectivityToDevice(unsigned int idx, uint32_t* value) {
    mergedCustomConnectivityUpdateGroup2[idx].Connectivity = value;
}
void updateDaleInit(unsigned long long timestep) {
     {
        // merged custom connectivity update group 2
        for(unsigned int g = 0; g < 9; g++) {
            const auto *group = &mergedCustomConnectivityUpdateGroup2[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                const unsigned int rowStartIdx = i * group->rowStride;
                uint32_t* _rowConnectivity = group->Connectivity + (group->NumRowWords * i);
                for(int32_t _i = 0; _i < group->NumRowWords; _i++) {
                    _rowConnectivity[_i] = 0;
                }
                
                if(i < group->numTrgNeurons && !group->AllowRecurrent)
                 {
                    _rowConnectivity[i / 32] |= (1 << (i % 32));
                }
                
                for(int j = 0; j < group->rowLength[i]; j++) {
                     {
                        _rowConnectivity[group->ind[rowStartIdx + j] / 32] |= (1 << (group->ind[rowStartIdx + j] % 32));
                    }
                }
                
            }
        }
    }
}
void updateDalePrune(unsigned long long timestep) {
     {
        // merged custom connectivity update group 1
        for(unsigned int g = 0; g < 9; g++) {
            const auto *group = &mergedCustomConnectivityUpdateGroup1[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                const unsigned int rowStartIdx = i * group->rowStride;
                group->NumPruned[i] = 0;
                uint32_t* _rowConnectivity = group->Connectivity + (group->NumRowWords * i);
                for(int j = 0; j < group->rowLength[i]; j++) {
                    const uint32_t idx = (uint32_t)rowStartIdx + j;
                     {
                        const bool _prune = (group->Sign > 0 && group->g[idx] < 0.0f) || (group->Sign < 0 && group->g[idx] > 0.0f);
                        if(_prune)
                         {
                            if(standardUniformDistribution(hostRNG) < (9.000000000e-01f))
                             {
                                group->g[idx] = group->Sign * 1e-5f * fabs(standardUniformDistribution(hostRNG));
                            }
                            else
                             {
                                _rowConnectivity[group->ind[rowStartIdx + j] / 32] &= ~(1 << (group->ind[rowStartIdx + j] % 32));
                                 {
                                    const unsigned lastIdx = rowStartIdx + group->rowLength[i] - 1;
                                    group->ind[idx] = group->ind[lastIdx];
                                    group->g[idx] = group->g[lastIdx];
                                    group->dependentVar0[idx] = group->dependentVar0[lastIdx];
                                    group->dependentVar1[idx] = group->dependentVar1[lastIdx];
                                    group->dependentVar2[idx] = group->dependentVar2[lastIdx];
                                    group->dependentVar3[idx] = group->dependentVar3[lastIdx];
                                    group->dependentVar4[idx] = group->dependentVar4[lastIdx];
                                    group->dependentVar5[idx] = group->dependentVar5[lastIdx];
                                    group->dependentVar6[idx] = group->dependentVar6[lastIdx];
                                    group->dependentVar7[idx] = group->dependentVar7[lastIdx];
                                    group->dependentVar8[idx] = group->dependentVar8[lastIdx];
                                    group->dependentVar9[idx] = group->dependentVar9[lastIdx];
                                    group->dependentVar10[idx] = group->dependentVar10[lastIdx];
                                    group->dependentVar11[idx] = group->dependentVar11[lastIdx];
                                    group->dependentVar12[idx] = group->dependentVar12[lastIdx];
                                    group->dependentVar13[idx] = group->dependentVar13[lastIdx];
                                    group->rowLength[i]--;
                                    j--;
                                }
                                ;
                                group->NumPruned[i]++;
                            }
                            
                        }
                        
                    }
                }
                
            }
        }
    }
}
void updateDaleRewire(unsigned long long timestep) {
     {
        // merged custom connectivity update group 0
        for(unsigned int g = 0; g < 9; g++) {
            const auto *group = &mergedCustomConnectivityUpdateGroup0[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                const unsigned int rowStartIdx = i * group->rowStride;
                uint32_t* _rowConnectivity = group->Connectivity + (group->NumRowWords * i);
                while(group->NumPruned[i] > 0)
                 {
                    const uint32_t _j = hostRNG() % group->numTrgNeurons;
                    if(_rowConnectivity[_j / 32] & (1 << (_j % 32)))
                     {
                        continue;
                    }
                    
                     {
                        assert(group->rowLength[i] < group->rowStride);
                        const unsigned newIdx = rowStartIdx + group->rowLength[i];
                        group->ind[newIdx] = _j;
                        group->g[newIdx] = group->Sign * 1e-5f * fabs(standardUniformDistribution(hostRNG));
                        group->dependentVar0[newIdx] = 0;
                        group->dependentVar1[newIdx] = 0;
                        group->dependentVar2[newIdx] = 0;
                        group->dependentVar3[newIdx] = 0;
                        group->dependentVar4[newIdx] = 0;
                        group->dependentVar5[newIdx] = 0;
                        group->dependentVar6[newIdx] = 0;
                        group->dependentVar7[newIdx] = 0;
                        group->dependentVar8[newIdx] = 0;
                        group->dependentVar9[newIdx] = 0;
                        group->dependentVar10[newIdx] = 0;
                        group->dependentVar11[newIdx] = 0;
                        group->dependentVar12[newIdx] = 0;
                        group->dependentVar13[newIdx] = 0;
                        group->rowLength[i]++;
                    }
                    ;
                    _rowConnectivity[_j / 32] |= (1 << (_j % 32));
                    group->NumPruned[i]--;
                }
                
            }
        }
    }
}
void updateGradientLearn(unsigned long long timestep) {
     {
        // merged custom WU update group 0
        for(unsigned int g = 0; g < 11; g++) {
            const auto *group = &mergedCustomUpdateWUGroup0[g]; 
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                for(unsigned int s = 0; s < group->rowLength[i]; s++) {
                    const unsigned int idSyn = (i * group->rowStride) + s;
                    float _lVarInit = group->VarInit[idSyn];
                    float _lSurprise = group->Surprise[idSyn];
                    float _lStep = group->Step[idSyn];
                    float _lS = group->S[idSyn];
                    float _lInitDone = group->InitDone[idSyn];
                    float _lM = group->M[idSyn];
                    float _lVariable = group->Variable[idSyn];
                    float _lGradient = group->Gradient[idSyn];
                    if(_lInitDone < 0.5f)
                     {
                        _lVarInit = _lVariable;
                        _lInitDone = 1.0f;
                    }
                    
                    float _deltaSample = _lGradient;
                    _lStep += 1.0f;
                    _lM = ((9.900000000e-01f) * _lM) + (1.0f - (9.900000000e-01f)) * _deltaSample;
                    float _deviation = _deltaSample - _lM;
                    _lS = ((9.999900000e-01f) * _lS) + (1.0f - (9.999900000e-01f)) * _deviation * _deviation + (1.000000000e-08f);
                    float _momentScale1 = 1.0f / (1.0f - pow((9.900000000e-01f), _lStep));
                    float _momentScale2 = 1.0f / (1.0f - pow((9.999900000e-01f), _lStep));
                    float _mHat = _lM * _momentScale1;
                    float _sHat = _lS * _momentScale2;
                    _lSurprise = _deviation * _deviation;
                    _lVariable -= (group->Alpha * _mHat) / (sqrt(_sHat) + (1.000000000e-08f));
                    _lGradient = 0;
                    _lVariable -= group->Alpha * 1e-05f * (_lVariable - _lVarInit);
                    group->Variable[idSyn] = _lVariable;
                    group->Gradient[idSyn] = _lGradient;
                    group->VarInit[idSyn] = _lVarInit;
                    group->Surprise[idSyn] = _lSurprise;
                    group->Step[idSyn] = _lStep;
                    group->S[idSyn] = _lS;
                    group->InitDone[idSyn] = _lInitDone;
                    group->M[idSyn] = _lM;
                }
            }
        }
    }
}
