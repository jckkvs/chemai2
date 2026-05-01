<!-- frontend/src/views/MLView.vue -->
<template>
  <div class="ml-container">
    <div class="header">
      <div class="title-area">
        <h2>🚀 ML Studio</h2>
        <p class="subtitle">Configure pipelines, descriptors, and train multiple algorithms</p>
      </div>
      <el-button type="success" size="large" :loading="store.isLoading" @click="handleRun" icon="VideoPlay" :disabled="!store.hasData">
        Start Training
      </el-button>
    </div>

    <el-row :gutter="20">
      <!-- Config Panel -->
      <el-col :span="11">
        <el-tabs type="border-card" class="config-tabs shadow">
          <!-- Pipeline Tab -->
          <el-tab-pane>
            <template #label>
              <span class="tab-label"><el-icon><Setting /></el-icon> Pipeline</span>
            </template>
            
            <el-form :model="store.pipelineConfig" label-width="150px" label-position="left">
              <el-form-item label="CV Folds">
                <el-input-number v-model="store.pipelineConfig.cv_folds" :min="2" :max="10" />
              </el-form-item>
              
              <el-form-item label="Numerical Scaler">
                <el-select v-model="store.pipelineConfig.num_scaler" class="full-width">
                  <el-option label="Standard Scaler" value="standard" />
                  <el-option label="MinMax Scaler" value="minmax" />
                  <el-option label="Robust Scaler" value="robust" />
                  <el-option label="No Scaling" value="none" />
                </el-select>
              </el-form-item>

              <el-form-item label="Polynomial Features">
                <el-switch v-model="store.pipelineConfig.do_polynomial" />
                <el-input-number v-if="store.pipelineConfig.do_polynomial" v-model="store.pipelineConfig.poly_degree" :min="2" :max="3" style="margin-left: 10px" />
              </el-form-item>

              <el-divider>Algorithms Selection</el-divider>
              
              <div v-if="loadingModels" class="loading-models">
                <el-icon class="is-loading"><Loading /></el-icon> Loading available models...
              </div>
              <el-checkbox-group v-else v-model="store.pipelineConfig.selected_models" class="model-selection">
                <el-checkbox v-for="m in availableModels" :key="m.key" :label="m.key">
                  {{ m.name }}
                </el-checkbox>
              </el-checkbox-group>
            </el-form>
          </el-tab-pane>

          <!-- Descriptors Tab -->
          <el-tab-pane>
            <template #label>
              <span class="tab-label"><el-icon><Operation /></el-icon> Descriptors</span>
            </template>
            <DescriptorSelector v-model="store.pipelineConfig.descriptors" />
          </el-tab-pane>

          <!-- Constraints Tab -->
          <el-tab-pane>
            <template #label>
              <span class="tab-label"><el-icon><TrendingUp /></el-icon> Constraints</span>
            </template>
            <MonotonicConstraints v-model="store.pipelineConfig.monotonic_constraints" />
          </el-tab-pane>

          <!-- Mixture Tab -->
          <el-tab-pane>
            <template #label>
              <span class="tab-label"><el-icon><Blender /></el-icon> Mixture</span>
            </template>
            <MixtureInput />
          </el-tab-pane>
        </el-tabs>
      </el-col>

      <!-- Execution / Status -->
      <el-col :span="13">
        <el-card class="status-card shadow">
          <template #header>
            <div class="card-header">
              <span>📊 Execution Status</span>
            </div>
          </template>
          
          <div v-if="store.isLoading" class="running-state animate-pulse">
            <el-progress type="circle" :percentage="progress" status="success" :stroke-width="12" />
            <p class="status-text">Processing chemical features and training models...</p>
            <div class="current-step">{{ currentStep }}</div>
          </div>

          <div v-else-if="result" class="result-summary animate-fade-in">
            <el-result
              icon="success"
              title="Pipeline Successfully Trained"
              :sub-title="`Champion Model: ${result.best_model}`"
            >
              <template #extra>
                <el-button type="primary" size="large" @click="viewResults">View Interactive Report</el-button>
              </template>
            </el-result>

            <el-descriptions title="Optimization Metrics" :column="2" border class="score-summary">
              <el-descriptions-item label="Best Metric ({{ store.taskType === 'regression' ? 'R²' : 'Acc' }})">
                <span class="score-val">{{ result.score.toFixed(4) }}</span>
              </el-descriptions-item>
              <el-descriptions-item label="CV Mean">
                <span class="score-val">{{ average(result.cv_scores).toFixed(4) }}</span>
              </el-descriptions-item>
              <el-descriptions-item label="CV Std Dev">
                <span class="score-val">{{ stdDev(result.cv_scores).toFixed(4) }}</span>
              </el-descriptions-item>
              <el-descriptions-item label="Training Time">
                <span class="score-val">{{ (result.time || 0).toFixed(1) }}s</span>
              </el-descriptions-item>
            </el-descriptions>
          </div>

          <el-empty v-else description="No analysis performed. Select data and configure your pipeline." />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useChemaiStore } from '../store/chemai'
import { getModels } from '../api/client'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { Setting, Operation, TrendingUp, Loading, VideoPlay } from '@element-plus/icons-vue'
import DescriptorSelector from '../components/ml/DescriptorSelector.vue'
import MonotonicConstraints from '../components/ml/MonotonicConstraints.vue'
import MixtureInput from '../components/ml/MixtureInput.vue'

const store = useChemaiStore()
const router = useRouter()
const progress = ref(0)
const result = ref<any>(null)
const availableModels = ref<any[]>([])
const loadingModels = ref(false)
const currentStep = ref('Initializing...')

onMounted(async () => {
  await store.initialize()
  await store.fetchPipelineConfig()
  
  loadingModels.value = true
  try {
    availableModels.value = await getModels(store.taskType)
  } catch (e) {
    console.error('Failed to load models', e)
  } finally {
    loadingModels.value = false
  }
})

const handleRun = async () => {
  result.value = null
  progress.value = 0
  currentStep.value = 'Extracting Molecular Descriptors...'
  
  const res = await store.runAnalysis(store.pipelineConfig)
  
  if (res) {
    // Fake progress animation for premium feel
    const timer = setInterval(() => {
      progress.value += 5
      if (progress.value >= 40) currentStep.value = 'Training Cross-Validation Folds...'
      if (progress.value >= 70) currentStep.value = 'Optimizing Hyperparameters...'
      if (progress.value >= 90) currentStep.value = 'Finalizing Report...'
      
      if (progress.value >= 100) {
        clearInterval(timer)
        result.value = res
        ElMessage.success('Analysis completed successfully!')
      }
    }, 150)
  }
}

const viewResults = () => {
  router.push('/results')
}

const average = (arr: number[]) => arr && arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0
const stdDev = (arr: number[]) => {
  if (!arr || arr.length < 2) return 0
  const avg = average(arr)
  return Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / (arr.length - 1))
}
</script>

<style scoped>
.ml-container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 30px; }
.subtitle { color: #909399; margin-top: 4px; font-size: 0.95rem; }
.config-tabs, .status-card { min-height: 650px; border-radius: 12px; }
.tab-label { display: flex; align-items: center; gap: 8px; font-weight: 600; }
.full-width { width: 100%; }
.model-selection { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; }
.loading-models { padding: 40px; text-align: center; color: #909399; }
.running-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 450px; }
.status-text { margin-top: 30px; color: #409eff; font-weight: bold; font-size: 1.1rem; }
.current-step { margin-top: 10px; color: #909399; font-size: 0.9rem; font-style: italic; }
.result-summary { padding: 20px; }
.score-summary { margin-top: 30px; }
.score-val { font-family: 'JetBrains Mono', monospace; font-weight: bold; color: #409eff; }
.shadow { box-shadow: 0 8px 24px 0 rgba(0,0,0,0.08) !important; }
.animate-fade-in { animation: fadeIn 0.8s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
</style>
