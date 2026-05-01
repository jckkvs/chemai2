<!-- frontend/src/views/DataUploadView.vue -->
<template>
  <div class="data-upload-container">
    <div class="header-section">
      <h2>📂 Data Management</h2>
      <p class="subtitle">Upload your CSV/Excel files or select from standard chemical benchmarks</p>
    </div>

    <el-tabs type="border-card" class="upload-tabs shadow">
      <!-- Custom Upload Tab -->
      <el-tab-pane>
        <template #label>
          <span class="tab-label"><el-icon><Upload /></el-icon> Upload Custom Data</span>
        </template>
        
        <div 
          class="upload-zone" 
          @dragover.prevent 
          @drop="handleDrop"
          :class="{ 'dragging': isDragging }"
        >
          <input type="file" ref="fileInput" @change="handleFileSelect" accept=".csv,.xlsx,.xls" hidden />
          <el-button type="primary" size="large" @click="$refs.fileInput.click()" icon="Upload">
            Select CSV/Excel File
          </el-button>
          <p class="hint">or drag & drop here</p>
        </div>
      </el-tab-pane>

      <!-- Benchmark Tab -->
      <el-tab-pane>
        <template #label>
          <span class="tab-label"><el-icon><Collection /></el-icon> Benchmark Datasets</span>
        </template>
        <BenchmarkSelector />
      </el-tab-pane>
    </el-tabs>

    <!-- Loading -->
    <div v-if="store.isLoading" class="loading-overlay">
      <el-icon class="is-loading"><Loading /></el-icon>
      <p>Processing Data...</p>
    </div>

    <!-- Error -->
    <div v-if="store.error" class="error-alert">
      <el-alert :title="store.error" type="error" show-icon />
    </div>

    <!-- Results Display -->
    <div v-if="store.hasData" class="data-result mt-8 animate-fade-in">
      <el-card class="status-card shadow">
        <template #header>
          <div class="card-header">
            <span>✅ {{ store.filename }}</span>
            <el-tag type="success">{{ store.rows }} rows × {{ store.cols }} cols</el-tag>
          </div>
        </template>

        <!-- Metrics Cards -->
        <el-row :gutter="20" class="metrics-row">
          <el-col :span="6">
            <div class="metric-card">
              <span class="val">{{ store.metrics.rows?.toLocaleString() || store.rows }}</span>
              <span class="lbl">Rows</span>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="metric-card">
              <span class="val">{{ store.cols }}</span>
              <span class="lbl">Columns</span>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="metric-card">
              <span class="val">{{ (store.metrics.missing_rate * 100).toFixed(1) }}%</span>
              <span class="lbl">Missing Rate</span>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="metric-card">
              <span class="val">{{ store.metrics.numeric_cols }}</span>
              <span class="lbl">Numeric Cols</span>
            </div>
          </el-col>
        </el-row>

        <!-- Preview Table -->
        <div class="preview-section">
          <h3>📊 Data Preview</h3>
          <el-table :data="store.preview" border stripe size="small" height="300">
            <el-table-column v-for="col in store.columns" :key="col" :prop="col" :label="col" min-width="150">
              <template #header>
                <div class="col-header">
                  {{ col }}
                  <el-tag v-if="col === store.targetCol" size="small" effect="dark" class="target-badge">Target</el-tag>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <!-- Task Configuration -->
        <div class="config-section">
          <h3>🎯 Analytical Configuration</h3>
          <el-form inline class="config-form">
            <el-form-item label="Target Column">
              <el-select v-model="store.targetCol" @change="applyTarget" placeholder="Select target" style="width: 250px">
                <el-option v-for="c in store.columns" :value="c" :key="c">{{ c }}</el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="Task Type">
              <el-radio-group v-model="store.taskType" @change="applyTask">
                <el-radio-button label="regression">Regression</el-radio-button>
                <el-radio-button label="classification">Classification</el-radio-button>
              </el-radio-group>
            </el-form-item>
          </el-form>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useChemaiStore } from '../store/chemai'
import BenchmarkSelector from '../components/ml/BenchmarkSelector.vue'
import { Upload, Collection, Loading } from '@element-plus/icons-vue'

const store = useChemaiStore()
const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)

onMounted(async () => {
  await store.initialize()
})

const handleFileSelect = (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) store.uploadFile(file)
}

const handleDrop = (e: DragEvent) => {
  isDragging.value = false
  const file = e.dataTransfer?.files[0]
  if (file) store.uploadFile(file)
}

const applyTarget = () => {
  store.updateConfig(store.targetCol)
}

const applyTask = () => {
  store.updateConfig(store.targetCol, store.taskType)
}
</script>

<style scoped>
.data-upload-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header-section { margin-bottom: 30px; }
.subtitle { color: #909399; margin-top: 5px; }

.upload-zone {
  border: 2px dashed #409eff; padding: 60px; text-align: center;
  margin-bottom: 30px; border-radius: 12px; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  background: rgba(64, 158, 255, 0.02);
}
.upload-zone.dragging { background: rgba(64, 158, 255, 0.1); border-color: #66b1ff; transform: scale(1.01); }
.hint { color: #909399; margin-top: 15px; font-size: 0.9rem; }

.loading-overlay { text-align: center; padding: 40px; }
.loading-overlay .el-icon { font-size: 40px; color: #409eff; margin-bottom: 10px; }

.status-card { border-radius: 12px; }
.card-header { display: flex; justify-content: space-between; align-items: center; font-weight: bold; }

.metrics-row { margin-bottom: 30px; }
.metric-card { 
  background: #f5f7fa; padding: 20px; border-radius: 8px; text-align: center;
  transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .val { display: block; font-size: 1.8rem; font-weight: bold; color: #409eff; }
.metric-card .lbl { color: #909399; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }

.preview-section { margin-bottom: 30px; }
.preview-section h3 { margin-bottom: 15px; font-size: 1.1rem; color: #303133; }
.col-header { display: flex; align-items: center; gap: 8px; }
.target-badge { font-weight: normal; }

.config-section h3 { margin-bottom: 20px; font-size: 1.1rem; color: #303133; }
.config-form { background: #fcfdfe; padding: 20px; border-radius: 8px; border: 1px solid #ebeef5; }

.shadow { box-shadow: 0 4px 12px 0 rgba(0,0,0,0.05) !important; }
</style>
