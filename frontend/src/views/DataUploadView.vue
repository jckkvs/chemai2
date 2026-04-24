<!-- frontend/src/views/DataUploadView.vue -->
<template>
  <div class="data-upload-container">
    <h2>📂 Data Management</h2>

    <!-- Upload Zone -->
    <div 
      class="upload-zone" 
      @dragover.prevent 
      @drop="handleDrop"
      :class="{ 'dragging': isDragging }"
      @click="$refs.fileInput.click()"
    >
      <input type="file" ref="fileInput" @change="handleFileSelect" accept=".csv,.xlsx,.xls" hidden />
      <button class="upload-btn">
        Select CSV/Excel File
      </button>
      <p class="hint">or drag & drop here</p>
    </div>

    <!-- Loading -->
    <div v-if="store.isLoading" class="loading">
      <div class="spinner"></div>
      <p>Processing...</p>
    </div>

    <!-- Error -->
    <div v-if="store.error" class="error-msg">
      ❌ {{ store.error }}
    </div>

    <!-- Results Display -->
    <div v-if="store.hasData" class="data-result animate-fade-in">
      <div class="status-bar success">
        ✅ {{ store.filename }} ({{ store.rows }} rows × {{ store.cols }} cols)
      </div>

      <!-- Metrics Cards -->
      <div class="metrics-row">
        <div class="metric-card">
          <span class="val">{{ store.metrics.rows?.toLocaleString() || store.rows }}</span>
          <span class="lbl">Rows</span>
        </div>
        <div class="metric-card">
          <span class="val">{{ store.cols }}</span>
          <span class="lbl">Columns</span>
        </div>
        <div class="metric-card">
          <span class="val">{{ (store.metrics.missing_rate * 100).toFixed(1) }}%</span>
          <span class="lbl">Missing Rate</span>
        </div>
        <div class="metric-card">
          <span class="val">{{ store.metrics.numeric_cols }}</span>
          <span class="lbl">Numeric Cols</span>
        </div>
      </div>

      <!-- Preview Table -->
      <div class="preview-table-wrapper">
        <table class="preview-table">
          <thead>
            <tr>
              <th v-for="col in Object.keys(store.preview[0] || {})" :key="col">
                {{ col }}
                <span v-if="col === store.targetCol" class="target-badge">Target</span>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, idx) in store.preview" :key="idx">
              <td v-for="(val, col) in row" :key="col">
                {{ val === null ? '—' : val }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Task Configuration -->
      <div class="config-row">
        <div class="config-item">
            <label>Target Column:</label>
            <select v-model="selectedTarget" @change="applyTarget">
                <option v-for="c in store.columns" :value="c" :key="c">{{ c }}</option>
            </select>
        </div>
        <div class="config-item">
            <label>Task Type:</label>
            <select v-model="store.taskType" @change="applyTask">
                <option value="regression">Regression</option>
                <option value="classification">Classification</option>
            </select>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useChemaiStore } from '../store/chemai'

const store = useChemaiStore()
const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const selectedTarget = ref('')

onMounted(async () => {
  await store.initialize()
  if (store.targetCol) selectedTarget.value = store.targetCol
})

watch(() => store.targetCol, (val) => {
    selectedTarget.value = val
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
  store.updateConfig(selectedTarget.value)
}

const applyTask = () => {
  store.updateConfig(store.targetCol, store.taskType)
}
</script>

<style scoped>
.data-upload-container { padding: 40px; max-width: 1000px; margin: 0 auto; font-family: 'Inter', sans-serif; }
h2 { color: #2d3748; margin-bottom: 30px; font-weight: 800; }

.upload-zone {
  border: 2px dashed #42b983; padding: 60px; text-align: center;
  margin-bottom: 30px; border-radius: 12px; transition: all 0.3s;
  background-color: #f8fafc; cursor: pointer;
}
.upload-zone:hover, .upload-zone.dragging { background: #f0fff4; border-color: #2f855a; transform: translateY(-2px); }
.upload-btn { padding: 12px 24px; background: #42b983; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; }
.hint { color: #718096; margin-top: 15px; font-size: 0.9rem; }

.loading { text-align: center; padding: 40px; }
.spinner { border: 4px solid #f3f3f3; border-top: 4px solid #42b983; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

.error-msg { color: #e53e3e; background: #fff5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #fed7d7; }
.data-result { background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); padding: 24px; border: 1px solid #e2e8f0; }
.status-bar { padding: 15px; background: #f0fff4; border-left: 4px solid #48bb78; margin-bottom: 25px; font-weight: 600; }

.metrics-row { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }
.metric-card { background: #f8fafc; padding: 20px; border-radius: 8px; flex: 1; min-width: 150px; text-align: center; border: 1px solid #edf2f7; }
.metric-card .val { display: block; font-size: 1.8rem; font-weight: 800; color: #2b6cb0; }
.metric-card .lbl { color: #718096; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

.preview-table-wrapper { overflow-x: auto; margin-bottom: 30px; border-radius: 8px; border: 1px solid #e2e8f0; }
.preview-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.preview-table th, .preview-table td { border: 1px solid #edf2f7; padding: 12px; text-align: left; }
.preview-table th { background: #f1f5f9; font-weight: 700; color: #475569; }
.target-badge { background: #ebf8ff; color: #2b6cb0; font-size: 0.7rem; padding: 2px 8px; border-radius: 9999px; margin-left: 8px; border: 1px solid #bee3f8; }

.config-row { display: flex; gap: 40px; align-items: center; padding: 20px; background: #f8fafc; border-radius: 8px; border: 1px solid #edf2f7; }
.config-item { display: flex; align-items: center; gap: 12px; }
.config-item label { font-weight: 600; color: #475569; }
select { padding: 8px 12px; border-radius: 6px; border: 1px solid #cbd5e0; background: white; font-weight: 500; }

.animate-fade-in { animation: fadeIn 0.4s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
