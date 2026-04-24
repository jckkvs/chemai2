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
    >
      <input type="file" ref="fileInput" @change="handleFileSelect" accept=".csv,.xlsx,.xls" hidden />
      <button @click="$refs.fileInput.click()" class="upload-btn">
        Select CSV/Excel File
      </button>
      <p class="hint">or drag & drop here</p>
    </div>

    <!-- Loading -->
    <div v-if="store.isLoading" class="loading">
      <div class="spinner"></div>
      <p>Uploading...</p>
    </div>

    <!-- Error -->
    <div v-if="store.error" class="error-msg">
      ❌ {{ store.error }}
    </div>

    <!-- Results Display (migrated from _render_data_load) -->
    <div v-if="store.hasData" class="data-result">
      <div class="status-bar success">
        ✅ {{ store.filename }} ({{ store.rows }} rows × {{ store.cols }} cols)
      </div>

      <!-- Metrics Cards (migrated from _update_metrics) -->
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

      <!-- Preview Table (migrated from _show_preview) -->
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

      <!-- Task Configuration (migrated from _on_target_change) -->
      <div class="config-row">
        <label>Target Column:</label>
        <select v-model="selectedTarget" @change="applyTarget">
          <option v-for="c in store.columns" :value="c" :key="c">{{ c }}</option>
        </select>
        <label>Task Type:</label>
        <select v-model="store.taskType" @change="applyTask">
          <option value="regression">Regression</option>
          <option value="classification">Classification</option>
        </select>
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
  if (!store.sessionId) await store.initSession()
  if (store.targetCol) selectedTarget.value = store.targetCol
})

watch(() => store.targetCol, (newVal) => {
    selectedTarget.value = newVal
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
.data-upload-container { padding: 20px; max-width: 1000px; margin: 0 auto; }
.upload-zone {
  border: 2px dashed #42b983; padding: 40px; text-align: center;
  margin-bottom: 20px; border-radius: 8px; transition: background 0.2s;
}
.upload-zone.dragging { background: #f0fff4; border-color: #2f855a; }
.upload-btn { padding: 10px 20px; background: #42b983; color: white; border: none; border-radius: 4px; cursor: pointer; }
.hint { color: #718096; margin-top: 10px; }
.loading { text-align: center; padding: 20px; }
.spinner { border: 4px solid #f3f3f3; border-top: 4px solid #42b983; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.error-msg { color: #e53e3e; background: #fff5f5; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
.status-bar { padding: 10px; background: #f0fff4; border-left: 4px solid #48bb78; margin-bottom: 15px; }
.metrics-row { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card { background: #f7fafc; padding: 15px; border-radius: 6px; min-width: 100px; text-align: center; }
.metric-card .val { display: block; font-size: 1.5rem; font-weight: bold; color: #2b6cb0; }
.metric-card .lbl { color: #718096; font-size: 0.85rem; }
.preview-table-wrapper { overflow-x: auto; margin-bottom: 20px; }
.preview-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.preview-table th, .preview-table td { border: 1px solid #e2e8f0; padding: 8px; text-align: left; }
.preview-table th { background: #edf2f7; }
.target-badge { background: #ebf8ff; color: #2b6cb0; font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; margin-left: 6px; }
.config-row { display: flex; gap: 15px; align-items: center; padding: 15px; background: #f7fafc; border-radius: 6px; }
select { padding: 6px; border-radius: 4px; border: 1px solid #cbd5e0; }
</style>
