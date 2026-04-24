<template>
  <div class="data-upload-container">
    <h2>📂 データ管理</h2>

    <!-- アップロードゾーン -->
    <div 
      class="upload-zone" 
      @dragover.prevent 
      @drop="handleDrop"
      :class="{ 'dragging': isDragging }"
      @click="fileInput?.click()"
    >
      <input type="file" ref="fileInput" @change="handleFileSelect" accept=".csv,.xlsx,.xls" hidden />
      <button class="upload-btn">
        CSV / Excel を選択
      </button>
      <p class="hint">またはここにドラッグ＆ドロップ</p>
    </div>

    <!-- ローディング -->
    <div v-if="store.isLoading" class="loading">
      <div class="spinner"></div>
      <p>読み込み中...</p>
    </div>

    <!-- エラー -->
    <div v-if="store.error" class="error-msg">
      ❌ {{ store.error }}
    </div>

    <!-- 結果表示（_render_data_load 移植） -->
    <div v-if="store.hasData" class="data-result animate-fade-in">
      <div class="status-bar success">
        ✅ {{ store.filename }} ({{ store.rows }}行 × {{ store.cols }}列)
      </div>

      <!-- メトリクスカード (_update_metrics 移植) -->
      <div class="metrics-row">
        <div class="metric-card">
          <span class="val">{{ store.metrics.rows?.toLocaleString() || store.rows }}</span>
          <span class="lbl">行数</span>
        </div>
        <div class="metric-card">
          <span class="val">{{ store.cols }}</span>
          <span class="lbl">列数</span>
        </div>
        <div class="metric-card">
          <span class="val">{{ (store.metrics.missing_rate * 100).toFixed(1) }}%</span>
          <span class="lbl">欠損率</span>
        </div>
        <div class="metric-card">
          <span class="val">{{ store.metrics.numeric_cols }}</span>
          <span class="lbl">数値列</span>
        </div>
      </div>

      <!-- プレビューテーブル (_show_preview 移植) -->
      <div class="preview-table-wrapper">
        <table class="preview-table">
          <thead>
            <tr>
              <th v-for="col in Object.keys(store.preview[0] || {})" :key="col">
                {{ col }}
                <span v-if="col === store.targetCol" class="target-badge">目的変数</span>
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

      <!-- タスク設定 (_on_target_change 移植) -->
      <div class="config-row">
        <div class="config-item">
          <label>目的変数:</label>
          <select v-model="selectedTarget" @change="applyTarget">
            <option v-for="c in store.columns" :value="c" :key="c">{{ c }}</option>
          </select>
        </div>
        <div class="config-item">
          <label>タスク:</label>
          <select v-model="store.taskType" @change="applyTask">
            <option value="regression">回帰</option>
            <option value="classification">分類</option>
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
  if (!store.sessionId) await store.initSession()
  if (store.targetCol) {
      selectedTarget.value = store.targetCol
  }
})

// Update local select when store changes (e.g. after upload)
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
.data-upload-container { padding: 40px; max-width: 1100px; margin: 0 auto; font-family: 'Inter', sans-serif; }
h2 { margin-bottom: 30px; font-weight: 800; color: #2d3748; }

.upload-zone {
  border: 2px dashed #48bb78; 
  padding: 60px; 
  text-align: center;
  margin-bottom: 30px; 
  border-radius: 16px; 
  transition: all 0.3s ease;
  background-color: #f0fff4;
  cursor: pointer;
}
.upload-zone:hover, .upload-zone.dragging { 
  background: #c6f6d5; 
  border-color: #2f855a; 
  transform: translateY(-2px);
}
.upload-btn { 
    padding: 12px 28px; 
    background: #38a169; 
    color: white; 
    border: none; 
    border-radius: 8px; 
    cursor: pointer; 
    font-weight: 700;
    font-size: 1rem;
}
.hint { color: #4a5568; margin-top: 15px; font-size: 0.95rem; }

.loading { text-align: center; padding: 40px; color: #38a169; }
.spinner { border: 4px solid #e2e8f0; border-top: 4px solid #38a169; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

.error-msg { color: #c53030; background: #fff5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #feb2b2; }

.data-result { background: white; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); padding: 25px; border: 1px solid #e2e8f0; }
.status-bar { padding: 15px; background: #f0fff4; border-left: 5px solid #48bb78; margin-bottom: 25px; font-weight: 600; border-radius: 4px; }

.metrics-row { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }
.metric-card { 
    background: #f7fafc; 
    padding: 20px; 
    border-radius: 12px; 
    flex: 1;
    min-width: 140px; 
    text-align: center;
    border: 1px solid #edf2f7;
}
.metric-card .val { display: block; font-size: 2rem; font-weight: 900; color: #2c5282; }
.metric-card .lbl { color: #718096; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

.preview-table-wrapper { overflow-x: auto; margin-bottom: 30px; border-radius: 8px; border: 1px solid #e2e8f0; }
.preview-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.preview-table th, .preview-table td { border: 1px solid #edf2f7; padding: 12px; text-align: left; }
.preview-table th { background: #f8fafc; color: #4a5568; font-weight: 700; }
.target-badge { background: #ebf8ff; color: #2b6cb0; font-size: 0.7rem; padding: 3px 8px; border-radius: 9999px; margin-left: 8px; border: 1px solid #bee3f8; }

.config-row { display: flex; gap: 30px; align-items: center; padding: 20px; background: #f7fafc; border-radius: 12px; border: 1px solid #edf2f7; }
.config-item { display: flex; align-items: center; gap: 12px; }
.config-item label { font-weight: 700; color: #4a5568; }
select { padding: 8px 12px; border-radius: 6px; border: 1px solid #cbd5e0; background: white; font-weight: 500; }

.animate-fade-in { animation: fadeIn 0.4s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
