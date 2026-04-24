<template>
  <div class="data-upload-component">
    <div class="card shadow-xl border border-slate-700 bg-slate-900/40 rounded-2xl overflow-hidden backdrop-blur-md">
      <!-- Header -->
      <div class="px-6 py-4 bg-slate-800/60 border-b border-slate-700 flex items-center justify-between">
        <h3 class="text-lg font-bold text-sky-400 flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          データ読み込み
        </h3>
        <div v-if="uploading" class="text-xs text-slate-400 flex items-center gap-2">
          <div class="w-3 h-3 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
          処理中...
        </div>
      </div>

      <!-- Upload Zone -->
      <div class="p-6">
        <div 
          class="upload-zone border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer"
          :class="[isDragging ? 'border-sky-400 bg-sky-400/5' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/30']"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="handleDrop"
          @click="$refs.fileInput.click()"
        >
          <input type="file" ref="fileInput" @change="handleFileSelect" accept=".csv,.xlsx,.xls" hidden />
          <div class="flex flex-col items-center gap-4">
            <div class="p-4 bg-slate-800 rounded-full text-slate-400">
              <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><polyline points="9 15 12 12 15 15"/></svg>
            </div>
            <div>
              <p class="text-slate-200 font-semibold">CSV / Excel ファイルを選択</p>
              <p class="text-slate-500 text-sm mt-1">またはここにドラッグ＆ドロップ</p>
            </div>
          </div>
        </div>

        <!-- Status Summary -->
        <div v-if="status" class="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-in">
          <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
            <p class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">ファイル名</p>
            <p class="text-sm font-bold text-slate-200 truncate">{{ status.filename }}</p>
          </div>
          <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
            <p class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">データ規模</p>
            <p class="text-sm font-bold text-slate-200">{{ status.rows.toLocaleString() }} 行 × {{ status.cols }} 列</p>
          </div>
          <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
            <p class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">検出された目的変数</p>
            <span class="inline-block px-2 py-0.5 bg-sky-500/20 text-sky-400 text-xs font-bold rounded border border-sky-500/30">{{ status.target_col }}</span>
          </div>
          <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
             <p class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">ステータス</p>
             <p class="text-xs text-emerald-400 font-bold flex items-center gap-1">
               <span class="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></span> Ready to Analyze
             </p>
          </div>
        </div>

        <!-- Preview Table -->
        <div v-if="status && status.preview" class="mt-8 animate-fade-in">
          <h4 class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
            データプレビュー (先頭5行)
          </h4>
          <div class="overflow-x-auto border border-slate-700 rounded-xl bg-slate-950/50">
            <table class="w-full text-left border-collapse">
              <thead>
                <tr class="bg-slate-800/80">
                  <th v-for="col in status.columns" :key="col" class="px-4 py-3 text-[11px] font-bold text-slate-400 uppercase border-b border-slate-700 min-w-[120px]">
                    {{ col }}
                    <span v-if="col === status.target_col" class="ml-1 text-[9px] bg-sky-500 text-slate-900 px-1 rounded">TARGET</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, i) in status.preview" :key="i" class="border-b border-slate-800 hover:bg-slate-800/30 transition-colors">
                  <td v-for="col in status.columns" :key="col" class="px-4 py-3 text-sm text-slate-300 font-mono">
                    {{ row[col] }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Error Toast -->
    <div v-if="error" class="fixed bottom-6 right-6 bg-rose-500 text-white px-6 py-4 rounded-xl shadow-2xl flex items-center gap-3 animate-slide-in">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span class="font-medium">{{ error }}</span>
      <button @click="error = ''" class="ml-2 hover:text-slate-200">✕</button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';

const isDragging = ref(false);
const uploading = ref(false);
const status = ref(null);
const error = ref('');
const fileInput = ref(null);

const API_BASE = 'http://localhost:8000';

const handleFileSelect = (e) => {
  const file = e.target.files[0];
  if (file) uploadFile(file);
};

const handleDrop = (e) => {
  isDragging.value = false;
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
};

const uploadFile = async (file) => {
  uploading.value = true;
  error.value = '';
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const res = await axios.post(`${API_BASE}/api/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    if (res.data.success) {
      status.value = res.data;
    } else {
      error.value = 'アップロードに失敗しました。';
    }
  } catch (err) {
    console.error(err);
    error.value = err.response?.data?.detail || 'サーバーとの通信に失敗しました。';
  } finally {
    uploading.value = false;
  }
};

onMounted(async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/data/status`);
    if (res.data.loaded) {
      // Re-fetch or simulate state if needed, but for now we wait for upload
    }
  } catch (e) {
    console.warn("API status check failed");
  }
});
</script>

<style scoped>
.animate-fade-in { animation: fadeIn 0.5s ease-out; }
.animate-slide-in { animation: slideIn 0.3s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }

.upload-zone {
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
