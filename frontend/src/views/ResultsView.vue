<template>
  <div class="results-view max-w-6xl mx-auto space-y-6">
    <el-card v-if="results.status === 'completed'" class="bg-slate-900/60 border-0 text-slate-100 shadow-xl">
      <template #header>
        <div class="flex items-center gap-2 text-xl font-bold text-emerald-400">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          Analysis Summary
        </div>
      </template>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div class="p-6 bg-slate-800/40 rounded-2xl border border-slate-700">
              <p class="text-xs text-slate-500 uppercase font-bold tracking-widest mb-2">Champion Model</p>
              <h2 class="text-3xl font-black text-white">{{ results.best_model }}</h2>
          </div>
          <div class="p-6 bg-slate-800/40 rounded-2xl border border-slate-700">
              <p class="text-xs text-slate-500 uppercase font-bold tracking-widest mb-2">Performance Score</p>
              <h2 class="text-3xl font-black text-sky-400">{{ results.score?.toFixed(4) }}</h2>
          </div>
      </div>

      <div v-if="results.cv_scores" class="mb-10">
          <h4 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">CV Score Distribution</h4>
          <div class="flex gap-2">
              <div v-for="(score, i) in results.cv_scores" :key="i" class="flex-1">
                  <div class="h-1 bg-slate-700 rounded-full overflow-hidden">
                      <div class="h-full bg-emerald-500" :style="{ width: (score * 100) + '%' }"></div>
                  </div>
                  <div class="text-[10px] text-slate-500 mt-1 text-center">Fold {{ i+1 }}</div>
              </div>
          </div>
      </div>

      <div v-if="results.feature_importances" class="feature-table">
        <h4 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Top Feature Importances</h4>
        <el-table :data="results.feature_importances" border stripe class="rounded-xl overflow-hidden">
          <el-table-column prop="name" label="Feature Name" />
          <el-table-column prop="value" label="Relative Importance">
              <template #default="scope">
                  <div class="flex items-center gap-3">
                      <el-progress :percentage="scope.row.value * 1000" :show-text="false" class="flex-1" color="#0ea5e9" />
                      <span class="text-xs font-mono text-slate-400">{{ scope.row.value.toFixed(4) }}</span>
                  </div>
              </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>

    <el-empty v-else description="Run an analysis to generate results" class="bg-slate-900/40 rounded-3xl p-20 border border-dashed border-slate-800" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { getResults } from '../api/client';

const results = ref<any>({ status: 'pending' });

onMounted(async () => {
  try {
    const res = await getResults();
    results.value = res;
  } catch (e) {
    console.warn("No results found in session", e);
  }
});
</script>

<style scoped>
:deep(.el-table) {
    background-color: transparent;
    --el-table-bg-color: transparent;
    --el-table-tr-bg-color: transparent;
    --el-table-header-bg-color: #1e293b;
    --el-table-text-color: #e2e8f0;
    --el-table-border-color: #334155;
}
</style>
