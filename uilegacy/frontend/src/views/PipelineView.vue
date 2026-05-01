<template>
  <div class="pipeline-config">
    <el-card class="shadow-lg border-0 bg-slate-900/60 backdrop-blur-md text-slate-100">
      <template #header>
        <span class="text-xl font-bold text-sky-400">⚙️ Pipeline Configuration</span>
      </template>
      
      <el-form :model="config" label-width="150px" label-position="left">
        <el-divider content-position="left">Validation</el-divider>
        <el-form-item label="CV Folds">
          <el-input-number v-model="config.cv_folds" :min="2" :max="10" />
          <div class="ml-4 text-xs text-slate-500 italic">Recommended: 5-10</div>
        </el-form-item>

        <el-divider content-position="left">Preprocessing</el-divider>
        <el-form-item label="Numerical Scaler">
          <el-select v-model="config.num_scaler" style="width: 200px;">
            <el-option label="Standard (Mean=0, Std=1)" value="standard" />
            <el-option label="Robust (Outlier focus)" value="robust" />
            <el-option label="MinMax (Range 0-1)" value="minmax" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="Feature Selector">
          <el-select v-model="config.feature_selector" style="width: 200px;">
            <el-option label="None" value="none" />
            <el-option label="Lasso (L1 Penalty)" value="select_from_model_lasso" />
            <el-option label="RF (Tree importance)" value="select_from_model_rf" />
          </el-select>
        </el-form-item>

        <el-divider content-position="left">Algorithms</el-divider>
        <el-form-item label="Models">
          <el-checkbox-group v-model="config.selected_models">
            <el-checkbox label="RandomForest" border />
            <el-checkbox label="XGBoost" border />
            <el-checkbox label="LightGBM" border />
          </el-checkbox-group>
        </el-form-item>

        <el-form-item class="mt-10">
          <el-button 
            type="primary" 
            @click="execute" 
            :loading="running"
            class="w-full h-12 text-lg font-bold bg-sky-600 hover:bg-sky-500 border-0"
          >
            {{ running ? 'Analyzing...' : '🚀 Execute Analysis' }}
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { ElMessage, ElNotification } from 'element-plus';
import { runPipeline } from '../api/client';
import { useRouter } from 'vue-router';

const router = useRouter();
const config = ref({
  cv_folds: 5,
  num_scaler: 'standard',
  feature_selector: 'none',
  selected_models: ['RandomForest', 'XGBoost']
});
const running = ref(false);

async function execute() {
  running.value = true;
  try {
    const res = await runPipeline(config.value);
    ElNotification({
      title: 'Analysis Complete',
      message: `Best Model: ${res.best_model} | Score: ${res.score}`,
      type: 'success',
      duration: 5000
    });
    router.push('/results');
  } catch (err: any) {
    ElMessage.error(err.response?.data?.detail || 'Pipeline execution failed');
  } finally {
    running.value = false;
  }
}
</script>

<style scoped>
.pipeline-config {
  max-width: 800px;
  margin: 0 auto;
}
:deep(.el-divider__text) {
    background-color: transparent;
    color: #94a3b8;
    font-weight: bold;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.1em;
}
:deep(.el-form-item__label) {
    color: #cbd5e1;
}
</style>
