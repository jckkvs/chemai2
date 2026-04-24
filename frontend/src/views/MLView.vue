<template>
  <div class="ml-view">
    <el-card>
      <template #header>
        <div class="card-header">
          <el-icon><DataBoard /></el-icon>
          <span>機械学習解析設定</span>
        </div>
      </template>
      <el-form :model="config" label-width="120px">
        <el-form-item label="CV Folds">
          <el-input-number v-model="config.cv_folds" :min="2" :max="10" />
        </el-form-item>
        <el-form-item label="モデル選択">
          <el-checkbox-group v-model="config.selected_models">
            <el-checkbox label="rf">Random Forest</el-checkbox>
            <el-checkbox label="xgb">XGBoost</el-checkbox>
            <el-checkbox label="lgbm">LightGBM</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="running" @click="runAnalysis">解析開始</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { DataBoard } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'
import { useRouter } from 'vue-router'

const router = useRouter()
const config = ref({
  target_col: '',
  task_type: 'regression',
  cv_folds: 5,
  selected_models: ['rf'],
  num_scaler: 'standard'
})
const running = ref(false)

const runAnalysis = async () => {
  running.value = true
  try {
    // 目的変数を取得
    const info = await axios.get('http://localhost:8000/api/data/info')
    config.value.target_col = info.data.target_col
    config.value.task_type = info.data.task_type
    
    await axios.post('http://localhost:8000/api/analysis/run', config.value)
    ElMessage.success('解析が完了しました')
    router.push('/results')
  } catch (error) {
    ElMessage.error('解析に失敗しました')
  } finally {
    running.value = false
  }
}
</script>
