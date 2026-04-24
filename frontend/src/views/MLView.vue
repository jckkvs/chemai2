<template>
  <div class="ml-container">
    <div class="header">
      <h2>機械学習パイプライン設定</h2>
      <el-button type="success" size="large" :loading="isRunning" @click="runML" icon="VideoPlay">
        実行開始
      </el-button>
    </div>

    <el-row :gutter="20">
      <!-- 設定パネル -->
      <el-col :span="10">
        <el-card class="config-card">
          <template #header>
            <div class="card-header">
              <span>⚙️ パラメータ設定</span>
            </div>
          </template>
          
          <el-form :model="config" label-width="140px" label-position="left">
            <el-form-item label="CV分割数">
              <el-input-number v-model="config.cv_folds" :min="2" :max="10" />
            </el-form-item>
            
            <el-form-item label="数値スケーラー">
              <el-select v-model="config.num_scaler" class="full-width">
                <el-option label="Standard" value="standard" />
                <el-option label="MinMax" value="minmax" />
                <el-option label="Robust" value="robust" />
                <el-option label="None" value="none" />
              </el-select>
            </el-form-item>

            <el-form-item label="数値補完">
              <el-select v-model="config.num_imputer" class="full-width">
                <el-option label="Median" value="median" />
                <el-option label="Mean" value="mean" />
                <el-option label="Zero" value="constant" />
              </el-select>
            </el-form-item>

            <el-form-item label="カテゴリ追加">
              <el-select v-model="config.cat_encoder" class="full-width">
                <el-option label="OneHot" value="onehot" />
                <el-option label="Label" value="label" />
                <el-option label="Target" value="target" />
              </el-select>
            </el-form-item>

            <el-form-item label="多項式特徴量">
              <el-switch v-model="config.do_polynomial" />
              <el-input-number v-if="config.do_polynomial" v-model="config.poly_degree" :min="2" :max="3" style="margin-left: 10px" />
            </el-form-item>

            <el-divider>アルゴリズム選択</el-divider>
            
            <el-checkbox-group v-model="config.selected_models">
              <el-checkbox label="RandomForest">RF (ランダムフォレスト)</el-checkbox>
              <el-checkbox label="XGBoost">XGB (XGBoost)</el-checkbox>
              <el-checkbox label="LightGBM">LGB (LightGBM)</el-checkbox>
              <el-checkbox label="Ridge">Ridge (リッジ回帰)</el-checkbox>
            </el-checkbox-group>
          </el-form>
        </el-card>
      </el-col>

      <!-- 進行状況 / 簡易結果 -->
      <el-col :span="14">
        <el-card class="status-card" v-if="isRunning || result">
          <template #header>
            <div class="card-header">
              <span>🚀 実行ステータス</span>
            </div>
          </template>
          
          <div v-if="isRunning" class="running-state">
            <el-progress type="circle" :percentage="progress" status="success" />
            <p>モデル訓練中...</p>
          </div>

          <div v-if="result" class="result-summary">
            <el-result
              icon="success"
              title="分析完了"
              :sub-title="`最良モデル: ${result.best_model}`"
            >
              <template #extra>
                <el-button type="primary" @click="$router.push('/results')">詳細レポートを見る</el-button>
              </template>
            </el-result>

            <el-descriptions title="スコアサマリー" :column="1" border>
              <el-descriptions-item label="検証スコア (R2/Acc)">{{ result.score.toFixed(4) }}</el-descriptions-item>
              <el-descriptions-item label="CV平均スコア">{{ result.cv_scores.reduce((a,b)=>a+b,0)/result.cv_scores.length }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'

const router = useRouter()
const isRunning = ref(false)
const progress = ref(0)
const result = ref<any>(null)

const config = ref({
  cv_folds: 5,
  num_scaler: 'standard',
  num_imputer: 'median',
  cat_encoder: 'onehot',
  selected_models: ['RandomForest', 'XGBoost'],
  do_polynomial: false,
  poly_degree: 2
})

const API_BASE = 'http://localhost:8000/api'
const sessionId = localStorage.getItem('chemai_session_id') || 'default_session'

const runML = async () => {
  isRunning.value = true
  result.value = null
  progress.value = 10
  
  try {
    const res = await axios.post(`${API_BASE}/pipeline/run`, {
      session_id: sessionId,
      ...config.value
    })
    
    // Simulate progress
    const timer = setInterval(() => {
      progress.value += 15
      if (progress.value >= 100) {
        clearInterval(timer)
        result.value = res.data
        isRunning.value = false
        ElMessage.success('分析が完了しました')
      }
    }, 500)
    
  } catch (e: any) {
    ElMessage.error('実行に失敗しました: ' + (e.response?.data?.detail || e.message))
    isRunning.value = false
  }
}
</script>

<style scoped>
.ml-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.config-card, .status-card { min-height: 500px; }
.full-width { width: 100%; }
.running-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; }
.result-summary { padding: 20px; }
.el-checkbox { margin-bottom: 10px; width: 180px; }
</style>
