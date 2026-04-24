<template>
  <div class="results-container">
    <div class="header">
      <h2>分析結果レポート</h2>
      <el-button type="primary" plain @click="fetchResults" icon="Refresh">
        最新の結果を取得
      </el-button>
    </div>

    <div v-if="!result" class="empty-state">
      <el-empty description="実行済みの分析結果がありません" />
    </div>

    <div v-else class="results-content">
      <!-- 概要カード -->
      <el-row :gutter="20">
        <el-col :span="8">
          <el-card class="metric-card shadow">
            <div class="label">最良モデル</div>
            <div class="value">{{ result.best_model }}</div>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="metric-card shadow">
            <div class="label">検証スコア (R2 / Acc)</div>
            <div class="value">{{ result.score.toFixed(4) }}</div>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="metric-card shadow">
            <div class="label">ステータス</div>
            <div class="value success">{{ result.status }}</div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 特徴量重要度 -->
      <el-card class="chart-card q-mt-lg">
        <template #header>
          <div class="card-header">
            <span>🔥 特徴量重要度 (Top 10)</span>
          </div>
        </template>
        <div class="chart-container">
          <v-chart class="chart" :option="importanceOption" autoresize />
        </div>
      </el-card>

      <!-- CVスコア分布 -->
      <el-card class="chart-card q-mt-lg">
        <template #header>
          <div class="card-header">
            <span>📈 交差検証スコア分布</span>
          </div>
        </template>
        <div class="chart-container small">
          <v-chart class="chart" :option="cvOption" autoresize />
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent
} from 'echarts/components'
import VChart from 'vue-echarts'

use([
  CanvasRenderer,
  BarChart,
  LineChart,
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent
])

const result = ref<any>(null)
const API_BASE = 'http://localhost:8000/api'
const sessionId = localStorage.getItem('chemai_session_id') || 'default_session'

const fetchResults = async () => {
  try {
    const res = await axios.get(`${API_BASE}/results`, { params: { session_id: sessionId } })
    if (res.data.status === 'completed') {
      result.value = res.data
    }
  } catch (e: any) {
    ElMessage.error('結果の取得に失敗しました')
  }
}

onMounted(fetchResults)

const importanceOption = computed(() => ({
  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
  xAxis: { type: 'value', boundaryGap: [0, 0.01] },
  yAxis: { type: 'category', data: result.value?.feature_importances.map(f => f.name).reverse() || [] },
  series: [{
    type: 'bar',
    data: result.value?.feature_importances.map(f => f.value).reverse() || [],
    itemStyle: { color: '#409eff' }
  }]
}))

const cvOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: result.value?.cv_scores.map((_, i) => `Fold ${i+1}`) || [] },
  yAxis: { type: 'value', min: 'dataMin' },
  series: [{
    data: result.value?.cv_scores || [],
    type: 'line',
    smooth: true,
    lineStyle: { color: '#67c23a' },
    areaStyle: { color: 'rgba(103, 194, 58, 0.2)' }
  }]
}))
</script>

<style scoped>
.results-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.metric-card { text-align: center; height: 120px; display: flex; flex-direction: column; justify-content: center; border-radius: 8px; }
.metric-card .label { color: #909399; font-size: 14px; margin-bottom: 8px; }
.metric-card .value { font-size: 24px; font-weight: bold; color: #303133; }
.metric-card .value.success { color: #67c23a; }
.chart-card { margin-top: 20px; border-radius: 8px; }
.chart-container { height: 450px; width: 100%; }
.chart-container.small { height: 300px; }
.chart { width: 100%; height: 100%; }
.empty-state { margin-top: 100px; }
.q-mt-lg { margin-top: 20px; }
</style>
