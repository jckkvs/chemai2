<template>
  <div class="results-container">
    <div class="header">
      <h2>📈 Analytical Report</h2>
      <el-button type="primary" plain @click="fetchResults" icon="Refresh">
        Refresh Results
      </el-button>
    </div>

    <div v-if="!result" class="empty-state">
      <el-empty description="No analytical results available. Please run the ML pipeline first." />
    </div>

    <div v-else class="results-content">
      <!-- Overview Cards -->
      <el-row :gutter="20">
        <el-col :span="8">
          <el-card class="metric-card shadow">
            <div class="label">Best Model</div>
            <div class="value">{{ result.best_model }}</div>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="metric-card shadow">
            <div class="label">Validation Score (R² / Acc)</div>
            <div class="value">{{ result.score.toFixed(4) }}</div>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="metric-card shadow">
            <div class="label">Status</div>
            <div class="value success">{{ result.status.toUpperCase() }}</div>
          </el-card>
        </el-col>
      </el-row>

      <!-- Feature Importance -->
      <el-card class="chart-card mt-lg shadow">
        <template #header>
          <div class="card-header">
            <span>🔥 Feature Importance (Top 10)</span>
          </div>
        </template>
        <div class="chart-container">
          <v-chart class="chart" :option="importanceOption" autoresize />
        </div>
      </el-card>

      <!-- CV Score Distribution -->
      <el-card class="chart-card mt-lg shadow">
        <template #header>
          <div class="card-header">
            <span>📊 Cross-Validation Scores</span>
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
import { api as chemaiApi } from '../api/client'
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

const fetchResults = async () => {
  try {
    const res = await chemaiApi.get(`/results`)
    if (res.data.status === 'completed') {
      result.value = res.data
    }
  } catch (e: any) {
    ElMessage.error('Failed to retrieve analysis results')
  }
}

onMounted(fetchResults)

const importanceOption = computed(() => ({
  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
  xAxis: { type: 'value', boundaryGap: [0, 0.01] },
  yAxis: { type: 'category', data: result.value?.feature_importances?.map(f => f.name).reverse() || [] },
  series: [{
    name: 'Importance',
    type: 'bar',
    data: result.value?.feature_importances?.map(f => f.value).reverse() || [],
    itemStyle: { 
      color: '#409eff',
      borderRadius: [0, 4, 4, 0]
    }
  }]
}))

const cvOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
  xAxis: { type: 'category', data: result.value?.cv_scores?.map((_, i) => `Fold ${i+1}`) || [] },
  yAxis: { type: 'value', min: 'dataMin' },
  series: [{
    name: 'Score',
    data: result.value?.cv_scores || [],
    type: 'line',
    smooth: true,
    symbol: 'circle',
    symbolSize: 8,
    lineStyle: { color: '#67c23a', width: 3 },
    areaStyle: { 
      color: {
        type: 'linear',
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [{ offset: 0, color: 'rgba(103, 194, 58, 0.4)' }, { offset: 1, color: 'rgba(103, 194, 58, 0)' }]
      }
    }
  }]
}))
</script>

<style scoped>
.results-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
.metric-card { text-align: center; height: 130px; display: flex; flex-direction: column; justify-content: center; border-radius: 12px; }
.metric-card .label { color: #909399; font-size: 0.9rem; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { font-size: 1.8rem; font-weight: bold; color: #303133; }
.metric-card .value.success { color: #67c23a; }
.chart-card { margin-top: 24px; border-radius: 12px; }
.chart-container { height: 500px; width: 100%; }
.chart-container.small { height: 350px; }
.chart { width: 100%; height: 100%; }
.empty-state { margin-top: 120px; }
.mt-lg { margin-top: 24px; }
.shadow { box-shadow: 0 4px 12px 0 rgba(0,0,0,0.05) !important; }
</style>
