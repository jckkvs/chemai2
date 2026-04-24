<template>
  <div class="eda-container">
    <div class="header">
      <h2>探索的データ分析 (EDA)</h2>
      <el-button type="primary" :loading="isLoading" @click="fetchAllEDA" icon="Refresh">
        分析を更新
      </el-button>
    </div>

    <el-tabs v-model="activeTab" class="eda-tabs">
      <!-- 統計サマリー -->
      <el-tab-pane label="📋 統計サマリー" name="stats">
        <el-table :data="statsData" border stripe style="width: 100%" height="500">
          <el-table-column prop="column" label="列名" width="180" fixed />
          <el-table-column prop="count" label="データ数" width="100" />
          <el-table-column prop="mean" label="平均値" width="120">
            <template #default="{ row }">{{ formatNum(row.mean) }}</template>
          </el-table-column>
          <el-table-column prop="std" label="標準偏差" width="120">
            <template #default="{ row }">{{ formatNum(row.std) }}</template>
          </el-table-column>
          <el-table-column prop="min" label="最小値" width="120">
            <template #default="{ row }">{{ formatNum(row.min) }}</template>
          </el-table-column>
          <el-table-column prop="max" label="最大値" width="120">
            <template #default="{ row }">{{ formatNum(row.max) }}</template>
          </el-table-column>
          <el-table-column prop="missing_rate" label="欠損率(%)" width="100">
            <template #default="{ row }">
              <el-tag :type="row.missing_rate > 5 ? 'danger' : 'success'">
                {{ row.missing_rate }}%
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="skew" label="歪度" width="100" />
          <el-table-column prop="kurtosis" label="尖度" width="100" />
        </el-table>
      </el-tab-pane>

      <!-- 相関行列 -->
      <el-tab-pane label="🔥 相関行列" name="corr">
        <div class="chart-container">
          <v-chart class="chart" :option="corrOption" autoresize />
        </div>
      </el-tab-pane>

      <!-- 次元削減 -->
      <el-tab-pane label="🌀 次元削減 (PCA/t-SNE)" name="dim">
        <el-row :gutter="20">
          <el-col :span="12">
            <div class="chart-container small">
              <h3>PCA (主成分分析)</h3>
              <v-chart class="chart" :option="pcaOption" autoresize />
            </div>
          </el-col>
          <el-col :span="12">
            <div class="chart-container small">
              <h3>t-SNE</h3>
              <v-chart class="chart" :option="tsneOption" autoresize />
            </div>
          </el-col>
        </el-row>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { HeatmapChart, ScatterChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  TitleComponent,
  LegendComponent
} from 'echarts/components'
import VChart from 'vue-echarts'

use([
  CanvasRenderer,
  HeatmapChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  TitleComponent,
  LegendComponent
])

const activeTab = ref('stats')
const isLoading = ref(false)
const statsData = ref([])
const corrData = ref({ columns: [], matrix: [] })
const dimData = ref({ pca: [], tsne: [], explained_variance: [] })

const API_BASE = 'http://localhost:8000/api'
const sessionId = localStorage.getItem('chemai_session_id') || 'default_session'

const fetchAllEDA = async () => {
  isLoading.ref = true
  try {
    const [stats, corr, dim] = await Promise.all([
      axios.get(`${API_BASE}/eda/stats`, { params: { session_id: sessionId } }),
      axios.get(`${API_BASE}/eda/correlation`, { params: { session_id: sessionId } }),
      axios.get(`${API_BASE}/eda/dim_reduction`, { params: { session_id: sessionId } })
    ])
    statsData.value = stats.data.stats
    corrData.value = corr.data
    dimData.value = dim.data
  } catch (e: any) {
    ElMessage.error('EDAデータの取得に失敗しました')
  } finally {
    isLoading.value = false
  }
}

onMounted(fetchAllEDA)

const formatNum = (v: number | null) => v !== null ? v.toFixed(4) : '-'

// ECharts Options
const corrOption = computed(() => ({
  tooltip: { position: 'top' },
  grid: { height: '80%', top: '10%' },
  xAxis: { type: 'category', data: corrData.value.columns, splitArea: { show: true } },
  yAxis: { type: 'category', data: corrData.value.columns, splitArea: { show: true } },
  visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: '5%', color: ['#d94e5d', '#eac736', '#50a3ba'] },
  series: [{
    name: 'Correlation',
    type: 'heatmap',
    data: corrData.value.matrix.flatMap((row, i) => row.map((val, j) => [j, i, val.toFixed(2)])),
    label: { show: true }
  }]
}))

const pcaOption = computed(() => ({
  tooltip: {},
  xAxis: { splitLine: { show: false } },
  yAxis: { splitLine: { show: false } },
  series: [{
    type: 'scatter',
    data: dimData.value.pca.map(p => [p.pc1, p.pc2])
  }]
}))

const tsneOption = computed(() => ({
  tooltip: {},
  xAxis: { splitLine: { show: false } },
  yAxis: { splitLine: { show: false } },
  series: [{
    type: 'scatter',
    data: dimData.value.tsne.map(p => [p.v1, p.v2])
  }]
}))
</script>

<style scoped>
.eda-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.eda-tabs { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1); }
.chart-container { height: 600px; width: 100%; display: flex; flex-direction: column; align-items: center; }
.chart-container.small { height: 400px; }
.chart { width: 100%; height: 100%; }
h3 { margin-bottom: 10px; color: #606266; }
</style>
