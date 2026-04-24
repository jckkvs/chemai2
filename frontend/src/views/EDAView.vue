<template>
  <div class="eda-view">
    <el-card>
      <template #header>
        <div class="card-header">
          <el-icon><DataAnalysis /></el-icon>
          <span>EDA・可視化</span>
        </div>
      </template>
      <div v-if="loading" class="loading-state">
        <el-skeleton :rows="10" animated />
      </div>
      <div v-else-if="summary" class="summary-content">
        <el-row :gutter="20">
          <el-col :span="8">
            <el-statistic title="行数" :value="summary.shape[0]" />
          </el-col>
          <el-col :span="8">
            <el-statistic title="列数" :value="summary.shape[1]" />
          </el-col>
        </el-row>
        <el-divider>欠損値状況</el-divider>
        <div v-for="(count, col) in summary.null_counts" :key="col" class="null-item">
            <span v-if="count > 0">{{ col }}: {{ count }}</span>
        </div>
      </div>
      <div v-else class="empty-state">
        <el-empty description="データがロードされていません" />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { DataAnalysis } from '@element-plus/icons-vue'
import axios from 'axios'

const summary = ref<any>(null)
const loading = ref(false)

const fetchSummary = async () => {
  loading.value = true
  try {
    const response = await axios.get('http://localhost:8000/api/eda/summary')
    summary.value = response.data
  } catch (error) {
    console.error(error)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchSummary()
})
</script>
