<!-- frontend/src/components/ml/BenchmarkSelector.vue -->
<template>
  <div class="benchmark-selector">
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <el-card 
        v-for="bench in benchmarks" 
        :key="bench.id" 
        class="bench-card" 
        :class="{ 'active': selected === bench.id }"
        @click="selectBenchmark(bench.id)"
        shadow="hover"
      >
        <div class="bench-header">
          <span class="name">{{ bench.name }}</span>
          <el-tag size="small" :type="bench.type === 'regression' ? 'primary' : 'success'">
            {{ bench.type }}
          </el-tag>
        </div>
        <p class="description">{{ bench.description }}</p>
        <div class="bench-footer">
          <span class="metric">Target: <b>{{ bench.target }}</b></span>
          <el-button 
            type="primary" 
            size="small" 
            plain 
            :loading="loadingId === bench.id"
            @click.stop="handleLoad(bench.id)"
          >
            Load
          </el-button>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getBenchmarks, loadBenchmark } from '../../api/client'
import { useChemaiStore } from '../../store/chemai'
import { ElMessage } from 'element-plus'

const store = useChemaiStore()
const benchmarks = ref<any[]>([])
const selected = ref('')
const loadingId = ref('')

onMounted(async () => {
  try {
    benchmarks.value = await getBenchmarks()
  } catch (e) {
    console.error('Failed to fetch benchmarks', e)
  }
})

const selectBenchmark = (id: string) => {
  selected.value = id
}

const handleLoad = async (id: string) => {
  loadingId.value = id
  try {
    const res = await loadBenchmark(id)
    store.setData(res)
    ElMessage.success(`Dataset ${id} loaded successfully`)
  } catch (e) {
    ElMessage.error('Failed to load benchmark dataset')
  } finally {
    loadingId.value = ''
  }
}
</script>

<style scoped>
.bench-card {
  cursor: pointer;
  transition: all 0.3s;
  border: 1px solid transparent;
  background: rgba(255, 255, 255, 0.02);
}
.bench-card:hover {
  transform: translateY(-2px);
  border-color: var(--el-color-primary-light-5);
}
.bench-card.active {
  border-color: var(--el-color-primary);
  background: rgba(64, 158, 255, 0.05);
}
.bench-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.name { font-weight: bold; font-size: 1.1rem; color: #409eff; }
.description { font-size: 0.85rem; color: #909399; margin-bottom: 15px; height: 3em; overflow: hidden; }
.bench-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
}
.metric { font-size: 0.75rem; color: #606266; }
</style>
